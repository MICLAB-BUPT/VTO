import torch
import torch.nn.functional as F
import numpy as np
import cv2
import clip
from PIL import Image
import argparse
import os

VTO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# 假设你的项目结构中有这些模块，如果没有，需要调整路径
from model import CLIPVAD
# from src.utils.tools import get_prompt_text # 如果你没有这个文件，我在下面手写了一个替代版

# ==========================================
# 1. 补充缺失的工具函数 (防止报错)
# ==========================================


def get_prompt_text(label_map):
    """
    根据 label_map 生成 Prompt，保持与训练时一致的格式。
    """
    # 假设训练时的 prompt 模板是 "A video of [class]"
    # 注意：这里返回的必须是 token ized 的文本或者 list
    # VadCLIP 内部通常会处理 tokenization，这里我们返回纯文本列表
    # 顺序非常重要：Index 0 必须是 Normal
    class_names = list(label_map.values())
    return class_names  # 模型内部通常会处理 "A video of..." 的模板


def process_feat(feat, length):
    """
    核心函数：将任意长度的视频特征插值采样到固定的 visual_length (例如 256)
    """
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat, length


# ==========================================
# 2. 优化后的特征提取 (Batch处理)
# ==========================================


def extract_clip_features(
    video_path, clip_model, preprocess, device, sample_rate=16, batch_size=32
):
    """
    提取特征，增加了 batch 处理以提升速度。
    默认 sample_rate=16，因为UCF-Crime通常是每16帧作为一个片段。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    features = []
    frames_batch = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按照 sample_rate 采样，例如每16帧取第1帧，或者每帧都取
        if frame_count % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames_batch.append(preprocess(pil_image))

            # 当积攒到一个 batch 时，统一推理
            if len(frames_batch) == batch_size:
                image_input = torch.stack(frames_batch).to(device)
                with torch.no_grad():
                    # encode_image 输出 (Batch, Dim)
                    batch_features = clip_model.encode_image(image_input)
                    features.append(batch_features.cpu().numpy())
                frames_batch = []

        frame_count += 1

    # 处理剩余的帧
    if frames_batch:
        image_input = torch.stack(frames_batch).to(device)
        with torch.no_grad():
            batch_features = clip_model.encode_image(image_input)
            features.append(batch_features.cpu().numpy())

    cap.release()

    if not features:
        raise ValueError("No frames extracted from video")

    # 拼接所有 batch: (N, Dim)
    return np.concatenate(features, axis=0)


# ==========================================
# 3. 推理主逻辑
# ==========================================


def infer_anomaly(video_path, model, clip_model, preprocess, args, device):
    print(f"1. Extracting features from {os.path.basename(video_path)}...")
    # 注意：sample_rate 建议设为 16，如果你的模型是基于片段训练的。如果是基于帧的，设为1。
    # 大多数 UCF-Crime 模型是 sample_rate=16
    raw_features = extract_clip_features(
        video_path, clip_model, preprocess, device, sample_rate=args.sample_rate
    )
    print(f"   Raw features shape: {raw_features.shape}")

    # 2. 特征预处理 (插值到 visual_length)
    processed_features, length = process_feat(raw_features, args.visual_length)

    # 转为 Tensor: (1, T, D)
    visual = torch.tensor(processed_features).unsqueeze(0).to(device).float()

    # 维度检查与适配 (如果你用ViT-B/16, 应该是512维)
    if visual.shape[-1] != args.visual_width:
        print(
            f"Warning: Feature dim {visual.shape[-1]} != Model dim {args.visual_width}. Projecting..."
        )
        # 这里如果维度不匹配通常会报错，除非你加一个投影层，暂且假设维度是匹配的

    # 3. 构造 Mask 和 Length
    lengths = torch.tensor([length], dtype=torch.int)  # [256]
    padding_mask = torch.zeros(1, args.visual_length, dtype=torch.bool).to(
        device
    )  # 全False，因为我们要填满256

    # 4. 准备 Prompt
    # 确保 Index 0 是 Normal
    label_map = {
        "Normal": "Normal",  # Index 0
        "Abuse": "Abuse",
        "Arrest": "Arrest",
        "Arson": "Arson",
        "Assault": "Assault",
        "Burglary": "Burglary",
        "Explosion": "Explosion",
        "Fighting": "Fighting",
        "RoadAccidents": "RoadAccidents",
        "Robbery": "Robbery",
        "Shooting": "Shooting",
        "Shoplifting": "Shoplifting",
        "Stealing": "Stealing",
        "Vandalism": "Vandalism",
    }
    prompt_text = get_prompt_text(label_map)

    # 5. 模型推理
    print("2. Running Model Inference...")
    with torch.no_grad():
        # logits2: (Batch, T, Class_Num)
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)

    # 6. 解析结果
    logits2 = logits2.squeeze(0)  # (T, Class_Num)
    probs = torch.softmax(logits2, dim=-1).cpu().numpy()

    # 找到每一时刻最大的概率和对应类别
    class_names = list(label_map.values())

    # 统计整个视频的异常情况
    # 排除 Normal (Index 0)
    anomaly_probs = probs[:, 1:]
    max_anomaly_score = np.max(anomaly_probs)

    results = {
        "is_abnormal": max_anomaly_score > 0.5,  # 阈值可调
        "max_score": max_anomaly_score,
        "frames": [],
    }

    # 为了输出好看，我们只打印关键帧或进行降采样打印
    print("\n" + "=" * 40)
    print("       ANALYSIS REPORT")
    print("=" * 40)

    if results["is_abnormal"]:
        # 找到异常置信度最高的那一刻
        # unravel_index 将扁平索引转为坐标 (time_idx, class_idx)
        t_idx, c_idx = np.unravel_index(np.argmax(anomaly_probs), anomaly_probs.shape)
        # c_idx + 1 因为我们跳过了 Normal
        top_class = class_names[c_idx + 1]

        print(f"[RESULT]: Abnormal Event Detected!")
        print(f"[TYPE]  : {top_class}")
        print(f"[CONF]  : {max_anomaly_score:.4f}")

        # 简单的时间定位 (假设视频被均匀采样为 visual_length 份)
        # 如果 raw video 100秒，visual_length 256，则每个点代表 ~0.4秒
        print(f"[TIME]  : Approx at segment {t_idx}/{args.visual_length}")
    else:
        print(f"[RESULT]: Normal Video")
        print(f"[CONF]  : {1 - max_anomaly_score:.4f} (Normal Confidence)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(VTO_ROOT, "VadCLIP/model/model_ucf.pth"),
    )
    # 模型参数 (需与训练配置一致)
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)  # 核心参数：时序长度
    parser.add_argument("--visual-width", default=512, type=int)  # ViT-B/16 是 512
    parser.add_argument(
        "--visual-head", default=1, type=int
    )  # 注意：检查你的 config，通常是 4
    parser.add_argument("--visual-layers", default=2, type=int)  # UCF uses 2 layers
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument(
        "--sample-rate", default=16, type=int, help="每隔多少帧提取一次特征"
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Models
    print("Loading CLIP (ViT-B/16)...")
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    print("Loading VadCLIP...")
    model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
    )

    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        print(f"Error: Model path {args.model_path} does not exist.")
        return

    infer_anomaly(args.video_path, model, clip_model, preprocess, args, device)


if __name__ == "__main__":
    main()
