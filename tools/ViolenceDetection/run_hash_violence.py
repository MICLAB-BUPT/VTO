# coding: utf-8
"""Run the same VadCLIP logic used by the registered ViolenceDetection tool."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/wyt/VIoTGPT")
sys.path.insert(0, str(ROOT / "tools"))
import VadCLIP.src.clip as clip
from VadCLIP.src.model import CLIPVAD

DEVICE = "cuda:0"
MODEL_PATH = ROOT / "tools/VadCLIP/models/model_ucf.pth"
CLASSES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]


def extract_features(video_path, clip_model, preprocess, sample_rate=16):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    features, batch = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_index % sample_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(preprocess(Image.fromarray(rgb)))
        if len(batch) == 32:
            tensor = torch.stack(batch).to(DEVICE)
            with torch.no_grad():
                value = clip_model.encode_image(tensor)
                value /= value.norm(dim=-1, keepdim=True)
            features.append(value.cpu().numpy())
            batch = []
    capture.release()
    if batch:
        tensor = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            value = clip_model.encode_image(tensor)
            value /= value.norm(dim=-1, keepdim=True)
        features.append(value.cpu().numpy())
    if not features:
        raise ValueError("No video frames were extracted")
    return np.concatenate(features, axis=0)


def resize_features(features, length=256):
    output = np.zeros((length, features.shape[1]), dtype=np.float32)
    ranges = np.linspace(0, len(features), length + 1, dtype=np.int32)
    for index in range(length):
        if ranges[index] != ranges[index + 1]:
            output[index] = np.mean(features[ranges[index]:ranges[index + 1]], axis=0)
        else:
            output[index] = features[min(ranges[index], len(features) - 1)]
    return output


def main():
    paths = [
        ROOT / "中移数据/检测视频/cad1b50d4cb898e53ac13232e7076d00.mp4",
        ROOT / "中移数据/检测视频/0d4847256dc8450a00cec8e9982f0c8f.mp4",
        ROOT / "中移数据/检测视频/26236371771329217978-打架视频.mp4",
        ROOT / "中移数据/检测视频/743fc281b80d-打架2.mp4",
    ]
    clip_model, preprocess = clip.load("ViT-B/16", device=DEVICE)
    model = CLIPVAD(14, 512, 256, 512, 1, 2, 8, 10, 10, DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    results = []

    for index, path in enumerate(paths, 1):
        started = time.time()
        print(f"[{index}/{len(paths)}] Processing {path.name}", flush=True)
        features = extract_features(path, clip_model, preprocess)
        visual = torch.tensor(resize_features(features)).unsqueeze(0).to(DEVICE).float()
        lengths = torch.tensor([256], dtype=torch.int)
        padding_mask = torch.zeros(1, 256, device=DEVICE)
        with torch.no_grad():
            _, _, logits = model(visual, padding_mask, CLASSES, lengths)
        probabilities = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
        anomaly_scores = 1 - probabilities[:, 0]
        peak_index = int(np.argmax(anomaly_scores))
        anomaly_score = float(anomaly_scores[peak_index])
        abnormal = probabilities[peak_index, 1:]
        class_index = int(np.argmax(abnormal))
        class_name = CLASSES[class_index + 1] if anomaly_score > 0.5 else "Normal"
        class_score = float(abnormal[class_index]) if anomaly_score > 0.5 else float(probabilities[peak_index, 0])
        record = {
            "source": str(path),
            "class_name": class_name,
            "class_score": round(class_score, 6),
            "anomaly_score": round(anomaly_score, 6),
            "threshold": 0.5,
            "sample_rate": 16,
            "sampled_frame_count": int(len(features)),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        results.append(record)
        print("RESULT", json.dumps(record, ensure_ascii=False), flush=True)

    output = ROOT / "中移数据/检测视频/打架斗殴检测结果.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
