# coding: utf-8
"""Batch runner equivalent to the registered FSDetect tool."""
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
COMPAT = ROOT / "tools/FireSmokeDetection/yolov5_compat"
sys.path.insert(0, str(COMPAT))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

SOURCE = ROOT / "中移数据/检测视频"
WEIGHTS = ROOT / "tools/FireSmokeDetection/best.pt"
DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 640
CONFIDENCE = 0.5
IOU = 0.5
RATIO = 0.2


def main():
    paths = sorted(
        p for p in SOURCE.glob("*.mp4")
        if ("火焰" in p.name or "fire" in p.name.lower())
        and "检测结果" not in p.name
    )
    model = attempt_load(str(WEIGHTS), map_location=DEVICE)
    names = model.module.names if hasattr(model, "module") else model.names
    model(torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE))
    records = []

    for video_index, path in enumerate(paths, 1):
        started = time.time()
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = SOURCE / f"{path.stem}_火焰检测结果.mp4"
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        fire_window, smoke_window = deque(maxlen=100), deque(maxlen=100)
        fire_detected = smoke_detected = False
        fire_frames = smoke_frames = frame_index = 0
        print(f"[{video_index}/{len(paths)}] {path.name} frames={total}", flush=True)

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            resized = letterbox(frame, new_shape=IMAGE_SIZE)[0]
            tensor = resized[:, :, ::-1].transpose(2, 0, 1)
            tensor = np.ascontiguousarray(tensor)
            tensor = torch.from_numpy(tensor).to(DEVICE).float() / 255.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                prediction = model(tensor)[0]
            detections = non_max_suppression(prediction, CONFIDENCE, IOU)[0]
            has_fire = has_smoke = False
            if detections is not None and len(detections):
                detections[:, :4] = scale_coords(
                    tensor.shape[2:], detections[:, :4], frame.shape
                ).round()
                for x1, y1, x2, y2, score, class_id in detections:
                    class_name = names[int(class_id)]
                    has_fire |= class_name == "fire"
                    has_smoke |= class_name == "smoke"
                    color = (0, 0, 255) if class_name == "fire" else (128, 128, 128)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(
                        frame, f"{class_name} {float(score):.2f}",
                        (int(x1), max(24, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
                    )
            fire_frames += int(has_fire)
            smoke_frames += int(has_smoke)
            fire_window.append(int(has_fire))
            smoke_window.append(int(has_smoke))
            if len(fire_window) == 100 and sum(fire_window) / 100 > RATIO:
                fire_detected = True
            if len(smoke_window) == 100 and sum(smoke_window) / 100 > RATIO:
                smoke_detected = True
            writer.write(frame)
            if frame_index % 500 == 0:
                print(f"  {frame_index}/{total}", flush=True)

        capture.release()
        writer.release()
        if frame_index < 100 and frame_index:
            fire_detected = fire_frames / frame_index > RATIO
            smoke_detected = smoke_frames / frame_index > RATIO
        if fire_detected and smoke_detected:
            result = "fire_and_smoke"
        elif fire_detected:
            result = "fire"
        elif smoke_detected:
            result = "smoke"
        else:
            result = "normal"
        record = {
            "source": str(path),
            "output_video": str(output_path),
            "result": result,
            "fire_frame_count": fire_frames,
            "smoke_frame_count": smoke_frames,
            "processed_frame_count": frame_index,
            "confidence_threshold": CONFIDENCE,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        records.append(record)
        print("RESULT", json.dumps(record, ensure_ascii=False), flush=True)

    summary_path = SOURCE / "火焰烟雾检测结果.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", summary_path, flush=True)


if __name__ == "__main__":
    main()
