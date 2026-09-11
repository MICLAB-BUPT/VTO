# coding: utf-8
"""Batch runner using the registered HumanFallDetection detector."""
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from HumanFallDetection.fall_core import FallDetectorMulti
from HumanFallDetection.config import (
    FPS, WINDOW_SIZE, V_THRESH, DY_THRESH, ASPECT_RATIO_THRESH,
)
from HumanFallDetection.utils.datasets import letterbox

SOURCE = ROOT / "中移数据/检测视频"


def main():
    paths = sorted(
        p for p in SOURCE.glob("*.mp4")
        if "倒地" in p.name and "检测结果" not in p.name
    )
    detector = FallDetectorMulti(
        fps=FPS,
        window_size=WINDOW_SIZE,
        v_thresh=V_THRESH,
        dy_thresh=DY_THRESH,
        ar_thresh=ASPECT_RATIO_THRESH,
    )
    records = []

    for video_index, path in enumerate(paths, 1):
        started = time.time()
        detector.trackers = {}
        detector.next_id = 1
        capture = cv2.VideoCapture(str(path))
        input_fps = capture.get(cv2.CAP_PROP_FPS) or FPS
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = capture.read()
        if not ok:
            records.append({"source": str(path), "error": "unable to read video"})
            continue
        shape = letterbox(first, 960, stride=64, auto=True)[0].shape
        output_path = SOURCE / f"{path.stem}_倒地检测结果.mp4"
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            input_fps, (shape[1], shape[0]),
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        event_frames = []
        last_event_frame = -10**9
        print(f"[{video_index}/{len(paths)}] {path.name} frames={total}", flush=True)

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            detector.process_frame(frame, writer=writer)
            is_fall_now = False
            for tracker in detector.trackers.values():
                if tracker.is_ready():
                    is_fall, _, _, _ = tracker.check_fall()
                    is_fall_now |= bool(is_fall)
            if is_fall_now and frame_index - last_event_frame > int(input_fps):
                event_frames.append(frame_index)
                last_event_frame = frame_index
            if frame_index % 250 == 0:
                print(f"  {frame_index}/{total}", flush=True)

        capture.release()
        writer.release()
        timestamps = [round(frame / input_fps, 3) for frame in event_frames]
        record = {
            "source": str(path),
            "output_video": str(output_path),
            "fall_detected": bool(event_frames),
            "fall_event_timestamps_seconds": timestamps,
            "processed_frame_count": frame_index,
            "input_fps": input_fps,
            "window_size": WINDOW_SIZE,
            "v_threshold": V_THRESH,
            "dy_threshold": DY_THRESH,
            "aspect_ratio_threshold": ASPECT_RATIO_THRESH,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        records.append(record)
        print("RESULT", json.dumps(record, ensure_ascii=False), flush=True)

    summary = SOURCE / "人员倒地检测结果.json"
    summary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", summary, flush=True)


if __name__ == "__main__":
    main()
