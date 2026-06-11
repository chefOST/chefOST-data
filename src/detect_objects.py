from __future__ import annotations

from pathlib import Path
from .utils import read_json, write_json


def _fallback_detect(image_path: Path) -> tuple[list[int], float]:
    from PIL import Image
    im = Image.open(image_path)
    w, h = im.size
    # Conservative central proposal. Replace with Grounding DINO adapter when installed.
    return [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)], 0.50


def detect_video_object(video: dict, config: dict, first_n: int = 5) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    frames = read_json(root / "metadata" / f"{video['video_id']}_frames.json", [])
    detections: list[dict] = []
    for frame in frames[:first_n]:
        image_path = root / frame["image_path"]
        bbox, conf = _fallback_detect(image_path)
        detections.append({
            "frame_id": frame["frame_id"],
            "object": video["object"],
            "confidence": conf,
            "bbox": bbox,
            "detector": "fallback_center_bbox",
        })
    write_json(root / "detections" / f"{video['video_id']}.json", detections)
    return detections


def run(video: dict, config: dict) -> list[dict]:
    return detect_video_object(video, config)
