from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw
from .utils import ensure_dir, read_json, write_json, relpath


def _bbox_area(b: list[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def track_and_mask(video: dict, config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    frames = read_json(root / "metadata" / f"{video['video_id']}_frames.json", [])
    detections = read_json(root / "detections" / f"{video['video_id']}.json", [])
    if not frames:
        return []
    if not detections:
        raise RuntimeError(f"No detections found for {video['video_id']}")

    det = max(detections, key=lambda d: d.get("confidence", 0.0))
    bbox = [int(x) for x in det["bbox"]]
    conf = float(det.get("confidence", 0.0))
    out_dir = ensure_dir(root / "masks" / video["video_id"])
    mask_meta: list[dict] = []

    for frame in frames:
        im = Image.open(root / frame["image_path"])
        mask = Image.new("L", im.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)
        mask_path = out_dir / f"{frame['frame_id']}.png"
        mask.save(mask_path)
        mask_meta.append({
            "frame_id": frame["frame_id"],
            "mask_path": relpath(mask_path, root),
            "bbox": bbox,
            "area": _bbox_area(bbox),
            "confidence": conf,
            "tracker": "fallback_static_bbox_mask",
        })

    write_json(root / "metadata" / f"{video['video_id']}_masks.json", mask_meta)
    return mask_meta


def run(video: dict, config: dict) -> list[dict]:
    return track_and_mask(video, config)
