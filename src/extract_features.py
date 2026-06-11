from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from .utils import ensure_dir, read_json, write_json, relpath


def _fallback_feature(image_path: Path) -> np.ndarray:
    im = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.asarray(im).astype(np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)
        hist.extend(h.astype(np.float32).tolist())
    gray = arr.mean(axis=2)
    texture = np.array([
        np.mean(np.abs(np.diff(gray, axis=0))),
        np.mean(np.abs(np.diff(gray, axis=1))),
    ], dtype=np.float32)
    return np.concatenate([means, stds, np.asarray(hist, dtype=np.float32), texture]).astype(np.float32)


def extract_video_features(video: dict, config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    crops = {c["frame_id"]: c for c in read_json(root / "metadata" / f"{video['video_id']}_masked_crops.json", [])}
    labels = read_json(root / "pseudo_labels" / f"{video['video_id']}.json", [])
    out_dir = ensure_dir(root / "features" / video["video_id"])
    rows: list[dict] = []
    for lab in labels:
        crop = crops.get(lab["frame_id"])
        if not crop:
            continue
        feature = _fallback_feature(root / crop["masked_crop_path"])
        out_path = out_dir / f"{lab['frame_id']}.npy"
        np.save(out_path, feature)
        rows.append({
            "frame_id": lab["frame_id"],
            "feature_path": relpath(out_path, root),
            "label": lab["label"],
            "is_uncertain": lab.get("is_uncertain", False),
            "reason": lab.get("reason"),
        })
    write_json(root / "metadata" / f"{video['video_id']}_features.json", rows)
    return rows


def run(video: dict, config: dict) -> list[dict]:
    return extract_video_features(video, config)
