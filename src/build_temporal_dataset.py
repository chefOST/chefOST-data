from __future__ import annotations

from pathlib import Path
from .utils import read_jsonl, write_jsonl


def build_temporal_dataset(config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    length = int(config.get("sequence", {}).get("length", 16))
    stride = int(config.get("sequence", {}).get("stride", 4))
    rows = read_jsonl(root / "frame_dataset.jsonl")
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], []).append(row)

    seqs: list[dict] = []
    for vid, items in by_video.items():
        items.sort(key=lambda r: r["timestamp"])
        if len(items) < length:
            continue
        for start in range(0, len(items) - length + 1, stride):
            window = items[start:start + length]
            seqs.append({
                "video_id": vid,
                "object": window[0]["object"],
                "task": window[0]["task"],
                "start_time": window[0]["timestamp"],
                "end_time": window[-1]["timestamp"],
                "feature_paths": [r["feature_path"] for r in window],
                "labels": [r["label"] for r in window],
            })
    write_jsonl(root / "temporal_dataset.jsonl", seqs)
    return seqs


def run(config: dict) -> list[dict]:
    return build_temporal_dataset(config)
