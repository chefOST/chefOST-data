from __future__ import annotations

import random
from pathlib import Path
from .utils import read_jsonl, write_jsonl, ensure_dir


def split_by_video(config: dict) -> dict[str, list[dict]]:
    root = Path(config.get("dataset_root", "dataset"))
    split_dir = ensure_dir(config.get("split_dir", "splits"))
    rows = read_jsonl(root / "frame_dataset.jsonl")
    videos = sorted({r["video_id"] for r in rows})
    rng = random.Random(config.get("split", {}).get("seed", 42))
    rng.shuffle(videos)

    ratios = config.get("split", {})
    train_r = float(ratios.get("train", 0.8))
    val_r = float(ratios.get("val", 0.1))
    n = len(videos)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    if n and n_train == 0:
        n_train = 1
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    video_sets = {
        "train": set(videos[:n_train]),
        "val": set(videos[n_train:n_train + n_val]),
        "test": set(videos[n_train + n_val:]),
    }
    out: dict[str, list[dict]] = {}
    for name, vids in video_sets.items():
        split_rows = [r for r in rows if r["video_id"] in vids]
        out[name] = split_rows
        write_jsonl(split_dir / f"{name}.jsonl", split_rows)
    return out


def run(config: dict) -> dict[str, list[dict]]:
    return split_by_video(config)
