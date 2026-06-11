from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from .utils import read_json, write_json


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(float(e.sum()), 1e-12)


def _heuristic_scores(image_path: Path, states: list[str]) -> dict[str, float]:
    im = Image.open(image_path).convert("RGB").resize((128, 128))
    arr = np.asarray(im).astype(np.float32) / 255.0
    non_black = arr[arr.mean(axis=2) > 0.03]
    if non_black.size == 0:
        non_black = arr.reshape(-1, 3)
    brightness = float(non_black.mean())
    redness = float(non_black[:, 0].mean() - non_black[:, 1].mean())
    # Edge/texture proxy: higher values tend to indicate sliced/diced/beaten textures.
    gray = arr.mean(axis=2)
    texture = float(np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1))))

    logits = []
    n = max(len(states) - 1, 1)
    for i, state in enumerate(states):
        progress = i / n
        logit = -abs(texture * 6.0 - progress)
        if state in {"whole", "peeled"}:
            logit += (1.0 - texture * 4.0) * 0.4
        if state in {"sliced", "diced", "beaten"}:
            logit += texture * 2.0
        if state == "cooked":
            logit += (redness + (0.45 - brightness)) * 1.5
        if state == "cracked":
            logit += texture * 0.8
        logits.append(logit)
    probs = _softmax(np.asarray(logits, dtype=np.float32))
    return {s: round(float(p), 4) for s, p in zip(states, probs)}


def score_video_states(video: dict, config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    crops = read_json(root / "metadata" / f"{video['video_id']}_masked_crops.json", [])
    states = config["objects"][video["object"]]["states"]
    rows: list[dict] = []
    for crop in crops:
        scores = _heuristic_scores(root / crop["masked_crop_path"], states)
        raw_label = max(scores, key=scores.get)
        rows.append({"frame_id": crop["frame_id"], "scores": scores, "raw_label": raw_label})
    write_json(root / "clip_scores" / f"{video['video_id']}.json", rows)
    return rows


def run(video: dict, config: dict) -> list[dict]:
    return score_video_states(video, config)
