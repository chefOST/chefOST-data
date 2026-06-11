from __future__ import annotations

from collections import Counter
from pathlib import Path
from .utils import read_json, write_json, frame_sort_key


def _top_margin(scores: dict[str, float]) -> tuple[float, float]:
    vals = sorted(scores.values(), reverse=True)
    if not vals:
        return 0.0, 0.0
    return vals[0], vals[0] - (vals[1] if len(vals) > 1 else 0.0)


def clean_video_labels(video: dict, config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    frames = {f["frame_id"]: f for f in read_json(root / "metadata" / f"{video['video_id']}_frames.json", [])}
    masks = {m["frame_id"]: m for m in read_json(root / "metadata" / f"{video['video_id']}_masks.json", [])}
    scores_rows = read_json(root / "clip_scores" / f"{video['video_id']}.json", [])
    order = config["objects"][video["object"]]["states"]
    rank = {s: i for i, s in enumerate(order)}
    q = config.get("quality", {})

    raw = []
    for row in scores_rows:
        scores = row["scores"]
        label = max(scores, key=scores.get)
        top, margin = _top_margin(scores)
        reasons = []
        m = masks.get(row["frame_id"], {})
        if float(m.get("confidence", 0.0)) < q.get("min_detection_confidence", 0.35):
            reasons.append("low_detection_confidence")
        if int(m.get("area", 0)) < q.get("min_mask_area", 100):
            reasons.append("small_mask_area")
        if top < q.get("min_clip_top_score", 0.20):
            reasons.append("low_clip_score")
        if margin < q.get("min_clip_margin", 0.05):
            reasons.append("low_clip_margin")
        raw.append({
            "frame_id": row["frame_id"],
            "timestamp": frames.get(row["frame_id"], {}).get("timestamp", 0.0),
            "raw_label": label,
            "raw_rank": rank.get(label, 0),
            "clip_scores": scores,
            "is_uncertain": bool(reasons),
            "reason": ",".join(reasons) if reasons else None,
        })
    raw.sort(key=frame_sort_key)

    # Majority smoothing in a 3-frame window.
    smoothed_ranks: list[int] = []
    for i in range(len(raw)):
        window = raw[max(0, i - 1): min(len(raw), i + 2)]
        counts = Counter(r["raw_rank"] for r in window)
        smoothed_ranks.append(counts.most_common(1)[0][0])

    # Enforce non-decreasing state progression to remove impossible backward jumps.
    cleaned: list[dict] = []
    current = 0
    for row, rnk in zip(raw, smoothed_ranks):
        current = max(current, rnk)
        clean_label = order[min(current, len(order) - 1)]
        cleaned.append({
            "frame_id": row["frame_id"],
            "timestamp": row["timestamp"],
            "label": clean_label,
            "raw_label": row["raw_label"],
            "clip_scores": row["clip_scores"],
            "is_uncertain": row["is_uncertain"],
            "reason": row["reason"],
        })

    write_json(root / "pseudo_labels" / f"{video['video_id']}.json", cleaned)
    return cleaned


def run(video: dict, config: dict) -> list[dict]:
    return clean_video_labels(video, config)
