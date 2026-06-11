from __future__ import annotations

from pathlib import Path
from .utils import read_json, write_json, write_jsonl


def build_video_records(video: dict, config: dict) -> dict:
    root = Path(config.get("dataset_root", "dataset"))
    frames = {f["frame_id"]: f for f in read_json(root / "metadata" / f"{video['video_id']}_frames.json", [])}
    masks = {m["frame_id"]: m for m in read_json(root / "metadata" / f"{video['video_id']}_masks.json", [])}
    crops = {c["frame_id"]: c for c in read_json(root / "metadata" / f"{video['video_id']}_masked_crops.json", [])}
    labels = {l["frame_id"]: l for l in read_json(root / "pseudo_labels" / f"{video['video_id']}.json", [])}
    feats = {f["frame_id"]: f for f in read_json(root / "metadata" / f"{video['video_id']}_features.json", [])}

    frame_rows = []
    for frame_id, frame in sorted(frames.items(), key=lambda kv: kv[1].get("timestamp", 0.0)):
        if frame_id not in labels or frame_id not in feats or frame_id not in crops:
            continue
        m = masks.get(frame_id, {})
        lab = labels[frame_id]
        frame_rows.append({
            "frame_id": frame_id,
            "timestamp": frame["timestamp"],
            "image_path": frame["image_path"],
            "mask_path": m.get("mask_path"),
            "masked_crop_path": crops[frame_id]["masked_crop_path"],
            "bbox": m.get("bbox"),
            "clip_scores": lab.get("clip_scores", {}),
            "pseudo_label_raw": lab.get("raw_label"),
            "pseudo_label_clean": lab.get("label"),
            "feature_path": feats[frame_id]["feature_path"],
            "is_uncertain": lab.get("is_uncertain", False),
            "reason": lab.get("reason"),
        })
    return {"video_id": video["video_id"], "object": video["object"], "task": video["task"], "frames": frame_rows}


def build_frame_dataset(videos: list[dict], config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    rows = []
    final_videos = []
    for video in videos:
        record = build_video_records(video, config)
        final_videos.append(record)
        write_json(root / "metadata" / f"{video['video_id']}_final.json", record)
        for fr in record["frames"]:
            if fr.get("is_uncertain"):
                continue
            rows.append({
                "video_id": video["video_id"],
                "frame_id": fr["frame_id"],
                "timestamp": fr["timestamp"],
                "object": video["object"],
                "task": video["task"],
                "masked_crop_path": fr["masked_crop_path"],
                "feature_path": fr["feature_path"],
                "label": fr["pseudo_label_clean"],
            })
    write_json(root / "metadata" / "final_videos.json", final_videos)
    write_jsonl(root / "frame_dataset.jsonl", rows)
    return rows


def run(videos: list[dict], config: dict) -> list[dict]:
    return build_frame_dataset(videos, config)
