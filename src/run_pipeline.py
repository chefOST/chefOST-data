from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/run_pipeline.py` from repository root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_yaml, read_json, write_json
from src import sample_frames, generate_prompts, detect_objects, track_masks
from src import create_masked_crops, score_states, clean_labels, extract_features
from src import build_frame_dataset, build_temporal_dataset, split_dataset


REQUIRED_DIRS = [
    "raw_videos", "frames", "detections", "masks", "masked_crops", "clip_scores",
    "pseudo_labels", "features", "sequences", "metadata",
]


def prepare_dirs(config: dict) -> None:
    root = Path(config.get("dataset_root", "dataset"))
    for d in REQUIRED_DIRS:
        ensure_dir(root / d)
    ensure_dir(config.get("split_dir", "splits"))


def load_videos(config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    videos = config.get("videos")
    if videos is None:
        videos = read_json(root / "metadata" / "video_metadata.json", [])
    if isinstance(videos, dict):
        videos = [videos]
    if not videos:
        raise RuntimeError(
            "No videos configured. Add `videos:` to the YAML or create "
            f"{root / 'metadata' / 'video_metadata.json'}."
        )
    for v in videos:
        for key in ["video_id", "video_path", "object", "task"]:
            if key not in v:
                raise ValueError(f"Video metadata missing `{key}`: {v}")
        if v["object"] not in config.get("objects", {}):
            raise ValueError(f"Unknown object {v['object']!r}; add it to config.objects")
    write_json(root / "metadata" / "video_metadata.json", videos)
    return videos


def run_video(video: dict, config: dict, skip_existing: bool = False) -> None:
    root = Path(config.get("dataset_root", "dataset"))
    vid = video["video_id"]
    print(f"[video] {vid}: sample frames")
    if not skip_existing or not (root / "metadata" / f"{vid}_frames.json").exists():
        sample_frames.run(video, config)

    print(f"[video] {vid}: generate prompts")
    generate_prompts.run(video, config)

    print(f"[video] {vid}: detect object")
    if not skip_existing or not (root / "detections" / f"{vid}.json").exists():
        detect_objects.run(video, config)

    print(f"[video] {vid}: track masks")
    if not skip_existing or not (root / "metadata" / f"{vid}_masks.json").exists():
        track_masks.run(video, config)

    print(f"[video] {vid}: create masked crops")
    if not skip_existing or not (root / "metadata" / f"{vid}_masked_crops.json").exists():
        create_masked_crops.run(video, config)

    print(f"[video] {vid}: score states")
    if not skip_existing or not (root / "clip_scores" / f"{vid}.json").exists():
        score_states.run(video, config)

    print(f"[video] {vid}: clean pseudo-labels")
    clean_labels.run(video, config)

    print(f"[video] {vid}: extract features")
    extract_features.run(video, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChefOST pseudo-labeled video dataset")
    parser.add_argument("--config", default="configs/dataset_config.yaml")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    prepare_dirs(config)
    videos = load_videos(config)

    for video in videos:
        run_video(video, config, skip_existing=args.skip_existing)

    print("[dataset] build frame_dataset.jsonl")
    build_frame_dataset.run(videos, config)
    print("[dataset] build temporal_dataset.jsonl")
    build_temporal_dataset.run(config)
    print("[dataset] create train/val/test split")
    split_dataset.run(config)
    print("Done.")


if __name__ == "__main__":
    main()
