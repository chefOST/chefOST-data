from __future__ import annotations

from pathlib import Path
from .utils import write_json


def generate_prompts_for_video(video: dict, config: dict) -> list[dict]:
    obj = video["object"]
    states = config["objects"][obj]["states"]
    return [{"state": state, "prompt": f"a {state} {obj}"} for state in states]


def run(video: dict, config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    prompts = generate_prompts_for_video(video, config)
    write_json(root / "metadata" / f"{video['video_id']}_prompts.json", prompts)
    return prompts
