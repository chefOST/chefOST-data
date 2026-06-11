from __future__ import annotations

from pathlib import Path
from PIL import Image
from .utils import ensure_dir, read_json, write_json, relpath


def create_crops(video: dict, config: dict) -> list[dict]:
    root = Path(config.get("dataset_root", "dataset"))
    frames = {f["frame_id"]: f for f in read_json(root / "metadata" / f"{video['video_id']}_frames.json", [])}
    masks = read_json(root / "metadata" / f"{video['video_id']}_masks.json", [])
    out_dir = ensure_dir(root / "masked_crops" / video["video_id"])
    rows: list[dict] = []

    for m in masks:
        frame = frames[m["frame_id"]]
        image = Image.open(root / frame["image_path"]).convert("RGBA")
        mask = Image.open(root / m["mask_path"]).convert("L")
        dark = Image.new("RGBA", image.size, (0, 0, 0, 255))
        masked = Image.composite(image, dark, mask)
        bbox = tuple(int(x) for x in m["bbox"])
        crop = masked.crop(bbox).convert("RGB")
        crop_path = out_dir / f"{m['frame_id']}.png"
        crop.save(crop_path)
        row = dict(m)
        row["masked_crop_path"] = relpath(crop_path, root)
        rows.append(row)

    write_json(root / "metadata" / f"{video['video_id']}_masked_crops.json", rows)
    return rows


def run(video: dict, config: dict) -> list[dict]:
    return create_crops(video, config)
