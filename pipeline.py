#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoModelForImageTextToText, AutoProcessor

LABELS = ("raw", "partially_toasted", "toasted", "not_visible")
PROMPT = """These images show the same naan around one timestamp.
Classify the cooking state in the Target image. Choose exactly one:
raw — pale dough with little or only faint browning
partially_toasted — noticeable browning, but still mostly pale
toasted — widespread or dark brown spots
not_visible — naan is absent or cannot be assessed
Respond with exactly one label and nothing else."""


def ffmpeg_path(explicit=None):
    path = explicit or shutil.which("ffmpeg") or "/Applications/AnyTrans.app/Contents/Resources/ffmpeg"
    if not Path(path).exists():
        raise SystemExit("ffmpeg not found; install it or pass --ffmpeg")
    return path


def extract_frames(video, frames, ffmpeg):
    frames.mkdir(parents=True, exist_ok=True)
    if not list(frames.glob("*.jpg")):
        subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(video), "-vf", "fps=1", "-q:v", "2", str(frames / "%05d.jpg")], check=True)
    return sorted(frames.glob("*.jpg"))


def segment(frames, output, box, prompt_second):
    masks_dir, crops_dir, overlays_dir = (output / name for name in ("masks", "crops", "overlays"))
    for directory in (masks_dir, crops_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-tiny", device=device)
    state = predictor.init_state(str(frames), offload_video_to_cpu=True)
    start = prompt_second - 1
    predictor.add_new_points_or_box(state, frame_idx=start, obj_id=1, box=np.asarray(box))

    crop_paths = [None] * len(list(frames.glob("*.jpg")))
    for frame_idx, _, logits in predictor.propagate_in_video(state, start_frame_idx=start):
        image = np.asarray(Image.open(frames / f"{frame_idx + 1:05d}.jpg").convert("RGB"))
        mask = logits[0, 0].cpu().numpy() > 0
        if mask.sum() < 500:
            continue

        ys, xs = np.where(mask)
        pad = max(10, int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.1))
        x1, x2 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad + 1)
        y1, y2 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad + 1)

        Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / f"{frame_idx + 1:05d}.png")
        masked = np.where(mask[..., None], image, 127).astype(np.uint8)
        crop_path = crops_dir / f"{frame_idx + 1:05d}.jpg"
        Image.fromarray(masked[y1:y2, x1:x2]).save(crop_path)
        crop_paths[frame_idx] = crop_path

        overlay = image.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(overlays_dir / f"{frame_idx + 1:05d}.jpg")

    return crop_paths


def parse_label(text):
    text = text.lower().strip()
    for label in LABELS:
        if re.search(rf"\b{label}\b", text):
            return label
    return "not_visible"


def classify(crops):
    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, _attn_implementation="eager").to(device)

    predictions = []
    for index, crop in enumerate(crops):
        if crop is None:
            predictions.append("not_visible")
            continue
        images, content = [], []
        for name, i in (("Before", max(0, index - 2)), ("Target", index), ("After", min(len(crops) - 1, index + 2))):
            if crops[i]:
                images.append(Image.open(crops[i]).convert("RGB"))
                content += [{"type": "text", "text": f"{name}:"}, {"type": "image"}]
        content.append({"type": "text", "text": PROMPT})
        prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=5)
        answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        predictions.append(parse_label(answer))
    return predictions


def smooth(predictions, streak_needed=3):
    order = {label: i for i, label in enumerate(LABELS[:3])}
    result, current, candidate, streak = [], None, None, 0
    for prediction in predictions:
        if prediction == "not_visible":
            candidate, streak = None, 0
            result.append(prediction)
            continue
        if prediction == candidate:
            streak += 1
        else:
            candidate, streak = prediction, 1
        if streak >= streak_needed and (current is None or order[prediction] >= order[current]):
            current = prediction
        result.append(current or "not_visible")
    return result


def write_csv(path, values):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "state"])
        writer.writerows(enumerate(values, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--prompt-second", type=int, required=True)
    parser.add_argument("--box", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    frames = args.output / "frames"
    frame_paths = extract_frames(args.video, frames, ffmpeg_path(args.ffmpeg))
    crops_dir = args.output / "crops"
    if crops_dir.exists():
        crops = [crops_dir / path.with_suffix(".jpg").name for path in frame_paths]
        crops = [path if path.exists() else None for path in crops]
    else:
        crops = segment(frames, args.output, args.box, args.prompt_second)
    raw = classify(crops)
    write_csv(args.output / "predictions_raw.csv", raw)
    write_csv(args.output / "predictions.csv", smooth(raw))

    thirds = max(1, len(frame_paths) // 3)
    fixed = ["raw" if i < thirds else "partially_toasted" if i < 2 * thirds else "toasted" for i in range(len(frame_paths))]
    write_csv(args.output / "fixed_labels.csv", fixed)


if __name__ == "__main__":
    main()
