#!/usr/bin/env python3
"""Run one whole video through OpenAI and save visible state transitions."""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

FALLBACK_FFMPEG = "/Applications/AnyTrans.app/Contents/Resources/ffmpeg"


def ffmpeg_path(explicit=None):
    path = explicit or shutil.which("ffmpeg") or FALLBACK_FFMPEG
    if not Path(path).is_file():
        raise RuntimeError("ffmpeg not found; install it or pass --ffmpeg")
    return path


def video_fps(video, ffmpeg):
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    match = re.search(r"(?:, | )(\d+(?:\.\d+)?) fps(?:,| )", probe.stderr)
    if not match:
        raise RuntimeError("could not read video frame rate")
    return float(match.group(1))


def extract_frames(video, output, sample_fps, ffmpeg):
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    if not list(frames.glob("*.jpg")):
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", f"fps={sample_fps},scale=640:-2", "-q:v", "3", "-start_number", "0",
            str(frames / "%06d.jpg"),
        ], check=True)
    return sorted(frames.glob("*.jpg"))


def image_part(path):
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}


def schema():
    return {
        "type": "object",
        "properties": {
            "transitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame": {"type": "integer"},
                        "object": {"type": "string"},
                        "state_before": {"type": "string"},
                        "state_after": {"type": "string"},
                    },
                    "required": ["frame", "object", "state_before", "state_after"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["transitions"],
        "additionalProperties": False,
    }


def call_openai(model, content, api_key):
    body = json.dumps({
        "model": model,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "video_transitions", "strict": True, "schema": schema()}},
    }).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"OpenAI API error {error.code}: {error.read().decode(errors='replace')}") from error
    return raw, time.monotonic() - started


def response_text(response):
    if response.get("output_text"):
        return response["output_text"]
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content["text"]
    raise RuntimeError("API response contained no output text")


def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY")
    ffmpeg = ffmpeg_path(args.ffmpeg)
    source_fps = video_fps(args.video, ffmpeg)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(args.video, args.output, args.sample_fps, ffmpeg)
    content = [{"type": "input_text", "text": """Watch these frames from one complete cooking video in chronological order. Identify visible physical state transitions of food or ingredients, such as whole to cut, separate ingredients to mixed batter, raw to cooked, or unpeeled to peeled. Return the first supplied source frame where each new state is visibly established. Ignore people, cabinets, appliances, flames, containers, and objects merely appearing, disappearing, opening, closing, or moving. Report only changes supported by visible evidence; do not report an action without a resulting food-state change. Use concise state names. The source frame number is written before every image."""}]
    for index, frame in enumerate(frames):
        source_frame = round(index * source_fps / args.sample_fps)
        content += [{"type": "input_text", "text": f"Source frame {source_frame}:"}, image_part(frame)]
    raw, elapsed = call_openai(args.model, content, api_key)
    result = json.loads(response_text(raw))
    (args.output / "raw_response.json").write_text(json.dumps(raw, indent=2) + "\n")
    (args.output / "transitions.json").write_text(json.dumps(result, indent=2) + "\n")
    metadata = {
        "video": str(args.video.resolve()),
        "model": args.model,
        "source_fps": source_fps,
        "sample_fps": args.sample_fps,
        "sampled_frames": len(frames),
        "elapsed_seconds": round(elapsed, 2),
        "usage": raw.get("usage"),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved raw response, transitions, and run metadata to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("full-run-output"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_VLM_MODEL", "gpt-6-astra"))
    parser.add_argument("--sample-fps", type=float, default=1)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
