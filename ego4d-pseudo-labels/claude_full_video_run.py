#!/usr/bin/env python3
"""Run one whole video through Claude and save visible food-state transitions."""

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from full_video_run import extract_frames, ffmpeg_path, video_fps


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def call_claude(model, content, api_key):
    body = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": content}],
    }).encode()
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Anthropic API error {error.code}: {error.read().decode(errors='replace')}") from error
    return raw, time.monotonic() - started


def run(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("set ANTHROPIC_API_KEY")
    ffmpeg = ffmpeg_path(args.ffmpeg)
    source_fps = video_fps(args.video, ffmpeg)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(args.video, args.output, args.sample_fps, ffmpeg)
    if len(frames) > 100:
        raise RuntimeError(f"Anthropic accepts at most 100 images; got {len(frames)}. Lower --sample-fps.")

    content = [{"type": "text", "text": f"""Watch these frames from one complete {args.recipe}-making video in chronological order. Identify every visible physical state transition of food or ingredients, such as whole to cut, separate ingredients to combined, unmixed to mixed, unpeeled to peeled, or loose ingredients to assembled food. Return the first supplied source frame where each new state is visibly established. Ignore people, cabinets, appliances, packaging, and objects merely moving. Do not infer cooking or other changes that are not visibly shown.

Return only JSON in exactly this shape:
{{"transitions":[{{"frame":123,"object":"food or ingredient","state_before":"previous visible state","state_after":"new visible state"}}]}}"""}]
    for index, frame in enumerate(frames):
        source_frame = round(index * source_fps / args.sample_fps)
        content.append({"type": "text", "text": f"Source frame {source_frame}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(frame.read_bytes()).decode(),
            },
        })

    raw, elapsed = call_claude(args.model, content, api_key)
    text = "".join(block["text"] for block in raw.get("content", []) if block.get("type") == "text")
    result = parse_json(text)
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
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_VLM_MODEL", "claude-fable-5-1"))
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--sample-fps", type=float, default=0.25)
    parser.add_argument("--ffmpeg")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
