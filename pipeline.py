#!/usr/bin/env python3
"""Conservative 1 FPS object-state pseudo-labeling with mandatory human review."""

import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

FALLBACK_FFMPEG = "/Applications/AnyTrans.app/Contents/Resources/ffmpeg"
SPECIAL_STATES = {"occluded", "uncertain"}


def load_manifest(path):
    manifest = json.loads(path.read_text())
    required = ("video", "object", "states", "transitions")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"manifest is missing: {', '.join(missing)}")
    names = [state["name"] for state in manifest["states"]]
    if len(names) != len(set(names)) or SPECIAL_STATES & set(names):
        raise ValueError("physical state names must be unique and cannot be occluded/uncertain")
    unknown = {value for values in manifest["transitions"].values() for value in values} - set(names)
    if unknown or set(manifest["transitions"]) - set(names):
        raise ValueError(f"transitions contain unknown states: {sorted(unknown)}")
    video = Path(manifest["video"])
    manifest["video"] = video if video.is_absolute() else (path.parent / video).resolve()
    return manifest


def ffmpeg_path(explicit=None):
    path = explicit or shutil.which("ffmpeg") or FALLBACK_FFMPEG
    if not Path(path).is_file():
        raise RuntimeError("ffmpeg not found; install it or pass --ffmpeg")
    return path


def extract_frames(manifest, output, ffmpeg):
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    expected = frames / "000000.jpg"
    if not expected.exists():
        command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
        start = float(manifest.get("start_second", 0))
        if start:
            command += ["-ss", str(start)]
        command += ["-i", str(manifest["video"])]
        if manifest.get("duration_seconds"):
            command += ["-t", str(manifest["duration_seconds"])]
        command += ["-vf", "fps=1", "-q:v", "2", "-start_number", "0", str(frames / "%06d.jpg")]
        subprocess.run(command, check=True)
    paths = sorted(frames.glob("*.jpg"))
    if not paths:
        raise RuntimeError("ffmpeg extracted no frames")
    return paths


def windows(length, size=12, overlap=4):
    if size <= overlap:
        raise ValueError("window size must exceed overlap")
    starts = list(range(0, length, size - overlap))
    result = [(start, min(length, start + size)) for start in starts]
    return [item for item in result if item[0] == 0 or item[1] - item[0] > overlap]


def image_part(path):
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"}


def response_text(response):
    if response.get("output_text"):
        return response["output_text"]
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content["text"]
    raise RuntimeError(f"API response has no text output: {response}")


def openai_json(model, content, schema, api_key, attempts=3):
    body = json.dumps({
        "model": model,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_schema", "name": "result", "strict": True, "schema": schema}},
    }).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response_text(json.load(response)))
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"OpenAI API error {error.code}: {detail}") from error
            if attempt == attempts - 1:
                raise RuntimeError(f"OpenAI API error {error.code}: {detail}") from error
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"OpenAI API request failed: {error}") from error
        time.sleep(2**attempt)


def state_prompt(manifest, indices):
    state_lines = "\n".join(f'- {s["name"]}: {s["description"]}' for s in manifest["states"])
    return f"""Track only this object: {manifest['object']}.
Its identity must remain the same across every image. Label each attached 1 FPS frame.

Allowed visible physical states:
{state_lines}
- occluded: the tracked object still exists but its visible state cannot be assessed
- uncertain: identity or state cannot be established from visible evidence

Rules:
- Use only visible evidence; never infer temperature, taste, or hidden progress.
- Compare adjacent frames and keep labels temporally consistent.
- Do not silently switch to another similar object.
- Choose only an allowed label. Prefer uncertain over guessing.
- Return exactly one entry for every sample index: {indices}.
- Evidence must be a short description of what is visibly decisive.
"""


def label_schema(labels):
    return {
        "type": "object",
        "properties": {"frames": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "sample_index": {"type": "integer"},
                "state": {"type": "string", "enum": labels},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "identity_match": {"type": "boolean"},
                "evidence": {"type": "string"},
            },
            "required": ["sample_index", "state", "confidence", "identity_match", "evidence"],
            "additionalProperties": False,
        }}},
        "required": ["frames"],
        "additionalProperties": False,
    }


def label_frames(manifest, frames, output, model, request_json):
    labels = [s["name"] for s in manifest["states"]] + sorted(SPECIAL_STATES)
    cache = output / "raw_batches"
    cache.mkdir(exist_ok=True)
    proposals = defaultdict(list)
    for start, end in windows(len(frames)):
        cache_file = cache / f"{start:06d}-{end - 1:06d}.json"
        if cache_file.exists():
            result = json.loads(cache_file.read_text())
        else:
            indices = list(range(start, end))
            content = [{"type": "input_text", "text": state_prompt(manifest, indices)}]
            for index in indices:
                content += [{"type": "input_text", "text": f"Sample {index}, timestamp {index}s:"}, image_part(frames[index])]
            result = request_json(model, content, label_schema(labels))
            cache_file.write_text(json.dumps(result, indent=2) + "\n")
        expected = set(range(start, end))
        returned = [item.get("sample_index") for item in result.get("frames", [])]
        if set(returned) != expected or len(returned) != len(expected):
            raise RuntimeError(f"model returned wrong frame indices for window {start}:{end}: {returned}")
        for item in result["frames"]:
            if item["state"] not in labels:
                raise RuntimeError(f"model returned unknown state: {item['state']}")
            if not item["identity_match"]:
                item = {**item, "state": "uncertain"}
            proposals[item["sample_index"]].append(item)
    return proposals


def consensus(proposals, count, min_confidence=0.80, min_agreement=2 / 3):
    timeline = []
    for index in range(count):
        items = proposals[index]
        votes = Counter(item["state"] for item in items)
        state, vote_count = max(votes.items(), key=lambda pair: (pair[1], sum(x["confidence"] for x in items if x["state"] == pair[0])))
        winning = [item for item in items if item["state"] == state]
        confidence = sum(item["confidence"] for item in winning) / len(winning)
        agreement = vote_count / len(items)
        if confidence < min_confidence or agreement < min_agreement:
            state = "uncertain"
        timeline.append({
            "sample_index": index,
            "timestamp_sec": index,
            "state": state,
            "confidence": round(confidence, 3),
            "agreement": round(agreement, 3),
            "evidence": winning[0]["evidence"],
        })
    return timeline


def enforce_consistency(timeline, transitions, minimum_run=2):
    physical = set(transitions)
    result = [dict(item) for item in timeline]

    index = 0
    while index < len(result):
        state = result[index]["state"]
        end = index + 1
        while end < len(result) and result[end]["state"] == state:
            end += 1
        if state in physical and end - index < minimum_run:
            for item in result[index:end]:
                item["state"] = "uncertain"
                item["evidence"] = "rejected one-frame physical-state flicker"
        index = end

    previous = None
    for item in result:
        state = item["state"]
        if state not in physical:
            continue
        if previous is not None and state != previous and state not in transitions[previous]:
            item["state"] = "uncertain"
            item["evidence"] = f"rejected illegal transition {previous} -> {state}"
            continue
        previous = state
    return result


def find_changes(timeline, transitions):
    physical = set(transitions)
    previous = None
    changes = []
    for item in timeline:
        state = item["state"]
        if state not in physical:
            continue
        if previous and state != previous:
            changes.append({
                "frame": item["sample_index"],
                "timestamp_sec": item["timestamp_sec"],
                "from_state": previous,
                "to_state": state,
                "confidence": item["confidence"],
                "evidence": item["evidence"],
            })
        previous = state
    return changes


def verify_schema():
    return {
        "type": "object",
        "properties": {
            "verified": {"type": "boolean"},
            "identity_match": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "first_new_state_sample": {"type": "integer"},
            "evidence": {"type": "string"},
        },
        "required": ["verified", "identity_match", "confidence", "first_new_state_sample", "evidence"],
        "additionalProperties": False,
    }


def verify_changes(changes, manifest, frames, model, request_json):
    verified = []
    for change in changes:
        boundary = change["frame"]
        indices = list(range(max(0, boundary - 2), min(len(frames), boundary + 3)))
        prompt = f"""Verify a proposed visible state change for this exact tracked object: {manifest['object']}.
Proposed transition: {change['from_state']} -> {change['to_state']}.
Candidate boundary: sample {boundary}; this means the first 1 FPS sample visibly showing the new state.
Reject it if identity changes, evidence is ambiguous, the old state is not visible before it, or the new state is not visible at/after it.
The attached samples are {indices}. Use only visible evidence."""
        content = [{"type": "input_text", "text": prompt}]
        for index in indices:
            content += [{"type": "input_text", "text": f"Sample {index}:"}, image_part(frames[index])]
        check = request_json(model, content, verify_schema())
        accepted = check["verified"] and check["identity_match"] and check["confidence"] >= 0.90 and check["first_new_state_sample"] == boundary
        verified.append({**change, "vlm_verified": accepted, "verification_confidence": check["confidence"], "verification_evidence": check["evidence"]})
    return verified


def write_jsonl(path, rows):
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def write_review(path, changes):
    fields = ["frame", "timestamp_sec", "from_state", "to_state", "confidence", "evidence", "vlm_verified", "verification_confidence", "verification_evidence", "approved"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for change in changes:
            writer.writerow({**change, "approved": ""})


def finalize(review_path, destination):
    with review_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    unanswered = [row["frame"] for row in rows if row["approved"].strip().lower() not in {"yes", "no"}]
    if unanswered:
        raise RuntimeError(f"review every transition with approved=yes or no; unanswered frames: {', '.join(unanswered)}")
    accepted = [{key: value for key, value in row.items() if key != "approved"} for row in rows if row["approved"].strip().lower() == "yes"]
    write_jsonl(destination, accepted)
    return len(accepted)


def make_requester(use_modal=False):
    if use_modal:
        try:
            import modal
        except ImportError as error:
            raise RuntimeError("install Modal with: python3 -m pip install modal") from error
        remote = modal.Function.from_name("ego4d-labeler", "openai_json")
        return remote.remote

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY or use --modal")
    return lambda model, content, schema: openai_json(model, content, schema, api_key)


def run(args):
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if not manifest["video"].is_file():
        raise RuntimeError(f"video not found: {manifest['video']}")
    request_json = make_requester(args.modal)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = extract_frames(manifest, args.output, ffmpeg_path(args.ffmpeg))
    proposals = label_frames(manifest, frames, args.output, args.model, request_json)
    raw = consensus(proposals, len(frames))
    timeline = enforce_consistency(raw, manifest["transitions"])
    changes = verify_changes(find_changes(timeline, manifest["transitions"]), manifest, frames, args.model, request_json)
    write_jsonl(args.output / "timeline.jsonl", timeline)
    write_review(args.output / "changes_for_review.csv", changes)
    print(f"Wrote {len(timeline)} frame labels and {len(changes)} proposed changes to {args.output}")
    print("Review changes_for_review.csv, set every approved cell to yes/no, then run finalize.")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    label = subparsers.add_parser("label")
    label.add_argument("--manifest", type=Path, default=Path("selected_video.json"))
    label.add_argument("--output", type=Path, default=Path("output"))
    label.add_argument("--model", default=os.environ.get("OPENAI_VLM_MODEL", "gpt-5.2"))
    label.add_argument("--ffmpeg")
    label.add_argument("--modal", action="store_true", help="use the private deployed Modal function instead of a local API key")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--review", type=Path, default=Path("output/changes_for_review.csv"))
    finalize_parser.add_argument("--output", type=Path, default=Path("output/state_changes.jsonl"))
    args = parser.parse_args()
    if args.command == "label":
        run(args)
    else:
        count = finalize(args.review, args.output)
        print(f"Wrote {count} approved state changes to {args.output}")


if __name__ == "__main__":
    main()
