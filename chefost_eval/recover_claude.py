"""Create auditable Claude derivatives by trimming JSON key whitespace only.

Raw responses and prediction values are never changed. Missing values, ambiguous
keys, incomplete responses, and invalid native frame/object fields are rejected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_keys(text):
    """Parse strict JSON and return key-only normalization plus JSON-pointer edits."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    payload = json.loads(text, object_pairs_hook=unique)
    changes = []

    def pointer(value):
        return str(value).replace("~", "~0").replace("/", "~1")

    def walk(value, path):
        if isinstance(value, list):
            return [walk(item, path + "/" + str(index)) for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            canonical = key.strip()
            if not canonical or canonical in result:
                raise ValueError(f"Empty or colliding normalized JSON key at {path}: {key!r}")
            if canonical != key:
                changes.append({"source_pointer": path + "/" + pointer(key),
                                "derived_pointer": path + "/" + pointer(canonical),
                                "from_key": key, "to_key": canonical,
                                "value_sha256": _digest(item)})
            result[canonical] = walk(item, path + "/" + pointer(canonical))
        return result

    return walk(payload, ""), changes


def validate_payload(payload, samples, target_object):
    if (not isinstance(payload, dict) or set(payload) != {"transitions"}
            or not isinstance(payload["transitions"], list)):
        raise ValueError("Response must contain exactly a transitions list")
    if not isinstance(target_object, str) or not target_object.strip():
        raise ValueError("A native target object is required")
    allowed = {row["source_frame"] for row in samples}
    seen, previous = set(), -1
    for row in payload["transitions"]:
        if not isinstance(row, dict) or set(row) != {"frame", "object", "state_before", "state_after"}:
            raise ValueError("Transition has missing or unexpected fields")
        if type(row["frame"]) is not int or row["frame"] not in allowed:
            raise ValueError("Transition frame was not supplied to Claude")
        if row["frame"] < previous:
            raise ValueError("Transitions are not in source-frame order")
        previous = row["frame"]
        for key in ("object", "state_before", "state_after"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("Transition strings must be nonempty")
        if row["object"] != target_object:
            raise ValueError("Transition object does not match the native target")
        signature = _digest(row)
        if signature in seen:
            raise ValueError("Duplicate transition")
        seen.add(signature)


def recover(raw_path, samples_path, target_object, output_dir, *, provenance_files=None):
    raw_path, samples_path, output_dir = map(Path, (raw_path, samples_path, output_dir))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("stop_reason") != "end_turn":
        raise ValueError("Only completed native responses can be structurally recovered")
    text = "".join(block["text"] for block in raw.get("content", []) if block.get("type") == "text").strip()
    if text.startswith("```"):
        if "\n" not in text or not text.endswith("```"):
            raise ValueError("Incomplete fenced JSON response")
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    payload, changes = normalize_keys(text)
    if not changes:
        raise ValueError("No JSON key whitespace to recover")
    samples_record = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = samples_record["samples"]
    validate_payload(payload, samples, target_object)
    files = {"raw_response": raw_path, "samples": samples_path,
             **{name: Path(path) for name, path in (provenance_files or {}).items()}}
    sources = {name: {"path": str(path.resolve()), "sha256": _file_hash(path)}
               for name, path in files.items()}
    metadata = {"schema_version": 1, "operation": "strip_json_key_whitespace_only",
                "prediction_values_changed": False, "model_inference_performed": False,
                "raw_response_id": raw.get("id"), "raw_model": raw.get("model"),
                "target_object": target_object, "sources": sources, "changes": changes,
                "validation": "complete response; exact transition schema; supplied source frames; native target; ordering; uniqueness",
                "helper_sha256": _file_hash(__file__)}
    # Validate every source and all changes before creating any derivative.
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "transitions.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata["derivative"] = {"path": str(output.resolve()), "sha256": _file_hash(output)}
    (output_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload, metadata
