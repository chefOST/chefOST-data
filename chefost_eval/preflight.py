"""Validate saved evaluation inputs without scoring or loading model providers.

This checks the normal manifest loader and each listed prediction artifact.
An inputs_ready report says nothing about external GPT/Qwen runtime access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import load_predictions
from .labels import Vocabulary, read_json, sha256
from .manifest import DATASETS, load_run

INPUT_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, ImportError)


def inspect_manifest(manifest_path):
    path = Path(manifest_path).resolve()
    report = {"schema_version": 1, "scope": "inputs_only", "manifest": str(path),
              "status": "blocked", "loader": {}, "artifacts": [], "blockers": [],
              "runtime_access_checked": False, "scores_computed": False}
    try:
        config = read_json(path)
        report["manifest_sha256"] = sha256(path)
        if not isinstance(config, dict):
            raise ValueError("Manifest must be a JSON object")
    except INPUT_ERRORS as exc:
        report["blockers"].append({"stage": "manifest", "error": str(exc)})
        return report
    report["method"] = config.get("method")
    validated_hashes = {}

    # Keep the authoritative loader's gates intact, including unresolved alignment
    # and full-release coverage. Never change the user's manifest to make it pass.
    try:
        loaded = load_run(path, metadata_only=True)
        videos, metadata = loaded[1], loaded[-1]
        if metadata["input_sha256"][str(path)] != report["manifest_sha256"]:
            raise ValueError("Manifest changed during preflight")
        validated_hashes = metadata["input_sha256"]
        report["loader"] = {"status": "passed", "videos": len(videos),
            "ingredient_pairs": sum(len(v["ingredients"]) for v in videos),
            "evaluation_scope": metadata["scope"],
            "omitted_available_videos": metadata["omitted_available_videos"]}
        del loaded, videos
    except INPUT_ERRORS as exc:
        report["loader"] = {"status": "blocked", "error": str(exc)}
        report["blockers"].append({"stage": "loader", "error": str(exc)})

    # A first loader blocker must not conceal corrupted or malformed artifacts
    # later in the manifest. Reuse native adapters, inspecting one file at a time.
    try:
        root = (path.parent / config["data_root"]).resolve()
        records = config["videos"]
        if not isinstance(records, list):
            raise ValueError("videos must be a list")
    except INPUT_ERRORS as exc:
        report["blockers"].append({"stage": "artifact_inventory", "error": str(exc)})
        return report
    object_vocabularies = {}
    for record in records:
        try:
            dataset = record["dataset"]
            video = record["video_id"]
            if dataset not in object_vocabularies:
                object_file = root / "MOSCATO/vocabulary" / DATASETS[dataset][1]
                object_vocabularies[dataset] = Vocabulary(read_json(object_file), "o")
            specs = record["predictions"]
            if not isinstance(specs, list) or not specs:
                raise ValueError("Each video needs saved prediction sources")
        except INPUT_ERRORS as exc:
            report["blockers"].append({"stage": "artifact_inventory", "error": str(exc)})
            continue
        for spec in specs:
            row = {"dataset": dataset, "video_id": video, "status": "blocked"}
            try:
                source = (path.parent / spec["path"]).resolve()
                row["path"] = str(source)
                if report["loader"].get("status") == "passed":
                    row.update(status="passed", sha256=validated_hashes[str(source)],
                               validation="authoritative_manifest_loader")
                    report["artifacts"].append(row)
                    continue
                row["sha256"] = sha256(source)
                if row["sha256"] != spec.get("sha256"):
                    raise ValueError("Prediction checksum mismatch")
                streams = load_predictions(spec, path.parent, config["method"],
                                           object_vocabularies[dataset])
                row.update(status="passed", streams=len(streams),
                           windows=sum(len(s["windows"]) for s in streams.values()))
                diagnostics = {name: s["diagnostics"] for name, s in streams.items()
                               if s.get("diagnostics")}
                if diagnostics:
                    row["native_diagnostics"] = diagnostics
                del streams
            except INPUT_ERRORS as exc:
                row["error"] = str(exc)
                report["blockers"].append({"stage": "artifact", "dataset": dataset,
                                           "video_id": video, "error": str(exc)})
            report["artifacts"].append(row)
    report["artifact_counts"] = {"listed": len(report["artifacts"]),
        "passed": sum(row["status"] == "passed" for row in report["artifacts"])}
    if not report["blockers"]:
        report["status"] = "inputs_ready"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, help="New preflight JSON report; never overwritten")
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("Output exists; use a new path")
    report = inspect_manifest(args.manifest)
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")
    return 0 if report["status"] == "inputs_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
