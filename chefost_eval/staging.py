"""Freeze portable evaluation inputs without copying videos or model weights."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

from .manifest import DATASETS


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_bundle_file(root, relative, expected):
    """Resolve one staged file beneath its mount and verify its frozen bytes."""
    root = Path(root).resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"Staged input must use a relative path: {relative}")
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Staged input escapes its root: {relative}")
    if "bytes" in expected and path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Staged input size mismatch: {relative}")
    if file_hash(path) != expected["sha256"]:
        raise ValueError(f"Staged input checksum mismatch: {relative}")
    return path


def validate_bundle_files(root, files):
    """Check every frozen file through the same guard used by score workers."""
    for relative, expected in files.items():
        validate_bundle_file(root, relative, expected)


def _source(base, value):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _predicted_labels(spec, source):
    kind = spec.get("format", "windows")
    if kind in {"mosp_npz", "vidosc_npz"}:
        labels = spec["state_labels"]
    else:
        payload = read_json(source)
        if kind == "claude_transitions":
            labels = [row["state_after"] for row in payload["transitions"]]
        elif kind == "tubeletgraph_timeline":
            identifiers = {identifier for row in payload["intervals"]
                           for identifier in row["resolved_semantic_ids"]}
            labels = [payload["obj_info"][identifier]["desc"] for identifier in identifiers]
        elif kind == "windows":
            streams = [payload["broadcast"]] if "broadcast" in payload else payload["objects"].values()
            labels = [label for stream in streams for window in stream["windows"]
                      for label in window.get("labels", window.get("probabilities", {}))]
        else:
            raise ValueError(f"Unsupported prediction format: {kind}")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Mapping inventory contains an empty or invalid native label")
    return set(labels)


def readiness_jobs(readiness_path):
    readiness = read_json(readiness_path)
    jobs = []
    for method in ("mosp", "vidosc", "claude"):
        row = readiness["full_release"][method]
        if row["status"] != "inputs_ready":
            raise ValueError(f"Full inputs not ready: {method}")
        jobs.append({"id": method + "-full41", "method": method,
                     "variant": method, "scope": "full_r2_release", **row["manifest"]})
    variants = (("mosp", "mosp"), ("vidosc", "vidosc"), ("claude", "claude"),
                ("tubeletgraph_gpt4.1", "gpt-4.1"), ("tubeletgraph_gpt5.6_sol", "gpt-5.6-sol"))
    for name, variant in variants:
        row = readiness["matched_four_video_comparison"][name]
        if row["status"] != "inputs_ready":
            raise ValueError(f"Comparison inputs not ready: {name}")
        method = "tubeletgraph" if name.startswith("tubeletgraph") else name
        slug = name.replace("_", "-").replace(".", "")
        jobs.append({"id": slug + "-matched4", "method": method,
                     "variant": variant, "scope": "subset_comparison", **row["manifest"]})
    return jobs


def stage_manifests(jobs, destination, campaign_id):
    """Copy unique bytes, rewrite only paths, and preserve source fingerprints."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", campaign_id):
        raise ValueError("Invalid campaign ID")
    if not jobs or len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Jobs must be nonempty and have unique IDs")
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", job["id"]):
            raise ValueError("Invalid job ID")
        if file_hash(job["path"]) != job["sha256"]:
            raise ValueError(f"Prepared manifest changed: {job['id']}")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    files, copied = {}, {}
    labels, vocabulary_paths = {}, {}
    staged_jobs = []

    def freeze(source, relative, expected=None):
        source = Path(source).resolve()
        digest = file_hash(source)
        if expected is not None and digest != expected:
            raise ValueError(f"Input checksum mismatch: {source}")
        relative = Path(relative)
        target = destination / relative
        name = relative.as_posix()
        if name in files:
            if files[name]["sha256"] != digest:
                raise ValueError(f"Different bytes for shared bundle path: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_hash(target) != digest:
                raise ValueError(f"Copy checksum mismatch: {name}")
            files[name] = {"sha256": digest, "bytes": target.stat().st_size}
        copied[str(source)] = name
        return name

    for job in jobs:
        path = Path(job["path"]).resolve()
        original = read_json(path)
        config = copy.deepcopy(original)
        if config["method"] != job["method"]:
            raise ValueError("Job method disagrees with prepared manifest")
        actual_scope = "subset_comparison" if config.get("comparison_subset") else "full_r2_release"
        if config.get("diagnostic_subset") or actual_scope != job["scope"]:
            raise ValueError("Launch jobs must preserve the prepared comparison scope")
        data_root = _source(path.parent, config["data_root"])
        datasets = sorted({row["dataset"] for row in config["videos"]})
        for dataset in datasets:
            annotation, objects, states = DATASETS[dataset]
            for source_relative in ("MOSCATO/annotations/ground_truth/" + annotation,
                                    "MOSCATO/vocabulary/" + objects,
                                    "MOSCATO/vocabulary/" + states):
                freeze(data_root / source_relative, "data/" + source_relative)
            vocabulary_paths[dataset] = "data/MOSCATO/vocabulary/" + states
            labels.setdefault(dataset, set())
        release = _source(path.parent, config["r2_manifest"])
        freeze(release, "shared/r2-manifest.json", config["r2_manifest_sha256"])
        config["data_root"] = "../data"
        config["r2_manifest"] = "../shared/r2-manifest.json"
        for video in config["videos"]:
            for spec in video["predictions"]:
                source = _source(path.parent, spec["path"])
                suffix = ".npz" if spec.get("format") in {"mosp_npz", "vidosc_npz"} else ".json"
                blob = "blobs/" + spec["sha256"] + suffix
                freeze(source, blob, spec["sha256"])
                labels[video["dataset"]].update(_predicted_labels(spec, source))
                spec["path"] = "../" + blob
        config["portable_source_manifest"] = {"path": str(path), "sha256": job["sha256"]}
        config["campaign_id"] = campaign_id
        manifest_relative = "manifests/" + job["id"] + ".json"
        manifest_path = destination / manifest_relative
        manifest_path.parent.mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        files[manifest_relative] = {"sha256": file_hash(manifest_path), "bytes": manifest_path.stat().st_size}
        staged_jobs.append({key: job[key] for key in ("id", "method", "variant", "scope")}
                           | {"manifest": manifest_relative,
                              "manifest_sha256": files[manifest_relative]["sha256"],
                              "source_manifest_sha256": job["sha256"],
                              "videos": len(config["videos"]), "datasets": datasets})
    mapping = {"vocabularies": vocabulary_paths,
               "labels": {dataset: sorted(values) for dataset, values in labels.items()}}
    mapping_path = destination / "mapping-input.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files["mapping-input.json"] = {"sha256": file_hash(mapping_path), "bytes": mapping_path.stat().st_size}
    campaign = {"schema_version": 1, "campaign_id": campaign_id, "jobs": staged_jobs,
                "mapping_input": "mapping-input.json", "files": files,
                "file_count": len(files), "total_bytes": sum(value["bytes"] for value in files.values()),
                "unique_mapping_labels": {dataset: len(values) for dataset, values in labels.items()}}
    (destination / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readiness", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    campaign = stage_manifests(readiness_jobs(args.readiness), args.destination, args.campaign_id)
    print(json.dumps({"campaign_id": campaign["campaign_id"], "jobs": len(campaign["jobs"]),
                      "files": campaign["file_count"], "bytes": campaign["total_bytes"],
                      "unique_mapping_labels": campaign["unique_mapping_labels"]}))


if __name__ == "__main__":
    main()
