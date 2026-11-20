"""Load a pinned evaluation set and saved predictions without inference."""

from __future__ import annotations

import math
from pathlib import Path

from .adapters import load_predictions
from .labels import Vocabulary, read_json, sha256

DATASETS = {
    "CMU": ("CMU/gt_annotations_cmu.json", "CMU/object_dict.json", "CMU/state_dict.json"),
    "EGTEA": ("EGTEA/gt_annotations_egtea.json", "EGTEA/obj_dict.json", "EGTEA/state_dict.json"),
}


def load_run(manifest_path):
    path = Path(manifest_path).resolve()
    config = read_json(path)
    if config.get("schema_version") != 1 or config.get("method") not in {"mosp", "vidosc", "claude", "tubeletgraph"}:
        raise ValueError("Manifest requires schema_version=1 and a supported method")
    if not config.get("annotation_revision") or not config.get("prediction_revision"):
        raise ValueError("Pin annotation and prediction revisions")
    root = (path.parent / config["data_root"]).resolve()
    release_path = path.parent / config["r2_manifest"]
    if sha256(release_path) != config.get("r2_manifest_sha256"):
        raise ValueError("R2 manifest checksum mismatch")
    release = read_json(release_path)
    available = {}
    for entry in release["entries"]:
        if entry.get("download_status") not in {"downloaded_verified_identity", "downloaded_user_selected_identity"}:
            continue
        key = (entry["dataset"], entry["ground_truth_video_id"])
        if key in available:
            raise ValueError("Duplicate R2 release video identity")
        available[key] = entry
    if type(config.get("diagnostic_subset", False)) is not bool:
        raise ValueError("diagnostic_subset must be an explicit boolean")
    policy = read_json(Path(__file__).with_name("ingredients.json"))["datasets"]
    vocabularies, object_vocabularies, annotations, hashes, videos = {}, {}, {}, {}, []
    seen = set()
    input_hashes = {str(release_path): sha256(release_path), str(path): sha256(path)}
    for record in config["videos"]:
        dataset, video_id = record["dataset"], record["video_id"]
        key = (dataset, video_id)
        if key in seen or key not in available:
            raise ValueError(f"Duplicate or unavailable R2 video: {key}")
        seen.add(key)
        if dataset not in DATASETS:
            raise ValueError("Supported released ingredient policies are CMU and EGTEA")
        if dataset not in annotations:
            annotation_file, objects_file, states_file = DATASETS[dataset]
            gt_path = root / "MOSCATO/annotations/ground_truth" / annotation_file
            obj_path = root / "MOSCATO/vocabulary" / objects_file
            state_path = root / "MOSCATO/vocabulary" / states_file
            annotations[dataset] = read_json(gt_path)
            object_vocabularies[dataset] = Vocabulary(read_json(obj_path), "o")
            vocabularies[dataset] = Vocabulary(read_json(state_path))
            hashes[dataset] = sha256(state_path)
            for source in (gt_path, obj_path, state_path):
                input_hashes[str(source)] = sha256(source)
        annotation = annotations[dataset][video_id]
        presence = annotation["object"]
        n = len(presence)
        if record.get("alignment_verified") is not True or not record.get("alignment_evidence"):
            raise ValueError("Each video requires verified annotation-to-source alignment and evidence")
        frames = record.get("gt_source_frames")
        if frames is None and record.get("annotation_indexing") == "source_frame":
            frames = list(range(n))
        if (not isinstance(frames, list) or len(frames) != n
                or any(type(frame) is not int or frame < 0 for frame in frames)
                or any(a >= b for a, b in zip(frames, frames[1:]))):
            raise ValueError("Supply a strictly increasing source-frame index for every GT row")
        fps = record["source_fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("source_fps must be finite and positive")
        if record.get("constant_frame_rate") is not True:
            raise ValueError("This source-frame timeline adapter requires verified constant frame rate")
        objects = object_vocabularies[dataset]
        canonical_presence = []
        for names in presence:
            canonical = {objects.canonical(name) for name in names}
            if None in canonical:
                raise ValueError("Unknown GT object")
            canonical_presence.append(canonical)
        state_rows = {}
        for name, states in annotation["state"].items():
            canonical = objects.canonical(name)
            if canonical is None or len(states) != n:
                raise ValueError("Invalid GT state timeline")
            destination = state_rows.setdefault(canonical, [frozenset()] * n)
            for index, labels in enumerate(states):
                if not isinstance(labels, list):
                    raise ValueError("GT frame states must be lists")
                if labels:
                    destination[index] = destination[index].union(labels)
        predictions = {}
        specs = record["predictions"]
        if not isinstance(specs, list) or not specs:
            raise ValueError("Each video needs saved prediction sources")
        for spec in specs:
            source = (path.parent / spec["path"]).resolve()
            if sha256(source) != spec.get("sha256"):
                raise ValueError("Prediction checksum mismatch")
            loaded = load_predictions(spec, path.parent, config["method"], objects)
            if set(loaded) & set(predictions):
                raise ValueError("Duplicate prediction streams")
            predictions.update(loaded)
            input_hashes[str(source)] = sha256(source)
        identified_states = {name for name, rows in state_rows.items() if any(rows)}
        targets = (set().union(*canonical_presence) | identified_states) & set(policy[dataset]["include"])
        ingredients = {}
        for name in sorted(targets):
            stream = predictions.get("*" if config["method"] == "vidosc" else name)
            if stream is None:
                # A missing run is not an initial-state abstention; fail visibly.
                raise ValueError(f"Missing saved prediction stream for {dataset}/{video_id}/{name}")
            states = state_rows.get(name, [frozenset()] * n)
            visible_presence = [name in values for values in canonical_presence]
            # Either GT object presence or a known GT state identifies the
            # ingredient for outer bounds; state annotations do not prove visibility.
            present = [value or bool(labels) for value, labels in zip(visible_presence, states)]
            if record.get("visibility_source") == "gt_object_presence":
                visible = visible_presence
            else:
                visible = record.get("visibility", {}).get(name)
                if not isinstance(visible, list) or len(visible) != n or any(type(x) is not bool for x in visible):
                    raise ValueError("Timing requires an explicit visibility mask or declared GT-presence visibility source")
            ingredients[name] = {"gt": states,
                                 "present": present, "visible": visible, **stream}
        videos.append({"key": dataset + "/" + video_id, "dataset": dataset, "video_id": video_id,
                       "frames": frames, "fps": fps, "ingredients": ingredients})
    if not videos:
        raise ValueError("No videos selected")
    omitted = set(available) - seen
    if omitted and not config.get("diagnostic_subset", False):
        raise ValueError("Final evaluation must include all available R2 videos; explicitly mark diagnostic_subset for a partial check")
    metadata = {"annotation_revision": config["annotation_revision"], "prediction_revision": config["prediction_revision"],
                "input_sha256": input_hashes, "r2_manifest_sha256": config["r2_manifest_sha256"],
                "selected_videos": [v["key"] for v in videos],
                "scope": "diagnostic_subset" if config.get("diagnostic_subset", False) else "full_r2_release",
                "omitted_available_videos": [dataset + "/" + video for dataset, video in sorted(omitted)]}
    return config, videos, vocabularies, hashes, metadata
