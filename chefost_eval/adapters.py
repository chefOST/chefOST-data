"""Convert saved native outputs to explicit half-open source-frame windows."""

from __future__ import annotations

import math

from .labels import normalized, read_json


def validate_stream(stream, *, probabilities):
    available = stream.get("available_from_frame")
    if type(available) is not int or available < 0:
        raise ValueError("Declare available_from_frame independently of scoring thresholds")
    windows = stream.get("windows")
    if not isinstance(windows, list):
        raise ValueError("A prediction stream requires windows")
    previous_end = -1
    for window in windows:
        start, end = window.get("start_frame"), window.get("end_frame")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start or start < previous_end:
            raise ValueError("Prediction windows must be ordered, nonoverlapping, half-open source-frame intervals")
        previous_end = end
        if start < available:
            raise ValueError("Prediction windows precede declared initial availability")
        key = "probabilities" if probabilities else "labels"
        values = window.get(key)
        if (probabilities and not isinstance(values, dict)) or (not probabilities and not isinstance(values, list)):
            raise ValueError(f"Prediction window requires {key}")
        for label in values:
            normalized(label)
            if probabilities:
                value = values[label]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("State probabilities must be finite numbers in [0,1]")
    return stream


def load_predictions(spec, base, method, object_vocabulary):
    path = base / spec["path"]
    kind = spec.get("format", "windows")
    if kind == "windows":
        payload = read_json(path)
    elif kind in {"mosp_npz", "vidosc_npz"}:
        import numpy as np
        if (kind == "mosp_npz") != (method == "mosp"):
            raise ValueError("NPZ adapter does not match method")
        with np.load(path, allow_pickle=False) as archive:
            values = archive["state_probabilities" if method == "mosp" else "state"]
        labels = spec["state_labels"]
        starts, end = spec["window_start_frames"], spec["end_frame"]
        if not labels or len(labels) != len(set(labels)) or not starts:
            raise ValueError("NPZ adapter requires unique state labels and explicit source-frame window starts")
        objects = spec["object_labels"] if method == "mosp" else ["*"]
        if values.shape != (len(starts), len(objects) * len(labels)):
            raise ValueError("Saved probability shape differs from declared vocabulary/timeline")
        if method == "vidosc" and not np.allclose(values.sum(axis=1), 1, atol=1e-5):
            raise ValueError("VidOSC adapter requires native softmax probabilities before vocabulary projection")
        streams = {}
        ends = [*starts[1:], end]
        for index, obj in enumerate(objects):
            streams[obj] = {"available_from_frame": starts[0], "windows": [
                {"start_frame": start, "end_frame": stop, "probabilities": dict(zip(labels,
                    [float(x) for x in values[t, index * len(labels):(index + 1) * len(labels)]]))}
                for t, (start, stop) in enumerate(zip(starts, ends))]}
        payload = {"broadcast": streams["*"]} if method == "vidosc" else {"objects": streams}
    elif kind == "claude_transitions":
        if method != "claude":
            raise ValueError("Claude adapter requires method=claude")
        source = read_json(path)
        grouped = {obj: [] for obj in spec["object_labels"]}
        for event in source["transitions"]:
            obj = object_vocabulary.canonical(event["object"])
            if obj not in grouped:
                raise ValueError("Claude transition object is absent from declared native targets")
            normalized(event["state_after"])
            grouped[obj].append(event)
        streams = {}
        for obj, events in grouped.items():
            events.sort(key=lambda event: event["frame"])
            frames = [event["frame"] for event in events]
            if len(frames) != len(set(frames)):
                raise ValueError("Ambiguous duplicate Claude transition timestamp")
            streams[obj] = {"available_from_frame": frames[0] if frames else spec["end_frame"],
                "windows": [{"start_frame": event["frame"], "end_frame": end, "labels": [event["state_after"]]}
                            for event, end in zip(events, [*frames[1:], spec["end_frame"]])]}
        payload = {"objects": streams}
    elif kind == "tubeletgraph_timeline":
        if method != "tubeletgraph":
            raise ValueError("TubeletGraph adapter requires method=tubeletgraph")
        source = read_json(path)
        windows = []
        for interval in source["intervals"]:
            labels = [source["obj_info"][identifier]["desc"]
                      for identifier in interval["resolved_semantic_ids"]]
            windows.append({"start_frame": interval["source_start"], "end_frame": interval["source_end"],
                            "labels": labels})
        incomplete = [interval["source_start"] for interval in source["intervals"]
                      if interval.get("semantic_status") != "complete"]
        payload = {"objects": {spec["object_label"]: {"available_from_frame": source["source_start"],
            "windows": windows, "diagnostics": {"native_status": source.get("status"),
                "incomplete_semantic_interval_starts": incomplete,
                "coverage_diagnostics": source.get("coverage_diagnostics", {})}}}}
    else:
        raise ValueError(f"Unknown prediction format: {kind}")
    if method == "vidosc":
        if set(payload) != {"broadcast"}:
            raise ValueError("VidOSC must supply exactly one broadcast stream, never a GT-selected ingredient")
        return {"*": validate_stream(payload["broadcast"], probabilities=True)}
    if set(payload) != {"objects"} or not isinstance(payload["objects"], dict):
        raise ValueError("Predictions require object-keyed streams")
    result = {}
    for obj, stream in payload["objects"].items():
        canonical = object_vocabulary.canonical(obj)
        if canonical is None or canonical in result:
            raise ValueError("Unknown or duplicate canonical prediction object")
        result[canonical] = validate_stream(stream, probabilities=method == "mosp")
    return result
