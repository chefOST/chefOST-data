"""Frame-macro content metrics and pooled, label-agnostic timing F1."""

from __future__ import annotations

import math
from statistics import fmean, pstdev

THRESHOLDS = (0.25, 0.5, 0.75)


def mean(values):
    values = [value for value in values if value is not None]
    return fmean(values) if values else None


def match_events(gt, predicted, tolerance):
    """Maximum-cardinality, minimum-total-error one-to-one temporal matching.

    With sorted times and absolute-distance costs an optimal noncrossing
    matching exists. This dynamic program retains its count and total error.
    """
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Timing tolerance must be finite and nonnegative")
    gt, predicted = sorted(gt), sorted(predicted)
    previous = [(0, 0.0)] * (len(predicted) + 1)
    for truth in gt:
        current = [(0, 0.0)]
        for j, prediction in enumerate(predicted, 1):
            options = [previous[j], current[j - 1]]
            error = abs(truth - prediction)
            if error <= tolerance:
                count, cost = previous[j - 1]
                options.append((count + 1, cost + error))
            current.append(max(options, key=lambda item: (item[0], -item[1])))
        previous = current
    return previous[-1]


def intervals(mask):
    start = None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None


def events(states, times, start, end):
    return [times[t] for t in range(start + 1, end) if states[t] != states[t - 1]]


def selected(window, threshold):
    if window is None:
        return ()
    if "probabilities" in window:
        return tuple(label for label, probability in window["probabilities"].items()
                     if probability >= threshold)
    return tuple(window["labels"])


def frame_windows(frames, windows):
    result, position = [], 0
    for frame in frames:
        while position < len(windows) and windows[position]["end_frame"] <= frame:
            position += 1
        window = windows[position] if position < len(windows) else None
        result.append(window if window and window["start_frame"] <= frame else None)
    return result


def evaluate(videos, vocabularies, *, assisted=None, embeddings=None, method):
    """Evaluate validated videos prepared by ``load_run``.

    Each ingredient owns GT, presence, timing-visible mask, declared initial
    available frame and windows. VidOSC broadcasts one stream during loading.
    """
    if not videos:
        raise ValueError("No evaluation videos")
    thresholds = THRESHOLDS if method in {"mosp", "vidosc"} else (None,)
    gaps = []
    prepared = []
    for video in videos:
        times = [frame / video["fps"] for frame in video["frames"]]
        vocabulary = vocabularies[video["dataset"]]
        items = []
        for ingredient, item in video["ingredients"].items():
            gt_cache = {}
            gt = []
            for labels in item["gt"]:
                key = tuple(sorted(labels))
                if key not in gt_cache:
                    gt_cache[key] = vocabulary.ground_truth(labels)
                gt.append(gt_cache[key])
            present = [i for i, value in enumerate(item["present"]) if value]
            if not present:
                continue
            left, right = present[0], present[-1]
            eligible = [left <= i <= right and bool(states) for i, states in enumerate(gt)]
            timing_mask = [value and item["visible"][i] for i, value in enumerate(eligible)]
            for start, end in intervals(timing_mask):
                ev = events(gt, times, start, end)
                gaps.extend(b - a for a, b in zip(ev, ev[1:]))
            available = item["available_from_frame"]
            scored = [value and frame >= available for value, frame in zip(eligible, video["frames"])]
            timed = [value and frame >= available for value, frame in zip(timing_mask, video["frames"])]
            items.append((ingredient, item, gt, scored, timed,
                          frame_windows(video["frames"], item["windows"])))
        prepared.append((video, times, items))
    tolerance = pstdev(gaps) if len(gaps) >= 2 else 0.0
    output = {"method": method, "thresholds": list(thresholds),
              "timing_tolerance_seconds": tolerance, "gt_interevent_gap_count": len(gaps),
              "vidosc_assignment": "broadcast_to_every_gt_ingredient" if method == "vidosc" else None,
              "embedding": embeddings.metadata if embeddings else None, "scores": []}
    for threshold in thresholds:
        for mode in ("vocabulary_only", "model_assisted") if assisted else ("vocabulary_only",):
            per_video, counts = {}, {"tp": 0, "fp": 0, "fn": 0, "total_matched_error_seconds": 0.0}
            for video, times, items in prepared:
                vocabulary = vocabularies[video["dataset"]]
                per_ingredient = {}
                for name, item, gt, scored, timed, windows in items:
                    raw = [selected(window, threshold) for window in windows]
                    prediction_cache = {}
                    predictions = []
                    for labels in raw:
                        if labels not in prediction_cache:
                            prediction_cache[labels] = (vocabulary.strict(labels) if mode == "vocabulary_only"
                                                       else assisted.map(video["dataset"], labels))
                        predictions.append(prediction_cache[labels])
                    p, r, c = [], [], []
                    scored_count = 0
                    for index, eligible in enumerate(scored):
                        if not eligible:
                            continue
                        scored_count += 1
                        hit = len(predictions[index] & gt[index])
                        if predictions[index]:
                            p.append(hit / len(predictions[index]))
                        r.append(hit / len(gt[index]))
                        if embeddings:
                            c.append(embeddings.similarity(raw[index], [label for _, label in gt[index]]))
                    timing = {"tp": 0, "fp": 0, "fn": 0, "total_matched_error_seconds": 0.0}
                    for start, end in intervals(timed):
                        ge = events(gt, times, start, end)
                        pe = events(predictions, times, start, end)
                        matched, error = match_events(ge, pe, tolerance)
                        timing["tp"] += matched
                        timing["fn"] += len(ge) - matched
                        timing["fp"] += len(pe) - matched
                        timing["total_matched_error_seconds"] += error
                    for key in counts:
                        counts[key] += timing[key]
                    per_ingredient[name] = {"precision": mean(p), "recall": mean(r),
                        "cosine_similarity": mean(c), "eligible_frames": scored_count,
                        "predicted_frames": len(p), "prediction_coverage": len(p) / scored_count if scored_count else None,
                        "initial_available_from_frame": item["available_from_frame"], "timing": timing,
                        "native_diagnostics": item.get("diagnostics", {})}
                per_video[video["key"]] = {metric: mean(row[metric] for row in per_ingredient.values())
                    for metric in ("precision", "recall", "cosine_similarity")}
                per_video[video["key"]]["ingredients"] = per_ingredient
                per_video[video["key"]]["contributing_ingredients"] = {
                    metric: sum(row[metric] is not None for row in per_ingredient.values())
                    for metric in ("precision", "recall", "cosine_similarity")}
            aggregate = {metric: mean(row[metric] for row in per_video.values())
                         for metric in ("precision", "recall", "cosine_similarity")}
            total = 2 * counts["tp"] + counts["fp"] + counts["fn"]
            counts["f1"] = 2 * counts["tp"] / total if total else None
            output["scores"].append({"threshold": threshold, "mapping": mode, **aggregate,
                "timing": counts, "videos": per_video,
                "contributing_videos": {metric: sum(row[metric] is not None for row in per_video.values())
                    for metric in ("precision", "recall", "cosine_similarity")}})
    return output
