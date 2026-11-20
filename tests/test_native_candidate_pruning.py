import pytest

from chefost_eval.adapters import load_predictions
from chefost_eval.labels import Vocabulary
from chefost_eval.scoring import THRESHOLDS, selected


def load(tmp_path, values, method="mosp"):
    np = pytest.importorskip("numpy")
    labels = ["raw", "cooked", "cut", "other"]
    path = tmp_path / "probabilities.npz"
    np.savez(path, **{"state_probabilities" if method == "mosp" else "state": values})
    return load_predictions({"path": path.name, "format": method + "_npz",
                             "object_labels": ["eggs"], "state_labels": labels,
                             "window_start_frames": list(range(len(values))), "end_frame": len(values)},
                            tmp_path, method, Vocabulary({"o2i": {"eggs": 0}}, "o"))


def test_pruning_preserves_all_threshold_selections_and_equality(tmp_path):
    values = [[0.249, 0.25, 0.5, 0.75], [0, 0.1, 0.2, 0.24]]
    stream = load(tmp_path, values)["eggs"]
    labels = ["raw", "cooked", "cut", "other"]
    for original, window in zip(values, stream["windows"]):
        for threshold in THRESHOLDS:
            assert selected(window, threshold) == tuple(
                label for label, value in zip(labels, original) if value >= threshold)
    assert stream["windows"][0]["probabilities"] == {"cooked": .25, "cut": .5, "other": .75}
    assert stream["windows"][1]["probabilities"] == {}
    assert stream["available_from_frame"] == 0


@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf"), 1.1])
def test_invalid_native_values_are_rejected_before_pruning(tmp_path, invalid):
    with pytest.raises(ValueError, match="finite numbers"):
        load(tmp_path, [[invalid, .25, .5, .75]])


def test_vidosc_sum_is_checked_before_pruning_without_renormalization(tmp_path):
    window = load(tmp_path, [[.1, .15, .25, .5]], method="vidosc")["*"]["windows"][0]
    assert window["probabilities"] == {"cut": .25, "other": .5}
    assert sum(window["probabilities"].values()) == .75
    with pytest.raises(ValueError, match="native softmax"):
        load(tmp_path, [[.1, .15, .25, .6]], method="vidosc")


def test_target_filter_does_not_reindex_columns_or_hide_invalid_unselected_scores(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "p.npz"
    spec = {"path": path.name, "format": "mosp_npz", "state_labels": ["raw", "cooked"],
            "object_labels": ["pan", "eggs"], "window_start_frames": [0], "end_frame": 1}
    vocabulary = Vocabulary({"o2i": {"pan": 0, "eggs": 1}}, "o")
    np.savez(path, state_probabilities=[[.9, .1, .25, .75]])
    result = load_predictions(spec, tmp_path, "mosp", vocabulary, target_objects={"eggs"})
    assert set(result) == {"eggs"}
    assert result["eggs"]["windows"][0]["probabilities"] == {"raw": .25, "cooked": .75}
    np.savez(path, state_probabilities=[[float("nan"), .1, .25, .75]])
    with pytest.raises(ValueError, match="finite numbers"):
        load_predictions(spec, tmp_path, "mosp", vocabulary, target_objects={"eggs"})
    np.savez(path, state_probabilities=[[.9, .1, .25, .75]])
    spec["object_labels"][0] = "unknown pan"
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        load_predictions(spec, tmp_path, "mosp", vocabulary, target_objects={"eggs"})
