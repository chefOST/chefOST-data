"""Regression cases for preserved training selection, separate from benchmark scores."""
import numpy as np
import pytest

from training_support.vidosc_shray.validation import ValidationCounts, choose_policy
from training_support.vidosc_shray.bootstrap import configure


def test_validation_counts_preserve_raw_frame_weighting_and_unknown_gaps():
    meta = {"fps": 2, "raw_frame_count": 4, "sample_anchors": [0, 2]}
    annotation = {"fps": 2, "states": [["raw"], ["raw"], ["cooked"], []]}
    counts = ValidationCounts(meta, annotation, ["raw", "cooked"], {})
    result = counts.measure(np.array([[True, False], [True, True]]))
    assert result["micro_f1"] == pytest.approx(6 / 7)
    assert result["precision"] == pytest.approx(3 / 4)
    assert result["macro_f1_gt_present"] == pytest.approx(0.9)


def test_legacy_validation_selection_keeps_original_grid_and_tie_order():
    meta = {"fps": 2, "raw_frame_count": 4, "sample_anchors": [0, 2]}
    annotation = {"fps": 2, "states": [["raw"], ["raw"], ["cooked"], ["cooked"]]}
    counts = ValidationCounts(meta, annotation, ["raw", "cooked"], {})
    selected = choose_policy({"a": np.array([[0.8, 0.2], [0.2, 0.8]])}, {"a": counts})
    assert selected["micro_f1"] == 1
    assert selected["precision"] == 1
    assert selected["threshold"] == pytest.approx(0.25)
    assert selected["smoothing"] == 1


def test_legacy_validation_rejects_incompatible_frame_clock():
    with pytest.raises(ValueError, match="clock"):
        ValidationCounts({"fps": 30, "raw_frame_count": 1, "sample_anchors": [0]},
                         {"fps": 10, "states": [["raw"]]}, ["raw"], {})


def test_training_support_requires_explicit_model_checkout(monkeypatch):
    monkeypatch.delenv("CHEFOST_VIDOSC_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="CHEFOST_VIDOSC_SOURCE"):
        configure("broader")
