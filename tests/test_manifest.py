import hashlib
import json

import pytest

from chefost_eval.__main__ import main
from chefost_eval.adapters import load_predictions, validate_stream
from chefost_eval.labels import Vocabulary
from chefost_eval.manifest import load_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def manifest(tmp_path):
    write(tmp_path / "MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json",
          {"test": {"object": [["eggs"], ["eggs"]], "state": {"eggs": [["raw"], ["cooked"]]}}})
    write(tmp_path / "MOSCATO/vocabulary/CMU/object_dict.json", {"o2i": {"eggs": 0}})
    write(tmp_path / "MOSCATO/vocabulary/CMU/state_dict.json", {"s2i": {"raw": 0, "cooked": 1}})
    r2hash = write(tmp_path / "r2.json", {"entries": [{"dataset": "CMU", "ground_truth_video_id": "test",
                                                     "download_status": "downloaded_verified_identity"}]})
    phash = write(tmp_path / "predictions.json", {"broadcast": {"available_from_frame": 0,
        "windows": [{"start_frame": 0, "end_frame": 2, "probabilities": {"raw": .6, "cooked": .4}}]}})
    data = {"schema_version": 1, "method": "vidosc", "data_root": ".", "annotation_revision": "test",
            "prediction_revision": "test", "r2_manifest": "r2.json", "r2_manifest_sha256": r2hash,
            "videos": [{"dataset": "CMU", "video_id": "test", "source_fps": 1,
                        "constant_frame_rate": True, "alignment_verified": True,
                        "alignment_evidence": "synthetic one row per source frame",
                        "annotation_indexing": "source_frame", "visibility_source": "gt_object_presence",
                        "predictions": [{"path": "predictions.json", "sha256": phash, "format": "windows"}]}]}
    write(tmp_path / "manifest.json", data)
    return tmp_path / "manifest.json"


def test_saved_prediction_cli_and_broadcast(manifest, tmp_path):
    main([str(manifest), "--output", str(tmp_path / "scores.json")])
    result = json.loads((tmp_path / "scores.json").read_text())
    assert result["vidosc_assignment"] == "broadcast_to_every_gt_ingredient"
    assert result["scores"][0]["precision"] == .5
    assert result["scores"][0]["recall"] == 1
    assert result["scores"][2]["recall"] == 0
    with pytest.raises(SystemExit):
        main([str(manifest), "--output", str(tmp_path / "scores.json")])


def test_alignment_must_be_verified(manifest):
    data = json.loads(manifest.read_text())
    data["videos"][0]["alignment_verified"] = False
    write(manifest, data)
    with pytest.raises(ValueError, match="verified annotation"):
        load_run(manifest)


def test_prediction_hash_mismatch_fails(manifest):
    (manifest.parent / "predictions.json").write_text('{}')
    with pytest.raises(ValueError, match="Prediction checksum"):
        load_run(manifest)


def test_broadcast_every_ingredient_no_best_assignment(manifest):
    root = manifest.parent
    write(root / "MOSCATO/vocabulary/CMU/object_dict.json", {"o2i": {"eggs": 0, "bread slices": 1}})
    write(root / "MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json", {"test": {
        "object": [["eggs", "bread slices"]] * 2,
        "state": {"eggs": [["raw"]] * 2, "bread slices": [["cooked"]] * 2}}})
    _, videos, _, _, _ = load_run(manifest)
    ingredients = videos[0]["ingredients"]
    assert set(ingredients) == {"eggs", "bread slices"}
    assert ingredients["eggs"]["windows"] is ingredients["bread slices"]["windows"]


def test_overlapping_windows_rejected():
    with pytest.raises(ValueError, match="nonoverlapping"):
        validate_stream({"available_from_frame": 0, "windows": [
            {"start_frame": 0, "end_frame": 2, "labels": ["raw"]},
            {"start_frame": 1, "end_frame": 3, "labels": ["cooked"]}]}, probabilities=False)


def test_claude_does_not_backfill_state_before(tmp_path):
    write(tmp_path / "claude.json", {"transitions": [
        {"frame": 5, "object": "eggs", "state_before": "raw", "state_after": "cooked"}]})
    streams = load_predictions({"path": "claude.json", "format": "claude_transitions",
                                "object_labels": ["eggs"], "end_frame": 10}, tmp_path, "claude",
                               Vocabulary({"o2i": {"eggs": 0}}, "o"))
    assert streams["eggs"]["available_from_frame"] == 5
    assert streams["eggs"]["windows"] == [{"start_frame": 5, "end_frame": 10, "labels": ["cooked"]}]


def test_mosp_npz_preserves_object_columns_for_selectable_states(tmp_path):
    np = pytest.importorskip("numpy")
    np.savez(tmp_path / "p.npz", state_probabilities=[[.7, .3, .2, .8], [.1, .9, .6, .4]])
    streams = load_predictions({"path": "p.npz", "format": "mosp_npz", "state_labels": ["raw", "cooked"],
                                "object_labels": ["eggs", "bread"], "window_start_frames": [0, 3],
                                "end_frame": 6}, tmp_path, "mosp", Vocabulary({"o2i": {"eggs": 0, "bread": 1}}, "o"))
    assert streams["bread"]["windows"][0]["probabilities"] == {"cooked": .8}
    assert streams["eggs"]["windows"][1]["start_frame"] == 3


def test_partial_release_must_be_explicit(manifest):
    config = json.loads(manifest.read_text())
    release = json.loads((manifest.parent / "r2.json").read_text())
    release["entries"].append({"dataset": "CMU", "ground_truth_video_id": "omitted",
                               "download_status": "downloaded_verified_identity"})
    config["r2_manifest_sha256"] = write(manifest.parent / "r2.json", release)
    write(manifest, config)
    with pytest.raises(ValueError, match="all available R2"):
        load_run(manifest)
    config["diagnostic_subset"] = True
    write(manifest, config)
    metadata = load_run(manifest)[-1]
    assert metadata["scope"] == "diagnostic_subset"
    assert metadata["omitted_available_videos"] == ["CMU/omitted"]


def test_export_mapping_input_excludes_gt(manifest, tmp_path):
    main([str(manifest), "--export-mapping-input", str(tmp_path / "mapper-input.json")])
    payload = json.loads((tmp_path / "mapper-input.json").read_text())
    assert set(payload) == {"vocabularies", "labels"}
    assert payload["labels"] == {"CMU": ["cooked", "raw"]}
    assert "test" not in json.dumps(payload["labels"])


def test_tubeletgraph_incomplete_native_output_diagnostics(tmp_path):
    write(tmp_path / "timeline.json", {"source_start": 3, "status": "incomplete",
        "obj_info": {"0": {"desc": "raw egg"}}, "coverage_diagnostics": {"missing": 1},
        "intervals": [{"source_start": 3, "source_end": 5, "resolved_semantic_ids": ["0"],
                       "semantic_status": "incomplete"}]})
    streams = load_predictions({"path": "timeline.json", "format": "tubeletgraph_timeline", "object_label": "eggs"},
                               tmp_path, "tubeletgraph", Vocabulary({"o2i": {"eggs": 0}}, "o"))
    assert streams["eggs"]["diagnostics"]["incomplete_semantic_interval_starts"] == [3]
    assert streams["eggs"]["windows"][0]["labels"] == ["raw egg"]


def test_state_only_gt_ingredient_not_silently_dropped(manifest):
    write(manifest.parent / "MOSCATO/annotations/ground_truth/CMU/gt_annotations_cmu.json", {
        "test": {"object": [[], []], "state": {"eggs": [["raw"], ["cooked"]]}}})
    _, videos, _, _, _ = load_run(manifest)
    egg = videos[0]["ingredients"]["eggs"]
    assert egg["present"] == [True, True]
    assert egg["visible"] == [False, False]


def test_explicit_comparison_subset_retains_scope_and_omissions(manifest):
    config = json.loads(manifest.read_text())
    release_path = manifest.parent / "r2.json"
    release = json.loads(release_path.read_text())
    release["entries"].append({"dataset": "CMU", "ground_truth_video_id": "outside_comparison",
                              "download_status": "downloaded_verified_identity"})
    config["r2_manifest_sha256"] = write(release_path, release)
    config["comparison_subset"] = True
    config["subset_reason"] = "User-approved matched four-video comparison"
    write(manifest, config)
    metadata = load_run(manifest)[-1]
    assert metadata["scope"] == "subset_comparison"
    assert metadata["subset_reason"] == config["subset_reason"]
    assert metadata["omitted_available_videos"] == ["CMU/outside_comparison"]


@pytest.mark.parametrize("reason", [None, "", "  ", 3])
def test_comparison_subset_requires_reason(manifest, reason):
    config = json.loads(manifest.read_text())
    config.update(comparison_subset=True, subset_reason=reason)
    write(manifest, config)
    with pytest.raises(ValueError, match="subset_reason"):
        load_run(manifest)


def test_comparison_subset_cannot_be_labeled_diagnostic(manifest):
    config = json.loads(manifest.read_text())
    config.update(comparison_subset=True, subset_reason="Matched comparison", diagnostic_subset=True)
    write(manifest, config)
    with pytest.raises(ValueError, match="both a diagnostic"):
        load_run(manifest)


def test_committed_synthetic_manifest_example():
    from pathlib import Path
    config, videos, _, _, metadata = load_run(Path(__file__).parent / "fixtures/minimal/manifest.json")
    assert config["method"] == "vidosc"
    assert len(videos) == 1
    assert metadata["scope"] == "full_r2_release"


def test_metadata_only_loader_keeps_gates_without_retaining_prediction_arrays(manifest):
    config, videos, _, _, metadata = load_run(manifest, metadata_only=True)
    assert config["method"] == "vidosc"
    assert videos[0]["ingredients"] == {"eggs": {}}
    assert "frames" not in videos[0]
    assert metadata["scope"] == "full_r2_release"
    config["videos"][0]["alignment_verified"] = False
    write(manifest, config)
    with pytest.raises(ValueError, match="verified annotation"):
        load_run(manifest, metadata_only=True)
