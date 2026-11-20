import hashlib
import json
from pathlib import Path
import shutil

import pytest

from chefost_eval.preflight import inspect_manifest, main


@pytest.fixture
def manifest(tmp_path):
    source = Path(__file__).parent / "fixtures/minimal"
    shutil.copytree(source, tmp_path / "fixture")
    return tmp_path / "fixture/manifest.json"


def write(path, value):
    path.write_text(json.dumps(value))


def test_input_preflight_never_scores_or_loads_providers(manifest, monkeypatch):
    import chefost_eval.scoring as scoring
    import chefost_eval.embeddings as embeddings

    def forbidden(*args, **kwargs):
        pytest.fail("Input preflight must not score or initialize models")

    monkeypatch.setattr(scoring, "evaluate", forbidden)
    monkeypatch.setattr(embeddings, "QwenEmbeddings", forbidden)
    report = inspect_manifest(manifest)
    assert report["status"] == "inputs_ready"
    assert report["loader"]["videos"] == 1
    assert report["artifact_counts"] == {"listed": 1, "passed": 1}
    assert report["runtime_access_checked"] is False
    assert report["scores_computed"] is False


def test_alignment_remains_blocked_while_other_artifacts_are_checked(manifest):
    config = json.loads(manifest.read_text())
    config["videos"][0]["alignment_verified"] = False
    write(manifest, config)
    predictions = manifest.parent / "predictions.json"
    predictions.write_text('{}')
    report = inspect_manifest(manifest)
    assert report["status"] == "blocked"
    assert "verified annotation" in report["loader"]["error"]
    assert report["artifacts"][0]["error"] == "Prediction checksum mismatch"
    assert json.loads(manifest.read_text())["videos"][0]["alignment_verified"] is False


def test_native_shape_is_checked_after_unresolved_alignment(manifest):
    config = json.loads(manifest.read_text())
    config["videos"][0]["alignment_verified"] = False
    predictions = manifest.parent / "predictions.json"
    write(predictions, {"broadcast": {"available_from_frame": 0, "windows": [
        {"start_frame": 2, "end_frame": 1, "probabilities": {"raw": .5}}]}})
    config["videos"][0]["predictions"][0]["sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    write(manifest, config)
    report = inspect_manifest(manifest)
    assert "nonoverlapping" in report["artifacts"][0]["error"]


def test_loader_full_scope_requirement_is_not_bypassed(manifest):
    config = json.loads(manifest.read_text())
    release_path = manifest.parent / "r2.json"
    release = json.loads(release_path.read_text())
    release["entries"].append({"dataset": "CMU", "ground_truth_video_id": "missing",
                               "download_status": "downloaded_verified_identity"})
    write(release_path, release)
    config["r2_manifest_sha256"] = hashlib.sha256(release_path.read_bytes()).hexdigest()
    write(manifest, config)
    report = inspect_manifest(manifest)
    assert report["status"] == "blocked"
    assert "all available R2" in report["loader"]["error"]


def test_report_output_is_immutable_and_exit_status_explicit(manifest, tmp_path):
    output = tmp_path / "preflight.json"
    assert main([str(manifest), "--output", str(output)]) == 0
    with pytest.raises(SystemExit):
        main([str(manifest), "--output", str(output)])
    config = json.loads(manifest.read_text())
    config["videos"][0]["alignment_verified"] = False
    write(manifest, config)
    assert main([str(manifest), "--output", str(tmp_path / "blocked.json")]) == 2
