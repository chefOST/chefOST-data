import json
from pathlib import Path

import pytest

from chefost_eval.staging import validate_bundle_files, file_hash


def test_bundle_validation_accepts_symlinked_volume_root(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    source = actual / "source.json"
    source.write_text('{"value": 1}')
    mount = tmp_path / "cache"
    mount.symlink_to(actual, target_is_directory=True)
    validate_bundle_files(mount, {"source.json": {"bytes": source.stat().st_size,
                                                 "sha256": file_hash(source)}})


def test_bundle_validation_rejects_escape_and_bad_content(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    with pytest.raises(ValueError, match="escapes"):
        validate_bundle_files(root, {"../outside.json": {"bytes": 2, "sha256": file_hash(outside)}})
    inside = root / "inside.json"
    inside.write_text('{}')
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle_files(root, {"inside.json": {"bytes": 2, "sha256": "0" * 64}})

from chefost_eval.manifest import load_run
from chefost_eval.staging import file_hash, stage_manifests


def job():
    path = Path(__file__).parent / "fixtures/minimal/manifest.json"
    return {"id": "vidosc-example", "method": "vidosc", "variant": "vidosc",
            "scope": "full_r2_release", "path": str(path), "sha256": file_hash(path)}


def test_bundle_is_portable_and_deduplicates_prediction_bytes(tmp_path):
    first = job()
    second = {**first, "id": "vidosc-second"}
    staged = tmp_path / "inputs"
    result = stage_manifests([first, second], staged, "test-campaign")
    assert len(result["jobs"]) == 2
    assert len(list((staged / "blobs").iterdir())) == 1
    moved = tmp_path / "relocated"
    staged.rename(moved)
    for item in result["jobs"]:
        _, videos, _, _, metadata = load_run(moved / item["manifest"], metadata_only=True)
        assert len(videos) == 1
        assert metadata["scope"] == "full_r2_release"
    mapping = json.loads((moved / "mapping-input.json").read_text())
    assert set(mapping) == {"labels", "vocabularies"}
    assert mapping["labels"] == {"CMU": ["cooked", "raw"]}
    assert "test" not in json.dumps(mapping["labels"])


def test_bundle_refuses_modified_prepared_manifest_before_writes(tmp_path):
    with pytest.raises(ValueError, match="manifest changed"):
        stage_manifests([{**job(), "sha256": "0" * 64}], tmp_path / "inputs", "test")
    assert not (tmp_path / "inputs").exists()


def test_bundle_refuses_overwriting_existing_destination(tmp_path):
    with pytest.raises(FileExistsError):
        stage_manifests([job()], tmp_path, "test")
