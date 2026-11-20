import hashlib
import json

import pytest

from chefost_eval.recover_claude import normalize_keys, recover


def test_key_whitespace_only_preserves_values_and_audit_paths():
    payload, changes = normalize_keys('{"transitions":[{" state_after":"  Raw Egg  "}]}')
    assert payload == {"transitions": [{"state_after": "  Raw Egg  "}]}
    assert changes[0]["source_pointer"] == "/transitions/0/ state_after"
    assert changes[0]["derived_pointer"] == "/transitions/0/state_after"


@pytest.mark.parametrize("text", ['{"a":1," a":2}', '{"a":1,"a":2}', '{" ":1}'])
def test_ambiguous_or_empty_keys_rejected(text):
    with pytest.raises(ValueError):
        normalize_keys(text)


def setup_files(tmp_path, row):
    raw = tmp_path / "raw.json"
    samples = tmp_path / "samples.json"
    raw.write_text(json.dumps({"stop_reason": "end_turn", "id": "raw-id", "model": "native-model",
                              "content": [{"type": "text", "text": json.dumps({"transitions": [row]})}]}))
    samples.write_text(json.dumps({"samples": [{"source_frame": 10}]}))
    return raw, samples


def test_recovery_never_changes_source_files(tmp_path):
    row = {"frame": 10, "object": "eggs", "state_before": "whole", " state_after": " cracked "}
    raw, samples = setup_files(tmp_path, row)
    before = raw.read_bytes()
    payload, audit = recover(raw, samples, "eggs", tmp_path / "derived")
    assert raw.read_bytes() == before
    assert audit["sources"]["raw_response"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert payload["transitions"][0]["state_after"] == " cracked "
    assert audit["prediction_values_changed"] is False
    with pytest.raises(FileExistsError):
        recover(raw, samples, "eggs", tmp_path / "derived")


def test_missing_after_value_is_not_invented(tmp_path):
    raw, samples = setup_files(tmp_path, {"frame": 10, " object": "eggs", "state_before": "whole"})
    with pytest.raises(ValueError, match="missing"):
        recover(raw, samples, "eggs", tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_recovery_still_enforces_native_frames(tmp_path):
    raw, samples = setup_files(tmp_path, {"frame": 11, "object": "eggs", "state_before": "whole", " state_after": "cracked"})
    with pytest.raises(ValueError, match="not supplied"):
        recover(raw, samples, "eggs", tmp_path / "derived")
