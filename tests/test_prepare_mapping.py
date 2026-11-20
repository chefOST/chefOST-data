import json
from types import SimpleNamespace

import pytest

from chefost_eval.prepare_mapping import prepare


def inputs(tmp_path, extra=None):
    vocab = tmp_path / "states.json"
    vocab.write_text(json.dumps({"s2i": {"cooked": 0, "chopped": 1}}))
    spec = {"vocabularies": {"CMU": "states.json"}, "labels": {"CMU": ["cooked and blue"]}}
    spec.update(extra or {})
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(spec))
    return path


def test_mapping_cache_preserves_partial_unmatched_and_resumes(tmp_path):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        assert kwargs["model"] == "gpt-6.2-sol"
        assert "temperature" not in kwargs
        data = json.loads(kwargs["input"][1]["content"])
        assert set(data) == {"canonical_states", "predicted_phrases"}
        return SimpleNamespace(id="response-1", status="completed", output_text=json.dumps({
            "mappings": [{"text": "cooked and blue", "states": ["cooked"], "unmatched": True}]}))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    path, output = inputs(tmp_path), tmp_path / "cache.json"
    cache = prepare(path, output, client=client)
    assert cache["entries"]["CMU"]["cooked and blue"] == {"states": ["cooked"], "unmatched": True}
    assert prepare(path, output, client=client) == cache
    assert len(calls) == 1


def test_mapping_rejects_ground_truth_input_before_api_call(tmp_path):
    path = inputs(tmp_path, {"ground_truth": {"frame": ["cooked"]}})
    with pytest.raises(ValueError, match="only vocabularies"):
        prepare(path, tmp_path / "cache.json", client=object())


@pytest.mark.parametrize("states", [["invented"], []])
def test_mapping_rejects_unknown_canonical_and_preserves_empty_as_unmatched(tmp_path, states):
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        id="r", status="completed", output_text=json.dumps({"mappings": [
            {"text": "cooked and blue", "states": states, "unmatched": False}]}))))
    path, output = inputs(tmp_path), tmp_path / "cache.json"
    if states:
        with pytest.raises(ValueError, match="invalid canonical"):
            prepare(path, output, client=client)
        assert not output.exists()
    else:
        assert prepare(path, output, client=client)["entries"]["CMU"]["cooked and blue"]["unmatched"] is True
