"""Execute the real worker path against a symlinked mount and local test inputs."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from chefost_eval.labels import sha256
from chefost_eval.prepare_mapping import MODEL, PROMPT_SHA, SETTINGS, prepare
from chefost_eval.staging import stage_manifests


class Image:
    @classmethod
    def debian_slim(cls, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class App:
    def __init__(self, *args):
        pass

    def function(self, **kwargs):
        return lambda function: function

    def local_entrypoint(self):
        return lambda function: function


@pytest.fixture
def worker(monkeypatch, tmp_path):
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    modal = SimpleNamespace(App=App, Image=Image,
        Volume=SimpleNamespace(from_name=lambda *a, **k: volume),
        Secret=SimpleNamespace(from_name=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
    source = Path(__file__).parents[1] / "chefost_eval/cloud_run.py"
    spec = importlib.util.spec_from_file_location("test_cloud_worker_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = tmp_path / "actual"
    actual.mkdir()
    mount = tmp_path / "cache"
    mount.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(module, "_campaign_root", lambda name: mount / "campaigns" / name)
    original = Path(__file__).parent / "fixtures/minimal/manifest.json"
    campaign = stage_manifests([{"id": "vidosc", "method": "vidosc", "variant": "vidosc",
        "scope": "full_r2_release", "path": str(original), "sha256": sha256(original)}],
        actual / "campaigns/test/inputs", "test")
    return module, actual / "campaigns/test", campaign["jobs"][0]


def test_score_worker_completes_through_symlinked_mount(worker, monkeypatch):
    module, root, job = worker
    vocab_hash = sha256(root / "inputs/data/MOSCATO/vocabulary/CMU/state_dict.json")
    mapping = {"model": MODEL, "prompt_sha256": PROMPT_SHA,
        "vocabulary_sha256": {"CMU": vocab_hash}, "entries": {"CMU": {
            label: {"states": [label], "unmatched": False} for label in ("raw", "cooked")}}}
    (root / "mapping").mkdir()
    (root / "mapping/assisted-mapping.json").write_text(json.dumps(mapping))

    class Artifact:
        def __init__(self, name, **kwargs):
            self.qualified_name = "chefOST/full-run-evals/" + name + ":v0"

        def add_file(self, path, **kwargs):
            assert Path(path).is_file()

        def wait(self, **kwargs):
            pass

    calls = []
    run = SimpleNamespace(id="worker-test", url="https://example.test/worker", summary={},
        log=lambda values: calls.append(values), log_artifact=lambda value: value,
        finish=lambda **kwargs: calls.append({"finished": kwargs}))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs,
        Table=lambda **kwargs: kwargs, Artifact=Artifact))

    class Embeddings:
        metadata = {"model": "synthetic"}

        def __init__(self, *args, **kwargs):
            pass

        def prefetch(self, sets, *, progress, **kwargs):
            progress(len(sets), len(sets), .01)

        def similarity(self, predicted, truth):
            return float(set(predicted) == set(truth))

    monkeypatch.setattr("chefost_eval.embeddings.QwenEmbeddings", Embeddings)
    result = module.score_job("test", job, "test-revision")
    assert result["status"] == "complete"
    assert run.summary["phase"] == "complete"
    scores = json.loads((root / "results/vidosc/scores.json").read_text())
    assert len(scores["scores"]) == 6
    assert scores["scores"][0]["precision"] == .5
    assert scores["scores"][0]["recall"] == 1


@pytest.mark.parametrize("change", ["escape", "checksum"])
def test_worker_rejects_bad_manifest_before_wandb_or_models(worker, monkeypatch, change):
    module, root, job = worker
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: pytest.fail("Invalid manifest reached W&B")))
    if change == "escape":
        outside = root / "outside.json"
        outside.write_text('{}')
        job = {**job, "manifest": "../outside.json", "manifest_sha256": sha256(outside)}
    else:
        job = {**job, "manifest_sha256": "0" * 64}
    with pytest.raises(ValueError, match="escapes|checksum"):
        module.score_job("test", job, "test-revision")


def test_copied_complete_mapping_cache_needs_zero_new_requests(tmp_path):
    source, retry = tmp_path / "source", tmp_path / "retry"
    source.mkdir()
    retry.mkdir()
    vocab = retry / "states.json"
    vocab.write_text(json.dumps({"s2i": {"cooked": 0}}))
    request = retry / "mapping-input.json"
    request.write_text(json.dumps({"vocabularies": {"CMU": "states.json"},
                                   "labels": {"CMU": ["cooked", "warm"]}}))
    cache = {"model": MODEL, "prompt_sha256": PROMPT_SHA,
        "vocabulary_sha256": {"CMU": sha256(vocab)}, "generation_settings": SETTINGS,
        "entries": {"CMU": {"cooked": {"states": ["cooked"], "unmatched": False},
                             "warm": {"states": [], "unmatched": True}}},
        "responses": [{"id": "original-response"}]}
    (source / "mapping.json").write_text(json.dumps(cache))
    output = retry / "mapping.json"
    shutil.copyfile(source / "mapping.json", output)
    client = SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kwargs: pytest.fail("Complete copied cache made a paid request")))
    assert prepare(request, output, client=client) == cache
