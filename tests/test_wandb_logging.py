from types import SimpleNamespace

from chefost_eval.wandb_logging import publish_result, run_id, score_rows


def result():
    return {"method": "vidosc", "provenance": {"evaluator_revision": "abc"},
        "timing_tolerance_seconds": 2,
        "scores": [{"threshold": .25, "mapping": "vocabulary_only", "precision": .5,
                    "recall": .75, "cosine_similarity": .8,
                    "timing": {"f1": .4, "tp": 2, "fp": 3, "fn": 3},
                    "videos": {"CMU/a": {"ingredients": {"egg": {"precision": .5, "recall": .75,
                        "cosine_similarity": .8, "eligible_frames": 100, "prediction_coverage": 1}}}}}]}


def test_deterministic_run_ids_and_explicit_operating_point():
    assert run_id("campaign", "a") == run_id("campaign", "a")
    assert run_id("campaign", "a") != run_id("campaign", "b")
    assert score_rows(result()) == [[.25, "vocabulary_only", .5, .75, .8, .4, 2, 3, 3]]


def test_publishes_metrics_tables_and_pinned_artifact(tmp_path):
    calls, artifacts = [], []

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.metadata, self.files = name, metadata, []
            self.qualified_name = "chefOST/full-run-evals/" + name + ":v0"
            artifacts.append(self)
        def add_file(self, path, name): self.files.append((path, name))
        def wait(self, timeout): return self

    run = SimpleNamespace(id="job", summary={}, log=calls.append, log_artifact=lambda a: a)
    wandb = SimpleNamespace(Artifact=Artifact, Table=lambda **kw: kw)
    paths = [tmp_path / name for name in ["scores.json", "manifest.json", "mapping.json"]]
    qualified = publish_result(run, wandb, result(), *paths)
    assert qualified.endswith(":v0")
    assert run.summary["vocabulary_only/threshold_0_25/precision"] == .5
    assert calls[0]["evaluation/results"]["columns"][0:2] == ["threshold", "mapping"]
    assert artifacts[0].metadata["provenance"]["evaluator_revision"] == "abc"
    assert [name for _, name in artifacts[0].files] == ["scores.json", "manifest.json", "assisted-mapping.json"]
