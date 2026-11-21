import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chefost_eval.embeddings import QwenEmbeddings, cosine, label_text, required_label_sets
from chefost_eval.labels import Vocabulary


def test_batched_prefetch_reuses_exact_vectors_and_persistent_cache(tmp_path, monkeypatch):
    calls = []

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.asarray([[1, len(text)] for text in texts], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    progress = []
    provider = QwenEmbeddings("test-revision", cache_dir=tmp_path, batch_size=2)
    report = provider.prefetch([["raw"], ["cooked"], ["raw"], [], ["cut", "wet"]],
                               progress=lambda *args: progress.append(args))
    assert report["requested_unique_texts"] == report["newly_embedded_texts"] == 3
    assert [len(texts) for texts, _ in calls] == [2, 1]
    assert all(kwargs["prompt"] == "" and kwargs["normalize_embeddings"] is True for _, kwargs in calls)
    expected = cosine([1, len(label_text(["raw"]))], [1, len(label_text(["cooked"]))])
    assert provider.similarity(["raw"], ["cooked"]) == expected
    assert provider.vectors[label_text(["raw"])].dtype == np.float32
    assert progress[-1][:2] == (3, 3)
    resumed = QwenEmbeddings("test-revision", cache_dir=tmp_path)
    assert resumed.prefetch([["raw"], ["cooked"]])["newly_embedded_texts"] == 0
    assert resumed.similarity(["raw"], ["cooked"]) == expected
    assert len(calls) == 2
    with pytest.raises(ValueError, match="metadata differs"):
        QwenEmbeddings("different-revision", cache_dir=tmp_path)


def test_required_sets_follow_scored_support_and_all_thresholds():
    vocab = Vocabulary({"s2i": {"raw": 0, "cooked": 1}})
    video = {"dataset": "CMU", "frames": list(range(5)), "ingredients": {"egg": {
        "present": [True] * 5, "gt": [["raw"], [], ["cooked"], ["cooked"], []],
        "available_from_frame": 2,
        "windows": [
            {"start_frame": 0, "end_frame": 2, "probabilities": {"excluded": 1}},
            {"start_frame": 2, "end_frame": 4, "probabilities": {"cut": .8, "wet": .3}},
            {"start_frame": 4, "end_frame": 5, "probabilities": {"unknown_gt": 1}},
        ]}}}
    assert required_label_sets([video], {"CMU": vocab}, "mosp") == [
        ("cooked",), ("cut",), ("cut", "wet")]


def test_corrupt_cache_vector_cannot_silently_score(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **k: object()))
    provider = QwenEmbeddings("revision", cache_dir=tmp_path)
    with provider._path(label_text(["raw"])).open("wb") as handle:
        np.save(handle, np.array([float("nan"), 1], dtype=np.float32))
    with pytest.raises(ValueError, match="persisted"):
        provider.prefetch([["raw"]])
