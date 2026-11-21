"""Complete-label-set embeddings, with optional offline cached vectors."""

from __future__ import annotations

import json
import math
import hashlib
import time
from pathlib import Path

QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"


def label_text(labels):
    return json.dumps(sorted(set(labels)), ensure_ascii=False)


def required_label_sets(videos, vocabularies, method):
    """Collect complete texts needed by cosine on the scorer's eligible support.

    Visit source windows once per ingredient rather than reselecting labels for
    every frame. GT sets are canonical; output sets remain native and unmapped.
    """
    from .scoring import THRESHOLDS, frame_windows, selected
    thresholds = THRESHOLDS if method in {"mosp", "vidosc"} else (None,)
    result = set()
    for video in videos:
        vocabulary = vocabularies[video["dataset"]]
        for item in video["ingredients"].values():
            present = [i for i, value in enumerate(item["present"]) if value]
            if not present:
                continue
            left, right = present[0], present[-1]
            frame_to_window = frame_windows(video["frames"], item["windows"])
            seen_windows, gt_cache = set(), {}
            for i in range(left, right + 1):
                if video["frames"][i] < item["available_from_frame"] or not item["gt"][i]:
                    continue
                raw_gt = tuple(sorted(set(item["gt"][i])))
                if raw_gt not in gt_cache:
                    gt_cache[raw_gt] = tuple(sorted(label for _, label in vocabulary.ground_truth(raw_gt)))
                result.add(gt_cache[raw_gt])
                window = frame_to_window[i]
                if window is not None and id(window) not in seen_windows:
                    seen_windows.add(id(window))
                    for threshold in thresholds:
                        labels = tuple(sorted(set(selected(window, threshold))))
                        if labels:
                            result.add(labels)
    return sorted(result)


def cosine(a, b):
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Embedding dimensions must agree and be nonempty")
    if any(not math.isfinite(float(x)) for x in [*a, *b]):
        raise ValueError("Embedding values must be finite")
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(x) ** 2 for x in b))
    if not na or not nb:
        raise ValueError("Zero embedding vector")
    return max(-1.0, min(1.0, sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)))


class CachedEmbeddings:
    def __init__(self, raw):
        if raw.get("model") != QWEN_MODEL or not raw.get("revision"):
            raise ValueError("Embedding cache must identify Qwen3-Embedding-8B and its revision")
        if raw.get("text_format") != "sorted_unique_labels_json":
            raise ValueError("Unexpected embedding text format")
        if raw.get("instruction") != "":
            raise ValueError("Complete-label-set similarity uses no retrieval instruction")
        self.metadata = {k: raw[k] for k in ("model", "revision", "text_format", "instruction")}
        self.vectors = raw.get("vectors", {})
        self.similarities = {}

    def similarity(self, predicted, truth):
        if not predicted:
            return 0.0
        texts = [label_text(predicted), label_text(truth)]
        for text in texts:
            if text not in self.vectors:
                raise ValueError(f"Missing cached embedding for complete label set: {text}")
        key = tuple(texts)
        if key not in self.similarities:
            self.similarities[key] = cosine(*(self.vectors[text] for text in texts))
        return self.similarities[key]


class QwenEmbeddings:
    """Loaded only on an explicit --qwen-revision request, never by default."""

    def __init__(self, revision, device=None, *, cache_dir=None, batch_size=32):
        if not revision:
            raise ValueError("Pin an explicit Qwen model revision")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Embedding batch size must be a positive integer")
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(QWEN_MODEL, revision=revision, device=device,
                                           tokenizer_kwargs={"padding_side": "left"})
        self.vectors = {}
        self.similarities = {}
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.metadata = {"model": QWEN_MODEL, "revision": revision,
                         "text_format": "sorted_unique_labels_json", "instruction": ""}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = self.cache_dir / "metadata.json"
            if metadata_path.exists():
                if json.loads(metadata_path.read_text()) != self.metadata:
                    raise ValueError("Embedding cache metadata differs from selected model/encoding")
            else:
                metadata_path.write_text(json.dumps(self.metadata, indent=2) + "\n")

    def _path(self, text):
        return self.cache_dir / (hashlib.sha256(text.encode()).hexdigest() + ".npy")

    def _load(self, text):
        if text in self.vectors:
            return True
        if self.cache_dir and self._path(text).exists():
            import numpy as np
            vector = np.load(self._path(text), allow_pickle=False)
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError("Invalid persisted embedding vector")
            self.vectors[text] = vector
            return True
        return False

    def prefetch(self, label_sets, *, batch_size=None, progress=None):
        """Embed distinct complete sets in batches, retaining exact float32 values."""
        import numpy as np
        size = self.batch_size if batch_size is None else batch_size
        if type(size) is not int or size < 1:
            raise ValueError("Embedding batch size must be a positive integer")
        texts = sorted({label_text(labels) for labels in label_sets if labels})
        missing = [text for text in texts if not self._load(text)]
        start, done = time.monotonic(), 0
        if progress:
            progress(0, len(missing), 0.0)
        for offset in range(0, len(missing), size):
            batch = missing[offset:offset + size]
            vectors = self.encoder.encode(batch, batch_size=size, normalize_embeddings=True, prompt="")
            if len(vectors) != len(batch):
                raise ValueError("Embedding model returned a different batch length")
            for text, value in zip(batch, vectors):
                vector = np.asarray(value, dtype=np.float32)
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all() or not np.any(vector):
                    raise ValueError("Invalid model embedding vector")
                self.vectors[text] = vector
                if self.cache_dir:
                    destination = self._path(text)
                    temporary = destination.with_suffix(".npy.tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, vector, allow_pickle=False)
                    temporary.replace(destination)
            done += len(batch)
            if progress:
                progress(done, len(missing), time.monotonic() - start)
        return {"requested_unique_texts": len(texts), "newly_embedded_texts": len(missing),
                "seconds": time.monotonic() - start}

    def similarity(self, predicted, truth):
        if not predicted:
            return 0.0
        texts = [label_text(predicted), label_text(truth)]
        if any(not self._load(text) for text in texts):
            self.prefetch([predicted, truth])
        key = tuple(texts)
        if key not in self.similarities:
            self.similarities[key] = cosine(*(self.vectors[text] for text in texts))
        return self.similarities[key]
