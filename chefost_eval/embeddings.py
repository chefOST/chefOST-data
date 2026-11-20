"""Complete-label-set embeddings, with optional offline cached vectors."""

from __future__ import annotations

import json
import math

QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"


def label_text(labels):
    return json.dumps(sorted(set(labels)), ensure_ascii=False)


def cosine(a, b):
    if len(a) != len(b) or not a:
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

    def __init__(self, revision, device=None):
        if not revision:
            raise ValueError("Pin an explicit Qwen model revision")
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(QWEN_MODEL, revision=revision, device=device,
                                           tokenizer_kwargs={"padding_side": "left"})
        self.vectors = {}
        self.similarities = {}
        self.metadata = {"model": QWEN_MODEL, "revision": revision,
                         "text_format": "sorted_unique_labels_json", "instruction": ""}

    def similarity(self, predicted, truth):
        if not predicted:
            return 0.0
        texts = [label_text(predicted), label_text(truth)]
        missing = [text for text in texts if text not in self.vectors]
        if missing:
            vectors = self.encoder.encode(missing, normalize_embeddings=True, prompt="")
            self.vectors.update(zip(missing, [v.tolist() for v in vectors]))
        key = tuple(texts)
        if key not in self.similarities:
            self.similarities[key] = cosine(*(self.vectors[text] for text in texts))
        return self.similarities[key]
