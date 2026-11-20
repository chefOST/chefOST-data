"""Fixed vocabulary and cached model-assisted normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(label):
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Labels must be nonempty strings")
    return " ".join(label.casefold().split())


class Vocabulary:
    def __init__(self, raw, kind="s"):
        names = raw.get(kind + "2i")
        if not isinstance(names, dict) or not names:
            raise ValueError("Vocabulary needs canonical label IDs")
        self.names = frozenset(names)
        self.lookup = {}
        for name in self.names:
            self._add(name, name)
        aliases = raw.get("all2one", {})
        if not isinstance(aliases, dict):
            raise ValueError("all2one must be a dictionary")
        for alias, canonical in aliases.items():
            if canonical not in self.names:
                raise ValueError(f"Unknown canonical alias target: {canonical}")
            # An empty dictionary key is not a positive prediction or GT state.
            if isinstance(alias, str) and alias.strip():
                self._add(alias, canonical)

    def _add(self, label, canonical):
        key = normalized(label)
        if key in self.lookup and self.lookup[key] != canonical:
            raise ValueError(f"Ambiguous normalized vocabulary entry: {label}")
        self.lookup[key] = canonical

    def canonical(self, label):
        return self.lookup.get(normalized(label))

    def strict(self, labels):
        return frozenset(("canonical", self.canonical(label)) if self.canonical(label)
                         else ("unmatched", normalized(label)) for label in labels)

    def ground_truth(self, labels):
        result = set()
        for label in labels:
            name = self.canonical(label)
            if name is None:
                raise ValueError(f"Unknown GT state: {label}")
            result.add(("canonical", name))
        return frozenset(result)


class AssistedMapping:
    """A frozen, auditable cache; model calls are never made during scoring."""

    def __init__(self, raw, vocabularies, vocabulary_hashes):
        if raw.get("model") != "gpt-6.2-sol":
            raise ValueError("Assisted mappings must identify the selected gpt-6.2-sol model")
        prompt_hash = raw.get("prompt_sha256", "")
        if len(prompt_hash) != 64 or any(c not in "0123456789abcdef" for c in prompt_hash):
            raise ValueError("Assisted mappings require prompt_sha256")
        if raw.get("vocabulary_sha256") != vocabulary_hashes:
            raise ValueError("Assisted mapping vocabulary revision mismatch")
        self.raw = raw
        self.vocabularies = vocabularies
        self.entries = raw.get("entries")
        if not isinstance(self.entries, dict):
            raise ValueError("Assisted mappings require dataset-keyed entries")

    def map(self, dataset, labels):
        result = set()
        entries = self.entries.get(dataset, {})
        for label in labels:
            if label not in entries:
                raise ValueError(f"Missing cached assisted mapping: {dataset}/{label}")
            entry = entries[label]
            if (not isinstance(entry, dict) or set(entry) != {"states", "unmatched"}
                    or not isinstance(entry["states"], list)
                    or type(entry["unmatched"]) is not bool):
                raise ValueError(f"Invalid assisted mapping: {dataset}/{label}")
            for state in entry["states"]:
                if state not in self.vocabularies[dataset].names:
                    raise ValueError(f"Assisted mapping invented canonical state: {state}")
                result.add(("canonical", state))
            if entry["unmatched"] or not entry["states"]:
                result.add(("unmatched", normalized(label)))
        return frozenset(result)
