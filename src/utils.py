from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def relpath(path: str | Path, root: str | Path) -> str:
    return os.path.relpath(str(path), str(root))


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from e
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def frame_sort_key(frame: dict) -> tuple:
    return (float(frame.get("timestamp", 0.0)), str(frame.get("frame_id", "")))
