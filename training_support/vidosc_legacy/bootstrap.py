"""Locate model and data-loader code without duplicating it in the evaluator repo."""
import os
from pathlib import Path
import sys


def configure():
    source = os.environ.get("CHEFOST_VIDOSC_SOURCE")
    if not source:
        raise RuntimeError("Set CHEFOST_VIDOSC_SOURCE to the VidOSC repository checkout")
    root = Path(source).expanduser().resolve()
    model_dir = root / "VidOSC"
    if not (model_dir / "model.py").is_file():
        raise ValueError("CHEFOST_VIDOSC_SOURCE must contain VidOSC/model.py")
    for path in (root / "third_party/compat", root / "third_party/lookforthechange_cpu", model_dir):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root
