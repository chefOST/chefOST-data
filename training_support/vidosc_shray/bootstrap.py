"""Locate the model checkout without copying its inference implementation."""
import os
import sys
from pathlib import Path

def configure(stage):
    value = os.environ.get("CHEFOST_VIDOSC_SOURCE")
    if not value:
        raise RuntimeError("Set CHEFOST_VIDOSC_SOURCE to the VidOSC checkout containing experiments/shray_multistate")
    source = Path(value).expanduser().resolve() / "experiments" / "shray_multistate"
    if not (source / "temporal_states.py").is_file():
        raise FileNotFoundError(source / "temporal_states.py")
    selected = source / stage
    sys.path[:0] = [str(selected), str(source / "broader"), str(source)]
    return selected
