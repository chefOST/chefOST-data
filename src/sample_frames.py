from __future__ import annotations

from pathlib import Path
from .utils import ensure_dir, write_json, relpath


def sample_video_frames(video: dict, config: dict) -> list[dict]:
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError("OpenCV is required for frame sampling: pip install opencv-python") from e

    root = Path(config.get("dataset_root", "dataset"))
    video_path = Path(video["video_path"])
    if not video_path.is_absolute():
        video_path = root / video_path
    out_dir = ensure_dir(root / "frames" / video["video_id"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = float(config.get("fps", 1))
    step = max(int(round(source_fps / target_fps)), 1)

    frames: list[dict] = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / source_fps
            frame_id = f"frame_{saved_idx:03d}"
            out_path = out_dir / f"{frame_id}.png"
            cv2.imwrite(str(out_path), frame)
            frames.append({
                "frame_id": frame_id,
                "timestamp": round(timestamp, 3),
                "image_path": relpath(out_path, root),
            })
            saved_idx += 1
        frame_idx += 1
    cap.release()

    write_json(root / "metadata" / f"{video['video_id']}_frames.json", frames)
    return frames


def run(video: dict, config: dict) -> list[dict]:
    return sample_video_frames(video, config)
