# ChefOST dataset generation

Builds a pseudo-labeled cooking video dataset for object state understanding.

## Usage

1. Put raw videos under `dataset/raw_videos/`.
2. Add video metadata either in `configs/dataset_config.yaml` under `videos:` or in `dataset/metadata/video_metadata.json`:

```json
[
  {
    "video_id": "onion_001",
    "video_path": "raw_videos/onion_001.mp4",
    "title": "How to dice an onion",
    "object": "onion",
    "task": "dice onion"
  }
]
```

3. Create and activate an isolated Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run the pipeline:

```bash
python src/run_pipeline.py --config configs/dataset_config.yaml
```

## Outputs

- `dataset/frame_dataset.jsonl`
- `dataset/temporal_dataset.jsonl`
- `splits/train.jsonl`, `splits/val.jsonl`, `splits/test.jsonl`
- `dataset/metadata/video_metadata.json`
- per-video frames, masks, crops, scores, pseudo-labels, features, and final metadata.

The detector/tracker/VLM/feature modules expose fallback implementations so the pipeline can run locally. Replace the internals of `detect_objects.py`, `track_masks.py`, `score_states.py`, and `extract_features.py` with Grounding DINO, SAM2, CLIP/SigLIP, and DINOv2 adapters when those models are available.
