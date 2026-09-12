# Ego4D pseudo-labels

Segment an object in a video with SAM2, classify its visual state once per second with `HuggingFaceTB/SmolVLM-500M-Instruct`, and write fixed and predicted labels to CSV.

## Setup

Python 3.10+ and `ffmpeg` are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

SAM2 needs one bounding-box prompt on a frame where the target is visible. Coordinates are `x1 y1 x2 y2` in source pixels.

```bash
python pipeline.py \
  --video ~/ego4d_data/v2/clips/a8918018-a4a5-451b-bcff-8e7b636e6702_00m42s-01m42s.mp4 \
  --output output \
  --prompt-second 4 \
  --box 690 280 930 455
```

Outputs:

- `output/masks/`: binary SAM2 masks
- `output/crops/`: masked RGB crops sent to SmolVLM
- `output/predictions_raw.csv`: direct SmolVLM labels
- `output/predictions.csv`: labels after three-frame monotonic smoothing
- `output/fixed_labels.csv`: fixed thirds (`raw`, `partially_toasted`, `toasted`) for comparison

The accepted model labels are `raw`, `partially_toasted`, `toasted`, and `not_visible`. Video files and generated output are ignored by git.
