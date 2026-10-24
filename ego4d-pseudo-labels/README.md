# Ego4D object-state pseudo-labels

Conservative 1 FPS labeling of one visible cooking-state change. Frames are classified in overlapping temporal windows, illegal/flickering transitions are rejected, each surviving transition gets a second VLM check, and nothing becomes final without human approval.

## Selected video

`selected_video.json` tracks the same chapati for the first 30 seconds of Ego4D clip `a8918018-a4a5-451b-bcff-8e7b636e6702_00m42s-01m42s.mp4`.

The object is suitable because its surface visibly progresses from pale to browned. The manifest explicitly excludes the other dough disc and finished chapatis in view.

## Run through Modal

Python 3.10+ and `ffmpeg` are required. Store the OpenAI key in Modal so it is never injected into the local labeling process:

```bash
python3 -m pip install -r requirements.txt
modal setup
modal secret create openai-secret OPENAI_API_KEY=sk-...
modal deploy modal_app.py

# Override this when a stronger vision-capable model is available to your account.
export OPENAI_VLM_MODEL=gpt-5.2
python3 pipeline.py label --modal --manifest selected_video.json --output output
```

Frame extraction stays local. Only the image windows, prompts, and schemas cross Modal's private RPC boundary. The deployed function reads `OPENAI_API_KEY` from the Modal secret and calls OpenAI; the key never returns locally.

W&B development work belongs to entity `chefOST`, project `ego4d-pseudo-labels`. The Modal `wandb` secret supplies `WANDB_API_KEY`; `ensure_wandb_project` verifies access and creates the project without launching a run. Do not reuse run IDs from other projects.

To bypass Modal, omit `--modal` and set `OPENAI_API_KEY` locally. The script uses `/Applications/AnyTrans.app/Contents/Resources/ffmpeg` as a local fallback when `ffmpeg` is not on `PATH`.

Outputs:

- `output/frames/`: exact 1 FPS samples
- `output/raw_batches/`: cached structured VLM responses, so reruns do not rebill completed windows
- `output/timeline.jsonl`: conservative frame-to-state labels
- `output/changes_for_review.csv`: proposed state-change pairs and independent verification

Review every row in `changes_for_review.csv` against the neighboring frames and set `approved` to `yes` or `no`. Then:

```bash
python3 pipeline.py finalize
```

This writes `output/state_changes.jsonl`, containing only human-approved frame-to-object-state-change pairs. A transition frame means the first 1 FPS sample where the new state is visibly established, not the unknowable exact instant between samples.

## Full-video production run

Run a complete video through one production model and save frame-based state transitions plus the untouched API response:

```bash
export OPENAI_API_KEY=...
python3 full_video_run.py /path/to/video.avi --output full-run-output/video-name
```

The default 1 FPS sampling covers the whole video and reports source-frame numbers at one-second precision. Outputs are `transitions.json`, `raw_response.json`, and `run.json` (model, runtime, token usage, and sampling details).

## Checks

```bash
python3 -m unittest -v
```

A VLM cannot guarantee zero errors. Mandatory abstention and human approval are therefore part of the pipeline rather than an optional claim of accuracy.
