"""Private Modal bridge to OpenAI; the API key never enters the local process."""

import os
import sys
from pathlib import Path

import modal

app = modal.App("ego4d-labeler")
WANDB_ENTITY = "chefOST"
WANDB_PROJECT = "ego4d-pseudo-labels"
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    str(Path(__file__).with_name("pipeline.py")), "/root/pipeline.py"
)
wandb_image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=600,
)
def openai_json(model, content, schema):
    sys.path.insert(0, "/root")
    from pipeline import openai_json as request

    return request(model, content, schema, os.environ["OPENAI_API_KEY"])


@app.function(
    image=wandb_image,
    secrets=[modal.Secret.from_name("wandb")],
    timeout=120,
)
def ensure_wandb_project():
    import wandb

    api = wandb.Api()
    try:
        project = api.project(WANDB_PROJECT, entity=WANDB_ENTITY)
    except wandb.errors.CommError:
        api.create_project(WANDB_PROJECT, entity=WANDB_ENTITY)
        project = api.project(WANDB_PROJECT, entity=WANDB_ENTITY)
    return {"entity": project.entity, "project": project.name}
