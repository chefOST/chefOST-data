"""Durable, bounded-concurrency evaluation campaign in chefOST/full-run-evals.

Inputs must already be staged in the evaluation volume. Launch with
`modal run --detach chefost_eval/cloud_run.py ...`; the local entrypoint spawns
a durable CPU coordinator and returns its call receipt immediately.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import modal

os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")
app = modal.App("chefost-full-run-evaluations")
volume = modal.Volume.from_name("chefost-evaluation-cache", create_if_missing=False)
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])
openai_secret = modal.Secret.from_name("tubelet-openai", required_keys=["OPENAI_API_KEY"])
cpu_image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai==3.57.0", "numpy==2.4.6", "wandb==0.30.0")
    .add_local_python_source("chefost_eval"))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.15.0", "sentence-transformers==6.3.0", "transformers==5.23.0", "wandb==0.30.0")
    .env({"HF_HOME": "/cache/huggingface"})
    .add_local_python_source("chefost_eval"))
QWEN_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"


def _campaign_root(campaign_id):
    if not campaign_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in campaign_id):
        raise ValueError("Campaign ID must be a simple slug")
    return Path("/cache/campaigns") / campaign_id


def _job_mapping(raw, hashes):
    selected = dict(raw)
    selected["vocabulary_sha256"] = {dataset: raw["vocabulary_sha256"][dataset] for dataset in hashes}
    if selected["vocabulary_sha256"] != hashes:
        raise ValueError("Campaign mapping vocabulary differs from job vocabulary")
    selected["entries"] = {dataset: raw["entries"][dataset] for dataset in hashes}
    return selected


def _code_hashes():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


@app.function(image=gpu_image, gpu="L40S", cpu=4, memory=65536, timeout=21600,
              volumes={"/cache": volume}, secrets=[wandb_secret], max_containers=3,
              scaledown_window=5, retries=0)
def score_job(campaign_id: str, job: dict, evaluator_revision: str):
    import wandb
    from chefost_eval.embeddings import QwenEmbeddings, required_label_sets
    from chefost_eval.labels import AssistedMapping, read_json, sha256
    from chefost_eval.manifest import load_run
    from chefost_eval.scoring import evaluate, THRESHOLDS
    from chefost_eval.wandb_logging import ENTITY, PROJECT, PROTOCOL, run_id, publish_result, write_json

    volume.reload()
    root = _campaign_root(campaign_id)
    _campaign_root(job["id"])
    output_dir = root / "results" / job["id"]
    state_path = output_dir / "state.json"
    if state_path.exists() and read_json(state_path).get("status") == "complete":
        return read_json(state_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = (root / "inputs" / job["manifest"]).resolve()
    if not manifest.is_relative_to(root / "inputs"):
        raise ValueError("Manifest must be within the staged input bundle")
    if sha256(manifest) != job["manifest_sha256"]:
        raise ValueError("Staged job manifest checksum mismatch")
    started = time.monotonic()
    run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id(campaign_id, job["id"]), resume="allow",
        name=job["id"], group=PROTOCOL + "-" + job["scope"], job_type="evaluation",
        tags=[campaign_id, job["method"], job["scope"]],
        config={"campaign_id": campaign_id, "protocol": PROTOCOL, **job,
                "evaluator_revision": evaluator_revision,
                "thresholds": list(THRESHOLDS) if job["method"] in {"mosp", "vidosc"} else [None],
                "evaluator_file_sha256": _code_hashes(),
                "mapping_model": "gpt-6.2-sol", "qwen_revision": QWEN_REVISION,
                "mapping_variants": ["vocabulary_only", "model_assisted"]},
        settings=wandb.Settings(init_timeout=120))

    def phase(name, **extra):
        run.summary["phase"] = name
        run.log({"progress/elapsed_seconds": time.monotonic() - started, **extra})
        write_json(state_path, {"status": "running", "phase": name, "run_url": run.url,
                               "elapsed_seconds": time.monotonic() - started})
        volume.commit()

    try:
        phase("loading_inputs")
        config, videos, vocabularies, hashes, metadata = load_run(manifest)
        global_mapping_path = root / "mapping" / "assisted-mapping.json"
        mapping = _job_mapping(read_json(global_mapping_path), hashes)
        mapping_path = output_dir / "assisted-mapping.json"
        write_json(mapping_path, mapping)
        assisted = AssistedMapping(mapping, vocabularies, hashes)
        metadata.update({"evaluator_revision": evaluator_revision,
                         "evaluator_file_sha256": _code_hashes(),
                         "original_manifest_sha256": job["source_manifest_sha256"],
                         "campaign_mapping_sha256": sha256(global_mapping_path),
                         "assisted_mapping_sha256": sha256(mapping_path),
                         "campaign_id": campaign_id, "job_id": job["id"]})
        phase("collecting_embedding_sets")
        label_sets = required_label_sets(videos, vocabularies, config["method"])
        run.summary["embedding/unique_sets"] = len(label_sets)
        phase("loading_qwen")
        provider = QwenEmbeddings(QWEN_REVISION, device="cuda", cache_dir=output_dir / "embeddings", batch_size=32)
        last_update = [0.0]

        def progress(done, total, elapsed):
            if done == total or time.monotonic() - last_update[0] >= 10:
                last_update[0] = time.monotonic()
                run.summary["phase"] = "embedding"
                run.log({"embedding/completed_sets": done, "embedding/total_sets": total,
                         "embedding/elapsed_seconds": elapsed})

        phase("embedding")
        provider.prefetch(label_sets, batch_size=32, progress=progress)
        volume.commit()
        phase("scoring")
        result = evaluate(videos, vocabularies, assisted=assisted, embeddings=provider, method=config["method"])
        result.update(schema_version=1, provenance=metadata, omitted_metrics=[])
        scores_path = output_dir / "scores.json"
        write_json(scores_path, result)
        volume.commit()
        phase("publishing")
        artifact = publish_result(run, wandb, result, scores_path, manifest, mapping_path)
        run.summary["phase"] = "complete"
        run.summary["evaluation_complete"] = True
        run.summary["elapsed_seconds"] = time.monotonic() - started
        state = {"status": "complete", "run_url": run.url, "artifact": artifact,
                 "elapsed_seconds": time.monotonic() - started, "scores_sha256": sha256(scores_path)}
        write_json(state_path, state)
        volume.commit()
        run.finish()
        return state
    except Exception as exc:
        run.summary["phase"] = "failed"
        run.summary["evaluation_complete"] = False
        run.summary["error_type"] = type(exc).__name__
        write_json(state_path, {"status": "failed", "run_url": run.url, "error_type": type(exc).__name__,
                               "elapsed_seconds": time.monotonic() - started})
        volume.commit()
        run.finish(exit_code=1)
        raise


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=86400,
              volumes={"/cache": volume}, secrets=[wandb_secret, openai_secret],
              max_containers=1, retries=0)
def run_campaign(campaign_id: str, evaluator_revision: str):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from types import SimpleNamespace
    import wandb
    from openai import OpenAI
    from chefost_eval.labels import read_json, sha256
    from chefost_eval.prepare_mapping import prepare, PROMPT_SHA
    from chefost_eval.staging import validate_bundle_files
    from chefost_eval.wandb_logging import ENTITY, PROJECT, PROTOCOL, run_id, write_json

    volume.reload()
    root = _campaign_root(campaign_id)
    campaign = read_json(root / "inputs" / "campaign.json")
    if campaign.get("campaign_id") != campaign_id:
        raise ValueError("Staged campaign identity mismatch")
    state_path = root / "campaign-state.json"
    if state_path.exists():
        raise RuntimeError("Campaign already has state; inspect existing calls rather than launching duplicate requests")
    run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id(campaign_id, "coordinator"),
        name=campaign_id, group=PROTOCOL + "-campaign", job_type="evaluation-coordinator",
        tags=[campaign_id, "coordinator"],
        config={"campaign_id": campaign_id, "evaluator_revision": evaluator_revision,
                "jobs": campaign["jobs"], "qwen_revision": QWEN_REVISION,
                "campaign_manifest_sha256": sha256(root / "inputs" / "campaign.json"),
                "mapping_model": "gpt-6.2-sol", "mapping_prompt_sha256": PROMPT_SHA},
        settings=wandb.Settings(init_timeout=120))
    started = time.monotonic()
    state = {"status": "running", "phase": "validating_inputs", "run_url": run.url, "jobs": {}}
    write_json(state_path, state)
    volume.commit()
    try:
        run.summary["phase"] = "validating_inputs"
        validate_bundle_files(root / "inputs", campaign["files"])
        mapping_input = root / "inputs" / campaign.get("mapping_input", "mapping-input.json")
        mapping_output = root / "mapping" / "assisted-mapping.json"
        mapping_output.parent.mkdir(parents=True, exist_ok=True)
        state["phase"] = "mapping"
        write_json(state_path, state)
        volume.commit()
        run.summary["phase"] = "mapping"
        total_labels = sum(len(labels) for labels in read_json(mapping_input)["labels"].values())
        run.summary["mapping/unique_labels"] = total_labels
        run.summary["mapping/total_batches"] = sum((len(labels) + 15) // 16
            for labels in read_json(mapping_input)["labels"].values())
        client = OpenAI(timeout=180, max_retries=2)
        original_create = client.responses.create
        completed = [0]

        def tracked_create(**kwargs):
            response = original_create(**kwargs)
            completed[0] += 1
            run.log({"mapping/completed_batches": completed[0],
                     "mapping/elapsed_seconds": time.monotonic() - started})
            return response

        prepare(mapping_input, mapping_output,
                client=SimpleNamespace(responses=SimpleNamespace(create=tracked_create)))
        volume.commit()
        mapping_artifact = wandb.Artifact("mapping-" + campaign_id, type="evaluation-mapping",
            metadata={"model": "gpt-6.2-sol", "prompt_sha256": PROMPT_SHA,
                      "sha256": sha256(mapping_output),
                      "campaign_manifest_sha256": sha256(root / "inputs" / "campaign.json")})
        mapping_artifact.add_file(str(mapping_output), name="assisted-mapping.json")
        mapping_artifact.add_file(str(root / "inputs" / "campaign.json"), name="campaign.json")
        artifact = run.log_artifact(mapping_artifact)
        artifact.wait(timeout=300)
        run.summary["mapping/artifact"] = artifact.qualified_name
        state["phase"] = "scoring"
        run.summary["phase"] = "scoring"
        calls = []
        for job in campaign["jobs"]:
            call = score_job.spawn(campaign_id, job, evaluator_revision)
            state["jobs"][job["id"]] = {"status": "submitted", "function_call_id": call.object_id,
                "wandb_run_id": run_id(campaign_id, job["id"])}
            calls.append((job["id"], call))
        write_json(state_path, state)
        volume.commit()
        errors = []
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            pending = {pool.submit(call.get): job_id for job_id, call in calls}
            for future in as_completed(pending):
                job_id = pending[future]
                try:
                    state["jobs"][job_id].update(future.result())
                except Exception as exc:
                    state["jobs"][job_id].update(status="failed", error_type=type(exc).__name__)
                    errors.append(job_id)
                write_json(state_path, state)
                volume.commit()
                run.log({"jobs/complete": sum(j["status"] == "complete" for j in state["jobs"].values()),
                         "jobs/failed": sum(j["status"] == "failed" for j in state["jobs"].values())})
        state.update(status="failed" if errors else "complete", phase="failed" if errors else "complete",
                     elapsed_seconds=time.monotonic() - started)
        run.summary.update({"phase": state["phase"], "evaluation_complete": not errors,
                            "elapsed_seconds": state["elapsed_seconds"]})
        write_json(state_path, state)
        volume.commit()
        run.finish(exit_code=1 if errors else 0)
        return state
    except Exception as exc:
        state.update(status="failed", phase="failed", error_type=type(exc).__name__)
        write_json(state_path, state)
        volume.commit()
        run.summary.update({"phase": "failed", "error_type": type(exc).__name__})
        run.finish(exit_code=1)
        raise


@app.local_entrypoint()
def main(campaign_id: str, evaluator_revision: str, receipt: str):
    path = Path(receipt)
    if path.exists():
        raise ValueError("Launch receipt exists; inspect it before resubmitting")
    _campaign_root(campaign_id)
    call = run_campaign.spawn(campaign_id, evaluator_revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"campaign_id": campaign_id, "function_call_id": call.object_id,
              "evaluator_revision": evaluator_revision,
              "project": "https://wandb.ai/chefOST/full-run-evals",
              "status": "submitted"}
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
