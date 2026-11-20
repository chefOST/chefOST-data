"""Verify model access on synthetic text only; never launch benchmark scoring.

Run with the Modal CLI from this repository. The existing `tubelet-openai`
secret supplies the API key; only sanitized status and model metadata return.
"""

from pathlib import Path
import json
import os

import modal

# Match the builder supported by the existing project Modal client.
os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")

app = modal.App("chefost-evaluation-preflight")
cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("openai==3.57.0", "huggingface-hub==2.4.0")
             .add_local_python_source("chefost_eval"))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.15.0", "sentence-transformers==6.3.0", "transformers==5.23.0")
             .env({"HF_HOME": "/cache/huggingface"})
             .add_local_python_source("chefost_eval"))
cache = modal.Volume.from_name("chefost-evaluation-cache", create_if_missing=True)


@app.function(image=cpu_image, timeout=240, secrets=[
    modal.Secret.from_name("tubelet-openai", required_keys=["OPENAI_API_KEY"])
])
def access_probe():
    import tempfile
    import importlib.metadata
    from huggingface_hub import model_info
    from openai import OpenAI
    from chefost_eval.prepare_mapping import prepare

    report = {"scope": "synthetic_runtime_preflight", "openai_sdk": importlib.metadata.version("openai")}
    try:
        client = OpenAI(timeout=90, max_retries=1)
        available = client.models.retrieve("gpt-6.2-sol")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vocab.json").write_text(json.dumps({"s2i": {"chopped": 0, "cooked": 1}}))
            (root / "input.json").write_text(json.dumps({
                "vocabularies": {"synthetic": "vocab.json"},
                "labels": {"synthetic": ["sliced"]}}))
            result = prepare(root / "input.json", root / "mapping.json", client=client)
        report["gpt"] = {"status": "passed", "model": available.id,
            "structured_mapping": result["entries"]["synthetic"]["sliced"],
            "prompt_sha256": result["prompt_sha256"], "generation_settings": result["generation_settings"]}
    except Exception as exc:
        # SDK error strings can include parts of an invalid key. Never return them.
        report["gpt"] = {"status": "failed", "error_type": type(exc).__name__,
                         "status_code": getattr(exc, "status_code", None),
                         "error_code": getattr(exc, "code", None)}
    info = model_info("Qwen/Qwen3-Embedding-8B")
    report["qwen"] = {"model": info.id, "revision": info.sha, "gated": info.gated}
    return report


@app.function(image=gpu_image, gpu="L40S", memory=65536, timeout=900,
              volumes={"/cache": cache}, max_containers=1, scaledown_window=2)
def embedding_probe(revision: str):
    import importlib.metadata
    import torch
    from chefost_eval.embeddings import QwenEmbeddings, label_text

    provider = QwenEmbeddings(revision, device="cuda")
    same = provider.similarity(["chopped"], ["chopped"])
    different = provider.similarity(["chopped"], ["cooked"])
    vector = provider.vectors[label_text(["chopped"])]
    if abs(same - 1) > 1e-5 or not vector or not torch.isfinite(torch.tensor(vector)).all():
        raise RuntimeError("Synthetic Qwen embedding smoke test failed")
    cache.commit()
    return {"status": "passed", **provider.metadata, "dimensions": len(vector),
            "gpu": torch.cuda.get_device_name(), "gpu_memory_peak_bytes": torch.cuda.max_memory_allocated(),
            "self_similarity": same, "different_state_similarity": different,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "sentence-transformers")}}


@app.local_entrypoint()
def main(output: str, check_embeddings: bool = True):
    path = Path(output)
    if path.exists():
        raise ValueError("Use a new output path for each runtime check")
    report = access_probe.remote()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if check_embeddings:
        report["qwen"] = embedding_probe.remote(report["qwen"]["revision"])
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"qwen": report["qwen"]}, indent=2))
    if report["gpt"]["status"] != "passed":
        raise RuntimeError("GPT access failed; see the sanitized runtime report")
