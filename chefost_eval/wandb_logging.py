"""Publish evaluation results without conflating thresholds or mapping variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ENTITY = "chefOST"
PROJECT = "full-run-evals"
PROTOCOL = "chefost-v1"


def run_id(campaign_id, job_id):
    return hashlib.sha256((campaign_id + "/" + job_id).encode()).hexdigest()[:24]


def score_rows(result):
    rows = []
    for score in result["scores"]:
        rows.append([score["threshold"], score["mapping"], score["precision"], score["recall"],
                     score["cosine_similarity"], score["timing"]["f1"], score["timing"]["tp"],
                     score["timing"]["fp"], score["timing"]["fn"]])
    return rows


def publish_result(run, wandb, result, output_path, manifest_path, mapping_path):
    columns = ["threshold", "mapping", "precision", "recall", "cosine_similarity",
               "timing_f1", "timing_tp", "timing_fp", "timing_fn"]
    run.log({"evaluation/results": wandb.Table(columns=columns, data=score_rows(result))})
    ingredient_rows = []
    for score in result["scores"]:
        threshold = "native" if score["threshold"] is None else str(score["threshold"]).replace(".", "_")
        prefix = score["mapping"] + "/threshold_" + threshold
        for metric in ("precision", "recall", "cosine_similarity"):
            run.summary[prefix + "/" + metric] = score[metric]
        run.summary[prefix + "/timing_f1"] = score["timing"]["f1"]
        for video, details in score["videos"].items():
            for ingredient, values in details["ingredients"].items():
                ingredient_rows.append([score["threshold"], score["mapping"], video, ingredient,
                    values["precision"], values["recall"], values["cosine_similarity"],
                    values["eligible_frames"], values["prediction_coverage"]])
    run.log({"evaluation/ingredients": wandb.Table(columns=["threshold", "mapping", "video", "ingredient",
        "precision", "recall", "cosine_similarity", "eligible_frames", "prediction_coverage"],
        data=ingredient_rows)})
    artifact = wandb.Artifact("evaluation-" + run.id, type="evaluation-results",
        metadata={"method": result["method"], "protocol": PROTOCOL,
                  "timing_tolerance_seconds": result["timing_tolerance_seconds"],
                  "provenance": result["provenance"]})
    for path, name in ((output_path, "scores.json"), (manifest_path, "manifest.json"),
                       (mapping_path, "assisted-mapping.json")):
        artifact.add_file(str(path), name=name)
    uploaded = run.log_artifact(artifact)
    uploaded.wait(timeout=300)
    run.summary["results_artifact"] = uploaded.qualified_name
    run.summary["timing_tolerance_seconds"] = result["timing_tolerance_seconds"]
    return uploaded.qualified_name


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
