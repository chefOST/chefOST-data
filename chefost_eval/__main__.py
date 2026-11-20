"""Score pinned saved predictions using the shared ChefOST protocol."""

import argparse
import json
from pathlib import Path

from .embeddings import CachedEmbeddings, QwenEmbeddings
from .labels import AssistedMapping, read_json, sha256
from .manifest import load_run
from .scoring import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON manifest with R2 release, GT alignment and hashed saved predictions")
    parser.add_argument("--output", help="New score JSON file; existing files are not overwritten")
    parser.add_argument("--export-mapping-input", help="Export unique native labels and vocabulary paths for the mapper, without scoring")
    parser.add_argument("--assisted-mapping", help="Frozen gpt-6.2-sol mapping cache; adds assisted scores")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--embedding-cache", help="Frozen complete-label-set Qwen embedding cache")
    group.add_argument("--qwen-revision", help="Explicitly load this Qwen3-Embedding-8B revision for cosine scoring")
    parser.add_argument("--device", help="Device for an explicitly requested Qwen model")
    args = parser.parse_args(argv)
    if not args.output and not args.export_mapping_input:
        parser.error("Supply --output or --export-mapping-input")
    output = Path(args.output) if args.output else None
    if output and output.exists():
        parser.error("Output exists; use a new path")
    config, videos, vocabularies, hashes, metadata = load_run(args.manifest)
    if args.export_mapping_input:
        from .manifest import DATASETS
        labels = {dataset: set() for dataset in vocabularies}
        for video in videos:
            for ingredient in video["ingredients"].values():
                for window in ingredient["windows"]:
                    labels[video["dataset"]].update(window.get("labels", window.get("probabilities", {})))
        data_root = (Path(args.manifest).resolve().parent / config["data_root"]).resolve()
        mapping_input = {"vocabularies": {dataset: str(data_root / "MOSCATO/vocabulary" / DATASETS[dataset][2])
                                          for dataset in vocabularies},
                         "labels": {dataset: sorted(values) for dataset, values in labels.items()}}
        with Path(args.export_mapping_input).open("x", encoding="utf-8") as handle:
            json.dump(mapping_input, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    if output is None:
        return
    assisted = AssistedMapping(read_json(args.assisted_mapping), vocabularies, hashes) if args.assisted_mapping else None
    if args.assisted_mapping:
        metadata["assisted_mapping_sha256"] = sha256(args.assisted_mapping)
    embeddings = (CachedEmbeddings(read_json(args.embedding_cache)) if args.embedding_cache
                  else QwenEmbeddings(args.qwen_revision, args.device) if args.qwen_revision else None)
    if args.embedding_cache:
        metadata["embedding_cache_sha256"] = sha256(args.embedding_cache)
    result = evaluate(videos, vocabularies, assisted=assisted, embeddings=embeddings, method=config["method"])
    omitted = []
    if embeddings is None:
        omitted.append("cosine_similarity: no embedding provider requested")
    if assisted is None:
        omitted.append("model_assisted_precision_recall: no mapping cache requested")
    result.update({"schema_version": 1, "provenance": metadata, "omitted_metrics": omitted})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
