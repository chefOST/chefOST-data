# chefOST-data

[MOSCATO](https://github.com/eelhami/iccv25_moscato) annotations and vocabularies for CMU, EGTEA and EgoPER. EgoPER is distinct from Ego4D.

- `MOSCATO/annotations/ground_truth/`: manually annotated evaluation labels.
- `MOSCATO/annotations/pseudo_labels/`: generated training labels.
- `MOSCATO/vocabulary/`: object, action and state dictionaries, and video metadata.

Multiple states can apply to an object simultaneously. Keep training pseudo-labels separate from evaluation ground truth.

## Evaluation

This repository is the sole home for ChefOST benchmark scoring, prediction adapters, metric tests and evaluation workflows. Model repositories export raw predictions; no other branch should contain a separate scoring workflow. The [shared specification](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md) defines the protocol.

Run `python -m chefost_eval manifest.json --output scores.json` on saved predictions. The [manifest loader](chefost_eval/manifest.py) validates the video selection, ingredient scope, source-frame alignment and input hashes; a [small synthetic manifest](tests/fixtures/minimal/manifest.json) demonstrates the format. Use `--help` for GPT mapping-cache and Qwen embedding options. [Tests](tests/) exercise the metric and adapter behavior without launching model runs. Publish scores and their provenance to the model's W&B evaluation project.

Use `python -m chefost_eval.preflight manifest.json --output preflight.json` to validate inputs without scoring or loading models. Explicit comparison subsets require `comparison_subset: true` and a `subset_reason`; omitted videos remain visible in provenance. The separate [runtime probe](chefost_eval/preflight_runtime.py) checks GPT access and Qwen embeddings using synthetic text only.

[Training support](training_support/) preserves validation-dependent model training and checkpoint selection separately from ChefOST benchmark metrics. Its VidOSC workflows use the model checkout selected by `CHEFOST_VIDOSC_SOURCE`; scoring implementations remain here.

## Annotations

Install [Git LFS](https://git-lfs.com/), then clone this repository and download its LFS objects:

```sh
git lfs install
git clone https://github.com/chefOST/chefOST-data.git
git -C chefOST-data lfs pull
```

Use this repository's `main` branch for shared annotations and vocabularies.

## Vocabularies

The JSON files in `MOSCATO/vocabulary/` contain canonical object/state IDs and normalization mappings. For example, see the [CMU state dictionary](MOSCATO/vocabulary/CMU/state_dict.json) and [EGTEA state dictionary](MOSCATO/vocabulary/EGTEA/state_dict.json).

- `all2one` maps label variants to a canonical name, such as CMU `whisked` to `beaten`.
- `one2all` lists the variants grouped under each canonical name.

These are fields in the imported MOSCATO vocabulary, not a separate ChefOST alias dictionary. The [MOSCATO paper, Section 3.2](https://openaccess.thecvf.com/content/ICCV2025/papers/Zameni_MOSCATO_Predicting_Multiple_Object_State_Change_Through_Actions_ICCV_2025_paper.pdf) describes vocabulary construction using GPT-4o summarization, `text-embedding-3-large` embeddings and k-means clustering. The files do not establish who authored or reviewed each individual mapping. Use the [shared evaluation specification](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md) for vocabulary-only and model-assisted scoring rules.

## Videos

Videos are hosted on R2:

- Bucket: `chefost-moscato-gt-videos`
- Endpoint: `https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com`
- Prefix: `moscato/releases/564f350e536734db/videos/`

Use the read-only credentials from the private team Discord channel. Keep credentials and downloaded videos outside Git. The prefix's `manifest.json` provides video paths, checksums and availability.

## Shared guidance

[Evaluation](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md) · [Organization](https://github.com/chefOST/.github/blob/main/profile/README.md) · [Contributing](https://github.com/chefOST/.github/blob/main/CONTRIBUTING.md) · [Agent instructions](AGENTS.md)
