# chefOST-data

[MOSCATO](https://github.com/eelhami/iccv25_moscato) annotations and vocabularies for CMU, EGTEA and EgoPER. EgoPER is distinct from Ego4D.

- `MOSCATO/annotations/ground_truth/`: manually annotated evaluation labels.
- `MOSCATO/annotations/pseudo_labels/`: generated training labels.
- `MOSCATO/vocabulary/`: object, action and state dictionaries, and video metadata.

Annotations use a 10 fps timeline. Multiple states can apply to an object simultaneously. Align video timestamps to that timeline; keep training pseudo-labels separate from evaluation ground truth.

## Annotations

Install [Git LFS](https://git-lfs.com/), then clone this repository and download its LFS objects:

```sh
git lfs install
git clone https://github.com/chefOST/chefOST-data.git
git -C chefOST-data lfs pull
```

Use this repository's `main` branch for shared annotations and vocabularies.

## Videos

Videos are hosted on R2:

- Bucket: `chefost-moscato-gt-videos`
- Endpoint: `https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com`
- Prefix: `moscato/releases/564f350e536734db/videos/`

Use the read-only credentials from the private team Discord channel. Keep credentials and downloaded videos outside Git. The prefix's `manifest.json` provides video paths, checksums and availability.

## Shared guidance

[Evaluation](https://github.com/chefOST/.github/blob/main/docs/Evaluation.md) · [Organization](https://github.com/chefOST/.github/blob/main/profile/README.md) · [Contributing](https://github.com/chefOST/.github/blob/main/CONTRIBUTING.md) · [Agent instructions](AGENTS.md)
