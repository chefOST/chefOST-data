# chefOST-data

Shared object-state annotations and vocabularies. Keep raw videos, extracted frames/features and temporary run outputs out of Git.

## MOSCATO

[MOSCATO](https://github.com/eelhami/iccv25_moscato) covers CMU, EGTEA and EgoPER (distinct from Ego4D).

- `MOSCATO/annotations/ground_truth/`: reference labels for evaluation.
- `MOSCATO/annotations/pseudo_labels/`: generated labels for training.
- `MOSCATO/vocabulary/`: object, action and state dictionaries, plus video metadata.

Multiple states can apply to one object at a time; verify video/frame alignment before scoring.

## Ego4D

- `Ego4D/annotations/pseudo_labels/`: final pseudo-labels for selected videos.
- `Ego4D/vocabulary/`: object/state dictionaries and video metadata.

These folders are empty placeholders. See the [generation pipeline](ego4d-pseudo-labels/README.md) for code and input manifests. Include video/object IDs, source timestamps and generation metadata with shared labels. Keep pseudo-labels separate from evaluation ground truth.

## Download

MOSCATO annotations are about 1.44 GB. Annotations use [Git LFS](https://git-lfs.com/).

```bash
brew install git-lfs # macOS; other platforms: git-lfs.com
git lfs install
git clone https://github.com/chefOST/chefOST-data.git
```

Already cloned? Run `git lfs pull` inside the repository.
