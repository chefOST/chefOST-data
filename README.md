# chefOST-data

[MOSCATO](https://github.com/eelhami/iccv25_moscato) annotations and vocabularies for CMU, EGTEA and EgoPER (distinct from Ego4D).

- `MOSCATO/annotations/ground_truth/`: reference labels for evaluation.
- `MOSCATO/annotations/pseudo_labels/`: generated labels for training; keep separate from evaluation ground truth.
- `MOSCATO/vocabulary/`: object, action and state dictionaries, plus video metadata.

Includes JSON and XLSX files; no videos, extracted features or evaluation code. Multiple states can apply to one object at a time; verify video/frame alignment before scoring.

The curated ground-truth release contains 87 recordings: CMU 29, EGTEA 12 and EgoPER 46. One CMU recording with an unresolved video/annotation length mismatch is excluded from ground truth, pseudo-labels and video metadata in both annotation formats. Historical runs and results are unchanged.

## Download

Annotations require [Git LFS](https://git-lfs.com/).

```bash
brew install git-lfs # macOS; other platforms: git-lfs.com
git lfs install
git clone https://github.com/chefOST/chefOST-data.git
```

Already cloned? Run `git lfs pull` inside the repository.
