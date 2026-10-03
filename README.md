# chefOST-data

[MOSCATO](https://github.com/eelhami/iccv25_moscato) annotations and vocabularies for CMU, EGTEA and EgoPER (distinct from Ego4D).

- `MOSCATO/annotations/ground_truth/`: reference labels for evaluation.
- `MOSCATO/annotations/pseudo_labels/`: generated labels for training; keep separate from evaluation ground truth.
- `MOSCATO/vocabulary/`: object, action and state dictionaries, plus video metadata.

Includes JSON and XLSX files; no videos, extracted features or evaluation code. Multiple states can apply to one object at a time; verify video/frame alignment before scoring.

## Download

About 1.44 GB. Annotations require [Git LFS](https://git-lfs.com/).

```bash
brew install git-lfs # macOS; other platforms: git-lfs.com
git lfs install
git clone https://github.com/chefOST/chefOST-data.git
```

Already cloned? Run `git lfs pull` inside the repository.

Raf’s original dataset-generation code and history are preserved on [`raf/dataset-generation`](https://github.com/chefOST/chefOST-data/tree/raf/dataset-generation).
