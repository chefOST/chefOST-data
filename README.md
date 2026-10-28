# chefOST-data

[MOSCATO](https://github.com/eelhami/iccv25_moscato) annotations and vocabularies for CMU, EGTEA and EgoPER (distinct from Ego4D).

- `MOSCATO/annotations/ground_truth/`: reference labels for evaluation.
- `MOSCATO/annotations/pseudo_labels/`: generated labels for training; keep separate from evaluation ground truth.
- `MOSCATO/vocabulary/`: object, action and state dictionaries, plus video metadata.

This Git repository includes JSON and XLSX files; videos are hosted separately on R2. Extracted features and evaluation code are not included. Multiple states can apply to one object at a time; verify video/frame alignment before scoring.

The curated ground-truth release contains 87 recordings: CMU 29, EGTEA 12 and EgoPER 46. One CMU recording with an unresolved video/annotation length mismatch is excluded from ground truth, pseudo-labels and video metadata in both annotation formats. Historical runs and results are unchanged.

## Download annotations

Annotations require [Git LFS](https://git-lfs.com/).

```bash
brew install git-lfs # macOS; other platforms: git-lfs.com
git lfs install
git clone https://github.com/chefOST/chefOST-data.git
```

Already cloned? Run `git lfs pull` inside the repository.

## Videos

The shared R2 bucket contains 41 MP4s (29 CMU, 12 EGTEA; 7.88 GB). EgoPER videos are still pending.

- Bucket: `chefost-moscato-gt-videos`
- Endpoint: `https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com`
- Video folder on R2: `moscato/releases/564f350e536734db/videos/`

For `<access_key_id>` and `<secret_key>`, use the read-only credentials sent in the private team Discord channel. These are the S3 Access Key ID and Secret Access Key; keep their values out of Git.

In your own Modal workspace/environment, create a Secret named `chefost-r2`:

```text
AWS_ACCESS_KEY_ID=<access_key_id>
AWS_SECRET_ACCESS_KEY=<secret_key>
```

Add this mount to your runner, using `volumes={"/videos": videos}` on its Modal function:

```python
import modal

videos = modal.CloudBucketMount(
    bucket_name="chefost-moscato-gt-videos",
    bucket_endpoint_url="https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com",
    key_prefix="moscato/releases/564f350e536734db/videos/",
    secret=modal.Secret.from_name("chefost-r2"),
    read_only=True,
)
```

`/videos/manifest.json` records relative paths, checksums and unavailable entries. Keep each model's preprocessing and source-video timing; save generated outputs in your own storage.

For local downloads with AWS CLI, configure the same Discord credentials in a separate profile (region `auto`, output `json`), then download the video folder outside Git:

```sh
aws configure --profile chefost-r2
aws --profile chefost-r2 --region auto \
  --endpoint-url https://2aee25b6b08cc209d7cab66f1d0942fd.r2.cloudflarestorage.com \
  s3 cp s3://chefost-moscato-gt-videos/moscato/releases/564f350e536734db/videos/ \
  /path/outside/git/moscato-videos/ --recursive
```
