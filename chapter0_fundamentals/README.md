# Chapter 0 on macOS with uv

This fork runs the Fundamentals notebooks natively on macOS: Apple GPU (MPS) on Apple silicon, CPU elsewhere. Python 3.12 and every dependency are pinned by `.python-version` and `uv.lock`; no Conda, Docker, CUDA or Hugging Face account is needed.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then:

```sh
git clone https://github.com/MO-DadhichiT-42760/ARENA_materials.git
cd ARENA_materials
bash install_uv.sh --chapter 0
uv run --locked jupyter lab
```

Open a notebook under `chapter0_fundamentals/exercises/`, starting with `part0_prereqs/0.0_Prerequisites_exercises.ipynb`, and select the **ARENA chapter 0 (uv)** kernel.

**VS Code or Cursor:** open the repository root folder, open a notebook, and choose **Select Kernel → Jupyter Kernel… → ARENA chapter 0 (uv)** (or **Python Environments → .venv**). The installer writes a git-ignored `.vscode/settings.json` so the editor uses `.venv` and resolves imports such as `part0_prereqs.utils`; an existing file is left untouched.

The installer is safe to rerun: it repairs an interrupted install and keeps completed downloads. Rerun it after moving the checkout. `--skip-data` installs only the environment.

## Network and data

Notebook images, animations and interactive plots (including their JavaScript) are checked in under `assets/`, so notebooks render offline. YouTube lectures and article links stay external.

The installer runs `scripts/prepare_chapter0.py`, which downloads MNIST, CIFAR-10 and ResNet34 weights into `exercises/data/` (git-ignored), verifies checksums, and builds the dataset snapshots the notebooks load. It never contacts Hugging Face: core preparation forces the Hugging Face libraries offline.

| Data | Hosts (in order of preference) |
| --- | --- |
| MNIST | `ossci-datasets.s3.amazonaws.com`, `storage.googleapis.com` |
| CIFAR-10 | `s3.amazonaws.com` / `fast-ai-imageclas.s3.amazonaws.com` (lossless mirror, verified against the original), then `www.cs.toronto.edu` / `cave.cs.toronto.edu` |
| ResNet34 | `download.pytorch.org` |

Downloads use the macOS system certificate store (uv's `system-certs` and Python's `truststore`), so a corporate proxy CA trusted in Keychain works without disabling verification. If every source is blocked, copy a prepared `exercises/data/` folder from another machine.

```sh
uv run --locked python scripts/prepare_chapter0.py                 # repair or finish data
uv run --locked python scripts/prepare_chapter0.py --verify-only   # check, no downloads
HF_HUB_OFFLINE=1 uv run --offline --locked jupyter lab             # work fully offline
```

W&B logging is off by default and the kernel sets `WANDB_MODE=offline`, so no API key is needed. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before PyTorch loads.

## Optional sections

| Section | How to enable | Requirement |
| --- | --- | --- |
| W&B sweeps | `ARENA_RUN_WANDB_SWEEPS=1 WANDB_MODE=online uv run --locked jupyter lab` | W&B account |
| CelebA GAN training | `prepare_chapter0.py --celeba`, then launch with `ARENA_RUN_CELEBA=1` | Hugging Face access, ~1.4 GB |
| GAN FID scoring | `prepare_chapter0.py --celeba --fid`, launch with `ARENA_RUN_CELEBA=1 ARENA_RUN_FID=1` | Inception weights, a trained CelebA generator |
| Distributed training | `ARENA_RUN_DISTRIBUTED=1` | 2+ NVIDIA GPUs; inactive on Macs |

Intel Macs get the last official Intel builds (PyTorch 2.2.2, torchvision 0.17.2); only Apple silicon has been run end to end. Other chapters keep the original `requirements.txt` setup: `bash install_uv.sh --chapter all`.

## Validation

```sh
uv run --locked pytest -q
uv run --locked python scripts/test_chapter0_notebooks.py
```

`pytest` runs the curriculum's own tests against the reference solutions, checks asset hashes and notebook setup cells, data preparation (including Hugging Face being unavailable), and MPS/CPU training steps, all with network access blocked. The notebook runner executes the seven solution notebooks offline in a real kernel with small data (`--full` for original sizes, `--device cpu|mps`, `--sections 0.0 0.4`); outputs go to the git-ignored `work/notebook-validation/`.

## Maintaining

Edit the master files under `infrastructure/chapters/chapter0_fundamentals/`, then regenerate:

```sh
cd infrastructure/core
uv run --locked python main.py --chapters='0.*' --verbose=false
```

Generation rewrites media URLs to the local `assets/` and sets the kernel. When upstream adds media, run `uv run --locked python scripts/chapter0_assets.py --fetch` while online, regenerate, and commit the new assets and `assets/manifest.json`.
