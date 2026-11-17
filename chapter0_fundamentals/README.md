# Chapter 0 on macOS with uv

This fork runs the Fundamentals notebooks locally on macOS. Apple silicon uses PyTorch's MPS backend; CPU is available on every supported Mac. Python 3.12 and all Python dependencies are pinned by `.python-version` and `uv.lock`. Conda, Docker, CUDA, Homebrew libraries and the other chapters' dependencies are not needed for this environment.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it isn't already installed, then run:

```sh
git clone https://github.com/MO-DadhichiT-42760/ARENA_materials.git
cd ARENA_materials
bash install_uv.sh --chapter 0
uv run --locked jupyter lab
```

Open a notebook under `chapter0_fundamentals/exercises/` and select **ARENA chapter 0 (uv)**. Start with `part0_prereqs/0.0_Prerequisites_exercises.ipynb`. Keep the notebook inside the checkout; the checkout folder can have any name. The setup cells find the repository by its contents and never install packages or run Linux commands.

The installer uses `uv sync --locked`, prepares the core datasets and ResNet34 weights, and registers its kernel inside `.venv`, without modifying your shell or user-wide Jupyter kernels. If you move the checkout, rerun the installer to refresh the kernel's absolute paths. To install only the environment (for example when transferring an offline cache), add `--skip-data`. Before training, run `uv run --locked python scripts/prepare_chapter0.py` or copy an existing `exercises/data/` folder. The installer reports training readiness only after preparation succeeds.

## Working with the restricted network

All embedded notebook images, animations and interactive HTML plots are included in `assets/` and use relative paths. The plots' JavaScript is included too. Source URLs and SHA-256 hashes are recorded in `assets/manifest.json`. Optional YouTube lectures are ordinary links and need an internet connection when opened. Reading links to external articles remain external.

Core setup does **not** contact Hugging Face or require a login/token. MNIST comes from PyTorch's public S3 mirror (with TensorFlow's public mirror as fallback). CIFAR-10 comes from a pinned, lossless fast.ai mirror, with the original university archive as fallback. The mirror's 60,000 images, labels and train/test memberships were checked against the canonical dataset. ResNet34 weights come from PyTorch. Files are verified before use; downloads retain HTTPS and certificate verification throughout redirects.

`prepare_chapter0.py` runs automatically during installation. It prepares MNIST for both loaders, CIFAR-10 and ResNet34 weights, then saves portable snapshots under `exercises/data/` (ignored by Git). It reuses existing valid snapshots from earlier versions, retries interrupted transfers, switches mirrors when a source returns 403, and rebuilds incomplete snapshots. Rerun the same installer or preparation command after a failure; completed resources are kept. Notebook dataset loaders read local snapshots and explain how to finish setup if a core snapshot is missing, instead of attempting a hidden Hub download.

The normal download hosts are `ossci-datasets.s3.amazonaws.com`, `s3.amazonaws.com` and `download.pytorch.org`. Fallback hosts are `storage.googleapis.com`, `fast-ai-imageclas.s3.amazonaws.com`, `www.cs.toronto.edu` and `cave.cs.toronto.edu`. Only the optional CelebA dataset requires access to Hugging Face. If all available sources are blocked on a managed network, copy a prepared `exercises/data/` folder from an allowed network.

To repair or verify an installation separately:

```sh
uv run --locked python scripts/prepare_chapter0.py
uv run --locked python scripts/prepare_chapter0.py --verify-only
```

`--verify-only` makes no downloads and checks that the core data and weights are present. To build a portable cache elsewhere, use `--data-dir /path/to/cache`, then copy its contents into `chapter0_fundamentals/exercises/data/` on the destination machine.

After installation and preparation, start Jupyter without network access:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --offline --locked jupyter lab
```

A first installation still needs access to the Python package index (or an already populated uv cache). `uv --offline` uses cached packages; it cannot download missing ones.

Installation and data preparation use the operating system's trusted certificates. The uv project enables `system-certs`, and the preparation CLI initializes Python's `truststore` before loading its HTTP clients. This supports managed networks whose proxy CA is already trusted in macOS Keychain, without disabling certificate or hostname verification. A proxy CA must be installed and trusted through your organization's normal device configuration; an untrusted certificate will still fail. No certificate environment-variable workaround is needed.

Weights & Biases logging is disabled by default in the training arguments. The kernel also defaults W&B to offline mode. No API key is needed for the core exercises. Training at the original notebook sizes may take time; start with smaller batches and fewer epochs if needed. MPS fallback is enabled before importing PyTorch for operators that need CPU implementations. See [PyTorch's MPS environment documentation](https://docs.pytorch.org/docs/stable/mps_environment_variables.html).

## Optional sections

These examples are opt-in so a default local run doesn't contact a cloud service or try unavailable NVIDIA hardware:

| Section | How to enable | Requirement |
| --- | --- | --- |
| W&B hyperparameter sweeps | `ARENA_RUN_WANDB_SWEEPS=1 WANDB_MODE=online uv run --locked jupyter lab` | W&B account and network access |
| CelebA GAN training | First `uv run --locked python scripts/prepare_chapter0.py --celeba`, then launch Jupyter with `ARENA_RUN_CELEBA=1` | About 1.4 GB of extra downloads; more training time and memory |
| GAN FID scoring | Prepare with `--celeba --fid`; launch with `ARENA_RUN_CELEBA=1 ARENA_RUN_FID=1` | Cached Inception weights and a trained CelebA generator |
| Distributed training | `ARENA_RUN_DISTRIBUTED=1` in an environment with 2+ NVIDIA GPUs | Uses CUDA/NCCL; the macOS notebooks leave this section inactive |

The root uv project supports macOS. Intel Macs select the last official Intel PyTorch/torchvision releases (2.2.2/0.17.2); dependency resolution was checked with uv's Intel target, but execution has only been tested on Apple silicon. Other chapters and cloud GPU installs retain their existing `requirements.txt` environment via `bash install_uv.sh --chapter all`.

## Validation

```sh
uv run --locked python scripts/chapter0_assets.py --check
uv run --locked pytest -q
uv run --locked python scripts/test_chapter0_notebooks.py
```

The test suite runs the curriculum's reference tests, verifies image decoding and local media references, checks dataset snapshots, and performs forward/backward training steps. Its tests block outbound socket connections. The three distributed test functions are excluded because they require multi-GPU workers. Preparation regressions cover fresh snapshots with Hugging Face forced to return 403, mirror fallback, checksum failures, interrupted saves, portable image storage, and offline reruns.

The notebook runner executes all seven **solution** notebooks in real Jupyter kernels with remote network connections blocked. Its default smoke profile uses at most 256 dataset examples, one epoch and smaller ray renders. It leaves the saved curriculum's sizes intact. Executed notebooks and a report are saved under `work/notebook-validation/`, ignored by Git. Use `--full` for original training sizes, `--device cpu` or `--device mps` to select hardware, and `--sections 0.0 0.4` to select sections. Exercise notebooks retain their intentionally incomplete exercises.

## Maintaining the materials

Edit the chapter's master Python files under `infrastructure/chapters/chapter0_fundamentals/`, then regenerate:

```sh
cd infrastructure/core
uv run --locked python main.py --chapters='0.*' --verbose=false
```

The generator automatically applies local media paths and the local kernel metadata to chapter 0 notebooks. The import-only backprop initialization is kept out of exercise notebooks. If adding upstream media, run `uv run --locked python scripts/chapter0_assets.py --fetch --localize --check` while connected, then check in the new assets and manifest. Ordinary setup never downloads the embedded media again.
