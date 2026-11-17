"""Prepare chapter 0 for offline use, without a Hugging Face account.

Core MNIST/CIFAR-10 data comes from public HTTPS sources and is converted locally
into the notebook snapshots. Add --celeba for optional Hugging Face face data,
or --fid for optional Inception weights. Completed downloads survive reruns.
"""

import argparse
import hashlib
import http.client
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "chapter0_fundamentals/exercises"
DATA = EXERCISES / "data"
sys.path.insert(0, str(EXERCISES))

# Public sources maintained by PyTorch / TensorFlow and the CIFAR authors.
MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
CIFAR_MIRRORS = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
)
# fast.ai's lossless PNG mirror: all 60,000 images/labels/splits were compared
# byte-for-byte with the canonical CIFAR-10 dataset before pinning this SHA-256.
CIFAR_IMAGE_MIRRORS = (
    "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
    "https://fast-ai-imageclas.s3.amazonaws.com/cifar10.tgz",
)
CIFAR_IMAGE_SHA256 = "637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441"
SPLIT_SIZES = {
    "mnist": {"train": 60_000, "test": 10_000},
    "cifar10": {"train": 50_000, "test": 10_000},
}


class PreparationError(RuntimeError):
    """A setup failure with an actionable, safe-to-display message."""


class HTTPSRedirectsOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise URLError("Refusing a redirect from HTTPS to an unverified HTTP connection")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def matches_hash(path: Path, expected: str, algorithm: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest().startswith(expected)


def download(urls: tuple[str, ...], destination: Path, checksum: str, algorithm="md5"):
    """Bounded HTTPS retries, mirror fallback, and verification before publishing a file."""
    if matches_hash(destination, checksum, algorithm):
        return
    from tqdm import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    opener = build_opener(HTTPSRedirectsOnly())
    failures = []
    for url in urls:
        if urlsplit(url).scheme != "https":
            raise ValueError("Dataset downloads must use HTTPS")
        host = urlsplit(url).hostname
        for attempt in range(3):
            print(f"Downloading {destination.name} from {host}...", flush=True)
            try:
                with opener.open(
                    Request(url, headers={"User-Agent": "ARENA-chapter0"}), timeout=30
                ) as response:
                    total = int(response.headers.get("Content-Length", 0)) or None
                    with (
                        temporary.open("wb") as stream,
                        tqdm(total=total, unit="B", unit_scale=True) as progress,
                    ):
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                            progress.update(len(chunk))
                if not matches_hash(temporary, checksum, algorithm):
                    raise OSError("download checksum mismatch (the file was not accepted)")
                temporary.replace(destination)
                return
            except (OSError, http.client.HTTPException) as error:
                temporary.unlink(missing_ok=True)
                # Never print signed URLs, tokens, or large HTTP tracebacks.
                reason = f"HTTP {error.code}" if isinstance(error, HTTPError) else str(error)
                failures.append(f"{host}: {reason}")
                if isinstance(error, HTTPError) and error.code in (401, 403, 404):
                    break
                if attempt < 2:
                    time.sleep(attempt + 1)
        print(f"Trying another source for {destination.name}...", flush=True)
    raise PreparationError(
        f"Could not download {destination.name}. {'; '.join(failures)}. "
        "Allow these HTTPS hosts on your network, or copy a prepared exercises/data folder "
        "from another machine and rerun the same command. Completed files are kept."
    )


def vision_data(name: str, data_dir: Path, train: bool):
    from torchvision import datasets

    image_folder = data_dir / "cifar10" / ("train" if train else "test")
    if name == "cifar10" and image_folder.is_dir():
        import numpy as np

        data = datasets.ImageFolder(image_folder)
        # ImageFolder groups by class. Mix deterministically so slices used by the
        # tutorials and smoke tests contain multiple classes, like the original batches.
        order = np.random.default_rng(0).permutation(len(data.samples))
        data.samples = [data.samples[i] for i in order]
        data.imgs = data.samples
        data.targets = [label for _, label in data.samples]
        return data
    cls = datasets.MNIST if name == "mnist" else datasets.CIFAR10
    return cls(data_dir, train=train, download=False)


def ensure_raw_data(name: str, data_dir: Path, verify_only=False):
    """Populate torchvision's local files using verified HTTPS downloads only."""
    from torchvision import datasets
    from torchvision.datasets.utils import extract_archive

    try:
        for split, count in SPLIT_SIZES[name].items():
            if len(vision_data(name, data_dir, train=split == "train")) != count:
                raise ValueError(f"Incomplete {name} {split} data")
        return
    except (OSError, RuntimeError, ValueError):
        if verify_only:
            raise PreparationError(
                f"Missing or incomplete local {name} data. Rerun preparation without --verify-only."
            ) from None
    if name == "mnist":
        raw = data_dir / "MNIST/raw"
        for filename, checksum in datasets.MNIST.resources:
            archive = raw / filename
            download(tuple(mirror + filename for mirror in MNIST_MIRRORS), archive, checksum)
            extract_archive(str(archive), str(raw))
    else:
        try:
            archive = data_dir / "cifar10.tgz"
            download(CIFAR_IMAGE_MIRRORS, archive, CIFAR_IMAGE_SHA256, "sha256")
            with tempfile.TemporaryDirectory(prefix=".cifar10-", dir=data_dir) as temporary:
                with tarfile.open(archive) as contents:
                    contents.extractall(temporary, filter="data")
                candidate = Path(temporary) / "cifar10"
                for split, count in SPLIT_SIZES[name].items():
                    if len(datasets.ImageFolder(candidate / split)) != count:
                        raise PreparationError(f"Incomplete CIFAR-10 {split} archive")
                destination = data_dir / "cifar10"
                if destination.exists():
                    shutil.rmtree(destination)
                candidate.replace(destination)
        except PreparationError:
            print(
                "CIFAR-10 mirror unavailable; trying the original university archive (may be slower)...",
                flush=True,
            )
            archive = data_dir / datasets.CIFAR10.filename
            download(CIFAR_MIRRORS, archive, datasets.CIFAR10.tgz_md5)
            extract_archive(str(archive), str(data_dir))
            # An incomplete PNG folder must not shadow the repaired canonical files.
            incomplete = data_dir / "cifar10"
            if incomplete.exists():
                shutil.rmtree(incomplete)
    # Catch incomplete extraction before saving any notebook snapshot.
    for split, count in SPLIT_SIZES[name].items():
        if len(vision_data(name, data_dir, train=split == "train")) != count:
            raise PreparationError(
                f"Incomplete {name} {split} data after extraction; rerun preparation."
            )


def validate_snapshot(path: Path, name: str):
    from datasets import load_from_disk

    data = load_from_disk(str(path))
    image_key = "img" if name == "cifar10" else "image"
    for split, count in SPLIT_SIZES[name].items():
        if split not in data or len(data[split]) != count:
            raise ValueError(f"Incomplete {name} {split} snapshot")
        if image_key not in data[split].features or "label" not in data[split].features:
            raise ValueError(f"Invalid {name} snapshot columns")
        for index in (0, count - 1):
            row = data[split][index]
            if row[image_key].size != ((28, 28) if name == "mnist" else (32, 32)):
                raise ValueError(f"Invalid {name} image dimensions")
            if not 0 <= row["label"] < 10:
                raise ValueError(f"Invalid {name} class label")
    if name == "cifar10" and len(data["train"].features["label"].names) != 10:
        raise ValueError("Invalid CIFAR-10 label metadata")
    return data


def image_records(name: str, data_dir: str, train: bool):
    """Stream public torchvision images into Arrow; never contact the Hub."""
    key = "img" if name == "cifar10" else "image"
    for image, label in vision_data(name, Path(data_dir), train):
        yield {key: image, "label": int(label)}


def prepare_core(name: str, data_dir: Path, verify_only=False):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image

    destination = data_dir / "datasets" / name
    valid = False
    try:
        validate_snapshot(destination, name)
        valid = True
    except (OSError, ValueError, KeyError, TypeError):
        if verify_only:
            raise PreparationError(
                f"Missing or incomplete {name} snapshot. Rerun preparation without --verify-only."
            ) from None
    # [0.2] also reads torchvision's MNIST files directly, even when the Arrow snapshot exists.
    if not valid or name == "mnist":
        ensure_raw_data(name, data_dir, verify_only)
    if valid:
        print(f"Ready: {name} (local cache)", flush=True)
        return
    print(f"Building {name} notebook snapshot locally...", flush=True)
    key = "img" if name == "cifar10" else "image"
    classes = (
        [str(n) for n in range(10)]
        if name == "mnist"
        else vision_data(name, data_dir, True).classes
    )
    features = Features({key: Image(), "label": ClassLabel(names=classes)})
    data = DatasetDict(
        {
            split: Dataset.from_generator(
                image_records,
                gen_kwargs={"name": name, "data_dir": str(data_dir), "train": split == "train"},
                features=features,
                cache_dir=str(data_dir / "conversion-cache"),
            )
            for split in SPLIT_SIZES[name]
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A killed/failed save never becomes the published snapshot; reruns repair old partial snapshots.
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "snapshot"
        data.save_to_disk(str(candidate))
        validate_snapshot(candidate, name)
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        candidate.replace(destination)
    print(f"Ready: {name}", flush=True)


def prepare_weights(verify_only=False):
    import torch
    from torchvision.models import ResNet34_Weights

    url = ResNet34_Weights.IMAGENET1K_V1.url
    filename = urlsplit(url).path.rsplit("/", 1)[-1]
    expected = filename.rsplit("-", 1)[-1].split(".")[0]
    destination = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if verify_only and not matches_hash(destination, expected, "sha256"):
        raise PreparationError(
            "Missing or corrupted ResNet34 weights. Rerun preparation without --verify-only."
        )
    download((url,), destination, expected, algorithm="sha256")
    print("Ready: ResNet34 weights (verified)", flush=True)


def prepare_celeba(data_dir: Path, verify_only=False):
    from datasets import load_dataset, load_from_disk

    destination = data_dir / "datasets/celeba"
    if destination.is_dir():
        load_from_disk(str(destination))
        print("Ready: CelebA (local cache)", flush=True)
        return
    if verify_only:
        raise PreparationError(
            "CelebA is not cached. Run preparation with --celeba while connected."
        )
    print(
        "Preparing optional CelebA (~1.4 GB); this step requires access to Hugging Face...",
        flush=True,
    )
    try:
        data = load_dataset("nielsr/CelebA-faces", token=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".celeba-", dir=destination.parent) as temporary:
            snapshot = Path(temporary) / "snapshot"
            data.save_to_disk(str(snapshot))
            load_from_disk(str(snapshot))
            snapshot.replace(destination)
    except (OSError, RuntimeError, ValueError) as error:
        raise PreparationError(
            "Core resources are ready, but optional CelebA could not be downloaded. "
            "A Hugging Face 403 is a network/access failure, not a reason to disable certificate checks. "
            "Use the MNIST GAN exercises now; retry --celeba on a network that allows Hugging Face."
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Prepare a portable data/model cache here (copy to exercises/data for notebook use)",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Check local core resources without downloading"
    )
    parser.add_argument("--celeba", action="store_true")
    parser.add_argument("--fid", action="store_true")
    args = parser.parse_args(argv)
    data_dir = (args.data_dir or DATA).expanduser().resolve()
    if args.data_dir:
        os.environ["HF_HOME"] = str(data_dir / "huggingface")
        os.environ["TORCH_HOME"] = str(data_dir / "torch")
    else:
        os.environ.setdefault("HF_HOME", str(data_dir / "huggingface"))
        os.environ.setdefault("TORCH_HOME", str(data_dir / "torch"))
    # Configure before importing clients; native Xet does not use Python's SSLContext.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import truststore

    truststore.inject_into_ssl()
    # Kept after truststore initialization: dataset/HTTP clients may cache SSLContext on import.
    from datasets import load_from_disk  # noqa: F401

    try:
        prepare_core("mnist", data_dir, args.verify_only)
        prepare_core("cifar10", data_dir, args.verify_only)
        prepare_weights(args.verify_only)
        print(
            "Core data and weights are ready for offline use (no Hugging Face login needed).",
            flush=True,
        )
        if args.celeba:
            prepare_celeba(data_dir, args.verify_only)
        if args.fid:
            if args.verify_only:
                raise PreparationError(
                    "--verify-only supports core data and --celeba; omit --fid for offline verification."
                )
            from torchmetrics.image.fid import FrechetInceptionDistance

            FrechetInceptionDistance(feature=2048)
    except (PreparationError, OSError) as error:
        parser.exit(
            1,
            f"\nPreparation incomplete: {error}\nRerun the same command to continue; completed resources are reused.\n",
        )


if __name__ == "__main__":
    main()
