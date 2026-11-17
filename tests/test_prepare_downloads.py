"""Fresh-cache preparation, blocked providers, interrupted saves and offline reuse."""

import hashlib
import io
from pathlib import Path
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from scripts import prepare_chapter0 as prepare


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


def test_403_uses_next_mirror_and_verifies_bytes(tmp_path, monkeypatch):
    payload = b"verified dataset bytes"
    urls = ("https://blocked.example/dataset", "https://allowed.example/dataset")
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request.full_url)
            assert timeout == 30
            if len(requests) == 1:
                raise HTTPError(request.full_url, 403, "Forbidden", {}, None)
            return Response(payload)

    monkeypatch.setattr(prepare, "build_opener", lambda *args: Opener())
    destination = tmp_path / "dataset.bin"
    checksum = hashlib.sha256(payload).hexdigest()
    prepare.download(urls, destination, checksum, "sha256")
    assert destination.read_bytes() == payload
    assert requests == list(urls)
    # A rerun needs neither the blocked service nor any network access.
    prepare.download(urls, destination, checksum, "sha256")
    assert len(requests) == 2


def test_corrupt_download_never_replaces_cache(tmp_path, monkeypatch):
    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"proxy error page")

    monkeypatch.setattr(prepare, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(prepare.time, "sleep", lambda _: None)
    destination = tmp_path / "archive.bin"
    destination.write_bytes(b"previous incomplete archive")
    with pytest.raises(prepare.PreparationError, match="checksum mismatch"):
        prepare.download(("https://example.com/data",), destination, "0" * 64, "sha256")
    assert destination.read_bytes() == b"previous incomplete archive"
    assert not destination.with_name("archive.bin.part").exists()


def test_https_is_required_including_redirects(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        prepare.download(("http://example.com/data",), tmp_path / "new", "0" * 32)
    with pytest.raises(URLError, match="Refusing"):
        prepare.HTTPSRedirectsOnly().redirect_request(
            Request("https://example.com/data"),
            None,
            302,
            "Redirect",
            {},
            "http://example.com/data",
        )


class TinyVisionData:
    classes: ClassVar[list[str]] = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, name, train):
        self.name, self.train = name, train

    def __len__(self):
        return 4 if self.train else 2

    def __iter__(self):
        from PIL import Image

        for i in range(len(self)):
            yield (
                Image.new(
                    "L" if self.name == "mnist" else "RGB",
                    (28, 28) if self.name == "mnist" else (32, 32),
                    color=i,
                ),
                i,
            )


@pytest.fixture
def tiny_data(monkeypatch):
    monkeypatch.setattr(
        prepare, "SPLIT_SIZES", {name: {"train": 4, "test": 2} for name in ("mnist", "cifar10")}
    )
    monkeypatch.setattr(
        prepare, "vision_data", lambda name, root, train: TinyVisionData(name, train)
    )
    # Simulate the reported failure for every possible Hub request.
    import datasets

    def forbidden(*args, **kwargs):
        raise PermissionError("Hugging Face HTTP 403")

    monkeypatch.setattr(datasets, "load_dataset", forbidden)


def test_fresh_core_snapshots_need_no_hugging_face(tmp_path, tiny_data, monkeypatch):
    import fundamentals_utils
    from datasets import load_from_disk
    from fundamentals_utils import CIFAR10, MNIST

    for name in ("mnist", "cifar10"):
        prepare.prepare_core(name, tmp_path)
        data = load_from_disk(str(tmp_path / "datasets" / name))
        assert len(data["train"]) == 4 and len(data["test"]) == 2
        prepare.prepare_core(name, tmp_path, verify_only=True)
    monkeypatch.setattr(fundamentals_utils, "LOCAL_DATA", tmp_path)
    assert MNIST(train=False)[0][0].shape == (1, 28, 28)
    assert CIFAR10(train=False)[0][0].shape == (3, 32, 32)
    assert CIFAR10(train=False).classes == TinyVisionData.classes


def test_failed_save_is_not_published_and_rerun_repairs_snapshot(tmp_path, tiny_data, monkeypatch):
    from datasets import DatasetDict

    destination = tmp_path / "datasets/mnist"
    destination.mkdir(parents=True)
    (destination / "incomplete.txt").write_text("interrupted earlier attempt")
    original = DatasetDict.save_to_disk

    def interrupted(self, path, **kwargs):
        Path(path).mkdir(parents=True)
        (Path(path) / "partial.arrow").write_bytes(b"partial")
        raise OSError("simulated interrupted save")

    monkeypatch.setattr(DatasetDict, "save_to_disk", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        prepare.prepare_core("mnist", tmp_path)
    assert (destination / "incomplete.txt").exists()
    assert not list(destination.parent.glob(".mnist-*"))
    monkeypatch.setattr(DatasetDict, "save_to_disk", original)
    prepare.prepare_core("mnist", tmp_path)
    prepare.validate_snapshot(destination, "mnist")
    assert not (destination / "incomplete.txt").exists()


def test_verify_only_missing_cache_does_not_download(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("verify-only must never download")

    monkeypatch.setattr(prepare, "download", unexpected)
    with pytest.raises(prepare.PreparationError, match="Rerun preparation"):
        prepare.prepare_core("cifar10", tmp_path, verify_only=True)


def test_notebook_missing_core_data_explains_setup_without_hub(tmp_path, monkeypatch):
    import fundamentals_utils

    monkeypatch.setattr(fundamentals_utils, "LOCAL_DATA", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Missing core data must not try Hugging Face")

    monkeypatch.setattr(fundamentals_utils, "load_dataset", forbidden)
    with pytest.raises(FileNotFoundError, match="scripts/prepare_chapter0.py"):
        fundamentals_utils.load_chapter_dataset("uoft-cs/cifar10")


def test_cifar_snapshot_embeds_images_and_mixes_classes(tmp_path, monkeypatch):
    from PIL import Image

    for split in ("train", "test"):
        for class_name in ("airplane", "automobile"):
            folder = tmp_path / "cifar10" / split / class_name
            folder.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (32, 32), color=i).save(folder / f"{i}.png")
    data = prepare.vision_data("cifar10", tmp_path, True)
    assert len(data) == 8 and len(set(data.targets[:4])) == 2
    # save_to_disk must embed PNG bytes so copying a snapshot does not depend on the raw folder.
    import shutil

    from datasets import ClassLabel, Dataset, DatasetDict, Features, load_from_disk
    from datasets import Image as HFImage

    features = Features({"img": HFImage(), "label": ClassLabel(names=data.classes)})
    snapshot = DatasetDict(
        {
            "train": Dataset.from_generator(
                prepare.image_records,
                gen_kwargs={"name": "cifar10", "data_dir": str(tmp_path), "train": True},
                features=features,
                cache_dir=str(tmp_path / "conversion"),
            )
        }
    )
    snapshot.save_to_disk(str(tmp_path / "portable"))
    shutil.rmtree(tmp_path / "cifar10")
    assert load_from_disk(str(tmp_path / "portable"))["train"][0]["img"].size == (32, 32)


def test_unavailable_png_mirror_uses_canonical_archive(tmp_path, monkeypatch):
    from torchvision.datasets import utils

    incomplete = tmp_path / "cifar10"
    incomplete.mkdir()

    def data(name, root, train):
        if incomplete.exists():
            raise RuntimeError("incomplete PNG folder")
        return TinyVisionData(name, train)

    monkeypatch.setattr(prepare, "vision_data", data)
    monkeypatch.setattr(prepare, "SPLIT_SIZES", {"cifar10": {"train": 4, "test": 2}})
    requests = []

    def download(urls, destination, checksum, algorithm="md5"):
        requests.append(urls)
        if urls == prepare.CIFAR_IMAGE_MIRRORS:
            raise prepare.PreparationError("HTTP 403")
        destination.write_bytes(b"verified canonical archive")

    monkeypatch.setattr(prepare, "download", download)
    monkeypatch.setattr(utils, "extract_archive", lambda *args: None)
    prepare.ensure_raw_data("cifar10", tmp_path)
    assert requests == [prepare.CIFAR_IMAGE_MIRRORS, prepare.CIFAR_MIRRORS]
    assert not incomplete.exists()
