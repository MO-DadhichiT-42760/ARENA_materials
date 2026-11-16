"""Download chapter 0 training data/weights once, then use the saved copies offline.

Default: MNIST (both loaders), CIFAR-10 and ResNet34 weights. Add --celeba for
GAN face training (~1.4 GB), or --fid for the optional Inception FID weights.
The cache is ignored by Git; copy exercises/data when moving to an offline Mac.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "chapter0_fundamentals/exercises"
DATA = EXERCISES / "data"
os.environ.setdefault("HF_HOME", str(DATA / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(DATA / "torch"))
sys.path.insert(0, str(EXERCISES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celeba", action="store_true")
    parser.add_argument("--fid", action="store_true")
    args = parser.parse_args()

    # This CLI owns the process: configure TLS before importing download clients,
    # which can cache SSLContext. Use OS trust (including managed proxy roots)
    # while retaining certificate and hostname verification for every download.
    import truststore

    truststore.inject_into_ssl()

    from datasets import load_dataset, load_from_disk
    from torchvision import datasets, models

    names = {"ylecun/mnist": "mnist", "uoft-cs/cifar10": "cifar10"}
    if args.celeba:
        names["nielsr/CelebA-faces"] = "celeba"
    for source, name in names.items():
        destination = DATA / "datasets" / name
        if destination.is_dir():
            load_from_disk(str(destination))  # verify that the previous save completed
            print(f"Ready: {name}", flush=True)
        else:
            print(f"Preparing {source}...", flush=True)
            data = load_dataset(source)
            temporary = destination.with_name(destination.name + ".partial")
            data.save_to_disk(str(temporary))
            temporary.rename(destination)
    for train in (True, False):
        datasets.MNIST(DATA, train=train, download=True)
    models.ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    if args.fid:
        from torchmetrics.image.fid import FrechetInceptionDistance

        FrechetInceptionDistance(feature=2048)
    print("Core data and weights are ready for offline use.")


if __name__ == "__main__":
    main()
