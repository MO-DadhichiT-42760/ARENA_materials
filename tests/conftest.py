import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "chapter0_fundamentals/exercises"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXERCISES))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", str(EXERCISES / "data/torch"))
os.environ.setdefault("HF_HOME", str(EXERCISES / "data/huggingface"))

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail on network access: the native validation must use local assets/data."""
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("Unexpected network access during offline validation")

    monkeypatch.setattr(socket.socket, "connect", blocked)
