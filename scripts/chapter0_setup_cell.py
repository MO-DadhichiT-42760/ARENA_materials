# Native local setup: run `uv sync --locked` once from the repository root.
# Select the "ARENA chapter 0 (uv)" kernel; this cell never installs or downloads.
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("WANDB_MODE", "offline")
chapter = "chapter0_fundamentals"
root_dir = next(
    (p for p in (Path.cwd(), *Path.cwd().parents) if (p / chapter / "exercises").is_dir()),
    None,
)
if root_dir is None:
    raise RuntimeError(
        "Open this notebook inside the cloned ARENA repository. See chapter0_fundamentals/README.md."
    )
exercises_dir = root_dir / chapter / "exercises"
os.environ.setdefault("HF_HOME", str(exercises_dir / "data/huggingface"))
os.environ.setdefault("TORCH_HOME", str(exercises_dir / "data/torch"))
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
os.chdir(exercises_dir)

# Include Plotly's JavaScript locally; figures also work without a CDN connection.
import plotly.io as pio

pio.renderers.default = "plotly_mimetype+notebook"
