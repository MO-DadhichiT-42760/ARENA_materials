"""Install a repository-local Jupyter kernel for the uv environment."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
subprocess.run(
    [
        sys.executable,
        "-m",
        "ipykernel",
        "install",
        "--sys-prefix",
        "--name",
        "arena-chapter0",
        "--display-name",
        "ARENA chapter 0 (uv)",
    ],
    check=True,
)
path = Path(sys.prefix) / "share/jupyter/kernels/arena-chapter0/kernel.json"
kernel = json.loads(path.read_text())
kernel["env"] = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "WANDB_MODE": "offline",
    "TORCH_HOME": str(root / "chapter0_fundamentals/exercises/data/torch"),
    "HF_HOME": str(root / "chapter0_fundamentals/exercises/data/huggingface"),
}
path.write_text(json.dumps(kernel, indent=2) + "\n")
print("Ready. Start Jupyter with: uv run --locked jupyter lab")
