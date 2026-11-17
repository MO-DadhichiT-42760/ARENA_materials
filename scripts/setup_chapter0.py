"""Install a repository-local Jupyter kernel for the uv environment and editor defaults."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ${workspaceFolder} keeps the settings valid if the checkout is moved or renamed.
EDITOR_SETTINGS = {
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    # Lets Pylance resolve `from part0_prereqs import ...` and `import fundamentals_utils`.
    "python.analysis.extraPaths": ["${workspaceFolder}/chapter0_fundamentals/exercises"],
}


def install_kernel(root: Path) -> None:
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


def write_editor_settings(root: Path) -> bool:
    """Write .vscode/settings.json for VS Code/Cursor unless the user already has one.

    An existing file is never merged into: it may be JSONC (comments) that json can't
    round-trip, and it is the user's. Returns True if a file was written.
    """
    path = root / ".vscode/settings.json"
    if path.exists():
        return False
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(EDITOR_SETTINGS, indent=4) + "\n")
    return True


def has_editor_settings(root: Path) -> bool:
    try:
        current = json.loads((root / ".vscode/settings.json").read_text())
    except (OSError, ValueError):  # missing, or JSONC with comments
        return False
    return all(current.get(key) == value for key, value in EDITOR_SETTINGS.items())


def main() -> None:
    install_kernel(ROOT)
    if write_editor_settings(ROOT):
        print("Wrote .vscode/settings.json (VS Code / Cursor: interpreter and import paths).")
    elif not has_editor_settings(ROOT):
        print("Kept existing .vscode/settings.json. For editor import resolution, add:")
        print(json.dumps(EDITOR_SETTINGS, indent=4))
    print("Ready. Start Jupyter with: uv run --locked jupyter lab")
    print("Or open this folder in VS Code / Cursor and pick the 'ARENA chapter 0 (uv)' kernel.")


if __name__ == "__main__":
    main()
