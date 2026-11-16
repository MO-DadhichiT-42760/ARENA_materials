"""Execute solution notebooks offline on a real Jupyter kernel.

The default smoke profile uses small datasets, one training epoch and small ray
renders. --full keeps the curriculum's original sizes. Optional online sweeps,
multi-NVIDIA-GPU work, CelebA and FID remain opt-in, as in the shipped notebooks.
Executed copies and a JSON report go under work/notebook-validation (Git ignored).
"""

import argparse
import json
import re
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "chapter0_fundamentals/exercises"


def prelude(device: str, full: bool) -> str:
    return f'''
import os, sys, socket
from pathlib import Path
sys.path.insert(0, {str(EXERCISES)!r})
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = {str(EXERCISES / "data/torch")!r}
os.environ["HF_HOME"] = {str(EXERCISES / "data/huggingface")!r}
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["WANDB_MODE"] = "disabled"
for flag in ("ARENA_RUN_WANDB_SWEEPS", "ARENA_RUN_DISTRIBUTED", "ARENA_RUN_CELEBA", "ARENA_RUN_FID"):
    os.environ.pop(flag, None)
# Jupyter communication is local; block exercise attempts to contact remote services.
_original_connect = socket.socket.connect
def offline_connect(self, address):
    if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(f"Unexpected network request: {{address}}")
    return _original_connect(self, address)
socket.socket.connect = offline_connect
import torch as t
t.set_num_threads(4)
if {device!r} == "mps":
    assert t.backends.mps.is_available(), "MPS is required for this validation run"
if not {full!r}:
    import fundamentals_utils as _fundamentals
    _original_load = _fundamentals.load_chapter_dataset
    def small_dataset(name, split=None):
        data = _original_load(name, split)
        if split is None:
            return type(data)({{key: value.select(range(min(len(value), 256))) for key, value in data.items()}})
        return data.select(range(min(len(data), 256)))
    _fundamentals.load_chapter_dataset = small_dataset
print("Offline kernel ready; MPS:", t.backends.mps.is_available())
'''


def small_source(source: str) -> str:
    source = re.sub(r'\bepochs(\s*:\s*int)?\s*=\s*[\d_]+', r'epochs\1 = 1', source)
    source = re.sub(
        r'\b(trainset_size|testset_size)(\s*:\s*int)?\s*=\s*[\d_]+', r'\1\2 = 256', source
    )
    source = re.sub(r'\b(num_pixels_y|num_pixels_z)\s*=\s*[\d_]+', r'\1 = 32', source)
    source = re.sub(r'\bnum_frames\s*=\s*[\d_]+', 'num_frames = 3', source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="*", help="section prefixes, e.g. 0.0 0.4")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "work/notebook-validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(EXERCISES.rglob("*solutions.ipynb"))
    if args.sections:
        paths = [
            p for p in paths if any(p.name.startswith(prefix + "_") for prefix in args.sections)
        ]
        if not paths:
            parser.error("No matching notebooks")
    report = []
    for path in paths:
        print(
            f"Executing {path.name} ({args.device}, {'full' if args.full else 'smoke'})...",
            flush=True,
        )
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                cell.pop("execution_count", None)
                cell.pop("outputs", None)
            else:
                if not args.full:
                    cell.source = small_source(cell.source)
                if args.device == "cpu":
                    cell.source = cell.source.replace(
                        't.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")',
                        't.device("cpu")',
                    ).replace(
                        '"cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"',
                        '"cpu"',
                    )
        notebook.cells.insert(0, nbformat.v4.new_code_cell(prelude(args.device, args.full)))
        started = time.monotonic()
        status = "passed"
        error = None
        try:
            NotebookClient(
                notebook,
                kernel_name="arena-chapter0",
                timeout=900,
                resources={"metadata": {"path": str(path.parent)}},
            ).execute()
        except Exception as exc:  # noqa: BLE001 - save the failing notebook and continue validation
            status = "failed"
            error = str(exc)
        destination = args.output / path.name
        nbformat.write(notebook, destination)
        row = {
            "notebook": path.name,
            "profile": "full" if args.full else "smoke",
            "device": args.device,
            "status": status,
            "seconds": round(time.monotonic() - started, 1),
        }
        if error:
            row["error"] = error
        report.append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{status.upper()} {path.name} ({row['seconds']}s)", flush=True)
        if error:
            print(error, flush=True)
    if any(row["status"] == "failed" for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
