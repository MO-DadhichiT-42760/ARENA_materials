#!/bin/bash
set -e

# =============================================================================
# Install using uv (https://docs.astral.sh/uv/). macOS defaults to the locked chapter 0 project.
# Use --chapter all for the legacy full-course requirements.txt environment.
# First clone the ARENA_materials repo and move into it:
#   git clone https://github.com/ARENA-education/ARENA_materials.git
#   cd ARENA_materials
# Then, from inside the ARENA_materials folder, run:
#   ./install_uv.sh                     # macOS: chapter 0; Linux: full course on RunPod
#   ./install_uv.sh --chapter 0          # locked native macOS environment and core data
#   ./install_uv.sh --chapter 0 --skip-data  # environment only (e.g. when transferring a cache)
#   ./install_uv.sh --platform vastai   # Vast.ai platform
#   ./install_uv.sh --platform local    # Your own machine (default on macOS, and on Windows under Git Bash)
# Safe to re-run: if a run is interrupted, running the same command again repairs the partial install.
# Creates the environment at ARENA_materials/.venv; activate it with:
#   source .venv/bin/activate           # Linux / macOS
#   source .venv/Scripts/activate       # Windows (Git Bash)
# On macOS and Windows, CUDA-only packages are swapped out in requirements.txt via environment markers
# (CPU jax, no liger-kernel), so GPU-heavy sections are slower or need a cloud GPU.
# =============================================================================

# Detect the OS
case "$(uname -s)" in
    Linux*) OS="linux" ;;
    Darwin*) OS="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
    *) echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac

# Defaults
if [[ "$OS" == "linux" ]]; then PLATFORM="runpod"; else PLATFORM="local"; fi
PYTHON_VERSION="3.13"
KERNEL_NAME="arena-env"
SKIP_DATA=false
# Native Macs install only the chapter being ported; cloud installs retain the full course.
if [[ "$OS" == "macos" ]]; then CHAPTER="0"; else CHAPTER="all"; fi

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data) SKIP_DATA=true; shift ;;
        --chapter)
            [[ $# -ge 2 ]] || { echo "--chapter requires 0 or all"; exit 1; }
            CHAPTER="$2"
            shift 2
            ;;
        --platform)
            [[ $# -ge 2 ]] || { echo "--platform requires a value"; exit 1; }
            PLATFORM="$2"
            shift 2
            ;;
        --no-llm-context) shift ;;  # no-op: this script never clones arena-llm-context; accepted for parity with install.sh
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
case "$PLATFORM" in
    runpod|vastai|local) ;;
    *) echo "Unknown platform: $PLATFORM (expected runpod, vastai or local)"; exit 1 ;;
esac

case "$CHAPTER" in
    0|all) ;;
    *) echo "Unknown chapter: $CHAPTER (expected 0 or all)"; exit 1 ;;
esac

# The repo this script lives in, wherever it was cloned
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
if [[ "$OS" == "windows" ]]; then
    VENV_BIN="$VENV_DIR/Scripts"
    VENV_PYTHON="$VENV_BIN/python.exe"
else
    VENV_BIN="$VENV_DIR/bin"
    VENV_PYTHON="$VENV_BIN/python"
fi

echo "=== Setup: os=$OS, platform=$PLATFORM ==="

# --- Install git ---
echo "=== Installing system packages ==="
# `dpkg --configure -a` finishes any package install that an earlier, interrupted run left half-done
if [[ "$PLATFORM" == "runpod" ]]; then
    dpkg --configure -a && apt update && apt install -y git curl
elif [[ "$PLATFORM" == "vastai" ]]; then
    sudo dpkg --configure -a && sudo apt update && sudo apt install -y git curl
else
    # On your own machine, leave system packages to you: just check the tools this script needs
    for tool in git curl; do
        if ! command -v "$tool" &> /dev/null; then
            echo "$tool not found; please install it and re-run (on macOS: xcode-select --install)"
            exit 1
        fi
    done
fi

# --- Install uv ---
# The uv installer puts uv in ~/.local/bin, which this shell may not have on its PATH yet (e.g. on a re-run)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    if [[ "$OS" == "windows" ]]; then
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
fi
echo "=== Using $(uv --version) ==="

# Chapter 0 has a reproducible, macOS-specific project and does not install the CUDA course stack.
if [[ "$CHAPTER" == "0" ]]; then
    if [[ "$OS" != "macos" ]]; then
        echo "The chapter 0 uv project currently targets macOS. Use --chapter all for the legacy full-course environment."
        exit 1
    fi
    uv sync --project "$REPO_DIR" --locked
    uv run --project "$REPO_DIR" --locked python "$REPO_DIR/scripts/setup_chapter0.py"
    uv run --project "$REPO_DIR" --locked python "$REPO_DIR/scripts/chapter0_assets.py" --check
    if [[ "$SKIP_DATA" == false ]]; then
        uv run --project "$REPO_DIR" --locked python "$REPO_DIR/scripts/prepare_chapter0.py"
        echo "=== Chapter 0 ready. From $REPO_DIR, run: uv run --locked jupyter lab ==="
    else
        echo "=== Environment ready. Prepare core data before training: uv run --locked python scripts/prepare_chapter0.py ==="
    fi
    exit 0
fi

# --- Create the venv and install Python deps ---
# uv downloads a standalone Python if the machine doesn't have this version. --seed adds pip to the
# venv, so `pip install` / `%pip install` in a notebook installs into this environment.
# The marker file is written only once every dependency has installed. If it's missing, an earlier run was
# interrupted (possibly mid-way through unpacking a package), so the venv is rebuilt from scratch rather than
# trusted. With uv's package cache already filled by that run, the rebuild is mostly copying files.
INSTALL_MARKER="$VENV_DIR/.arena-install-complete"
if [[ -f "$INSTALL_MARKER" ]] && "$VENV_PYTHON" -c "import sys; assert sys.version.startswith('$PYTHON_VERSION.')" 2>/dev/null; then
    echo "=== Reusing existing venv at $VENV_DIR ==="
else
    echo "=== Creating venv at $VENV_DIR (python $PYTHON_VERSION) ==="
    uv venv --clear --python "$PYTHON_VERSION" --seed "$VENV_DIR"
fi
rm -f "$INSTALL_MARKER"
echo "=== Installing Python dependencies from $REPO_DIR/requirements.txt ==="
uv pip install --python "$VENV_PYTHON" -r "$REPO_DIR/requirements.txt"
touch "$INSTALL_MARKER"
# Register the venv as a Jupyter kernel named "arena-env", so it shows up under that name in any Jupyter
"$VENV_PYTHON" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_NAME"

# --- Activate the venv in new shells ---
# Only on rented GPU boxes: on your own machine, we don't edit your shell config
ACTIVATE_LINE="source $VENV_BIN/activate"
if [[ "$PLATFORM" != "local" ]] && ! grep -qxF "$ACTIVATE_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$ACTIVATE_LINE" >> "$HOME/.bashrc"
fi

# --- VS Code workspace settings ---
echo "=== Configuring VS Code workspace settings ==="

# On rented boxes you open VS Code at ~, so the settings go in ~/.vscode. On your own machine they go in the
# repo's (gitignored) .vscode folder, and an existing settings file there is left alone.
if [[ "$PLATFORM" == "local" ]]; then
    SETTINGS_DIR="$REPO_DIR/.vscode"
else
    SETTINGS_DIR="$HOME/.vscode"
fi
# VS Code on Windows needs C:/... paths, not Git Bash's /c/... paths
to_native() { if [[ "$OS" == "windows" ]]; then cygpath -m "$1"; else echo "$1"; fi; }
if [[ "$PLATFORM" == "local" && -f "$SETTINGS_DIR/settings.json" ]]; then
    echo "=== Leaving existing $SETTINGS_DIR/settings.json unchanged ==="
else
    mkdir -p "$SETTINGS_DIR"
    cat > "$SETTINGS_DIR/settings.json" << EOF
{
    "python.defaultInterpreterPath": "$(to_native "$VENV_PYTHON")",
    "python.analysis.extraPaths": [
        "$(to_native "$REPO_DIR/chapter0_fundamentals/exercises")",
        "$(to_native "$REPO_DIR/chapter1_transformer_interp/exercises")",
        "$(to_native "$REPO_DIR/chapter2_rl/exercises")",
        "$(to_native "$REPO_DIR/chapter3_llm_evals/exercises")",
        "$(to_native "$REPO_DIR/chapter4_alignment_science/exercises")"
    ]
}
EOF
fi

echo "=== Done! Open a new shell, or run: source $VENV_BIN/activate ==="
