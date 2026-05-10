#!/usr/bin/env bash
# =============================================================================
# GA-LEM-Inverter environment setup
#
# Supports:
#   - macOS Intel / Apple Silicon
#   - Linux x86_64 / aarch64
#   - Windows x86_64 from Git Bash / MSYS2 / Cygwin
#
# The project depends on xarray-simlab 0.5.0 + fastscape 0.1.0.  That stack is
# not compatible with zarr 3.x, and current xarray-simlab also still calls
# numpy.in1d, so numpy must stay below 2.4.
# =============================================================================

set -Eeuo pipefail

PYTHON_VERSION="3.11"
ENV_DIR_NAME=".conda"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
RECREATE_ENV=1
INSTALL_JUPYTER_KERNEL=1
DRY_RUN=0
DIAGNOSE_ONLY=0
INSTALL_BASE_TOOLS=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { printf "%b[INFO]%b %s\n" "$BLUE" "$NC" "$*"; }
print_success() { printf "%b[SUCCESS]%b %s\n" "$GREEN" "$NC" "$*"; }
print_warning() { printf "%b[WARNING]%b %s\n" "$YELLOW" "$NC" "$*"; }
print_error() { printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$*" >&2; }

usage() {
    cat <<'EOF'
Usage: bash setup_environment.sh [options]

Options:
  --keep-existing     Do not delete an existing .conda environment; install/update into it.
  --recreate          Delete and recreate .conda before installing. This is the default.
  --no-jupyter        Skip Jupyter kernel registration.
  --dry-run           Print detected platform and pinned package plan only.
  --diagnose-only     Check host tools and print the package plan only.
  --install-base      Try to install missing host tools such as git, curl, and bash.
  -h, --help          Show this help.

Environment variables:
  CONDA_ROOT          Miniconda install directory. Defaults to "$HOME/miniconda3".
EOF
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --keep-existing)
                RECREATE_ENV=0
                ;;
            --recreate)
                RECREATE_ENV=1
                ;;
            --no-jupyter)
                INSTALL_JUPYTER_KERNEL=0
                ;;
            --dry-run)
                DRY_RUN=1
                ;;
            --diagnose-only)
                DIAGNOSE_ONLY=1
                ;;
            --install-base)
                INSTALL_BASE_TOOLS=1
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 2
                ;;
        esac
        shift
    done
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        print_error "$1 command not found"
        exit 1
    fi
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

try_install_base_tools() {
    if [ "$INSTALL_BASE_TOOLS" -eq 0 ]; then
        return 0
    fi

    local missing=()
    for tool in curl bash git; do
        if ! command_exists "$tool"; then
            missing+=("$tool")
        fi
    done

    if [ "${#missing[@]}" -eq 0 ]; then
        print_info "Base tools already available"
        return 0
    fi

    print_warning "Missing base tools: ${missing[*]}"

    if [ "$PLATFORM" = "macos" ]; then
        if command_exists brew; then
            print_info "Installing missing tools with Homebrew"
            brew install "${missing[@]}"
        else
            print_warning "Homebrew not found. Install Xcode Command Line Tools or Homebrew, then rerun."
            if ! command_exists git; then
                xcode-select --install 2>/dev/null || true
            fi
        fi
    elif [ "$PLATFORM" = "linux" ]; then
        if command_exists apt-get; then
            print_info "Installing missing tools with apt-get"
            sudo apt-get update
            sudo apt-get install -y "${missing[@]}"
        elif command_exists dnf; then
            print_info "Installing missing tools with dnf"
            sudo dnf install -y "${missing[@]}"
        elif command_exists yum; then
            print_info "Installing missing tools with yum"
            sudo yum install -y "${missing[@]}"
        elif command_exists pacman; then
            print_info "Installing missing tools with pacman"
            sudo pacman -Sy --needed --noconfirm "${missing[@]}"
        elif command_exists zypper; then
            print_info "Installing missing tools with zypper"
            sudo zypper install -y "${missing[@]}"
        else
            print_warning "No supported Linux package manager found. Install manually: ${missing[*]}"
        fi
    elif [ "$PLATFORM" = "windows" ]; then
        if command_exists winget.exe; then
            if ! command_exists git; then
                print_info "Installing Git for Windows with winget"
                winget.exe install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
            fi
            print_warning "If bash/curl are still missing, rerun from Git Bash or use setup_environment.ps1 from PowerShell."
        else
            print_warning "Use setup_environment.ps1 from PowerShell for Windows-native bootstrap."
        fi
    fi
}

run_host_diagnostics() {
    print_info "Host diagnostics"
    print_info "Shell: ${SHELL:-unknown}"
    print_info "uname: $(uname -a)"
    for tool in git bash curl pip python conda powershell pwsh winget.exe; do
        if command_exists "$tool"; then
            print_success "$tool: $(command -v "$tool")"
        else
            print_warning "$tool: not found"
        fi
    done
}

detect_platform() {
    local os_name arch_name
    os_name="$(uname -s)"
    arch_name="$(uname -m)"

    case "$os_name" in
        Darwin*)
            PLATFORM="macos"
            ;;
        Linux*)
            PLATFORM="linux"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            PLATFORM="windows"
            ;;
        *)
            print_error "Unsupported OS: $os_name"
            exit 1
            ;;
    esac

    case "$arch_name" in
        x86_64|amd64|AMD64)
            ARCH="x86_64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            print_error "Unsupported CPU architecture: $arch_name"
            exit 1
            ;;
    esac

    if [ "$PLATFORM" = "windows" ] && [ "$ARCH" != "x86_64" ]; then
        print_warning "Windows ARM64 is not directly supported by Miniconda here; using Windows x86_64 installer."
        ARCH="x86_64"
    fi
}

to_windows_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        printf "%s" "$1"
    fi
}

miniconda_installer_name() {
    case "$PLATFORM:$ARCH" in
        macos:arm64)
            printf "Miniconda3-latest-MacOSX-arm64.sh"
            ;;
        macos:x86_64)
            printf "Miniconda3-latest-MacOSX-x86_64.sh"
            ;;
        linux:arm64)
            printf "Miniconda3-latest-Linux-aarch64.sh"
            ;;
        linux:x86_64)
            printf "Miniconda3-latest-Linux-x86_64.sh"
            ;;
        windows:x86_64)
            printf "Miniconda3-latest-Windows-x86_64.exe"
            ;;
        *)
            print_error "No Miniconda installer mapping for $PLATFORM/$ARCH"
            exit 1
            ;;
    esac
}

find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    local candidates=(
        "$CONDA_ROOT/bin/conda"
        "$CONDA_ROOT/Scripts/conda.exe"
        "$CONDA_ROOT/condabin/conda.bat"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ] || [ -f "$candidate" ]; then
            printf "%s" "$candidate"
            return 0
        fi
    done

    return 1
}

install_miniconda_if_needed() {
    if CONDA_BIN="$(find_conda)"; then
        print_info "Using conda: $CONDA_BIN"
        return 0
    fi

    require_command curl
    require_command bash

    local installer installer_path url
    installer="$(miniconda_installer_name)"
    installer_path="$SCRIPT_DIR/$installer"
    url="https://repo.anaconda.com/miniconda/$installer"

    print_info "Miniconda not found. Downloading $installer"
    curl -L --fail --retry 3 -o "$installer_path" "$url"

    print_info "Installing Miniconda to $CONDA_ROOT"
    mkdir -p "$(dirname "$CONDA_ROOT")"

    if [ "$PLATFORM" = "windows" ]; then
        require_command cmd.exe
        local installer_win conda_root_win
        installer_win="$(to_windows_path "$installer_path")"
        conda_root_win="$(to_windows_path "$CONDA_ROOT")"
        MSYS2_ARG_CONV_EXCL="*" cmd.exe /c start /wait "" "$installer_win" \
            /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S "/D=$conda_root_win"
    else
        bash "$installer_path" -b -p "$CONDA_ROOT"
    fi

    rm -f "$installer_path"

    if ! CONDA_BIN="$(find_conda)"; then
        print_error "Conda installation finished but conda executable was not found."
        exit 1
    fi

    print_success "Miniconda installed: $CONDA_BIN"
}

set_env_paths() {
    ENV_PATH="$SCRIPT_DIR/$ENV_DIR_NAME"
    if [ "$PLATFORM" = "windows" ]; then
        CONDA_ENV_PREFIX="$(to_windows_path "$ENV_PATH")"
        ENV_PYTHON="$ENV_PATH/python.exe"
    else
        CONDA_ENV_PREFIX="$ENV_PATH"
        ENV_PYTHON="$ENV_PATH/bin/python"
    fi
}

is_target_env_active() {
    local active_prefix="${CONDA_PREFIX:-}"
    if [ -z "$active_prefix" ]; then
        return 1
    fi

    if [ "$active_prefix" = "$ENV_PATH" ] || [ "$active_prefix" = "$CONDA_ENV_PREFIX" ]; then
        return 0
    fi

    if [ "$PLATFORM" = "windows" ] && command -v cygpath >/dev/null 2>&1; then
        local active_unix
        active_unix="$(cygpath -u "$active_prefix" 2>/dev/null || true)"
        if [ "$active_unix" = "$ENV_PATH" ]; then
            return 0
        fi
    fi

    return 1
}

CONDA_PACKAGES=(
    "python=$PYTHON_VERSION"
    "numpy=2.3.5"
    "scipy=1.17.1"
    "matplotlib=3.10.9"
    "pandas=3.0.2"
    "scikit-image=0.26.0"
    "scikit-learn=1.8.0"
    "xarray=2026.4.0"
    "xarray-simlab=0.5.0"
    "fastscape=0.1.0"
    "zarr=2.18.7"
    "numcodecs=0.15.1"
    "numba=0.65.1"
    "llvmlite=0.47.0"
    "rasterio=1.4.4"
    "geopandas=1.1.3"
    "shapely=2.1.2"
    "affine=2.4.0"
    "pyproj=3.7.2"
    "cartopy=0.25.0"
    "libpysal=4.14.1"
    "esda=2.9.0"
    "seaborn=0.13.2"
    "tqdm=4.67.3"
    "ipywidgets=8.1.8"
    "notebook=7.5.6"
    "ipykernel=7.2.0"
    "psutil=7.2.2"
    "joblib=1.5.3"
    "typeguard=4.5.1"
    "pyyaml=6.0.3"
    "dask=2026.3.0"
    "plotly=6.6.0"
    "pytest=9.0.3"
    "black=26.3.1"
    "flake8=7.3.0"
    "mypy=1.20.2"
)

PIP_PACKAGES=(
    "torch==2.11.0"
    "torchvision==0.26.0"
    "lpips==0.1.4"
    "opencv-python==4.13.0.92"
    "pykrige==1.7.3"
    "scikit-opt==0.6.6"
)

print_plan() {
    print_info "Platform: $PLATFORM / $ARCH"
    print_info "Project dir: $SCRIPT_DIR"
    print_info "Conda root: $CONDA_ROOT"
    print_info "Environment: $ENV_PATH"
    print_info "Conda prefix: $CONDA_ENV_PREFIX"
    print_info "Recreate env: $RECREATE_ENV"
    print_info "Jupyter kernel: $INSTALL_JUPYTER_KERNEL"
    print_info "Conda packages:"
    printf "  %s\n" "${CONDA_PACKAGES[@]}"
    print_info "Pip packages:"
    printf "  %s\n" "${PIP_PACKAGES[@]}"
}

create_or_update_environment() {
    if [ -d "$ENV_PATH" ] && [ "$RECREATE_ENV" -eq 1 ]; then
        if is_target_env_active; then
            print_error "The target environment is currently active. Deactivate it before recreating."
            exit 1
        fi
        print_warning "Removing existing environment: $ENV_PATH"
        rm -rf "$ENV_PATH"
    fi

    if [ ! -d "$ENV_PATH" ]; then
        print_info "Creating conda environment at $ENV_PATH"
        "$CONDA_BIN" create -p "$CONDA_ENV_PREFIX" -y --override-channels -c conda-forge \
            --strict-channel-priority "${CONDA_PACKAGES[@]}"
    else
        print_info "Installing/updating conda packages in $ENV_PATH"
        "$CONDA_BIN" install -p "$CONDA_ENV_PREFIX" -y --override-channels -c conda-forge \
            --strict-channel-priority "${CONDA_PACKAGES[@]}"
    fi

    print_info "Upgrading pip"
    "$ENV_PYTHON" -m pip install --upgrade "pip>=24,<26"

    print_info "Installing pinned pip packages"
    "$ENV_PYTHON" -m pip install "${PIP_PACKAGES[@]}"
}

register_jupyter_kernel() {
    if [ "$INSTALL_JUPYTER_KERNEL" -eq 0 ]; then
        print_info "Skipping Jupyter kernel registration"
        return 0
    fi

    print_info "Registering Jupyter kernel"
    "$ENV_PYTHON" -m ipykernel install --user \
        --name=ga-lem-inverter \
        --display-name="GA-LEM-Inverter (Python $PYTHON_VERSION)"
}

verify_environment() {
    print_info "Verifying imports and Fastscape runtime compatibility"
    (
        cd "$SCRIPT_DIR"
        "$ENV_PYTHON" - <<'PY'
import importlib.metadata as md
import numpy as np
import scipy
import matplotlib
import skimage
import sklearn
import rasterio
import geopandas
import shapely
import pyproj
import pandas
import xsimlab
import fastscape
import zarr
import torch
import torchvision
import lpips
import cv2
import tqdm
import dask
import plotly
import pykrige
import cartopy
import libpysal
import esda
import seaborn

assert hasattr(zarr, "MemoryStore"), "zarr<3 is required for xarray-simlab"
assert hasattr(np, "in1d"), "numpy<2.4 is required for xarray-simlab"

from model_runner import run_fastscape_model

shape = (10, 10)
k_sp = np.full(shape, 6.92e-6)
uplift = np.full(shape, 5.0)
elevation = run_fastscape_model(
    k_sp=k_sp,
    uplift=uplift,
    k_diff=19.2,
    x_size=shape[1],
    y_size=shape[0],
    spacing=900,
    time_total=1e4,
)
assert elevation.shape == shape
assert np.isfinite(elevation).all()

packages = [
    "numpy", "scipy", "matplotlib", "scikit-image", "scikit-learn",
    "rasterio", "geopandas", "shapely", "pyproj", "pandas",
    "xarray-simlab", "fastscape", "zarr", "torch", "torchvision",
    "lpips", "opencv-python", "PyKrige",
]
for package in packages:
    print(f"{package}=={md.version(package)}")
try:
    import sko
    print(f"scikit-opt=={md.version('scikit-opt')}")
except Exception as exc:
    print(f"scikit-opt=={md.version('scikit-opt')} (optional import warning: {exc})")
print("Fastscape smoke test passed")
PY
    )
}

main() {
    parse_args "$@"

    require_command uname
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

    detect_platform
    set_env_paths
    run_host_diagnostics
    try_install_base_tools
    print_plan

    if [ "$DIAGNOSE_ONLY" -eq 1 ]; then
        print_success "Diagnosis completed; no environment changes made."
        exit 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        print_success "Dry run completed; no changes made."
        exit 0
    fi

    install_miniconda_if_needed
    create_or_update_environment
    register_jupyter_kernel
    verify_environment

    print_success "Environment setup completed."
    print_info "Use: conda activate $ENV_PATH"
    print_info "Or:  $ENV_PYTHON test_environment.py"
}

main "$@"
