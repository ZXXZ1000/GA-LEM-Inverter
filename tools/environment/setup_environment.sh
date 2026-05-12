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
CONDA_SOLVER="${CONDA_SOLVER:-libmamba}"
CONDA_REPODATA_FN="${CONDA_REPODATA_FN:-current_repodata.json}"
CONDA_REMOTE_CONNECT_TIMEOUT_SECS="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-30}"
CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-180}"
PROJECT_DIR="${PROJECT_DIR:-}"
RECREATE_ENV=1
INSTALL_JUPYTER_KERNEL=1
DRY_RUN=0
DIAGNOSE_ONLY=0
INSTALL_BASE_TOOLS=0
LOG_FILE=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { printf "%b[INFO]%b %s\n" "$BLUE" "$NC" "$*"; }
print_success() { printf "%b[SUCCESS]%b %s\n" "$GREEN" "$NC" "$*"; }
print_warning() { printf "%b[WARNING]%b %s\n" "$YELLOW" "$NC" "$*"; }
print_error() { printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$*" >&2; }

setup_logging() {
    LOG_FILE="$ENV_SCRIPT_DIR/setup_environment.log"
    : > "$LOG_FILE"
    exec > >(tee -a "$LOG_FILE") 2>&1
    print_info "Log file: $LOG_FILE"
}

usage() {
    cat <<'EOF'
Usage: bash tools/environment/setup_environment.sh [options]

Options:
  --keep-existing     Do not delete an existing .conda environment; install/update into it.
  --recreate          Delete and recreate .conda before installing. This is the default.
  --no-jupyter        Skip Jupyter kernel registration.
  --dry-run           Print detected platform and pinned package plan only.
  --diagnose-only     Check host tools and print the package plan only.
  --install-base      Try to install missing host tools such as git, curl, and bash.
  --project-dir PATH  Explicit project root. Use this when launching through wrappers.
  -h, --help          Show this help.

Environment variables:
  CONDA_ROOT          Miniconda install directory. Defaults to "$HOME/miniconda3".
  CONDA_SOLVER        Preferred conda solver when supported. Defaults to "libmamba".
  CONDA_REPODATA_FN   Preferred conda repodata file when supported. Defaults to "current_repodata.json".
  CONDA_REMOTE_CONNECT_TIMEOUT_SECS  Conda network connect timeout. Defaults to 30.
  CONDA_REMOTE_READ_TIMEOUT_SECS     Conda network read timeout. Defaults to 180.
  PROJECT_DIR         Explicit project root. Same effect as --project-dir.
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
            --project-dir)
                if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                    print_error "--project-dir requires a path"
                    exit 2
                fi
                PROJECT_DIR="$2"
                shift
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

resolve_project_root() {
    local candidate source dir target

    if [ -n "$PROJECT_DIR" ]; then
        candidate="$PROJECT_DIR"
    else
        source="${BASH_SOURCE[0]:-}"
        if [ -z "$source" ] || [ "$source" = "-" ] || [ ! -e "$source" ]; then
            print_error "Cannot determine project root from this launch mode."
            print_error "Run from the cloned repository, or pass: --project-dir /path/to/GA-LEM-Inverter"
            exit 1
        fi

        while [ -L "$source" ]; do
            dir="$(cd -P "$(dirname "$source")" >/dev/null 2>&1 && pwd)"
            target="$(readlink "$source")"
            if [[ "$target" == /* ]]; then
                source="$target"
            else
                source="$dir/$target"
            fi
        done
        ENV_SCRIPT_DIR="$(cd -P "$(dirname "$source")" >/dev/null 2>&1 && pwd)"
        candidate="$ENV_SCRIPT_DIR"
    fi

    if [ ! -d "$candidate" ]; then
        print_error "Project directory does not exist: $candidate"
        exit 1
    fi

    candidate="$(cd "$candidate" >/dev/null 2>&1 && pwd -P)"
    if [ -f "$candidate/config.ini" ] && [ -f "$candidate/runner.py" ]; then
        SCRIPT_DIR="$candidate"
    elif [ -f "$candidate/../../config.ini" ] && [ -f "$candidate/../../runner.py" ]; then
        SCRIPT_DIR="$(cd "$candidate/../.." >/dev/null 2>&1 && pwd -P)"
    else
        SCRIPT_DIR="$candidate"
    fi

    if [ -z "${ENV_SCRIPT_DIR:-}" ]; then
        ENV_SCRIPT_DIR="$SCRIPT_DIR/tools/environment"
    fi
}

validate_project_root() {
    local missing=()
    local file

    for file in runner.py main.py config.ini ga_lem_inverter; do
        if [ ! -f "$SCRIPT_DIR/$file" ]; then
            if [ ! -d "$SCRIPT_DIR/$file" ]; then
                missing+=("$file")
            fi
        fi
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        print_error "Project root check failed: $SCRIPT_DIR"
        print_error "Missing required project files: ${missing[*]}"
        print_error "Pass --project-dir /path/to/GA-LEM-Inverter if this script was launched from a wrapper."
        exit 1
    fi
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
    print_info "Current dir: $(pwd)"
    print_info "Project dir: $SCRIPT_DIR"
    print_info "Environment tools dir: $ENV_SCRIPT_DIR"
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

diagnose_conda_manager() {
    print_info "Conda manager diagnostics"
    if CONDA_BIN="$(find_conda)"; then
        print_success "Preferred conda manager: $CONDA_BIN"
        "$CONDA_BIN" --version || true
        print_info "Target environment prefix: $ENV_PATH"
    else
        print_warning "No conda manager found. Setup will install Miniconda to: $CONDA_ROOT"
        print_info "Target environment prefix after install: $ENV_PATH"
    fi
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

    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    return 1
}

install_miniconda_if_needed() {
    if CONDA_BIN="$(find_conda)"; then
        print_info "Using conda manager: $CONDA_BIN"
        print_info "Target packages are installed into: $ENV_PATH"
        return 0
    fi

    require_command curl
    require_command bash

    local installer installer_path url
    installer="$(miniconda_installer_name)"
    installer_path="$ENV_SCRIPT_DIR/$installer"
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

configure_conda_args() {
    local create_help install_help
    export CONDA_REMOTE_CONNECT_TIMEOUT_SECS
    export CONDA_REMOTE_READ_TIMEOUT_SECS

    create_help="$("$CONDA_BIN" create --help 2>/dev/null || true)"
    install_help="$("$CONDA_BIN" install --help 2>/dev/null || true)"

    CONDA_CREATE_ARGS=(--override-channels -c conda-forge --strict-channel-priority)
    CONDA_INSTALL_ARGS=(--override-channels -c conda-forge --strict-channel-priority)

    if printf "%s" "$create_help" | grep -q -- "--repodata-fn"; then
        CONDA_CREATE_ARGS+=(--repodata-fn "$CONDA_REPODATA_FN")
    fi
    if printf "%s" "$install_help" | grep -q -- "--repodata-fn"; then
        CONDA_INSTALL_ARGS+=(--repodata-fn "$CONDA_REPODATA_FN")
    fi

    if [ -n "$CONDA_SOLVER" ] && printf "%s" "$create_help" | grep -q -- "--solver"; then
        CONDA_CREATE_ARGS+=(--solver "$CONDA_SOLVER")
    fi
    if [ -n "$CONDA_SOLVER" ] && printf "%s" "$install_help" | grep -q -- "--solver"; then
        CONDA_INSTALL_ARGS+=(--solver "$CONDA_SOLVER")
    fi

    if printf "%s" "$install_help" | grep -q -- "--satisfied-skip-solve"; then
        CONDA_INSTALL_ARGS+=(--satisfied-skip-solve)
    fi

    print_info "Conda create options: ${CONDA_CREATE_ARGS[*]}"
    print_info "Conda install options: ${CONDA_INSTALL_ARGS[*]}"
    print_info "Conda network timeouts: connect=${CONDA_REMOTE_CONNECT_TIMEOUT_SECS}s read=${CONDA_REMOTE_READ_TIMEOUT_SECS}s"
}

run_conda() {
    local subcommand="$1"
    shift

    if "$CONDA_BIN" "$subcommand" "$@"; then
        return 0
    fi

    if [ -n "$CONDA_SOLVER" ]; then
        print_warning "conda $subcommand failed with solver '$CONDA_SOLVER'; retrying with conda's default solver."
        local filtered=()
        while [ "$#" -gt 0 ]; do
            if [ "$1" = "--solver" ]; then
                shift
                if [ "$#" -gt 0 ]; then
                    shift
                fi
                continue
            fi
            filtered+=("$1")
            shift
        done
        "$CONDA_BIN" "$subcommand" "${filtered[@]}"
        return $?
    fi

    return 1
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

assert_env_prefix() {
    if [ ! -x "$ENV_PYTHON" ] && [ ! -f "$ENV_PYTHON" ]; then
        print_error "Expected environment Python was not found: $ENV_PYTHON"
        print_error "The environment must be created under the project root: $ENV_PATH"
        exit 1
    fi

    "$ENV_PYTHON" - "$ENV_PATH" <<'PY'
import os
import sys

expected = os.path.normcase(os.path.realpath(os.path.abspath(sys.argv[1])))
actual = os.path.normcase(os.path.realpath(os.path.abspath(sys.prefix)))

if actual != expected:
    print(f"Environment prefix mismatch: expected {expected}, got {actual}", file=sys.stderr)
    sys.exit(1)

print(f"Environment prefix verified: {sys.prefix}")
PY
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
    "compilers=1.11.0"
    "make=4.4.1"
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
    print_info "Environment tools dir: $ENV_SCRIPT_DIR"
    print_info "Conda root: $CONDA_ROOT"
    print_info "Conda solver preference: $CONDA_SOLVER"
    print_info "Conda repodata preference: $CONDA_REPODATA_FN"
    print_info "Conda network timeouts: connect=${CONDA_REMOTE_CONNECT_TIMEOUT_SECS}s read=${CONDA_REMOTE_READ_TIMEOUT_SECS}s"
    print_info "Environment: $ENV_PATH"
    print_info "Conda prefix: $CONDA_ENV_PREFIX"
    print_info "Recreate env: $RECREATE_ENV"
    print_info "Jupyter kernel: $INSTALL_JUPYTER_KERNEL"
    print_info "Conda packages:"
    printf "  %s\n" "${CONDA_PACKAGES[@]}"
    print_info "Pip packages:"
    printf "  %s\n" "${PIP_PACKAGES[@]}"
}

diagnose_existing_environment() {
    print_info "Project-local environment diagnostics"
    if [ ! -d "$ENV_PATH" ]; then
        print_warning "Environment directory does not exist yet: $ENV_PATH"
        return 0
    fi

    print_success "Environment directory exists: $ENV_PATH"
    if [ ! -x "$ENV_PYTHON" ] && [ ! -f "$ENV_PYTHON" ]; then
        print_error "Environment Python is missing: $ENV_PYTHON"
        return 0
    fi

    "$ENV_PYTHON" - "$ENV_PATH" <<'PY'
import importlib.metadata as md
import os
import sys

expected = os.path.normcase(os.path.realpath(os.path.abspath(sys.argv[1])))
actual = os.path.normcase(os.path.realpath(os.path.abspath(sys.prefix)))

print(f"python_executable={sys.executable}")
print(f"sys_prefix={sys.prefix}")
print(f"expected_prefix={sys.argv[1]}")
print(f"prefix_match={actual == expected}")

checks = {
    "numpy": "2.3.5",
    "zarr": "2.18.7",
    "xarray-simlab": "0.5.0",
    "fastscape": "0.1.0",
    "notebook": "7.5.6",
    "ipykernel": "7.2.0",
}
ok = actual == expected
for package, expected_version in checks.items():
    try:
        actual_version = md.version(package)
    except md.PackageNotFoundError:
        print(f"{package}=MISSING expected={expected_version}")
        ok = False
        continue
    print(f"{package}={actual_version} expected={expected_version}")
    if actual_version != expected_version:
        ok = False

try:
    import numpy as np
    import zarr
    print(f"numpy_has_in1d={hasattr(np, 'in1d')}")
    print(f"zarr_has_MemoryStore={hasattr(zarr, 'MemoryStore')}")
    ok = ok and hasattr(np, "in1d") and hasattr(zarr, "MemoryStore")
except Exception as exc:
    print(f"compatibility_import_error={exc}")
    ok = False

sys.exit(0 if ok else 1)
PY
    local status=$?
    if [ "$status" -eq 0 ]; then
        print_success "Existing .conda environment matches critical checks"
    else
        print_warning "Existing .conda environment is missing or mismatched; setup will recreate/update it unless --diagnose-only was used."
    fi
    return 0
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
        run_conda create -p "$CONDA_ENV_PREFIX" -y \
            "${CONDA_CREATE_ARGS[@]}" "${CONDA_PACKAGES[@]}"
    else
        print_info "Installing/updating conda packages in $ENV_PATH"
        run_conda install -p "$CONDA_ENV_PREFIX" -y \
            "${CONDA_INSTALL_ARGS[@]}" "${CONDA_PACKAGES[@]}"
    fi

    assert_env_prefix

    print_info "Upgrading pip"
    "$ENV_PYTHON" -m pip install --upgrade "pip>=24,<26"

    print_info "Installing pinned pip packages"
    "$ENV_PYTHON" -m pip install "${PIP_PACKAGES[@]}"

    assert_env_prefix
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
    assert_env_prefix
    (
        cd "$SCRIPT_DIR"
        "$ENV_PYTHON" - <<'PY'
import importlib.metadata as md
import warnings
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

from ga_lem_inverter.pipeline.forward_model import run_fastscape_model

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

print("Initializing LPIPS alex model")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    lpips_model = lpips.LPIPS(net="alex", verbose=False)
del lpips_model
print("LPIPS alex model ready")
print("Fastscape smoke test passed")
PY
    )
}

build_pecube_engine() {
    local build_script="$SCRIPT_DIR/tools/environment/build_pecube.sh"
    if [ ! -f "$build_script" ]; then
        print_warning "Pecube build script not found: $build_script"
        return 0
    fi

    print_info "Building vendored Pecube engine"
    (
        cd "$SCRIPT_DIR"
        bash "$build_script"
    )
}

main() {
    parse_args "$@"

    require_command uname
    resolve_project_root
    validate_project_root
    setup_logging

    detect_platform
    set_env_paths
    run_host_diagnostics
    diagnose_conda_manager
    try_install_base_tools
    print_plan
    diagnose_existing_environment

    if [ "$DIAGNOSE_ONLY" -eq 1 ]; then
        print_success "Diagnosis completed; no environment changes made."
        print_info "Log file: $LOG_FILE"
        exit 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        print_success "Dry run completed; no changes made."
        exit 0
    fi

    install_miniconda_if_needed
    configure_conda_args
    create_or_update_environment
    register_jupyter_kernel
    verify_environment
    build_pecube_engine

    print_success "Environment setup completed."
    print_info "Use: conda activate $ENV_PATH"
    print_info "Or:  $ENV_PYTHON tools/environment/test_environment.py"
    print_info "Pecube executables: $SCRIPT_DIR/vendor/pecube/bin"
    print_info "Log file: $LOG_FILE"
}

main "$@"
