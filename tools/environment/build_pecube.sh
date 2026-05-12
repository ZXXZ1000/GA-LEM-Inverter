#!/usr/bin/env bash
# Build the vendored Pecube Fortran engine into vendor/pecube/bin.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd -P)"
VENDOR_ROOT="$PROJECT_ROOT/vendor/pecube"
SOURCE_DIR="$VENDOR_ROOT/source"
SOURCE_SRC_DIR="$SOURCE_DIR/src"
SOURCE_BIN_DIR="$SOURCE_DIR/bin"
BIN_DIR="$VENDOR_ROOT/bin"
LOCAL_CONDA_BIN="$PROJECT_ROOT/.conda/bin"

print_info() { printf "[INFO] %s\n" "$*"; }
print_success() { printf "[SUCCESS] %s\n" "$*"; }
print_error() { printf "[ERROR] %s\n" "$*" >&2; }

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        print_error "$1 command not found."
        print_error "Install a Fortran/C build toolchain first. On macOS with conda-forge:"
        print_error "  conda install -p ./.conda --override-channels -c conda-forge gfortran_osx-arm64 clang_osx-arm64 make"
        print_error "On Linux, install gfortran gcc make with your package manager."
        exit 1
    fi
}

main() {
    if [ ! -f "$SOURCE_SRC_DIR/Makefile" ]; then
        print_error "Pecube source Makefile not found: $SOURCE_SRC_DIR/Makefile"
        exit 1
    fi

    if [ -d "$LOCAL_CONDA_BIN" ]; then
        export PATH="$LOCAL_CONDA_BIN:$PATH"
    fi

    require_command make
    require_command gfortran
    if command -v gcc >/dev/null 2>&1; then
        c_compiler="$(command -v gcc)"
    elif command -v clang >/dev/null 2>&1; then
        c_compiler="$(command -v clang)"
    else
        print_error "gcc or clang command not found."
        print_error "Install a C compiler first. On macOS, install Xcode Command Line Tools or conda-forge compilers."
        exit 1
    fi
    fortran_compiler="$(command -v gfortran)"
    link_flags=""
    if [ "$(uname -s)" = "Darwin" ] && command -v xcrun >/dev/null 2>&1; then
        sdk_path="$(xcrun --show-sdk-path 2>/dev/null || true)"
        if [ -n "$sdk_path" ]; then
            link_flags="-Wl,-syslibroot,$sdk_path"
        fi
    fi

    mkdir -p "$BIN_DIR" "$SOURCE_BIN_DIR"
    print_info "Building Pecube from: $SOURCE_SRC_DIR"
    print_info "Fortran compiler: $fortran_compiler"
    print_info "C compiler: $c_compiler"
    if [ -n "$link_flags" ]; then
        print_info "Link flags: $link_flags"
    fi
    (
        cd "$SOURCE_SRC_DIR"
        make clean >/dev/null 2>&1 || true
        make Pecube Test Vtk \
            FF90="$fortran_compiler" \
            FF77="$fortran_compiler -fd-lines-as-comments" \
            CC="$c_compiler" \
            LINK="$fortran_compiler $link_flags"
    )

    for executable in Pecube Test Vtk; do
        if [ ! -x "$SOURCE_BIN_DIR/$executable" ]; then
            print_error "Expected executable was not built by Pecube Makefile: $SOURCE_BIN_DIR/$executable"
            exit 1
        fi
        cp "$SOURCE_BIN_DIR/$executable" "$BIN_DIR/$executable"
        chmod +x "$BIN_DIR/$executable"
        rm -f "$SOURCE_BIN_DIR/$executable"
        if [ ! -x "$BIN_DIR/$executable" ]; then
            print_error "Expected executable was not built: $BIN_DIR/$executable"
            exit 1
        fi
    done

    print_success "Pecube executables are ready in $BIN_DIR"
}

main "$@"
