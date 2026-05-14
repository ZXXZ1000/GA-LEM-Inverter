#!/usr/bin/env python
"""
测试环境配置脚本
验证所有必需的包是否可以正常导入，并检查关键兼容版本。
"""

import importlib.metadata as md
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path


VERSION_CHECKS = {
    "numpy": "2.3.5",
    "zarr": "2.18.7",
    "xarray-simlab": "0.5.0",
    "fastscape": "0.1.0",
}


ISOLATED_IMPORTS = {"sko"}


def import_package(import_name: str) -> tuple[bool, str]:
    """Import a package for environment validation.

    scikit-opt imports `sko.tools`, which calls multiprocessing.set_start_method
    at import time. Under pytest, another plugin may already have set the
    multiprocessing context, so validate it in a fresh Python process.
    """
    if import_name in ISOLATED_IMPORTS:
        result = subprocess.run(
            [sys.executable, "-c", f"import {import_name}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            return True, ""
        message = (result.stderr or result.stdout).strip()
        return False, message or f"subprocess import failed with exit code {result.returncode}"

    try:
        __import__(import_name)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def find_environment_tool(tool: str) -> str | None:
    """Find a tool in PATH or beside the current conda Python executable."""
    path = shutil.which(tool)
    if path:
        return path

    env_prefix = Path(sys.prefix).resolve()
    search_dirs = [
        Path(sys.executable).resolve().parent,
        env_prefix / "bin",
        env_prefix / "Scripts",
        env_prefix / "Library" / "mingw-w64" / "bin",
        env_prefix / "Library" / "usr" / "bin",
        env_prefix / "Library" / "bin",
    ]
    names = [tool]
    if tool == "make":
        names.append("mingw32-make")
    if os.name == "nt":
        names.extend([f"{name}.exe" for name in list(names)])
        names.extend([f"{name}.cmd" for name in list(names)])
        names.extend([f"{name}.bat" for name in list(names)])

    seen = set()
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists() and candidate.is_file():
                return str(candidate)
    return None


def check_environment() -> bool:
    """测试所有关键包的导入"""
    packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'xsimlab': 'xarray-simlab',
        'fastscape': 'fastscape',
        'rasterio': 'rasterio',
        'geopandas': 'geopandas',
        'shapely': 'shapely',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'torch': 'torch',
        'lpips': 'lpips',
        'cv2': 'opencv-python',
        'tqdm': 'tqdm',
        'pandas': 'pandas',
        'dask': 'dask',
        'plotly': 'plotly',
        'pykrige': 'pykrige',
        'sko': 'scikit-opt',
        'cartopy': 'cartopy',
        'configparser': 'configparser',
        'affine': 'affine',
        'pyproj': 'pyproj',
        'libpysal': 'libpysal',
        'esda': 'esda',
        'seaborn': 'seaborn',
        'psutil': 'psutil',
        'joblib': 'joblib',
        'typeguard': 'typeguard',
        'yaml': 'pyyaml'
    }
    
    success_count = 0
    total_count = len(packages)
    
    print("测试环境配置...")
    print("=" * 50)
    
    for import_name, package_name in packages.items():
        ok, error = import_package(import_name)
        if ok:
            version_text = ""
            try:
                version_text = f"=={md.version(package_name)}"
            except md.PackageNotFoundError:
                pass
            print(f"✓ {package_name}{version_text}")
            success_count += 1
        else:
            print(f"✗ {package_name}: {error}")
    
    compatibility_ok = True
    for package_name, expected_version in VERSION_CHECKS.items():
        try:
            actual_version = md.version(package_name)
        except md.PackageNotFoundError:
            print(f"✗ {package_name}: not installed")
            compatibility_ok = False
            continue
        if actual_version != expected_version:
            print(f"✗ {package_name}: expected {expected_version}, got {actual_version}")
            compatibility_ok = False

    try:
        import numpy as np
        import zarr

        if not hasattr(np, "in1d"):
            print("✗ numpy compatibility: numpy.in1d is missing; numpy must stay below 2.4")
            compatibility_ok = False
        if not hasattr(zarr, "MemoryStore"):
            print("✗ zarr compatibility: zarr.MemoryStore is missing; zarr must stay on 2.x")
            compatibility_ok = False
    except ImportError as e:
        print(f"✗ compatibility import check failed: {e}")
        compatibility_ok = False

    try:
        import lpips

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            lpips_model = lpips.LPIPS(net="alex", verbose=False)
        del lpips_model
        print("✓ LPIPS alex model initialized")
    except Exception as e:
        print(f"✗ LPIPS alex model: {e}")
        compatibility_ok = False

    tool_ok = True
    for tool in ("make", "gfortran"):
        path = find_environment_tool(tool)
        if path:
            print(f"✓ {tool}: {path}")
        else:
            print(f"✗ {tool}: not found; Pecube build requires the conda compilers package")
            tool_ok = False

    print("=" * 50)
    print(f"成功导入: {success_count}/{total_count} 个包")
    
    if success_count == total_count and compatibility_ok and tool_ok:
        print("所有包导入成功，关键兼容版本正确。")
        return True
    else:
        print("部分包导入失败或关键版本不兼容，请检查安装。")
        return False


def test_imports():
    assert check_environment()


if __name__ == "__main__":
    sys.exit(0 if check_environment() else 1)
