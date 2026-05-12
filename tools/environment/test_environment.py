#!/usr/bin/env python
"""
测试环境配置脚本
验证所有必需的包是否可以正常导入，并检查关键兼容版本。
"""

import importlib.metadata as md
import shutil
import sys
import warnings


VERSION_CHECKS = {
    "numpy": "2.3.5",
    "zarr": "2.18.7",
    "xarray-simlab": "0.5.0",
    "fastscape": "0.1.0",
}


def test_imports():
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
        try:
            module = __import__(import_name)
            version_text = ""
            try:
                version_text = f"=={md.version(package_name)}"
            except md.PackageNotFoundError:
                pass
            print(f"✓ {package_name}{version_text}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
    
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
        path = shutil.which(tool)
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

if __name__ == "__main__":
    sys.exit(0 if test_imports() else 1)
