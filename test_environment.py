#!/usr/bin/env python
"""
测试环境配置脚本
验证所有必需的包是否可以正常导入
"""

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
        'skimage': 'scikit-image',
        'yaml': 'pyyaml'
    }
    
    success_count = 0
    total_count = len(packages)
    
    print("测试环境配置...")
    print("=" * 50)
    
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
    
    print("=" * 50)
    print(f"成功导入: {success_count}/{total_count} 个包")
    
    if success_count == total_count:
        print("🎉 所有包导入成功！环境配置完成。")
        return True
    else:
        print("⚠️  部分包导入失败，请检查安装。")
        return False

if __name__ == "__main__":
    test_imports()