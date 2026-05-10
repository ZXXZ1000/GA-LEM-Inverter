# GA-LEM-Inverter 环境配置指南

## 概述

本文档提供了 GA-LEM-Inverter 项目的完整环境配置指南。项目使用 Python 3.11 和 Conda 环境管理，确保所有依赖包的正确安装和版本兼容性。

## 快速开始

### 一键配置脚本

我们提供了自动化配置脚本，可以一键完成基础工具诊断、Miniconda 安装、本地环境创建、依赖安装和运行验证。

```bash
# macOS / Linux / Windows Git Bash
bash setup_environment.sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1

# Windows CMD 也可以运行
setup_environment.bat
```

该脚本将自动执行以下操作：
1. 诊断 `git`、`bash`、`curl`、`pip`、`python`、`conda`、PowerShell 等基础工具状态
2. 检查并安装 Miniconda（如果不存在）
3. 创建本地 Conda 环境 `./.conda`
4. 从 conda-forge 和 pip 安装锁定版本的兼容依赖
5. 注册 Jupyter 内核
6. 运行导入检查和 Fastscape smoke test

只诊断不安装：

```bash
bash setup_environment.sh --diagnose-only
```

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1 -DiagnoseOnly
```

如果 macOS/Linux/Git Bash 缺少基础工具，可尝试：

```bash
bash setup_environment.sh --install-base
```

## 手动配置步骤

如果需要手动配置，请按以下步骤操作：

### 1. 安装 Miniconda

```bash
# 下载 Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# 安装到用户目录
bash Miniconda3-latest-MacOSX-arm64.sh -b -p ~/miniconda3

# 初始化 conda
~/miniconda3/bin/conda init zsh
~/miniconda3/bin/conda init bash

# 清理安装文件
rm Miniconda3-latest-MacOSX-arm64.sh
```

### 2. 创建本地 Conda 环境

```bash
# 在项目根目录下创建环境
~/miniconda3/bin/conda create -p ./.conda python=3.11 -y
```

### 3. 安装核心包

```bash
# 激活环境并安装 xarray-simlab 和 fastscape
./.conda/bin/conda install -c conda-forge xarray-simlab fastscape -y
```

### 4. 安装其他依赖包

```bash
# 安装 conda 可用的包
./.conda/bin/conda install -c conda-forge \
    numpy scipy matplotlib scikit-image scikit-learn \
    rasterio geopandas shapely affine pyproj \
    libpysal esda seaborn tqdm ipywidgets \
    notebook psutil joblib typeguard configparser \
    pyyaml dask plotly pytest black flake8 mypy -y

# 安装 pip 包
./.conda/bin/pip install torch lpips opencv-python pykrige scikit-opt
```

### 5. 注册 Jupyter 内核

```bash
# 使用环境的 Python 注册内核
./.conda/bin/python -m ipykernel install --user \
    --name=ga-lem-inverter \
    --display-name="GA-LEM-Inverter (Python 3.11)"
```

### 6. 验证环境

```bash
# 运行测试脚本
./.conda/bin/python test_environment.py
```

## 环境使用

### 激活环境

```bash
# 方法 1: 使用 conda 激活
source ~/.zshrc  # 重新加载 shell 配置
conda activate ./.conda

# 方法 2: 直接使用环境的 Python
./.conda/bin/python your_script.py
```

### 使用 Jupyter

1. 启动 Jupyter Notebook 或 JupyterLab：
   ```bash
   jupyter notebook
   # 或
   jupyter lab
   ```

2. 在内核选择器中选择 "GA-LEM-Inverter (Python 3.11)"

### 验证包导入

```python
# 测试关键包导入
import numpy as np
import scipy
import matplotlib.pyplot as plt
import xsimlab  # xarray-simlab
import fastscape
import rasterio
import geopandas as gpd
import shapely
import sklearn
import torch
import cv2  # opencv-python
import lpips
import pykrige
import sko  # scikit-opt

print("所有包导入成功！")
```

## 依赖包列表

### 核心科学计算库
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0

### 地理空间处理
- rasterio >= 1.2.0
- geopandas >= 0.9.0
- shapely >= 1.8.0
- affine >= 2.3.0
- pyproj >= 3.0.0
- pykrige >= 1.5.0
- cartopy >= 0.20.0

### 景观演化模型
- xarray-simlab >= 0.4.0 (导入名: xsimlab)
- fastscape >= 0.1.0

### 优化算法
- scikit-opt >= 0.6.0 (导入名: sko)

### 空间统计
- libpysal >= 4.5.0
- esda >= 2.4.0
- seaborn >= 0.11.2

### 深度学习
- torch >= 1.9.0
- lpips >= 0.1.4

### 图像处理
- opencv-python >= 4.5.0 (导入名: cv2)
- scikit-image >= 0.18.0

### 工具库
- tqdm >= 4.61.0
- ipywidgets >= 7.6.0
- notebook >= 6.4.0
- psutil >= 5.8.0
- joblib >= 1.0.0
- typeguard >= 2.12.0
- configparser >= 5.0.0
- pyyaml >= 6.0
- dask
- plotly >= 5.3.0

### 开发工具
- pytest >= 6.2.5
- black >= 21.9b0
- flake8 >= 3.9.0
- mypy >= 0.910

## 常见问题

### Q: conda 命令未找到
A: 确保已初始化 conda 并重新加载 shell 配置：
```bash
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

### Q: 包导入失败
A: 检查是否使用了正确的 Python 解释器：
```bash
which python  # 应该指向 ./.conda/bin/python
```

### Q: Jupyter 内核未显示
A: 重新注册内核：
```bash
jupyter kernelspec remove ga-lem-inverter -f
./.conda/bin/python -m ipykernel install --user \
    --name=ga-lem-inverter \
    --display-name="GA-LEM-Inverter (Python 3.11)"
```

### Q: 权限问题
A: 确保脚本有执行权限：
```bash
chmod +x setup_environment.sh
```

## 环境管理

### 更新包
```bash
# 更新 conda 包
./.conda/bin/conda update --all

# 更新 pip 包
./.conda/bin/pip list --outdated
./.conda/bin/pip install --upgrade package_name
```

### 导出环境
```bash
# 导出环境配置
./.conda/bin/conda env export > environment.yml
```

### 重建环境
```bash
# 删除现有环境
rm -rf ./.conda

# 重新运行配置脚本
bash setup_environment.sh
```

## 技术细节

### 环境结构
```
项目根目录/
├── .conda/                 # 本地 conda 环境
│   ├── bin/               # 可执行文件
│   ├── lib/               # Python 库
│   └── ...               # 其他环境文件
├── setup_environment.sh    # 配置脚本
├── test_environment.py    # 测试脚本
└── ENVIRONMENT_SETUP.md   # 本文档
```

### 内核配置
Jupyter 内核配置文件位置：
```
~/Library/Jupyter/kernels/ga-lem-inverter/kernel.json
```

配置内容：
```json
{
  "argv": [
    "/path/to/project/.conda/bin/python",
    "-Xfrozen_modules=off",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "GA-LEM-Inverter (Python 3.11)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
```

## 支持

如果遇到环境配置问题，请：
1. 检查本文档的常见问题部分
2. 运行 `test_environment.py` 获取详细错误信息
3. 检查系统兼容性（macOS ARM64）
4. 确保网络连接正常（用于下载包）
