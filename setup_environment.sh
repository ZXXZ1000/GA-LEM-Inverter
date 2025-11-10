#!/bin/bash
# =============================================================================
# GA-LEM-Inverter 环境配置脚本
# 一键创建conda环境并安装所有依赖包
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 命令未找到，请先安装 $1"
        exit 1
    fi
}

# =============================================================================
# 主函数
# =============================================================================

main() {
    print_info "开始配置 GA-LEM-Inverter 环境..."
    
    # 检查必要命令
    check_command "curl"
    check_command "bash"
    
    # 获取脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    cd "$SCRIPT_DIR"
    
    print_info "工作目录: $(pwd)"
    
    # =============================================================================
    # 1. 检查并安装 Miniconda
    # =============================================================================
    
    if [ ! -d "$HOME/miniconda3" ]; then
        print_info "安装 Miniconda..."
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
        bash Miniconda3-latest-MacOSX-arm64.sh -b -p ~/miniconda3
        rm Miniconda3-latest-MacOSX-arm64.sh
        
        # 初始化conda
        ~/miniconda3/bin/conda init zsh
        ~/miniconda3/bin/conda init bash
        
        print_success "Miniconda 安装完成"
    else
        print_info "Miniconda 已存在，跳过安装"
    fi
    
    # =============================================================================
    # 2. 创建本地conda环境
    # =============================================================================
    
    ENV_PATH="./.conda"
    
    if [ -d "$ENV_PATH" ]; then
        print_warning "环境已存在，删除旧环境..."
        rm -rf "$ENV_PATH"
    fi
    
    print_info "创建conda环境: $ENV_PATH"
    ~/miniconda3/bin/conda create -p "$ENV_PATH" python=3.11 -y
    
    # =============================================================================
    # 3. 安装核心包 (xarray-simlab 和 fastscape)
    # =============================================================================
    
    print_info "安装核心包: xarray-simlab, fastscape..."
    "$ENV_PATH/bin/conda" install -c conda-forge xarray-simlab fastscape -y
    
    # =============================================================================
    # 4. 安装conda可用的包
    # =============================================================================
    
    print_info "安装conda可用包..."
    "$ENV_PATH/bin/conda" install -c conda-forge \
        numpy scipy matplotlib scikit-image scikit-learn \
        rasterio geopandas shapely affine pyproj \
        libpysal esda seaborn tqdm ipywidgets \
        notebook psutil joblib typeguard configparser \
        pyyaml dask plotly pytest black flake8 mypy \
        cartopy pandas -y
    
    # =============================================================================
    # 5. 安装pip包
    # =============================================================================
    
    print_info "安装pip包..."
    "$ENV_PATH/bin/pip" install torch lpips opencv-python pykrige scikit-opt
    
    # =============================================================================
    # 6. 注册Jupyter内核
    # =============================================================================
    
    print_info "注册Jupyter内核..."
    "$ENV_PATH/bin/python" -m ipykernel install --user \
        --name=ga-lem-inverter \
        --display-name="GA-LEM-Inverter (Python 3.11)"
    
    # =============================================================================
    # 7. 验证环境
    # =============================================================================
    
    print_info "验证环境配置..."
    "$ENV_PATH/bin/python" -c "
import numpy, scipy, matplotlib, xsimlab, fastscape
import rasterio, geopandas, shapely, sklearn, torch
import lpips, cv2, tqdm, pandas, dask, plotly
import pykrige, sko
print('🎉 所有包导入成功！')
"
    
    if [ $? -eq 0 ]; then
        print_success "环境配置完成！"
        print_info "Jupyter内核 'GA-LEM-Inverter (Python 3.11)' 已注册"
        print_info "使用方法:"
        print_info "  1. 启动 Jupyter: jupyter notebook 或 jupyter lab"
        print_info "  2. 选择 'GA-LEM-Inverter (Python 3.11)' 内核"
        print_info "  3. 或者激活环境: conda activate ./.conda"
    else
        print_error "环境验证失败，请检查错误信息"
        exit 1
    fi
}

# 运行主函数
main "$@"