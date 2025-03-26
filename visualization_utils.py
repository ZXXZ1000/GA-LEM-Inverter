# visualization_utils.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import logging
from typing import Tuple, Optional, List, Dict

def setup_plot_style():
    """设置全局绘图样式"""
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.style'] = 'italic'

def plot_comparison(data1: np.ndarray, data2: np.ndarray, 
                   title1: str, title2: str, value1: str, value2: str, 
                   cmap: str = 'viridis', figsize: Tuple[int, int] = (15, 10), 
                   dpi: int = 300) -> plt.Figure:
    """绘制两个数据集的对比图"""
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # 获取data2的值范围
    vmin = np.nanmin(data2)
    vmax = np.nanmax(data2)
    
    # 绘制第一个数据集，使用data2的值范围
    im1 = ax1.imshow(data1, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    ax1.set_title(title1, fontsize=16, weight='bold', fontstyle='italic')
    ax1.set_xlabel('X', fontsize=14, weight='bold', fontstyle='italic')
    ax1.set_ylabel('Y', fontsize=14, weight='bold', fontstyle='italic')
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label(value1, fontsize=14, weight='bold', fontstyle='italic', labelpad=10)
    
    # 绘制第二个数据集
    im2 = ax2.imshow(data2, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    ax2.set_title(title2, fontsize=16, weight='bold', fontstyle='italic')
    ax2.set_xlabel('X', fontsize=14, weight='bold', fontstyle='italic')
    ax2.set_ylabel('Y', fontsize=14, weight='bold', fontstyle='italic')
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label(value2, fontsize=14, weight='bold', fontstyle='italic', labelpad=10)

    plt.tight_layout()
    return fig

def plot_uplift_distribution(uplift_data: np.ndarray) -> plt.Figure:
    """绘制隆升率分布图"""
    setup_plot_style()
    
    y_coords = np.arange(uplift_data.shape[0])
    x_coords = np.arange(uplift_data.shape[1])
    uplift_values = uplift_data[:, :]/10

    mean_uplift = np.mean(uplift_values, axis=0)
    std_uplift = np.std(uplift_values, axis=0)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax1 = plt.gca()
    
    ax1.plot(x_coords, mean_uplift, color='#B32626', label='Mean Uplift Rate', linewidth=2)
    ax1.fill_between(x_coords, mean_uplift - std_uplift, mean_uplift + std_uplift, 
                     color='gray', alpha=0.2, label='Standard Deviation Range')

    ax1.set_xlabel('Y Coordinate', fontsize=14, weight='bold', fontstyle='italic')
    ax1.set_ylabel('Uplift Rate (mm/y)', fontsize=14, weight='bold', 
                   fontstyle='italic', color='#B32626')
    ax1.set_title('Uplift Rate Distribution and 10Ma Total Uplift', 
                  fontsize=16, weight='bold', fontstyle='italic')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='#B32626')

    ax2 = ax1.twinx()
    total_uplift_10ma = mean_uplift * 10
    ax2.plot(x_coords, total_uplift_10ma, color='#1E88E5', 
             linestyle='--', label='10Ma Total Uplift')
    ax2.set_ylabel('10Ma Total Uplift (km)', fontsize=14, weight='bold', 
                   fontstyle='italic', color='#1E88E5')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right')

    return fig

def plot_uplift_distribution_y(uplift_data: np.ndarray) -> plt.Figure:
    """绘制沿Y轴的隆升率分布图"""
    setup_plot_style()
    
    y_coords = np.arange(uplift_data.shape[0])
    x_coords = np.arange(uplift_data.shape[1])
    uplift_values = uplift_data[:, :]/10

    mean_uplift = np.mean(uplift_values, axis=1)
    std_uplift = np.std(uplift_values, axis=1)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax1 = plt.gca()
    
    ax1.plot(y_coords, mean_uplift, color='#B32626', label='Mean Uplift Rate', linewidth=2)
    ax1.fill_between(y_coords, mean_uplift - std_uplift, mean_uplift + std_uplift, 
                        color='gray', alpha=0.2, label='Standard Deviation Range')

    ax1.set_xlabel('Y Coordinate', fontsize=14, weight='bold', fontstyle='italic')
    ax1.set_ylabel('Uplift Rate (mm/y)', fontsize=14, weight='bold', 
                    fontstyle='italic', color='#B32626')
    ax1.set_title('Uplift Rate Distribution and 10Ma Total Uplift', 
                    fontsize=16, weight='bold', fontstyle='italic')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='#B32626')

    ax2 = ax1.twinx()
    total_uplift_10ma = mean_uplift * 10
    ax2.plot(y_coords, total_uplift_10ma, color='#1E88E5', 
                linestyle='--', label='10Ma Total Uplift')
    ax2.set_ylabel('10Ma Total Uplift (km)', fontsize=14, weight='bold', 
                    fontstyle='italic', color='#1E88E5')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right')

    return fig

def plot_single_data(data: np.ndarray, title: str, cmap: str = 'terrain', 
                    origin: str = 'lower', vmin: Optional[float] = None, 
                    vmax: Optional[float] = None) -> plt.Figure:
    """绘制单个数据集的可视化图"""
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()
    
    im = ax.imshow(data, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=16, weight='bold', fontstyle='italic')
    ax.set_xlabel('X', fontsize=14, weight='bold', fontstyle='italic')
    ax.set_ylabel('Y', fontsize=14, weight='bold', fontstyle='italic')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Value', fontsize=14, weight='bold', fontstyle='italic', labelpad=10)
    
    return fig

def plot_3d_surface(data: np.ndarray, uplift: np.ndarray, 
                    title: str = "3D Surface Visualization") -> plt.Figure:
    """绘制3D地形和隆升率叠加图"""
    setup_plot_style()
    
    def create_smooth_surface(surface: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        rows, cols = surface.shape
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        X, Y = np.meshgrid(x, y)
        
        x_new = np.linspace(0, cols - 1, cols * scale_factor)
        y_new = np.linspace(0, rows - 1, rows * scale_factor)
        X_new, Y_new = np.meshgrid(x_new, y_new)
        
        Z_new = griddata((X.ravel(), Y.ravel()), surface.ravel(), 
                        (X_new, Y_new), method='cubic')
        return Z_new

    # 创建平滑表面
    smooth_data = create_smooth_surface(data)
    smooth_uplift = create_smooth_surface(uplift)

    fig = plt.figure(figsize=(15, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(np.arange(smooth_data.shape[1]), 
                       np.arange(smooth_data.shape[0]))

    # 绘制地形表面
    surf = ax.plot_surface(X, Y, smooth_data, cmap='terrain',
                          linewidth=0, antialiased=True)
    
    # 添加隆升率等值线
    levels = np.linspace(np.min(smooth_uplift), np.max(smooth_uplift), 20)
    contour = ax.contour(X, Y, smooth_uplift, levels, cmap='RdBu_r')

    ax.set_title(title, fontsize=16, weight='bold', fontstyle='italic')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

    return fig

def plot_optimization_history(fitness_history: List[float], save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制优化过程的适应度历史
    
    参数:
    - fitness_history: 适应度历史记录
    - save_path: 可选的保存路径
    
    返回:
    - fig: matplotlib图形对象
    """
    setup_plot_style()
    
    # 创建图形
    fig = plt.figure(figsize=(10, 6), dpi=300)
    
    # 绘制主曲线
    plt.plot(fitness_history, linewidth=2, label='Fitness')
    
    # 添加最佳适应度标记
    best_gen = np.argmin(fitness_history)
    best_fitness = np.min(fitness_history)
    plt.plot(best_gen, best_fitness, 'ro', 
            label=f'Best: {best_fitness:.4f} at gen {best_gen}')
    
    # 设置标题和标签
    plt.title('Optimization History', fontsize=16, weight='bold')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness Value', fontsize=14)
    
    # 添加网格和图例
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 如果提供了保存路径，保存图像
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved fitness history plot to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save fitness history plot: {e}")
    
    return fig


def display_array_info(name: str, array: np.ndarray, spacing: float):
    """显示数组的统计信息"""
    info = {
        "Shape": array.shape,
        "Spacing": spacing,
        "Min value": np.nanmin(array),
        "Max value": np.nanmax(array),
        "Has NaN": np.isnan(array).any(),
    }
    
    print(f"\n{'-'*50}")
    print(f"{name} Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    if info["Has NaN"]:
        print(f"Number of NaN values: {np.isnan(array).sum()}")
    print(f"{'-'*50}\n")

def display_tiff_info(tiff_path: str):
    """打印TIFF文件的详细信息"""
    import rasterio
    try:
        with rasterio.open(tiff_path) as src:
            print("\n=== TIFF文件基本信息 ===")
            print(f"文件路径: {tiff_path}")
            print(f"影像尺寸: {src.width} x {src.height} 像素")
            print(f"波段数量: {src.count}")
            
            print("\n=== 坐标系统信息 ===")
            print(f"投影: {src.crs}")
            
            xres, yres = src.res
            print("\n=== 分辨率信息 ===")
            print(f"X方向分辨率: {abs(xres):.2f} 米")
            print(f"Y方向分辨率: {abs(yres):.2f} 米")
            print(f"栅格大小: {abs(xres)/1000:.3f} x {abs(yres)/1000:.3f} 公里")
            
            bounds = src.bounds
            print("\n=== 数据范围 ===")
            print(f"左边界: {bounds.left:.2f}")
            print(f"右边界: {bounds.right:.2f}")
            print(f"上边界: {bounds.top:.2f}")
            print(f"下边界: {bounds.bottom:.2f}")
            print(f"宽度: {(bounds.right - bounds.left)/1000:.2f} km")
            print(f"高度: {(bounds.top - bounds.bottom)/1000:.2f} km")
            
            print("\n=== 数据特征 ===")
            print(f"数据类型: {src.dtypes[0]}")
            
            data = src.read(1)
            print(f"最小值: {np.nanmin(data):.2f}")
            print(f"最大值: {np.nanmax(data):.2f}")
            print(f"平均值: {np.nanmean(data):.2f}")
    except Exception as e:
        logging.error(f"读取TIFF文件信息时出错: {e}")