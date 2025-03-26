# result_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import logging

def visualize_and_save_results(best_full_res_uplift, output_path='results'):
    """
    可视化并保存结果。

    参数:
    - best_full_res_uplift: 最佳全分辨率抬升速率。
    - output_path: 输出路径。
    """
    try:
        # 保存结果
        np.save(f'{output_path}/best_full_res_uplift.npy', best_full_res_uplift)
        logging.info(f"最佳结果已保存到: {output_path}/best_full_res_uplift.npy")

        # 可视化结果
        plt.imshow(best_full_res_uplift, cmap='RdBu_r')
        plt.colorbar(label='Uplift Rate')
        plt.title('Best Full Resolution Uplift')
        plt.savefig(f'{output_path}/best_full_res_uplift.png')
        plt.show()
    except Exception as e:
        logging.error(f"可视化和保存结果出错: {e}")
        raise RuntimeError(f"可视化和保存结果出错: {e}")