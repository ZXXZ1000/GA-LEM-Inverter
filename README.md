# GA-LEM-Inverter
本代码库包含了一种基于遗传算法的景观演化模型（LEM）逆反演方法，用于从地形中解析构造隆升场。特点包括与Fastscape景观演化模型的耦合、采用感知相似性（LPIPS）的多维适应度函数以及降维策略。相关论文：[提交中]。
[English Version](README_EN.md)
# 景观演化模型遗传算法优化使用说明

## 环境安装
```bash
git clone https://github.com/ZXXZ1000/GA-LEM-Inverter.git
cd GA-LEM-Inverter
```
然后安装所需环境：

```bash
pip install -r requirements.txt
```

如果安装过程出现问题，可能需要单独安装一些依赖，例如GDAL库、Fastscape库等。

## 代码运行分为三种模式

### 1. 合成地形实验

合成地形实验用于测试算法性能，使用人工生成的隆升场来验证算法能否有效地恢复原始隆升场。

#### 运行步骤：
1. 确保所有代码文件在同一文件夹中
2. 直接运行：
   ```bash
   python run_synthetic_experiment.py
   ```

#### 主要参数（可在run_synthetic_experiment.py中修改）：
- `shape`: 地形栅格大小，默认(100, 100)
- `patterns`: 测试的隆升模式，有'simple'（简单）, 'medium'（中等）, 'complex'（复杂）三种
- `scale_factor`: 降维因子，越大运算越快但精度降低
- `ga_params`: 遗传算法参数
  - `pop`: 种群大小，默认100
  - `max_iter`: 最大迭代次数，默认150
  - `prob_cross`: 交叉概率，默认0.7
  - `prob_mut`: 变异概率，默认0.05
  - `lb`和`ub`: 隆升率上下限，单位mm/yr

#### 实验结果：
结果将保存在`synthetic_experiments`目录下，包含：
- 真实隆升场与反演隆升场的对比
- 目标地形与模拟地形的对比
- 适应度演化历史
- 误差分析图表

  
### 2. 降维因子敏感性实验

降维因子(K)敏感性实验用于评估不同降维因子对反演结果的影响，帮助用户选择最佳的降维因子值。

#### 运行步骤：

1. 确保所有代码文件在同一文件夹中
2. 直接运行：
   ```bash
   python k_sensitivity_experiment.py
   ```
#### 主要参数（可在k_sensitivity_experiment.py中修改）：

- k_values: 要测试的降维因子值列表，例如[3, 5, 7, 10, 15]
- repetitions: 每个K值重复实验的次数，用于获取统计显著性
- pattern: 测试的隆升模式，可选'simple'、'medium'或'complex'
- shape: 地形栅格大小
- ga_params: 遗传算法参数（与合成实验相同）

#### 实验结果：

结果将保存在sensitivity_experiments目录下，包含：

- 综合分析图表：展示K值与RMSE、计算时间、R²的关系
- 最佳K值推荐：基于精度和计算效率的综合评分
- 不同K值的DEM和隆升场对比可视化
- 详细的统计数据和性能指标

#### 结果解读：

- RMSE vs 参数数量：展示精度与参数数量的权衡关系
- K vs 计算时间：展示降维因子对计算效率的影响
- K vs R²：展示降维因子对拟合质量的影响
- 综合评分：结合精度和效率，推荐最佳K值

  
### 3. 真实地形测试

真实地形测试使用真实的DEM数据进行隆升场反演。

#### 运行步骤：
1. 根据config.ini文件准备实验数据
2. 运行：
   ```bash
   python main.py
   ```

#### config.ini主要参数说明：

**[Paths]部分**
- `terrain_path`: DEM文件路径（支持.tif格式）
- `fault_shp_path`: 断层shapefile文件路径
- `study_area_shp_path`: 研究区shapefile文件路径
- `output_path`: 结果输出目录

**[Model]部分**
- `k_sp_value`: 基础侵蚀系数，默认6.92e-6
- `ksp_fault`: 断层带侵蚀系数，默认2e-5
- `d_diff_value`: 坡地扩散系数，默认19.2
- `boundary_status`: 边界条件，通常使用"fixed_value"
- `area_exp`: 面积指数，默认0.43
- `slope_exp`: 坡度指数，默认1.0
- `time_total`: 总模拟时间（年），根据研究区地质历史设置，通常为百万年至千万年量级

**[GeneticAlgorithm]部分**
- `ga_pop_size`: 种群大小，越大探索能力越强但计算时间更长
- `ga_max_iter`: 最大迭代次数
- `ga_prob_cross`: 交叉概率
- `ga_prob_mut`: 变异概率
- `lb`和`ub`: 隆升率上下限，根据研究区实际情况设置，活跃造山带约1-10 mm/yr
- `n_jobs`: 并行计算的进程数，-1表示使用所有CPU核心
- `decay_rate`: 种群大小衰减率
- `patience`: 早停耐心值，连续多少代无改进后停止

**[Preprocessing]部分**
- `smooth_sigma`: 平滑系数
- `scale_factor`: 降维因子，通常为5-10，越大运算越快但精度降低
- `ratio`: DEM降采样比例，0-1之间
- `target_crs`: 目标坐标系统（如需重投影）

#### 运行结果：
结果将保存在指定的输出目录下，包含：
- 原始DEM和旋转后DEM的可视化
- 侵蚀系数场可视化
- 反演出的隆升率场
- 基于该隆升场模拟的地形
- 与目标地形的对比图
- 隆升率分布图
- 3D地形可视化
- 优化过程记录

## 常见问题解决

1. **内存不足**：
   - 减小`ratio`值降低DEM分辨率
   - 增大`scale_factor`值减小参数空间

2. **运行时间过长**：
   - 减小`ga_pop_size`和`ga_max_iter`
   - 增大`n_jobs`使用更多CPU核心并行计算
   - 增大`scale_factor`降低计算量

3. **收敛性问题**：
   - 根据研究区实际情况调整`lb`和`ub`限定更合理的搜索范围
   - 增大`patience`允许更长的无改进迭代
   - 检查侵蚀系数是否合理

4. **坐标系统错误**：
   - 确保所有输入文件使用相同的坐标系统
   - 使用`target_crs`参数进行重投影
   
## 引用

如果您在研究中使用了本工具，请引用我们的论文：引用信息将在论文发表后更新

## 联系方式

如有任何问题或建议，请联系：

- 邮箱：[xiangzhao@zju.edu.cn](xiangzhao@zju.edu.cn)
