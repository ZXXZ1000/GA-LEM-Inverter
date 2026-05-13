# TODO

## GA 优化可信度与稳定性

### 1. 修复最优个体和适应度错配

问题：GA 每代评估后会排序 `Y`，但记录 best 时可能拿到排序前或后续被变异改写的个体，导致 `best_x` 和 `best_y` 不一定对应。

修改方向：

- 每代适应度评估后先排序。
- 在 selection/crossover/mutation 之前，立即用排序后的 `self.Chrom[0].copy()` 和 `self.Y[0]` 记录本代最优。
- 记录 best 时必须 copy，避免后续种群操作污染最优个体。
- 增加测试，确保返回的 `best_x` 确实对应最低 fitness。

### 2. 明确隆升率离散编码

问题：旧实现直接用整数搜索值进入目标函数，物理语义不够明确；早期代码实际更接近“整数编码 1..10，对应 0.1..1.0 mm/yr”的设计。

修改方向：

- 保留整数编码搜索，避免过细浮点搜索。
- 显式加入 `uplift_scale` 或 `uplift_precision`。
- GA 内部搜索整数编码，例如 `1..10`。
- 进入 objective/FastScape 前解码成真实隆升率，例如乘以 `0.1` 得到 `0.1..1.0 mm/yr`。
- 输出、图件和 metrics 使用解码后的真实隆升率。

### 3. 修复 block_size 小矩阵边界问题

问题：`block_size` 是低分辨率隆升矩阵内部的空间交叉块大小，不是 `scale_factor/K/N`。当低分辨率矩阵很小时，当前计算可能得到 `0`，导致除零或交叉失败。

修改方向：

- `block_size` 至少为 `1`。
- 对 `LOW_RES_SHAPE` 做明确校验。
- 如果矩阵太小，交叉退化为点级、行级或列级操作。
- 不把 `block_size` 和 `scale_factor` 混用。

### 4. 强化地形 prior 初始化

问题：基于 DEM 分布初始化是合理的，可以减少完全随机搜索的难度；但平坦 DEM、异常值、NaN 或过强地形 prior 会带来除零、NaN 或搜索偏置。

修改方向：

- 保留 DEM prior 初始化。
- 对 `max-min=0`、NaN、Inf 做兜底。
- 使用混合初始化，而不是只依赖高程分布：
  - 一部分地形 prior 个体。
  - 一部分随机个体。
  - 一部分平滑随机场个体。
- 后续可把各类初始化比例放进配置。

### 5. 增加目标函数失败率控制

问题：当前 objective 如果 FastScape、LPIPS、shape 匹配等步骤失败，会返回惩罚值继续跑。单个坏个体这样处理可以接受，但系统性失败时 GA 可能假装完成并产生假收敛。

修改方向：

- 保留单个候选解失败惩罚。
- 统计每代失败率。
- 如果一代中失败率过高，或连续多代全失败，直接中止。
- 报错信息要包含真实异常摘要和失败样本数量。
- 返回 fitness 前统一检查有限数和值域。

### 6. 缓存适应度计算中的固定项

问题：目标 DEM 特征和 LPIPS 模型不应该在每个候选解里反复计算或加载，多进程下尤其浪费。

修改方向：

- 目标 DEM 的固定特征预计算。
- LPIPS 模型做进程级缓存。
- 固定输入相关的 CV/特征计算缓存。
- 高频评价日志从 INFO 降到 DEBUG。

## 建议执行顺序

1. 先修 best_x/best_y 错配，并加测试。
2. 再补明确的隆升率整数编码/解码机制。
3. 修 block_size 和初始化边界稳定性。
4. 增加 objective 失败率控制。
5. 最后做 LPIPS 和地形特征缓存。

## Pecube 坐标一致性修正

状态：已完成。

### 问题

当前 `main` 模式已经可以从 DEM 的 CRS/transform 自动推导 `lon0/lat0/dlon/dlat`，并把样品 `lon/lat` 接到 Pecube 输出上。

但这里仍有一个需要修正的科学问题：如果输入 DEM 是 UTM 等米制投影坐标，直接用 DEM 四角经纬度反推 `dlon/dlat`，本质上是把投影规则网格近似成规则经纬度网格。小范围 demo 可以跑通，但严格来说这不是完整的坐标一致性处理。

Pecube 要求：

```text
lon0, lat0 = topographic grid bottom-left origin, decimal degrees
dlon, dlat = longitude/latitude grid spacing, decimal degrees
```

所以 Pecube 输入的 `topo/uplift/temp` 矩阵应当对应一个规则经纬度网格，而不是直接把 UTM 规则网格当成经纬度规则网格。

### 当前临时状态

- `demo/data/demo_thermo_samples.csv` 使用真实 `lon,lat`。
- `main` 会读取 DEM CRS/transform，自动估算 Pecube 的 `lon0/lat0/dlon/dlat`。
- 样品点会按 Pecube 输出的 `Longitude/Latitude` 匹配。
- 这能验证 FastScape -> Pecube -> thermo loss -> GA 的完整链路。
- 但对真实研究区，尤其是较大区域或投影畸变明显区域，这个近似不应作为最终科学实现。

### 修改方向

1. 增加一个正式的 `FastScape projected grid -> Pecube geographic grid` 适配层。
2. 输入：
   - FastScape 输出的 `topography/uplift/temperature` 矩阵。
   - DEM 的 CRS、transform、bounds。
   - 用户样品 `lon/lat` 或投影坐标。
3. 输出：
   - 规则经纬度网格上的 `topo0/topo1/uplift0/uplift1/temp0/temp1`。
   - 与该经纬度网格严格一致的 `Pecube.in`：
     ```text
     lon0
     lat0
     dlon
     dlat
     nx
     ny
     ```
4. 对投影 DEM：
   - 先构造目标规则经纬度网格。
   - 用 `rasterio.warp.reproject` 或等价插值，把投影坐标下的矩阵重采样到经纬度网格。
5. 对已经是 EPSG:4326 的 DEM：
   - 可以直接使用原始 transform 推导 `lon0/lat0/dlon/dlat`。
6. 样品坐标处理：
   - `lon/lat` 样品直接使用。
   - `projected/dem_crs` 样品先转 EPSG:4326。
   - `grid_index` 样品先转 DEM 坐标，再转 EPSG:4326。
7. 增加校验：
   - 样品点必须落在 Pecube 经纬度网格范围内。
   - Pecube 输出 `Ages001.csv` 的 `Longitude/Latitude` 范围必须覆盖样品。
   - 输出 summary 里记录坐标处理方式和网格范围。

### 已实现

- 新增 `PecubeSpatialAdapter`，把 DEM CRS/transform 转成 Pecube 所需的规则 EPSG:4326 网格。
- EPSG:4326 DEM 直接使用原始规则经纬度网格，不额外重采样。
- 投影 DEM 使用 `rasterio.warp.calculate_default_transform` 和 `rasterio.warp.reproject` 把 `topography/uplift/temperature` 数组重采样到规则经纬度网格。
- `main` 模式自动使用该 adapter，Pecube 运行时写入的 `topo/uplift/temp` 与 `lon0/lat0/dlon/dlat` 保持同一网格。
- 样品 `dem_crs/projected` 坐标自动转 EPSG:4326；`grid_index` 在自动模式下解释为原始 DEM 像素索引，再转 EPSG:4326。
- Pecube 空间图使用真实 `lon0/lat0` 作为 extent，避免样品点和地形底图错位。
- 已增加坐标单元测试，覆盖 EPSG:4326 DEM、投影 DEM 重采样、投影样品坐标、DEM 像素索引样品。
