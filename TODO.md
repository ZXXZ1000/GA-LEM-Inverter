# TODO

## 当前完成状态

旧 TODO 中的 GA 可信度修复、隆升率整数编码、低分辨率 block 边界、地形 prior 初始化、objective 失败率控制、LPIPS/地形特征缓存、Pecube 坐标一致性适配均已完成，并已有对应测试覆盖。

当前主线已经完成：

```text
静态二维平均 uplift 场
  -> FastScape 连续正演
  -> 多步 topography 序列
  -> Pecube 热年代学预测
  -> thermo loss + terrain loss
  -> GA 联合优化
```

也就是说，现在可以相对明确地表述为：

```text
已实现 10 Ma 等时间窗内的平均空间 uplift 场联合反演原型。
```

但当前还没有实现：

```text
分阶段、随时间变化的 uplift history 反演。
```

## 下一阶段目标：时间变化 uplift history

### 1. 已确认的 FastScape Python 接口事实

原生 `fastscape.models.basic_model` 不直接支持：

```text
uplift__rate(time, y, x)
```

实际验证结果：

- `uplift__rate` 可以接受标量 `()`。
- `uplift__rate` 可以接受二维场 `(y, x)`。
- 直接传三维时间序列 `(time, y, x)` 会失败。

这说明不能把每个时间步的 uplift 场直接塞进当前 `basic_model`。

### 2. 显式初始 DEM 的使用方式

Python FastScape 可以使用显式初始 DEM，但不能直接把 `topography__elevation` 当作普通 input 塞给 `basic_model`。

正确方向是替换 `init_topography` process：

```text
basic_model.update_processes({
    "init_topography": InitialDEM
})
```

其中 `InitialDEM` 负责：

- 从输入读取 `initial_elevation(y, x)`。
- 校验 shape 与 FastScape 网格一致。
- 在 initialize 阶段把它写入 `SurfaceTopography.elevation`。

这使得后续可以做分段连续正演：

```text
stage 1 输入 DEM0 -> 输出 DEM1
stage 2 输入 DEM1 -> 输出 DEM2
stage 3 输入 DEM2 -> 输出 DEM3
```

### 3. 按 time 写 uplift 规则的使用方式

Python FastScape 也可以实现按时间变化的 uplift，但需要替换 `uplift` process，而不是直接使用 `BlockUplift`。

第一版建议实现：

```text
U(x, y, t) = U_base(x, y) * multiplier(stage)
```

对应自定义 process：

```text
basic_model.update_processes({
    "uplift": TimeScaledUplift
})
```

`TimeScaledUplift` 需要做：

- 读取基础空间 uplift 场 `U_base(y, x)`。
- 读取阶段边界 `stage_times`。
- 读取阶段倍率 `stage_multipliers`。
- 在每个 xsimlab time step 通过 `step_start` / `step_delta` 判断当前处于哪个阶段。
- 输出当前步的 uplift 位移：

```text
uplift_displacement = U_base * multiplier(stage) * dt
```

这样 FastScape 内部仍是连续正演，不需要把一次模拟硬拆成互不连续的小实验。

## 优化设计约束

不要第一版就优化完整的：

```text
U_stage_1(y, x), U_stage_2(y, x), U_stage_3(y, x) ...
```

原因：

- 参数量会成倍增加。
- 地形终态对多阶段 uplift 的响应高度耦合。
- Pecube 样品点通常稀疏，难以唯一约束每个阶段的完整空间场。
- GA 会在大量等价解里搜索，结果很可能不可辨识。

第一版应采用强参数化：

```text
U(x, y, t) = U_base(x, y) * m_stage
```

优化变量：

- `U_base(x, y)`：当前已经实现的二维空间 uplift 控制场。
- `m_stage`：少数几个阶段倍率，例如 3 个或 4 个。

先固定阶段时间：

```text
10-6 Ma: U_base * m1
6-3 Ma : U_base * m2
3-0 Ma : U_base * m3
```

稳定后再考虑让阶段边界时间也进入优化：

```text
t1, t2 + m1, m2, m3
```

## 实施顺序

1. 新增 `InitialDEM` process。
   - 验收：给定显式 DEM，FastScape 第一帧输出必须与输入 DEM 一致。

2. 新增 `TimeScaledUplift` process。
   - 验收：同一个 `U_base` 在不同阶段会按 `m_stage` 改变 uplift 位移。

3. 新增 `run_fastscape_time_scaled_series`。
   - 输入：`initial_dem`、`U_base`、`stage_times`、`stage_multipliers`、Ksp、边界和模型参数。
   - 输出：与 Pecube 对齐的 `topography_series` 和 `uplift_series`。

4. 扩展 GA 染色体。
   - 当前：只编码 `U_base(x, y)`。
   - 下一步：编码 `U_base(x, y) + m_stage`。
   - `m_stage` 应设置合理范围，例如 `0.5..1.5` 或 `0.25..2.0`，并使用独立 precision。

5. 扩展 Pecube 可视化。
   - 增加 stage uplift history 图。
   - 显示每个阶段的累计 uplift：

```text
cumulative_uplift_stage = U_base * m_stage * duration_stage
```

6. 增加产品验收测试。
   - 显式初始 DEM 测试。
   - time-scaled uplift 测试。
   - GA 解码 `U_base + m_stage` 测试。
   - Pecube 接收到多阶段 `uplift_series` 的测试。

## 当前不做

第一版不做完整自由时间场：

```text
U(time, y, x)
```

第一版也不直接引入 Fortran FastScape V3 的文件接口。当前优先在 Python xsimlab/FastScape 体系内完成可测试、可配置、可并行的实现。
