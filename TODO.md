# TODO

## 当前完成状态

旧 TODO 中的 GA 可信度修复、隆升率整数编码、低分辨率 block 边界、地形 prior 初始化、objective 失败率控制、LPIPS/地形特征缓存、Pecube 坐标一致性适配均已完成，并已有对应测试覆盖。

当前主线已经完成：

```text
U_base(x,y) 空间 uplift 场
  + 可选 m_stage 时间倍率
  -> FastScape 连续正演
  -> 多步 topography_series + uplift_series
  -> Pecube 热年代学预测
  -> thermo loss + terrain loss
  -> GA 联合优化
```

也就是说，现在可以相对明确地表述为：

```text
已实现 10 Ma 等时间窗内的平均空间 uplift 场联合反演原型，
并支持低维分阶段 uplift history：

U(x,y,t) = U_base(x,y) * m_stage
```

## 已实现的时间变化 uplift history

第一版不是优化多个自由 uplift 场，而是优化：

```text
[U_base 低分辨率控制点] + [m1, m2, m3 ...]
```

配置入口：

```ini
[UpliftHistory]
enabled = true
mode = stage_multiplier
stage_times_ma = 10,6,3,0
multiplier_min = 0.5
multiplier_max = 1.5
multiplier_precision = 0.1
normalize_time_weighted_mean = true
```

含义：

```text
10-6 Ma: U_base * m1
6-3 Ma : U_base * m2
3-0 Ma : U_base * m3
```

关键约束：

- `stage_times_ma` 必须从过去到现在递减，并且第一个值要与 `[Model] time_total` 对齐，最后一个值必须是 `0`。
- `normalize_time_weighted_mean = true` 时，`m_stage` 会按阶段时长归一化到加权均值为 `1`。
- 因此 `U_base(x,y)` 仍表示整个模拟时间窗的平均 uplift，`m_stage` 只表示不同时段构造活动强弱。
- FastScape 内部通过自定义 `TimeScaledUplift` process 在每个 xsimlab step 按当前时间选择倍率，仍然是连续正演，不是拆成多个互不连续的小实验。
- Pecube 接收真实的 `topography_series + uplift_series`，其中 `uplift_series` 会随阶段倍率变化。

新增输出：

```text
figures/uplift_history_summary.png
arrays/stage_uplift.npy
arrays/cumulative_stage_uplift.npy
arrays/stage_multipliers.npy
```

## 已确认的 FastScape Python 接口事实

原生 `fastscape.models.basic_model` 不直接支持：

```text
uplift__rate(time, y, x)
```

实际验证结果：

- `uplift__rate` 可以接受标量 `()`。
- `uplift__rate` 可以接受二维场 `(y, x)`。
- 直接传三维时间序列 `(time, y, x)` 会失败。

因此当前实现不是把三维数组直接塞给 `basic_model`，而是替换 `uplift` process：

```text
basic_model.update_processes({
    "uplift": TimeScaledUplift
})
```

`TimeScaledUplift` 每个 step 输出：

```text
uplift_displacement = U_base * multiplier(stage) * dt
```

## 当前不做

第一版仍不做完整自由时间场：

```text
U_stage_1(y, x), U_stage_2(y, x), U_stage_3(y, x)
```

原因：

- 参数量会成倍增加。
- 地形终态对多阶段 uplift 的响应高度耦合。
- Pecube 样品点通常稀疏，难以唯一约束每个阶段的完整空间场。
- GA 会在大量等价解里搜索，结果很可能不可辨识。

后续如果要开放两个或多个自由场，应优先考虑受限形式：

```text
U_stage_i(x,y) = U_base(x,y) * m_i + delta_i(x,y)
```

其中 `delta_i` 必须是低分辨率、强平滑、幅度受限、带正则的修正场。

## 后续任务

1. 用 `demo/configs/demo3_staged_smoke.ini` 跑一次短 smoke，确认 `uplift_history_summary.png`、Pecube loss history 和 `stage_multipliers.npy` 正常生成。
2. 用 `demo/configs/demo3_staged_medium_smoke.ini` 跑一次 2-3 小时中等验收，观察热年代学 loss 是否对 `m_stage` 有有效梯度。
3. 如果热约束对倍率仍不敏感，优先检查样品筛选、Pecube 热参数、`thermo_loss_weight/scale` 和 stage 时间节点，而不是马上增加自由 uplift field。
4. 研究是否从外部 thermal history / cooling history 反演结果中提取 `stage_times_ma` 先验。
