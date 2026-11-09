# 方法候选与方案 A 原型

- 创建：2026-11-05 · claude code (local)
- 状态：设计草案；`temporal_loss.py` 已通过冒烟测试，尚未接入正式训练。

## 1. 三条路线

| 方案 | 做法 | 推理开销 | 风险 | 优先级 |
| --- | --- | --- | --- | --- |
| **A. 训练期时序损失** | 任务损失 + 对齐后的时序一致性损失；网络结构不变 | **0** | 损失权重难调；对快速运动/遮挡敏感 | ★★★ |
| B. 轻量帧间模块 | feature EMA / 1–2 帧输入 / 轻量 recurrent conv；尽量可重参数化 | 小～中 | 参数/延迟预算；可能引入拖影 | ★★ |
| C. 输入多帧拼接 | 把上一帧与当前帧拼接输入 | 中（通道翻倍） | 参数量增加；对齐问题 | ★ |

**主线选择 A**：最符合 MobileIE “训练复杂、推理极简”的哲学；若 A 不足，再加 B。

## 2. 方案 A：损失设计

```text
L = L_task(out_t, gt_t)
  + λ_temp * L_temp(out_t, warp(out_{t-1}, flow_{t-1→t}))
  + λ_bright * |mean(out_t) - mean(out_{t-1})|        # 可选
```

- `L_task`：沿用 `LossLLE`（LVW + 余弦相似度 + PSNR 损失）；
- `L_temp`：Charbonnier，见 [`scripts/temporal_loss.py`](../scripts/temporal_loss.py)；
- `flow`：训练期计算，推理不用；合成视频可用已知平移，真实视频用 Farneback/RAFT 离线预计算；
- **遮挡处理**：前后向光流一致性检查生成 mask；越界区域置 0；
- `detach_prev=True`（默认）：只对当前帧回传梯度，省显存；后续对比 `False` 的效果。

### 已实现

- `TemporalConsistencyLoss`：支持 flow warp、mask、detach；
- `BrightnessConsistencyLoss`：全局亮度项；
- `translation_flow`：已知平移的 flow 生成；
- 冒烟测试：已知 2 px 平移 + 正确 flow + 边界 mask，损失 ~1e-6；无 flow 时 0.333。

### 待确定

1. **初始化**：从零训练，还是从 slim 权重转换/微调？当前只有 slim 权重，多分支训练态需要重新训练；
2. **λ 调度**：先用 0.1 / 0.5 / 1.0 三档扫描；时序损失量纲需与 `L_task` 对齐；
3. **训练数据**：LOL 静态图合成视频（随机平移/缩放/亮度变化 + 退化），是否需要真实视频参与；
4. **是否 detach**：显存与效果权衡；
5. **快速运动**：16/32 px/帧 下 flow 误差变大，需要 mask 或鲁棒损失（Charbonnier 已有一定鲁棒性）。

## 3. 方案 B（备选）

- feature EMA：`f_t = α f_{t-1} + (1-α) f_t`，α 可学习或固定；
- 1–2 帧输入：把上一帧（或其增强结果）与当前帧拼接，第一层通道 3→6；
- 可重参数化时序分支：训练期多分支、推理期折叠成单卷积（参考 NanoVSR / MobileIE 的 MBRConv 思路）；
- 必须报告：参数量、MACs、延迟、是否引入帧缓冲。

## 4. 方案 C（对照）

输入拼接最简单，但通道翻倍会直接增加第一层计算量；主要用于证明“显式多帧输入”不是必要条件，或作为强对照。

## 5. 训练与评测接入

1. 数据侧：新增“视频对”采样器（连续帧 + flow + occlusion mask），复用 `LLEData` 的读图/归一化；
2. 训练侧：复制 `main.py` 的训练循环到 `train_temporal.py`，加入 `L_temp`，其余超参与官方配置一致；
3. 评测侧：按 [`evaluation-protocol.md`](evaluation-protocol.md) 统一跑合成 + 真实视频；
4. 每个实验固定种子，报告闪烁-画质-效率三元组。

## 6. 修订记录

- 2026-11-05 · claude code (local)：创建；完成方案 A 损失原型与冒烟测试。
