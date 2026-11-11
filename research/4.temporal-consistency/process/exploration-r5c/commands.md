# exploration-r5c 实际执行入口

修改：2026-11-11T16:38:59.634516+08:00；作者：Codex  (sqs_A100)；原因：按完成的 guard 记录整理实际命令、退出状态及证据位置。

工作目录 /data/Users/zrh/repo/MobileIE-Neo；环境 PYTHONDONTWRITEBYTECODE=1。固定输出/记录不能覆盖，复验须新后缀。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5c/gpu-benchmark.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/benchmark_causal_r5.py --variants plain,spatial_post,tsm_fixed --direct-fixed-shift --out research/4.temporal-consistency/results/exploration-r5c-model-cost.json
```

退出状态：0；启动 Git：384cebd20f2ade55a87035b7a1112bdd14155fd5。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 1 --record research/4.temporal-consistency/process/exploration-r5c/gpu-eval-spatial-post.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_causal_r5.py --variants spatial_post --checkpoint-root .cache/checkpoints/exploration-r5c --out research/4.temporal-consistency/results/exploration-r5c-eval-spatial-post
```

退出状态：0；启动 Git：6ee6a8fd9d0849f236564c019ae10d2235a845ba。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5c/gpu-shift-numerics-full.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/check_shift_implementation_r5.py --out research/4.temporal-consistency/results/exploration-r5c-shift-numerics-full.json
```

退出状态：1；启动 Git：9c2bc212fb4e654da8e65ae14ca7f5d59f816aee。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5c/gpu-shift-numerics.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/check_shift_implementation_r5.py --out research/4.temporal-consistency/results/exploration-r5c-shift-numerics.json
```

退出状态：1；启动 Git：6ee6a8fd9d0849f236564c019ae10d2235a845ba。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 1 --record research/4.temporal-consistency/process/exploration-r5c/gpu-train-spatial-post.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants spatial_post --checkpoint-root .cache/checkpoints/exploration-r5c --protocol research/4.temporal-consistency/docs/exploration-r5c-protocol.md --out research/4.temporal-consistency/results/exploration-r5c-spatial-post
```

退出状态：0；启动 Git：3e4fdd55c77e326327ce1130707e1df0e358e3bc。

