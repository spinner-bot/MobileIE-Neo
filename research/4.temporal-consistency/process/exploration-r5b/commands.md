# exploration-r5b 实际执行入口

修改：2026-11-11T16:38:59.634516+08:00；作者：Codex  (sqs_A100)；原因：按完成的 guard 记录整理实际命令、退出状态及证据位置。

工作目录 /data/Users/zrh/repo/MobileIE-Neo；环境 PYTHONDONTWRITEBYTECODE=1。固定输出/记录不能覆盖，复验须新后缀。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5b/gpu-eval-plain.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_causal_r5.py --variants plain --checkpoint-root .cache/checkpoints/exploration-r5b --out research/4.temporal-consistency/results/exploration-r5b-eval-plain
```

退出状态：0；启动 Git：3e4fdd55c77e326327ce1130707e1df0e358e3bc。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r5b/gpu-eval-shift.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_causal_r5.py --variants tsm_learned --checkpoint-root .cache/checkpoints/exploration-r5b --out research/4.temporal-consistency/results/exploration-r5b-eval-shift
```

退出状态：0；启动 Git：46299f45d5147d6734bc7ff0fdbc9cdd582a4698。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r5b/gpu-eval-temporal.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_causal_r5.py --variants temporal_bilinear --checkpoint-root .cache/checkpoints/exploration-r5b --out research/4.temporal-consistency/results/exploration-r5b-eval-temporal
```

退出状态：0；启动 Git：3e4fdd55c77e326327ce1130707e1df0e358e3bc。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5b/gpu-train-plain.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants plain --synthetic-temporal-weight 1 --checkpoint-root .cache/checkpoints/exploration-r5b --out research/4.temporal-consistency/results/exploration-r5b-plain
```

退出状态：0；启动 Git：9cfb5a2a4c59566c772fda31b13c9eb4e5d67bc6。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r5b/gpu-train-shift.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants tsm_learned --synthetic-temporal-weight 1 --checkpoint-root .cache/checkpoints/exploration-r5b --out research/4.temporal-consistency/results/exploration-r5b-shift
```

退出状态：0；启动 Git：9cfb5a2a4c59566c772fda31b13c9eb4e5d67bc6。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r5b/gpu-train-temporal.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants temporal_bilinear --synthetic-temporal-weight 1 --checkpoint-root .cache/checkpoints/exploration-r5b --out research/4.temporal-consistency/results/exploration-r5b-temporal
```

退出状态：0；启动 Git：9cfb5a2a4c59566c772fda31b13c9eb4e5d67bc6。

