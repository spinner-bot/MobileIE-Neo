# exploration-r5 实际执行入口

修改：2026-11-11T16:38:59.634516+08:00；作者：Codex  (sqs_A100)；原因：按完成的 guard 记录整理实际命令、退出状态及证据位置。

工作目录 /data/Users/zrh/repo/MobileIE-Neo；环境 PYTHONDONTWRITEBYTECODE=1。固定输出/记录不能覆盖，复验须新后缀。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5/gpu-benchmark.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/benchmark_causal_r5.py --out research/4.temporal-consistency/results/exploration-r5-model-cost.json
```

退出状态：0；启动 Git：7a4ea312f7fe43f2ac47ea7e8ca34b831f9a93e0。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r5/gpu-cache.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/prepare_cache_r5.py --out research/4.temporal-consistency/results/exploration-r5-cache
```

退出状态：0；启动 Git：a67a7e0ea298c9216ac0847e72b2091dc08af7cd。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r5/gpu-cuda-check.json -- .venv/bin/python research/4.temporal-consistency/scripts/check_causal_gpu_r5.py --out research/4.temporal-consistency/results/exploration-r5-cuda-check.json
```

退出状态：0；启动 Git：b444da4eaa3f6255dcad37a488c06b91614bbab2。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r5/gpu-eval-shift.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_causal_r5.py --variants tsm_learned,tsm_fixed --out research/4.temporal-consistency/results/exploration-r5-eval-shift
```

退出状态：0；启动 Git：15934f8a3329ee38a80cebd246c2a74a041567c3。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5/gpu-eval-spatial.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_causal_r5.py --variants plain,spatial_bilinear,r2_task --out research/4.temporal-consistency/results/exploration-r5-eval-spatial
```

退出状态：0；启动 Git：15934f8a3329ee38a80cebd246c2a74a041567c3。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 1 --record research/4.temporal-consistency/process/exploration-r5/gpu-eval-temporal.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_causal_r5.py --variants temporal_bilinear,temporal_post --out research/4.temporal-consistency/results/exploration-r5-eval-temporal
```

退出状态：0；启动 Git：15934f8a3329ee38a80cebd246c2a74a041567c3。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r5/gpu-train-shift.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants tsm_learned,tsm_fixed --out research/4.temporal-consistency/results/exploration-r5-shift
```

退出状态：0；启动 Git：4908e92f2be8c08d2f624962b6bb2898a00d06e1。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5/gpu-train-spatial.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants plain,spatial_bilinear --out research/4.temporal-consistency/results/exploration-r5-spatial
```

退出状态：0；启动 Git：4908e92f2be8c08d2f624962b6bb2898a00d06e1。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 1 --record research/4.temporal-consistency/process/exploration-r5/gpu-train-temporal.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants temporal_bilinear,temporal_post --out research/4.temporal-consistency/results/exploration-r5-temporal
```

退出状态：0；启动 Git：4908e92f2be8c08d2f624962b6bb2898a00d06e1。

