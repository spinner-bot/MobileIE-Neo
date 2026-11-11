# R5 commands

Run from /data/Users/zrh/repo/MobileIE-Neo. Set PYTHONDONTWRITEBYTECODE=1 for Python jobs.

```bash
.venv/bin/python research/4.temporal-consistency/scripts/test_causal_fst_r5.py
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r5/gpu-cuda-check.json -- .venv/bin/python research/4.temporal-consistency/scripts/check_causal_gpu_r5.py --out research/4.temporal-consistency/results/exploration-r5-cuda-check.json
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r5/gpu-train-spatial.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants plain,spatial_bilinear --out research/4.temporal-consistency/results/exploration-r5-spatial
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 1 --record research/4.temporal-consistency/process/exploration-r5/gpu-train-temporal.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants temporal_bilinear,temporal_post --out research/4.temporal-consistency/results/exploration-r5-temporal
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r5/gpu-train-shift.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/train_causal_r5.py --variants tsm_learned,tsm_fixed --out research/4.temporal-consistency/results/exploration-r5-shift
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r5/gpu-cache.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/prepare_cache_r5.py --out research/4.temporal-consistency/results/exploration-r5-cache
```

训练 stdout/stderr 重定向 process/exploration-r5/train-{spatial,temporal,shift}.log，缓存到 cache.log。上面的固定 record/out 路径不得覆盖重跑；复验须新后缀并登记。
