# R6 实际命令

创建：2026-11-11T17:03:36.029417+08:00；作者：Codex  (sqs_A100)；原因：从guard记录重建可核验入口。

目录 /data/Users/zrh/repo/MobileIE-Neo；PYTHONDONTWRITEBYTECODE=1。记录与输出不可覆盖复验，应使用新后缀。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r6/gpu-centered.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/probe_centered_noise_r6b.py --out research/4.temporal-consistency/results/exploration-r6b-centered
```

allowed=True；returncode=0。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r6/gpu-moments-gpu2.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/probe_noise_moments_r6.py --out research/4.temporal-consistency/results/exploration-r6-moments
```

allowed=True；returncode=0。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 0 --record research/4.temporal-consistency/process/exploration-r6/gpu-moments.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/probe_noise_moments_r6.py --out research/4.temporal-consistency/results/exploration-r6-moments
```

allowed=False；returncode=not launched。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r6/gpu-rgb-components.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/probe_rgb_components_r6c.py --out research/4.temporal-consistency/results/exploration-r6c-rgb-components
```

allowed=True；returncode=0。

