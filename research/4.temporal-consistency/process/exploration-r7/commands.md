# R7实际命令

创建：2026-11-11T18:10:51.201750+08:00；作者：Codex  (sqs_A100)；原因：保存通道/融合/对齐的实际启动入口。

工作目录 /data/Users/zrh/repo/MobileIE-Neo；PYTHONDONTWRITEBYTECODE=1，原输出不能覆盖复验。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r7/gpu-aligned-features.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_aligned_mix_r7c.py --scope features --out research/4.temporal-consistency/results/exploration-r7c-features
```

退出0；启动Git 39712b679f8b4c69fdf48a397bab8d2cc9af287a。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r7/gpu-aligned-outputs.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_aligned_mix_r7c.py --scope outputs --out research/4.temporal-consistency/results/exploration-r7c-outputs
```

退出0；启动Git 1afecd7f8bade2dc78df61d433c7b24ef90fb0bd。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r7/gpu-channel-calibration.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/probe_channel_noise_r7.py --out research/4.temporal-consistency/results/exploration-r7-channel
```

退出0；启动Git 6c0d14003eff6039bbe7c467cc4ce6e2ed68dc1f。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r7/gpu-static-mix.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_static_mix_r7b.py --out research/4.temporal-consistency/results/exploration-r7b-static
```

退出0；启动Git 3839ebf21fc3903ebe35c80fa3aff427c2167804。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 3 --record research/4.temporal-consistency/process/exploration-r7/gpu-video-post.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_video_mix_r7b.py --methods body_post_first3_g05,body_post_selected_g05,body_post_uniform_g05,body_post_selected_g1 --out research/4.temporal-consistency/results/exploration-r7b-video-post
```

退出0；启动Git 41ade485634a3227bda1ee447be9a5ca4bf6fef5。

```bash
.venv/bin/python research/4.temporal-consistency/scripts/gpu_guard.py --gpu 2 --record research/4.temporal-consistency/process/exploration-r7/gpu-video-pre.json -- .venv/bin/python -u research/4.temporal-consistency/scripts/evaluate_video_mix_r7b.py --methods body_pre_first3_g05,body_pre_selected_g05,body_pre_uniform_g05,body_pre_selected_g1 --include-baseline --out research/4.temporal-consistency/results/exploration-r7b-video-pre
```

退出0；启动Git aeca0bc5323be23b7ce80522dbd7341ef90bdde8。

