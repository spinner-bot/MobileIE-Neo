"""One fixed CPU train-only check of finite strong-post supervision targets."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
import cv2
import numpy as np
import torch
from causal_reference_io_r75 import write, append, stamp, digest
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from half_motion_r4 import half_forward_backward
from video_metrics import warp_previous


def available_kib():
    return next(int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()
                if r.startswith('MemAvailable:'))


def mse(a, b):
    return float(np.mean(((a.astype(np.float64) - b.astype(np.float64)) / 255) ** 2))


def temporal_proxy(raw, post, gt):
    values, coverage = [[], []], []
    for t in range(1, 4):
        flow, mask = half_forward_backward(gt[t - 1], gt[t])
        previous_gt, bounds = warp_previous(gt[t - 1], flow)
        valid = mask & bounds
        valid[:8] = valid[-8:] = False
        valid[:, :8] = valid[:, -8:] = False
        if not valid.any():
            raise RuntimeError('No registered GT-derived diagnostic support; do not select another window')
        coverage.append(float(valid.mean()))
        for j, output in enumerate((raw, post)):
            previous, _ = warp_previous(output[t - 1], flow)
            error = (output[t] - previous) - (gt[t] - previous_gt)
            values[j].append(float(np.abs(error)[valid].mean() / 255))
    return [float(np.mean(v)) for v in values], float(np.mean(coverage))


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    os.nice(5)
    frozen = json.loads((BASE / 'process/source-freeze001.json').read_text())
    assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
    assert frozen['protocol_sha256'] == digest(BASE / 'preflight-protocol.md')
    assert available_kib() >= 8 * 1024 * 1024
    assert not torch.cuda.is_initialized()
    out = BASE / 'process/preflight001'
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'config.json', dict(at=stamp(), pid=os.getpid(), source_freeze=frozen,
          data_model_seeds=[0, 1], data_steps=[8000, 8015], batch=4, frames=4, crop=128,
          expected_F0_images=512, expected_teacher_DIS_pairs=384,
          expected_GT_half_FB_pairs=384, optimizer_updates=0, student_forwards=0,
          initial_CUDA_VISIBLE_DEVICES='', Torch_threads=2, OpenCV_threads=1,
          nice_priority=os.getpriority(os.PRIO_PROCESS, 0), host_available_kib=available_kib(),
          quality_gate='Train-only necessary signal, not model/DEV adoption'))
    counts = dict(F0_images=0, teacher_DIS_pairs=0, GT_half_FB_pairs=0, window_records=0)
    started, complete = time.perf_counter(), False
    grouped = defaultdict(list)
    try:
        pool = MixedPool()
        write(out / 'data.json', pool.info)
        for seed in (0, 1):
            teacher, provenance = load_spatial(seed)
            assert sum(p.numel() for p in teacher.parameters()) == 4075
            write(out / f'teacher-seed{seed}.json', provenance)
            for step in range(8000, 8016):
                assert available_kib() >= 4 * 1024 * 1024
                low, target, identities = pool.batch(seed, step)
                with torch.inference_mode():
                    raw = teacher(low.flatten(0, 1)).clamp(0, 1).unflatten(0, (4, 4))
                raw = raw.permute(0, 1, 3, 4, 2).contiguous().numpy() * 255
                gt = target.permute(0, 1, 3, 4, 2).contiguous().numpy() * 255
                counts['F0_images'] += 16
                for b in range(4):
                    motion = FixedDIS()
                    stream = RecursivePipeline(motion)
                    post, accepted = [], []
                    for frame in raw[b]:
                        value, field = stream.step(frame)
                        post.append(value)
                        if field is not None:
                            accepted.append(float(field[1].mean()))
                    post = np.stack(post)
                    assert np.array_equal(post[0], raw[b, 0]) and np.isfinite(post).all()
                    assert motion.pairs == 3
                    proxy, coverage = temporal_proxy(raw[b], post, gt[b])
                    row = dict(at=stamp(), model_data_seed=seed, data_step=step, batch_index=b,
                        domain=identities[b]['source'], identity=identities[b],
                        raw_all_MSE=mse(raw[b], gt[b]), post_all_MSE=mse(post, gt[b]),
                        raw_tail_MSE=mse(raw[b, 1:], gt[b, 1:]), post_tail_MSE=mse(post[1:], gt[b, 1:]),
                        raw_train_proxy_temporal=proxy[0], post_train_proxy_temporal=proxy[1],
                        GT_proxy_valid_fraction=coverage, teacher_motion_accepted_fraction=float(np.mean(accepted)))
                    append(out / 'per_window.jsonl', row)
                    grouped[row['domain'], seed].append(row)
                    counts['window_records'] += 1
                    counts['teacher_DIS_pairs'] += 3
                    counts['GT_half_FB_pairs'] += 3
                print(json.dumps(dict(seed=seed, data_step=step, **counts)), flush=True)
        summary = []
        for (domain, seed), rows in sorted(grouped.items()):
            assert len(rows) == 32
            mean = lambda key: sum(r[key] for r in rows) / len(rows)
            ratio = mean('post_tail_MSE') / max(mean('raw_tail_MSE'), 1e-12)
            time_ratio = mean('post_train_proxy_temporal') / max(mean('raw_train_proxy_temporal'), 1e-12)
            wins = sum(r['post_train_proxy_temporal'] < r['raw_train_proxy_temporal'] for r in rows)
            coverage = mean('GT_proxy_valid_fraction')
            summary.append(dict(domain=domain, seed=seed, windows=len(rows), tail_MSE_ratio=ratio,
                train_proxy_temporal_ratio=time_ratio, proxy_temporal_win_windows=wins,
                GT_proxy_valid_fraction=coverage, teacher_motion_accepted_fraction=mean('teacher_motion_accepted_fraction'),
                necessary_signal_passed=ratio <= 1.02 and time_ratio <= .98 and coverage >= .25 and wins >= 16))
        assert len(summary) == 4
        assert counts == dict(F0_images=512, teacher_DIS_pairs=384, GT_half_FB_pairs=384, window_records=128)
        write(out / 'summary.json', dict(at=stamp(), groups=summary,
              proceed_to_separate_training_protocol=all(r['necessary_signal_passed'] for r in summary),
              method_established=False, independent_videos=False, full_training_registered=False,
              scope='Finite train-only target reliability. GT-derived temporal proxy is neither physical flow nor DEV performance.'))
        assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
        assert not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), pid=os.getpid(), completed_normally=complete,
              seconds=time.perf_counter() - started, **counts, optimizer_updates=0,
              student_forwards=0, CUDA_initialized=torch.cuda.is_initialized(), new_DEV_rows=0))


if __name__ == '__main__':
    main()
