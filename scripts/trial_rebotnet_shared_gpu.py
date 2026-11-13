"""Two discarded native384 causal-reference updates after measured shared admission."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from prebind_shared_gpu import require_initial_binding, actual_device
from shared_gpu_admission import ROOT, assess, snapshot, sustained_samples, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bound = require_initial_binding()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit('Use a new owned versioned process directory.')
    out.mkdir(exist_ok=False, parents=True)
    import torch
    import cv2
    sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
    from rebotnet_reference import new_model, crop_predictions
    from rebotnet_resource_window import prepare
    gate = ROOT / 'research/5.prior-art-survey/process/causal-reference-code/native-qualification001.json'
    assert json.loads(gate.read_text())['passed']
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    complete, updates, videos, busy, selected, lock = False, 0, 0, 0, None, None
    compute_seconds = pause_seconds = 0.
    try:
        low, target, identities, data_info = prepare()
        cpu_model = new_model(0).train()
        assert not torch.cuda.is_initialized()
        write(out / 'config.json', dict(at=datetime.now().astimezone().isoformat(),
            revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
            source_commit='dff81d3cbd484b01ef2d42da2dd1ad1332598a09',
            protocol='research/4.temporal-consistency/docs/rebotnet-reference-preflight.md',
            pid=os.getpid(), discarded_updates=2, formal_updates=0, batch_videos=4,
            temporal_frames=4, crop=384, sequential_video_accumulation=True,
            objective='mean sqrt(error^2+1e-6), no output/history clamp or detach',
            optimizer='fresh Adam4e-4, clip1; state discarded', FP32=True, TF32=False,
            expected_total_increment_mib=4096, allocator_cap_mib=2048,
            target_compute_duty=.15, first_training_identities=identities, data=data_info,
            new_DEV_or_test_pixels=0, checkpoint_files=0))
        samples = sustained_samples()
        assessment = assess(samples)
        selected = next(r for r in assessment['candidates'] if r['uuid'] == bound)
        write(out / 'admission.json', dict(samples=samples, selected=selected,
              CPU_preparation_completed=True, initial_UUID_binding=True))
        if not selected['allowed']:
            raise SystemExit(75)
        lock = (ROOT / f'.cache/gpu-{selected["index"]}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immediate = snapshot()
        write(out / 'immediate-pre-CUDA.json', immediate)
        row = next(r for r in immediate['GPUs'] if r['uuid'] == bound)
        if row['total_mib'] - row['used_mib'] < selected['required_free_mib']:
            raise SystemExit(75)
        torch.cuda.set_per_process_memory_fraction(2048 / row['total_mib'], device=0)
        marker = torch.empty(1, dtype=torch.uint8, device='cuda')
        torch.cuda.synchronize()
        actual = actual_device(bound)
        write(out / 'actual-device.json', actual)
        assert actual['matched']
        del marker
        model = cpu_model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
        torch.cuda.reset_peak_memory_stats()

        def memory_guard(sample):
            gpu = next(r for r in sample['GPUs'] if r['uuid'] == bound)
            if gpu['total_mib'] - gpu['used_mib'] < selected['incumbent_growth_reserve_mib'] + 1024:
                raise RuntimeError('Incumbent memory reserve exhausted; stop own trial.')
            return gpu

        def observe(work_seconds, label):
            nonlocal busy, compute_seconds, pause_seconds
            immediate = snapshot()
            memory_guard(immediate)
            pause = max(2., work_seconds * (1/.15 - 1))
            time.sleep(pause)
            yielded = snapshot()
            gpu = memory_guard(yielded)
            busy = busy + 1 if gpu['utilization_percent'] >= 90 else 0
            compute_seconds += work_seconds
            pause_seconds += pause
            record = dict(at=datetime.now().astimezone().isoformat(), label=label,
                work_seconds=work_seconds, pause_seconds=pause,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                immediate_after_work=immediate, after_yield=yielded)
            with (out / 'observations.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if busy >= 3:
                raise RuntimeError('Sustained busy incumbent; stop own discarded trial.')

        losses = []
        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            errors = []
            for b in range(4):
                memory_guard(snapshot())
                torch.cuda.synchronize()
                began = time.perf_counter()
                inputs = low[b:b+1].cuda()
                targets = target[b:b+1].cuda()
                output = crop_predictions(model, inputs)
                loss = ((output-targets).square()+1e-6).sqrt().mean()
                assert torch.isfinite(loss)
                (loss/4).backward()
                value = float(loss.detach())
                del inputs, targets, output, loss
                torch.cuda.synchronize()
                seconds = time.perf_counter()-began
                videos += 1
                errors.append(value)
                observe(seconds, f'update{update+1}-video{b}')
            began = time.perf_counter()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize()
            updates += 1
            observe(time.perf_counter()-began, f'update{update+1}-optimizer')
            losses.append(dict(discarded_update=updates, four_train_crop_losses=errors,
                               gradient_norm_before_clip=norm))
            print(json.dumps(losses[-1]), flush=True)
        write(out / 'result.json', dict(passed=True, actual_GPU=selected['index'], UUID=bound,
            discarded_updates=updates, formal_updates=0, sequential_video_backwards=videos,
            registered_parameters=sum(p.numel() for p in model.parameters()), losses=losses,
            synchronized_work_seconds=compute_seconds, cooperative_pause_seconds=pause_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), final_snapshot=snapshot(),
            quality_rows=0, checkpoints_written=0,
            limits='Resource and finite-gradient trial only; no convergence, temporal benefit, phone runtime or incumbent non-slowdown measured. No state inherited.'))
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(ended=datetime.now().astimezone().isoformat(),
            completed_normally=complete, discarded_updates=updates, sequential_videos=videos,
            formal_updates=0, seconds=time.perf_counter()-started,
            CUDA_initialized=torch.cuda.is_initialized(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,
            scope='Child lifecycle includes CPU preparation, sustained sampling and pacing; not inference timing.'))
        if lock is not None:
            globals()['_held_project_lock'] = lock


if __name__ == '__main__':
    main()
