"""Resource-only wrapper around the original frozen complete R74 metric evaluator."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from shared_gpu_admission import ROOT, assess, snapshot, sustained_samples, write
sys.path.insert(0, str(ROOT/'research/4.temporal-consistency/scripts'))
import torch
import evaluate_cvvdp_motion_transport_r74 as evaluator

BASE = ROOT/'research/4.temporal-consistency'


def prerequisites():
    process = BASE/'process/exploration-r74'
    handoff = json.loads((process/'finalize002-resource-handoff.json').read_text())
    assert handoff['resource_wait_only'] and handoff['no_scientific_child_interrupted']
    assert handoff['original_controller_terminal']
    launch = json.loads((process/'finalize002-launch.json').read_text())
    # The original controller must have exited; a reused unrelated PID is harmless.
    proc = Path(f'/proc/{launch["pid"]}/stat')
    if proc.exists():
        assert proc.read_text().split()[21] != str(launch['start_ticks'])
    terminal = json.loads((process/'finalize002/runtime.json').read_text())
    assert not terminal['completed_normally']
    assert 'KeyboardInterrupt' in (process/'finalize002/failure.txt').read_text()
    assert (process/'finalize002/perceptual-resource-wait.json').is_file()
    for seed in [0, 1]:
        result = json.loads((BASE/f'results/exploration-r74-stage001-seed{seed}/runtime.json').read_text())
        assert result['completed_normally'] and result['motion_model_updates'] == dict(task=4000, proxy=4000)
        assert json.loads((BASE/f'results/exploration-r74-eval-seed{seed}/runtime.json').read_text())['completed_normally']
    quality = json.loads((BASE/'report/exploration-r74-quality/audit.json').read_text())
    assert quality['completed'] and quality['new_quality_rows'] == 80 and quality['all_quality_rows'] == 120
    renderer = json.loads((BASE/'report/exploration-r74-contacts/runtime.json').read_text())
    assert renderer['completed_normally'] and renderer['contacts'] == 20
    assert not (BASE/'results/exploration-r74-perceptual001').exists()
    frozen = json.loads((process/'finalize002/config.json').read_text())['source_sha256']
    assert len(frozen) == 42
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == sha for n, sha in frozen.items())
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    frozen = prerequisites()
    out = BASE/f'process/exploration-r74/shared-perception{args.attempt:03d}'
    out.mkdir(exist_ok=False)
    started, completed, lock, selected, scored, busy = time.perf_counter(), False, None, None, 0, 0
    original_predict = evaluator.predict
    original_admission = evaluator.cuda_headroom
    original_argv = sys.argv.copy()
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), source_sha256=frozen,
        resource_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_admission_sha256=hashlib.sha256((ROOT/'scripts/shared_gpu_admission.py').read_bytes()).hexdigest(),
        execution_amendment_sha256=hashlib.sha256((BASE/'docs/exploration-r74-shared-perception-execution.md').read_bytes()).hexdigest(),
        overridden_module_attributes=['cuda_headroom', 'predict scheduling around original call'],
        original_metric_calculations_unchanged=True, allocator_cap_mib=2048,
        expected_total_increment_mib=4096, target_compute_duty_fraction=.25,
        prior_same_shape_peak_allocated_bytes=360763392,
        selection='Original80 rows, no prefix selection or scientific retry.'))

    def admission():
        nonlocal lock, selected
        assert not torch.cuda.is_initialized()
        samples = sustained_samples(); assessment = assess(samples); selected = assessment['selected']
        record = dict(samples=samples, **assessment, allowed=selected is not None)
        write(out/'admission.json', record)
        if selected is None:
            return record
        lock = (ROOT/f'.cache/gpu-{selected["index"]}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh = snapshot(); write(out/'immediate-pre-CUDA.json', fresh)
        row = next(r for r in fresh['GPUs'] if r['uuid'] == selected['uuid'])
        if row['total_mib']-row['used_mib'] < selected['required_free_mib']:
            record['allowed'] = False; record['reason'] = 'Headroom changed before CUDA'
            write(out/'admission.json', record)
            return record
        os.environ['CUDA_VISIBLE_DEVICES'] = selected['uuid']
        torch.cuda.set_per_process_memory_fraction(2048/row['total_mib'], device=0)
        return record

    def paced_predict(metric, test, reference):
        nonlocal scored, busy
        torch.cuda.synchronize(); began = time.perf_counter()
        result = original_predict(metric, test, reference)
        torch.cuda.synchronize(); work_seconds = time.perf_counter()-began
        scored += 1
        immediate = snapshot()
        row = next(r for r in immediate['GPUs'] if r['uuid'] == selected['uuid'])
        if row['total_mib']-row['used_mib'] < selected['incumbent_growth_reserve_mib']+1024:
            raise RuntimeError('Incumbent memory reserve exhausted; no automatic metric retry')
        pause = max(2., 3*work_seconds)
        time.sleep(pause)
        after = snapshot()
        row = next(r for r in after['GPUs'] if r['uuid'] == selected['uuid'])
        busy = busy+1 if row['utilization_percent'] >= 90 else 0
        with (out/'resource-observations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(completed_original_predict=scored,
                synchronized_work_seconds=work_seconds, cooperative_pause_seconds=pause,
                immediate_after_compute=immediate, after_yield=after))+'\n')
        if busy >= 3 or row['total_mib']-row['used_mib'] < selected['incumbent_growth_reserve_mib']+1024:
            raise RuntimeError('Sustained incumbent load or low memory; no automatic metric retry')
        return result

    try:
        evaluator.cuda_headroom = admission
        evaluator.predict = paced_predict
        sys.argv = [str(BASE/'scripts/evaluate_cvvdp_motion_transport_r74.py'),
                    '--admission-attempt', str(10000+args.attempt)]
        evaluator.main()
        assert scored == 80
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == sha for n, sha in frozen.items())
        completed = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        evaluator.cuda_headroom, evaluator.predict = original_admission, original_predict
        sys.argv = original_argv
        write(out/'runtime.json', dict(completed_normally=completed, original_predict_calls=scored,
            CUDA_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-started,
            ended=datetime.now().astimezone().isoformat(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            scope='Resource wrapper lifecycle; not model/phone speed or incumbent throughput measurement.'))
        if lock is not None:
            globals()['_held_gpu_lock'] = lock


if __name__ == '__main__':
    main()
