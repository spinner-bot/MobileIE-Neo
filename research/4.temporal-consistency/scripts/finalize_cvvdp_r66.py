"""Finish reference table, wait without a GPU, then score every complete R65 output."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from cvvdp_adapter_r66 import BASE, ROOT
from dynamic_training_r34 import digest
from evaluate_cvvdp_references_r66 import write
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from finalize_task_fastdvd_r47 import wait_training
from gpu_guard import sample


def wait_normal_runtime(path, pid, identity):
    while True:
        if path.exists():
            record = read_stable(path)
            assert record['completed_normally'], 'Original dependency ended unsuccessfully; no automatic retry.'
            return record
        if process_identity(pid) != identity:
            time.sleep(1)
            assert path.exists(), 'Original dependency disappeared without terminal evidence.'
        time.sleep(5)


def main():
    process = BASE/'process/exploration-r66'; out = process/'finalize001'; out.mkdir(exist_ok=False)
    reference_launch = read_stable(process/'reference-launch002.json')
    guard = read_stable(process/'reference-guard002.json'); assert guard['allowed']
    ref_identity = process_identity(reference_launch['guard_pid'])
    assert 'returncode' in guard or (ref_identity and 'gpu_guard.py' in ref_identity['commandline'])
    original_launch = read_stable(BASE/'process/exploration-r65/finalize-launch.json')
    original_identity = process_identity(original_launch['pid'])
    original_runtime = BASE/'process/exploration-r65/finalize001/runtime.json'
    assert original_runtime.exists() or (original_identity and 'finalize_stablellve_task_r65.py' in original_identity['commandline'])
    names = ['finalize_cvvdp_r66.py', 'evaluate_cvvdp_stablellve_r66.py', 'evaluate_cvvdp_references_r66.py',
             'cvvdp_adapter_r66.py', 'summarize_cvvdp_r66.py', 'render_stablellve_task_r65.py',
             'render_dense_history_r64.py', 'prepare_cache_r5.py', 'half_motion_r4.py',
             'aligned_raw_history_r11.py', 'aggregation_controls_r20.py']
    frozen = {str((BASE/'scripts'/name).relative_to(ROOT)): digest(BASE/'scripts'/name) for name in names}
    for name in ['exploration-r66-perceptual-metric-protocol.md', 'exploration-r65-final-visual-protocol.md']:
        path = BASE/'docs'/name; frozen[str(path.relative_to(ROOT))] = digest(path)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), source_sha256=frozen,
          reference_guard_pid=reference_launch['guard_pid'], reference_guard_identity=ref_identity,
          original_R65_coordinator_pid=original_launch['pid'], original_R65_identity=original_identity,
          no_wall_cutoff=True, no_scientific_retry=True, no_GPU_lock_while_waiting=True,
          scope='Only registered reference/complete summaries, registered final R65 cached contacts and80 perceptual rows.'))
    start, complete, phase_B_code, render_code = time.perf_counter(), False, None, None
    render, render_stream = None, None
    try:
        wait_training(process/'reference-guard002.json', reference_launch['guard_pid'], ref_identity)
        record = read_stable(BASE/'results/exploration-r66-reference-attempt002/runtime.json')
        assert record['completed_normally'] and record['quality_rows'] == 140 and record['comparison_frames'] == 6272
        assert all(digest(ROOT/path) == sha for path, sha in frozen.items())
        with (out/'summarize-references.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str((BASE/'scripts/summarize_cvvdp_r66.py').relative_to(ROOT)),
                            '--phase', 'references'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        write(out/'state.json', dict(at=datetime.now().astimezone().isoformat(),
              stage='Reference140 complete; waiting original R65 normal completion with no GPU held.'))
        wait_normal_runtime(original_runtime, original_launch['pid'], original_identity)
        assert all(digest(ROOT/path) == sha for path, sha in frozen.items())
        command = ['./scripts/project-python', str((BASE/'scripts/render_stablellve_task_r65.py').relative_to(ROOT))]
        render_stream = (out/'render-R65.log').open('x')
        render = subprocess.Popen(command, cwd=ROOT, stdout=render_stream, stderr=subprocess.STDOUT)
        write(out/'render-R65-launch.json', dict(at=datetime.now().astimezone().isoformat(), pid=render.pid, command=command))
        # Resource-only waiting never holds a project GPU lock. A scientific failure is never retried.
        attempt = 0
        while True:
            chosen = None
            for gpu in [4, 7]:
                samples = [sample(gpu)]; time.sleep(1); samples.append(sample(gpu))
                if all(not r['compute_processes'] and r['memory_mib'] <= 64 and r['utilization_percent'] <= 5 for r in samples):
                    chosen = gpu; break
            if chosen is None:
                time.sleep(30); continue
            attempt += 1
            guard_path = out/f'phaseB-guard{attempt:03d}.json'
            command = ['./scripts/project-python', str((BASE/'scripts/gpu_guard.py').relative_to(ROOT)),
                       '--gpu', str(chosen), '--record', str(guard_path.relative_to(ROOT)), '--', './scripts/project-python',
                       str((BASE/'scripts/evaluate_cvvdp_stablellve_r66.py').relative_to(ROOT))]
            with (out/f'phaseB{attempt:03d}.log').open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                write(out/f'phaseB-launch{attempt:03d}.json', dict(at=datetime.now().astimezone().isoformat(),
                      gpu=chosen, guard_pid=child.pid, command=command, preguard_samples=samples))
                code = child.wait()
            admitted = read_stable(guard_path)
            if not admitted['allowed']:
                assert not (BASE/'results/exploration-r66-stablellve-attempt001').exists()
                time.sleep(30); continue
            phase_B_code = code; assert code == 0, 'Admitted scoring failed; no automatic scientific retry.'
            break
        assert all(digest(ROOT/path) == sha for path, sha in frozen.items())
        with (out/'summarize-complete.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str((BASE/'scripts/summarize_cvvdp_r66.py').relative_to(ROOT)),
                            '--phase', 'complete'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        render_code = render.wait(); assert render_code == 0
        assert all(digest(ROOT/path) == sha for path, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        if render_stream: render_stream.close()
        write(out/'runtime.json', dict(completed_normally=complete, phaseB_returncode=phase_B_code,
              R65_render_returncode=render_code, seconds=time.perf_counter()-start,
              actual_visual_review_still_required=True, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
