"""Follow the original CPU reference process, then finish all registered evidence."""
from datetime import datetime
import json
import subprocess
import time
import traceback

from prepare_vbm3d_r70 import BASE, ROOT, digest, write
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from gpu_guard import sample


def main():
    process = BASE/'process/exploration-r70'; out = process/'finalize001'; out.mkdir(exist_ok=False)
    names = ['prepare_vbm3d_r70.py', 'vbm3d_array_r70.cpp', 'classical_block_r70.py', 'check_classical_block_r70.py',
             'evaluate_classical_block_r70.py', 'summarize_classical_block_r70.py', 'render_classical_block_r70.py',
             'evaluate_cvvdp_classical_r70.py', 'finalize_classical_block_r70.py', 'response_cue_r45.py',
             'evaluate_spatial_l1_r69.py', 'drv_references_r59.py', 'dynamic_training_r34.py', 'prepare_cache_r5.py',
             'exploration_r2_data.py', 'video_metrics.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py',
             'render_stablellve_task_r65.py', 'render_dense_history_r64.py', 'aggregation_controls_r20.py',
             'cvvdp_adapter_r66.py', 'evaluate_cvvdp_references_r66.py', 'training_form_r41.py', 'gpu_guard.py']
    paths = [BASE/'scripts'/name for name in names]
    paths += [BASE/'docs/exploration-r70-classical-block-reference-protocol.md', process/'build001/completed.json']
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    launch = read_stable(process/'evaluation-launch001.json'); identity = process_identity(launch['pid'])
    runtime_path = BASE/'results/exploration-r70-classical-attempt001/runtime.json'
    assert identity == launch['identity'] or runtime_path.exists()
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), evaluation_launch=launch,
          source_sha256=frozen, no_wall_cutoff=True, no_automatic_scientific_retry=True,
          scope='Only fixed registered classical reference; all conditions retained, no parameter rescue or independent final pixels.'))
    start, complete, metric_code, render_code = time.perf_counter(), False, None, None
    render, render_stream = None, None
    def check_sources(): assert all(digest(ROOT/path) == sha for path, sha in frozen.items())
    def run(script, log, extra=()):
        check_sources()
        with (out/log).open('x') as stream:
            subprocess.run(['./scripts/project-python', str((BASE/'scripts'/script).relative_to(ROOT)), *extra],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        while not runtime_path.exists():
            current = process_identity(launch['pid'])
            if current != launch['identity']:
                time.sleep(1)
                assert runtime_path.exists(), 'Original CPU process disappeared without terminal evidence.'
                break
            time.sleep(5)
        runtime = read_stable(runtime_path)
        assert runtime['completed_normally'] and runtime['quality_rows'] == 80
        check_sources(); run('summarize_classical_block_r70.py', 'summarize-quality.log')
        render_stream = (out/'render.log').open('x')
        command = ['./scripts/project-python', str((BASE/'scripts/render_classical_block_r70.py').relative_to(ROOT))]
        render = subprocess.Popen(command, cwd=ROOT, stdout=render_stream, stderr=subprocess.STDOUT)
        write(out/'render-launch.json', dict(at=datetime.now().astimezone().isoformat(), pid=render.pid, command=command))
        attempt = 0
        while True:
            check_sources(); chosen = None
            for gpu in [4, 5, 6, 7]:
                samples = [sample(gpu)]; time.sleep(1); samples.append(sample(gpu))
                if all(not row['compute_processes'] and row['memory_mib'] <= 64 and row['utilization_percent'] <= 5 for row in samples):
                    chosen = gpu; break
            if chosen is None:
                write(out/'resource-wait.json', dict(at=datetime.now().astimezone().isoformat(), stage='quality complete, waiting for idle same-group metric GPU'))
                time.sleep(30); continue
            attempt += 1; guard_path = out/f'perceptual-guard{attempt:03d}.json'
            command = ['./scripts/project-python', str((BASE/'scripts/gpu_guard.py').relative_to(ROOT)),
                       '--gpu', str(chosen), '--record', str(guard_path.relative_to(ROOT)), '--', './scripts/project-python',
                       str((BASE/'scripts/evaluate_cvvdp_classical_r70.py').relative_to(ROOT))]
            with (out/f'perceptual{attempt:03d}.log').open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                write(out/f'perceptual-launch{attempt:03d}.json', dict(at=datetime.now().astimezone().isoformat(),
                      gpu=chosen, guard_pid=child.pid, identity=process_identity(child.pid), command=command, preguard_samples=samples))
                code = child.wait()
            admitted = read_stable(guard_path)
            if not admitted['allowed']:
                assert not (BASE/'results/exploration-r70-perceptual-attempt001').exists()
                time.sleep(30); continue
            metric_code = code; assert code == 0, 'Admitted metric failed; no automatic scientific retry.'
            break
        render_code = render.wait(); assert render_code == 0
        run('summarize_classical_block_r70.py', 'summarize-joint.log', ['--perceptual'])
        check_sources(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        if render is not None and render_code is None: render_code = render.wait()
        if render_stream: render_stream.close()
        write(out/'runtime.json', dict(completed_normally=complete, perceptual_returncode=metric_code,
              render_returncode=render_code, seconds=time.perf_counter()-start,
              actual_visual_review_still_required=True, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
