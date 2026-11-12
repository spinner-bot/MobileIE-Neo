"""Follow original guards, finish all final evaluations and fixed evidence once."""
from datetime import datetime
import json
import subprocess
import time
import traceback

from check_spatial_l1_r69 import write
from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from finalize_task_fastdvd_r47 import wait_training
from gpu_guard import sample


def main():
    process = BASE/'process/exploration-r69'; out = process/'finalize001'; out.mkdir(exist_ok=False)
    names = ['spatial_l1_r69.py', 'check_spatial_l1_r69.py', 'train_spatial_l1_r69.py',
             'evaluate_spatial_l1_r69.py', 'summarize_spatial_l1_r69.py', 'evaluate_cvvdp_spatial_l1_r69.py',
             'render_spatial_l1_r69.py', 'finalize_spatial_l1_r69.py', 'mixed_anchor_r40.py',
             'training_form_r41.py', 'dynamic_training_r34.py', 'native_training_r14.py', 'gpu_guard.py',
             'prepare_cache_r5.py', 'exploration_r2_data.py', 'video_metrics.py', 'train_exploration_r2.py',
             'drv_references_r59.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py',
             'render_stablellve_task_r65.py', 'render_dense_history_r64.py', 'aggregation_controls_r20.py',
             'cvvdp_adapter_r66.py', 'evaluate_cvvdp_references_r66.py', 'finalize_task_fastdvd_r47.py']
    paths = [BASE/'scripts'/name for name in names]
    paths += [BASE/'docs/exploration-r69-matched-spatial-L1-protocol.md',
              ROOT/'MobileIE/model/lle.py', ROOT/'MobileIE/model/utils.py']
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    dependencies = []
    for seed in [0, 1]:
        launch = read_stable(process/f'seed{seed}-launch.json'); guard = read_stable(process/f'seed{seed}-guard.json')
        identity = process_identity(launch['guard_pid'])
        assert guard['allowed'] and ('returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline']))
        dependencies.append(dict(seed=seed, pid=launch['guard_pid'], identity=identity))
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), dependencies=dependencies,
          source_sha256=frozen, no_wall_cutoff=True, no_automatic_scientific_retry=True,
          scope='Only final8000; all paired CPU outputs, all contacts and fixed metric scores. No independent final data.'))
    start, complete, evaluated, metric_code, render_code = time.perf_counter(), False, {}, None, None
    children, streams, render, render_stream = {}, {}, None, None
    def check_sources(): assert all(digest(ROOT/path) == sha for path, sha in frozen.items())
    def run(script, log, extra=()):
        check_sources()
        with (out/log).open('x') as stream:
            subprocess.run(['./scripts/project-python', str((BASE/'scripts'/script).relative_to(ROOT)), *extra],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        for dep in dependencies:
            seed = dep['seed']; wait_training(process/f'seed{seed}-guard.json', dep['pid'], dep['identity'])
            folder = BASE/f'results/exploration-r69-stage001-seed{seed}'
            runtime = read_stable(folder/'runtime.json'); marker = read_stable(folder/'step8000.json')
            assert runtime['completed_normally'] and runtime['updates'] == 8000
            assert marker['final_quality_eligible'] and marker['final_fold_gate_passed']
            assert marker['step'] == 8000 and digest(ROOT/marker['path']) == marker['sha256']
        check_sources()
        for seed in [0, 1]:
            command = ['./scripts/project-python', str((BASE/'scripts/evaluate_spatial_l1_r69.py').relative_to(ROOT)), '--seed', str(seed)]
            streams[seed] = (out/f'evaluate-seed{seed}.log').open('x')
            children[seed] = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
            write(out/f'evaluate-seed{seed}-launch.json', dict(at=datetime.now().astimezone().isoformat(),
                  pid=children[seed].pid, identity=process_identity(children[seed].pid), command=command))
        write(out/'state.json', dict(at=datetime.now().astimezone().isoformat(), stage='Both normal final8000; final CPU evaluations actually launched.'))
        for seed, child in children.items(): evaluated[seed] = child.wait()
        assert len(evaluated) == 2 and all(code == 0 for code in evaluated.values()), evaluated
        run('summarize_spatial_l1_r69.py', 'summarize-quality.log')
        render_stream = (out/'render.log').open('x')
        command = ['./scripts/project-python', str((BASE/'scripts/render_spatial_l1_r69.py').relative_to(ROOT))]
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
                write(out/'resource-wait.json', dict(at=datetime.now().astimezone().isoformat(), stage='quality complete; metric waiting for genuinely idle same-group GPU'))
                time.sleep(30); continue
            attempt += 1; guard_path = out/f'perceptual-guard{attempt:03d}.json'
            command = ['./scripts/project-python', str((BASE/'scripts/gpu_guard.py').relative_to(ROOT)),
                       '--gpu', str(chosen), '--record', str(guard_path.relative_to(ROOT)), '--', './scripts/project-python',
                       str((BASE/'scripts/evaluate_cvvdp_spatial_l1_r69.py').relative_to(ROOT))]
            with (out/f'perceptual{attempt:03d}.log').open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                write(out/f'perceptual-launch{attempt:03d}.json', dict(at=datetime.now().astimezone().isoformat(),
                      gpu=chosen, guard_pid=child.pid, identity=process_identity(child.pid), command=command, preguard_samples=samples))
                code = child.wait()
            admitted = read_stable(guard_path)
            if not admitted['allowed']:
                assert not (BASE/'results/exploration-r69-perceptual-attempt001').exists()
                time.sleep(30); continue
            metric_code = code; assert code == 0, 'Admitted metric failed; preserve, no automatic scientific retry.'
            break
        render_code = render.wait(); assert render_code == 0
        run('summarize_spatial_l1_r69.py', 'summarize-joint.log', ['--perceptual'])
        check_sources(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        if render is not None and render_code is None: render_code = render.wait()
        for stream in streams.values(): stream.close()
        if render_stream: render_stream.close()
        write(out/'runtime.json', dict(completed_normally=complete, evaluation_returncodes=evaluated,
              perceptual_returncode=metric_code, render_returncode=render_code, seconds=time.perf_counter()-start,
              actual_visual_review_still_required=True, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
