"""One immutable REDS reference chain; resource waits only, no scientific retries."""
import json
import subprocess
import time
import traceback

from gpu_guard import sample
from finalize_preserved_r37 import process_identity
from reds_references_r72 import (BASE, ROOT, PROCESS, check_sources, digest,
    frozen_sources, stamp, write)


def main():
    # Source preparation and actual inspection are prerequisites, not inferred from a launcher file.
    source = BASE / 'process/exploration-r71/fetch001/runtime.json'
    review = BASE / 'report/exploration-r71-reds-sources/actual-visual-review.json'
    terminal = json.loads(source.read_text()); inspected = json.loads(review.read_text())
    assert terminal['completed_normally'] and terminal['completed_members'] == 256
    assert inspected['actual_reviewed'] and inspected['contacts'] == 4
    out = PROCESS / 'finalize001'; out.mkdir(parents=True, exist_ok=False)
    frozen = frozen_sources()
    write(out / 'config.json', dict(started=stamp(), source_sha256=frozen,
          R71_terminal_sha256=digest(source), actual_review_sha256=digest(review),
          no_wall_cutoff=True, no_automatic_scientific_retry=True,
          scope='Fixed reference transfer only; sequential GPU seeds, CPU metrics after each GPU release. No training or independent final test.'))
    started = time.perf_counter(); complete = False; cpu_children = []; cpu_streams = []; render = None; render_stream = None

    def command(script, extra=()):
        return ['./scripts/project-python', str((BASE / 'scripts' / script).relative_to(ROOT)), *extra]

    def run(script, log, extra=()):
        check_sources(frozen)
        with (out / log).open('x') as stream:
            subprocess.run(command(script, extra), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def state(value):
        write(out / 'state.json', dict(at=stamp(), stage=value))

    def guarded(script, tag, extra=()):
        attempt = 0
        while True:
            check_sources(frozen); chosen = None
            for gpu in [4, 5, 6, 7]:
                samples = [sample(gpu)]; time.sleep(1); samples.append(sample(gpu))
                if all(not r['compute_processes'] and r['memory_mib'] <= 64 and r['utilization_percent'] <= 5 for r in samples):
                    chosen = gpu; break
            if chosen is None:
                write(out / f'{tag}-resource-wait.json', dict(at=stamp(), stage='Waiting for two fresh idle same-group samples'))
                time.sleep(30); continue
            attempt += 1
            record = out / f'{tag}-guard{attempt:03d}.json'
            cmd = command('gpu_guard.py', ['--gpu', str(chosen), '--record', str(record.relative_to(ROOT)),
                          '--', *command(script, [*extra, '--admission-attempt', str(attempt)])])
            log = out / f'{tag}-attempt{attempt:03d}.log'
            with log.open('x') as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                time.sleep(.2)
                write(out / f'{tag}-launch{attempt:03d}.json', dict(at=stamp(), guard_pid=child.pid,
                      identity=process_identity(child.pid), gpu=chosen, command=cmd, preguard_samples=samples))
                code = child.wait()
            if not record.exists():
                # A different project job can acquire the lock between samples and the guard.
                text = log.read_text()
                if code != 0 and 'BlockingIOError' in text and 'Resource temporarily unavailable' in text:
                    write(out / f'{tag}-lock-race{attempt:03d}.json', dict(at=stamp(),
                          resource_only=True, command_not_admitted=True, returncode=code))
                    time.sleep(5); continue
                raise RuntimeError(f'{tag} guard failed before audit, returncode={code}')
            admission = json.loads(record.read_text())
            if not admission['allowed']:
                time.sleep(5); continue
            if code == 75:
                internal_tag = 'metric' if tag == 'perceptual' else tag
                internal = PROCESS / f'{internal_tag}-internal-admission{attempt:03d}.json'
                assert internal.exists() and not json.loads(internal.read_text())['allowed']
                time.sleep(5); continue
            assert code == 0, f'{tag} admitted command failed; preserved, no automatic retry'
            return

    try:
        state('Full chain frozen; CPU metric interface and fixed input preparation next')
        run('check_reds_metric_r72.py', 'metric-cpu-gate.log')
        run('prepare_reds_r72.py', 'prepare-data.log')
        for seed in [0, 1]:
            state(f'Input preparation normal; frozen reference GPU seed{seed} pending fresh admission')
            guarded('infer_reds_r72.py', f'seed{seed}', ['--seed', str(seed)])
            neural = json.loads((BASE / f'results/exploration-r72-neural-seed{seed}/runtime.json').read_text())
            assert neural['completed_normally'] and (neural['CNN_sequences'], neural['CNN_centers']) == (40, 2560)
            stream = (out / f'quality-seed{seed}.log').open('x'); cpu_streams.append(stream)
            cmd = command('evaluate_reds_r72.py', ['--seed', str(seed)])
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT); cpu_children.append(child)
            time.sleep(.2)
            write(out / f'quality-seed{seed}-launch.json', dict(at=stamp(), pid=child.pid,
                  identity=process_identity(child.pid), command=cmd))
            state(f'GPU seed{seed} normal and released; original CPU quality child actually launched')
        codes = [child.wait() for child in cpu_children]
        assert codes == [0, 0], f'CPU quality failed {codes}; no retry'
        run('summarize_reds_r72.py', 'summarize-quality.log')
        render_stream = (out / 'render.log').open('x')
        cmd = command('render_reds_references_r72.py')
        render = subprocess.Popen(cmd, cwd=ROOT, stdout=render_stream, stderr=subprocess.STDOUT)
        write(out / 'render-launch.json', dict(at=stamp(), pid=render.pid, command=cmd))
        state('All quality normal; fixed contacts and separately admitted24fps perception next')
        guarded('evaluate_cvvdp_reds_r72.py', 'perceptual')
        run('summarize_reds_r72.py', 'summarize-joint.log', ['--include-perceptual'])
        assert render.wait() == 0
        check_sources(frozen); complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        for stream in cpu_streams:
            stream.close()
        if render_stream:
            render_stream.close()
        write(out / 'runtime.json', dict(completed_normally=complete, ended=stamp(),
              seconds=time.perf_counter() - started, actual_visual_review_still_required=True,
              scope='Coordinator terminal only; no new method, handset or independent final claim.'))


if __name__ == '__main__':
    main()
