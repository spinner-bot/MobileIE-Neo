"""Wait for the original complete training chain, evaluate once, render once.

This CPU-only continuation never starts, stops or restarts a training worker.
"""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
from pathlib import Path
import subprocess
import time
import traceback

from simple_io import BASE, ROOT, append, digest, identity, owned_cache, read, stamp, verify_training, write


def main():
    frozen = verify_training()
    target = read(BASE/'process/simple-execution-launch001.json')['parent']
    out, logs = BASE/'process/simple-after001', owned_cache('after001')
    out.mkdir(parents=True,exist_ok=False)
    logs.mkdir(parents=True,exist_ok=False)
    began,complete = time.perf_counter(),False
    source = {str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),BASE/'render_simple.py')}
    state = dict(at=stamp(),process=identity(),stage='waiting_original_training',children={})
    write(out/'config.json',dict(at=stamp(),process=identity(),original_training_parent=target,
          training_source_freeze=frozen,analysis_sources=source,GPU_used=False,
          modifies_or_restarts_training=False,algorithm_wall_cutoff=False,
          opens_independent_final=False,automatically_starts_cost_or_perceptual_experiment=False))

    def stage(command,name):
        env = os.environ.copy();env['CUDA_VISIBLE_DEVICES'] = ''
        with (logs/(name+'.log')).open('x') as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                     stdout=stream,stderr=subprocess.STDOUT)
        state.update(at=stamp(),stage=name)
        state['children'][name] = identity(child.pid)
        write(out/'state.json',state)
        append(out/'events.jsonl',dict(at=stamp(),action='start',stage=name,child=state['children'][name]))
        code = child.wait()
        state['children'][name]['exit_code'] = code
        write(out/'state.json',state)
        append(out/'events.jsonl',dict(at=stamp(),action='terminal',stage=name,code=code))
        assert code == 0,f'{name} failed with{code}; retained, no automatic inference retry'

    write(out/'state.json',state)
    try:
        runtime = BASE/'process/simple-chain001/runtime.json'
        while True:
            if runtime.exists():
                result = read(runtime)
                assert result['completed_normally'],'Original chain is terminal without full success; audit numeric workers separately'
                assert result['state']['stage'] == 'training_completed_pending_evaluation'
                break
            try:
                current = identity(target['pid'])
            except FileNotFoundError:
                if runtime.exists():
                    continue
                raise RuntimeError('Original orchestration handle missing without runtime; training workers may still run. Needs audit, no restart.')
            except OSError as error:
                append(out/'events.jsonl',dict(at=stamp(),action='observation_retry',reason=str(error),target=target))
                time.sleep(30)
                continue
            assert current['start_ticks'] == target['start_ticks'],'Original parent PID reused; no inference launch'
            assert current['state'] != 'Z','Original parent terminal without successful runtime; inspect children, do not restart'
            time.sleep(30)
        verify_training()
        assert all(digest(ROOT/p)==h for p,h in source.items())
        assert not (BASE/'results/simple-eval001').exists(),'Existing evaluation must be inspected, not duplicated'
        assert not (BASE/'report/simple-analysis001').exists(),'Existing analysis must be inspected'
        python = str(ROOT/'scripts/project-python')
        stage([python,str(BASE/'evaluate_simple.py')],'full-evaluation')
        assert read(BASE/'results/simple-eval001/runtime.json')['completed_normally']
        stage([python,str(BASE/'render_simple.py')],'complete-plots')
        assert not read(BASE/'report/simple-analysis001/render-record.json')['actual_viewed']
        state.update(at=stamp(),stage='results_ready_pending_actual_review_and_decision')
        write(out/'state.json',state)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json',dict(at=stamp(),process=identity(),completed_normally=complete,
              seconds=time.perf_counter()-began,state=state,GPU_used=False,
              human_or_agent_visual_review_done=False,candidate_or_paper_success_claimed=False,goal_complete=False))


if __name__ == '__main__':
    main()
