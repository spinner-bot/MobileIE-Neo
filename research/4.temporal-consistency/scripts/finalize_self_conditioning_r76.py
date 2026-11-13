"""Wait for complete primary results, then run one fixed supplemental control without retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from causal_reference_io_r75 import ROOT,BASE,write,digest,stamp
from self_conditioning_evaluation_r76 import require_primary_pipeline

PROCESS=BASE/'process/exploration-r76'


def handle_state(pid,ticks):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split()
    except (FileNotFoundError,ProcessLookupError):
        return 'missing'
    except OSError:
        return 'unknown'
    return 'live' if fields[21]==str(ticks) and fields[2]!='Z' else 'terminal'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--primary-pid',type=int,required=True)
    parser.add_argument('--metric-attempt',type=int,default=1)
    args=parser.parse_args()
    lock=(ROOT/'.cache/r76-finalize.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    out=PROCESS/f'finalize{args.attempt:03d}';out.mkdir(exist_ok=False)
    freeze=PROCESS/'evaluation-freeze001.json'
    frozen=json.loads(freeze.read_text())
    ready=False
    try:
        require_primary_pipeline();ready=True
    except (FileNotFoundError,json.JSONDecodeError):
        pass
    ticks=None
    if not ready:
        fields=Path(f'/proc/{args.primary_pid}/stat').read_text().split()
        argv=[x.decode() for x in Path(f'/proc/{args.primary_pid}/cmdline').read_bytes().split(b'\0') if x]
        assert fields[2]!='Z' and any(x.endswith('/finalize_causal_reference_r75.py') for x in argv)
        ticks=fields[21]
    write(out/'config.json',dict(at=stamp(),pid=os.getpid(),primary_pid=args.primary_pid,
        primary_start_ticks=ticks,primary_already_complete=ready,evaluation_freeze_sha256=digest(freeze),
        scope='Only R76 supplemental artifacts; no modification/restart of R75 and no scientific retries.'))
    started,complete,reason=time.perf_counter(),False,'error'

    def verify():
        assert all(digest(ROOT/path)==value for path,value in frozen['source_sha256'].items())

    def terminal(path,fields):
        if not path.exists():
            return False
        value=json.loads(path.read_text())
        if not value['completed_normally'] or any(value.get(k)!=v for k,v in fields.items()):
            raise RuntimeError(f'Existing incomplete stage retained without retry: {path}')
        return True

    def run(name,command,cpu=True):
        verify()
        env=os.environ.copy()
        if cpu:
            env['CUDA_VISIBLE_DEVICES']=''
        with (out/f'{name}.log').open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write(out/f'{name}-child.json',dict(at=stamp(),pid=child.pid,command=command))
        code=child.wait()
        write(out/f'{name}-terminal.json',dict(ended=stamp(),pid=child.pid,returncode=code))
        if code:
            raise RuntimeError(f'{name} exited{code}; no automatic replay or parameter adjustment.')

    python=str(ROOT/'scripts/project-python');scripts=BASE/'scripts'
    try:
        while not ready:
            verify()
            try:
                require_primary_pipeline();ready=True;break
            except (FileNotFoundError,json.JSONDecodeError):
                pass
            state=handle_state(args.primary_pid,ticks)
            if state in ('missing','terminal'):
                # Recheck artifacts in case the primary exited between the two observations.
                try:
                    require_primary_pipeline();ready=True;break
                except (FileNotFoundError,json.JSONDecodeError):
                    reason='primary_terminal_without_complete_artifacts'
                    raise RuntimeError('Original primary handle ended before prerequisites; inspect it, do not restart from this observation.')
            write(out/'waiting.json',dict(at=stamp(),primary_pid=args.primary_pid,start_ticks=ticks,
                primary_handle_state=state,verified_live=state=='live',primary_complete=False,
                scope='No R76 inference or GPU yet. Unknown observations wait on the same handle.'))
            time.sleep(30)
        verify();children=[]
        for seed in (0,1):
            done=BASE/f'results/exploration-r76-eval-seed{seed}/runtime.json'
            if terminal(done,dict(quality_rows=10,frame_rows=448,tile_CNN_calls=9728)):
                continue
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            command=[python,str(scripts/'evaluate_current_refinement_r76.py'),'--seed',str(seed)]
            name=f'quality-seed{seed}'
            with (out/f'{name}.log').open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            children.append((name,child))
            write(out/f'{name}-child.json',dict(at=stamp(),pid=child.pid,command=command))
        failures=[]
        for name,child in children:
            code=child.wait()
            write(out/f'{name}-terminal.json',dict(ended=stamp(),pid=child.pid,returncode=code))
            if code:
                failures.append((name,code))
        if failures:
            raise RuntimeError(f'Incomplete supplemental quality retained: {failures}')
        quality=BASE/'report/exploration-r76-quality/audit.json'
        if not quality.exists():
            run('quality-summary',[python,str(scripts/'summarize_self_conditioning_r76.py')])
        assert json.loads(quality.read_text())['completed']
        visuals=BASE/'report/exploration-r76-contacts/runtime.json'
        if not terminal(visuals,dict(contacts=20)):
            run('fixed-contacts',[python,str(scripts/'render_self_conditioning_r76.py')])
        metric=BASE/f'results/exploration-r76-perceptual{args.metric_attempt:03d}/runtime.json'
        if not terminal(metric,dict(quality_rows=20,comparison_frames=896)):
            run('perceptual',[python,str(ROOT/'scripts/prebind_shared_gpu.py'),'--gpu','0','--record',
                str(out/'metric-parent.json'),'--',python,str(scripts/'evaluate_cvvdp_self_conditioning_r76.py'),
                '--attempt',str(args.metric_attempt)],cpu=False)
        joint=BASE/'report/exploration-r76-joint/audit.json'
        if not joint.exists():
            run('joint-summary',[python,str(scripts/'summarize_self_conditioning_r76.py'),
                '--perceptual','--metric-attempt',str(args.metric_attempt)])
        assert json.loads(joint.read_text())['completed']
        verify();complete=True;reason='complete'
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,stop_reason=reason,
            seconds=time.perf_counter()-started,primary_results_rewritten=False,new_training=0,
            actual_visual_review_completed=False,
            scope='Supplemental numerical/render closure; actual image review and scientific interpretation remain manual.'))
        globals()['_held_pipeline_lock']=lock


if __name__=='__main__':
    main()
