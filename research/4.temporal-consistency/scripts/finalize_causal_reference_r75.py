"""Wait for verified original live seed handles, then run the registered complete final pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from causal_reference_io_r75 import ROOT,BASE,PROCESS,write,digest,stamp,seed_root
from causal_reference_evaluation_r75 import final_markers


class StageFailure(RuntimeError):
    pass


def pid_state(pid,start_ticks):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        if stat[21]!=str(start_ticks) or stat[2]=='Z':
            return False
        return True
    except (FileNotFoundError,ProcessLookupError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--train-pids',type=int,nargs=2,required=True)
    parser.add_argument('--metric-attempt',type=int,default=1)
    args=parser.parse_args()
    out=PROCESS/f'finalize{args.attempt:03d}';out.mkdir(exist_ok=False)
    frozen=json.loads((PROCESS/'evaluation-freeze001.json').read_text())
    handles=[]
    for seed,pid in enumerate(args.train_pids):
        terminal=seed_root(seed)/'final.json'
        if terminal.exists():
            handles.append(dict(seed=seed,pid=pid,start_ticks=None,already_final=True))
        else:
            stat=Path(f'/proc/{pid}/stat').read_text().split()
            command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            argv=[x.decode() for x in command if x]
            assert stat[2]!='Z' and any(x.endswith('/train_causal_reference_r75.py') for x in argv)
            assert argv[argv.index('--seed')+1]==str(seed)
            handles.append(dict(seed=seed,pid=pid,start_ticks=stat[21],already_final=False))
    write(out/'config.json',dict(at=stamp(),pid=os.getpid(),training_handles=handles,
        evaluation_freeze_sha256=digest(PROCESS/'evaluation-freeze001.json'),
        scope='No scientific retries, no wall-clock cutoffs; all final DEV stages, visual inspection remains manual.'))
    started,complete,reason=time.perf_counter(),False,'error'

    def verify():
        assert all(digest(ROOT/path)==sha for path,sha in frozen['source_sha256'].items())

    def run(name,command,cpu=True):
        verify()
        env=os.environ.copy()
        if cpu:
            env['CUDA_VISIBLE_DEVICES']=''
        with (out/f'{name}.log').open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        write(out/f'{name}-child.json',dict(at=stamp(),pid=child.pid,command=command,CPU_mask_empty=cpu))
        code=child.wait()
        write(out/f'{name}-terminal.json',dict(ended=stamp(),pid=child.pid,returncode=code))
        if code!=0:
            raise StageFailure(f'{name} exited {code}; inspect original logs, do not auto-retry.')

    def terminal(path,fields):
        if not path.exists():
            return False
        value=json.loads(path.read_text())
        if not value['completed_normally'] or any(value.get(k)!=v for k,v in fields.items()):
            raise StageFailure(f'Existing incomplete/unexpected stage retained: {path}')
        return True

    python=str(ROOT/'scripts/project-python')
    scripts=BASE/'scripts'
    try:
        while True:
            verify()
            waiting=[]
            for handle in handles:
                if (seed_root(handle['seed'])/'final.json').exists():
                    continue
                if not pid_state(handle['pid'],handle['start_ticks']):
                    reason='training_handle_terminal_without_final'
                    raise StageFailure('Training handle is missing/terminal without final. Inspect runtime; no restart inferred from a timeout.')
                waiting.append(handle)
            write(out/'waiting.json',dict(at=stamp(),verified_live_training_handles=waiting,
                final_count=2-len(waiting),scope='Live PID/start-time observation, not a completed model.'))
            if not waiting:
                break
            time.sleep(30)
        final_markers()
        children=[]
        verify()
        for seed in (0,1):
            done=BASE/f'results/exploration-r75-eval-seed{seed}/runtime.json'
            if terminal(done,dict(quality_rows=20,frame_rows=896,tile_CNN_calls=4864)):
                continue
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            name=f'quality-seed{seed}'
            command=[python,str(scripts/'evaluate_causal_reference_r75.py'),'--seed',str(seed)]
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
            raise StageFailure(f'Quality stages failed: {failures}; retain all outputs.')
        summary=BASE/'report/exploration-r75-quality/audit.json'
        if not summary.exists():
            run('quality-summary',[python,str(scripts/'summarize_causal_reference_r75.py')])
        assert json.loads(summary.read_text())['completed']
        rendered=BASE/'report/exploration-r75-contacts/runtime.json'
        if not terminal(rendered,dict(contacts=20)):
            run('fixed-contacts',[python,str(scripts/'render_causal_reference_r75.py')])
        metric=BASE/f'results/exploration-r75-perceptual{args.metric_attempt:03d}/runtime.json'
        if not terminal(metric,dict(quality_rows=40,comparison_frames=1792)):
            command=[python,str(ROOT/'scripts/prebind_shared_gpu.py'),'--gpu','0','--record',
                     str(out/'metric-parent.json'),'--',python,str(scripts/'evaluate_cvvdp_causal_r75.py'),
                     '--attempt',str(args.metric_attempt)]
            run('perceptual',command,cpu=False)
        joint=BASE/'report/exploration-r75-joint/audit.json'
        if not joint.exists():
            run('joint-summary',[python,str(scripts/'summarize_causal_reference_r75.py'),
                                '--perceptual','--metric-attempt',str(args.metric_attempt)])
        assert json.loads(joint.read_text())['completed']
        verify();complete=True;reason='complete'
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,stop_reason=reason,
            seconds=time.perf_counter()-started,actual_visual_review_completed=False,
            training_restarted=False,scope='Numerical/renderer pipeline only; manual20-contact review and scientific decision still required.'))


if __name__=='__main__':
    main()
