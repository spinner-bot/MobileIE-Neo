"""Observe identified fixed-budget runs and close full CPU comparisons once normal."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    process=BASE/'process/exploration-r48'
    folder=process/'finalize001';folder.mkdir(exist_ok=False)
    names=['finalize_joint_recursion_r48.py','train_joint_recursion_r48.py','joint_recursion_r48.py',
           'evaluate_joint_recursion_r48.py','summarize_joint_recursion_r48.py','response_cue_r45.py',
           'response_sequence_r45.py','budget_causal_r33.py','half_motion_r4.py','aligned_raw_history_r11.py',
           'aggregation_controls_r20.py','exploration_r2_data.py','video_metrics.py','mixed_anchor_r40.py',
           'dynamic_training_r34.py','training_form_r41.py','gpu_guard.py']
    frozen={n:digest(BASE/'scripts'/n) for n in names}
    dependencies=[]
    for seed in [0,1]:
        launch=read_stable(process/f'seed{seed}-launch.json')
        guard=read_stable(process/f'seed{seed}-guard.json');assert guard['allowed']
        identity=process_identity(launch['guard_pid'])
        assert 'returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline'])
        dependencies.append({'seed':seed,'pid':launch['guard_pid'],'identity':identity})
    write(folder/'config.json',{'at':datetime.now().astimezone().isoformat(),'dependencies':dependencies,
          'source_sha256':frozen,'scope':'Both seeds full8000 new model updates each and final fold before120 CPU rows; no retry or wall cutoff.'})
    started,complete,evaluated=time.perf_counter(),False,{}
    children,streams={},{}
    try:
        pending=list(dependencies)
        while pending:
            for dep in list(pending):
                seed=dep['seed'];guard=read_stable(process/f'seed{seed}-guard.json')
                if 'returncode' not in guard:
                    if process_identity(dep['pid'])!=dep['identity']:
                        time.sleep(1);guard=read_stable(process/f'seed{seed}-guard.json')
                        assert 'returncode' in guard,'Identified process vanished without terminal record.'
                    else:continue
                assert guard['returncode']==0,(seed,guard['returncode'])
                runtime=read_stable(BASE/f'results/exploration-r48-stage001-seed{seed}/runtime.json')
                assert runtime['completed_normally'] and runtime['model_updates']=={'plain':4000,'joint':4000}
                assert runtime['total_new_updates']==8000
                assert set(runtime['final_checkpoints'])=={'plain','joint'}
                assert all(v['new_step']==4000 and v['final_fold_passed'] for v in runtime['final_checkpoints'].values())
                pending.remove(dep)
            if pending:time.sleep(5)
        assert all(digest(BASE/'scripts'/n)==value for n,value in frozen.items())
        for seed in [0,1]:
            command=['./scripts/project-python',str(BASE/'scripts/evaluate_joint_recursion_r48.py'),'--seed',str(seed)]
            streams[seed]=(folder/f'evaluate-seed{seed}.log').open('x')
            child=subprocess.Popen(command,cwd=ROOT,stdout=streams[seed],stderr=subprocess.STDOUT)
            children[seed]=child
            write(folder/f'evaluate-seed{seed}-launch.json',{'at':datetime.now().astimezone().isoformat(),
                  'pid':child.pid,'command':command})
        for seed,child in children.items():evaluated[seed]=child.wait()
        assert len(evaluated)==2 and all(v==0 for v in evaluated.values())
        assert all(digest(BASE/'scripts'/n)==value for n,value in frozen.items())
        with (folder/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python',str(BASE/'scripts/summarize_joint_recursion_r48.py')],
                           cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        complete=True
    except BaseException:
        (folder/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        for stream in streams.values():stream.close()
        write(folder/'runtime.json',{'completed_normally':complete,'evaluation_returncodes':evaluated,
              'seconds':time.perf_counter()-started,'ended':datetime.now().astimezone().isoformat(),
              'scope':'Completeness only; gates A/B, cost and novelty remain distinct.'})


if __name__=='__main__':main()
