"""Wait for admitted seed1, queue untouched seed0, then finish fixed CPU evaluation."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from gpu_guard import sample


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def stamp():
    return datetime.now().astimezone().isoformat()


def wait_training(guard_path, pid, identity):
    while True:
        record=read_stable(guard_path)
        assert record['allowed'], 'New admission denied; preserve this attempt, no automatic retry.'
        if 'returncode' in record:
            assert record['returncode']==0, record['returncode']
            return
        current=process_identity(pid)
        if current!=identity:
            time.sleep(1)
            record=read_stable(guard_path)
            assert record.get('returncode')==0, 'Guard disappeared without a normal terminal record.'
            return
        time.sleep(5)


def check_training(seed):
    folder=BASE/f'results/exploration-r47-stage001-seed{seed}'
    runtime=read_stable(folder/'runtime.json')
    marker=read_stable(folder/'fastdvd-step8000.json')
    assert runtime['completed_normally'] and runtime['steps']==8000
    assert marker['step']==8000 and marker['final_quality_eligible']


def main():
    process=BASE/'process/exploration-r47'
    folder=process/'finalize001'
    folder.mkdir(exist_ok=False)
    launch=read_stable(process/'seed1-launch.json')
    guard=read_stable(process/'seed1-guard.json')
    assert guard['allowed']
    identity=process_identity(launch['guard_pid'])
    assert identity is not None and 'gpu_guard.py' in identity['commandline']
    rejected=read_stable(process/'seed0-guard.json')
    assert not rejected['allowed'] and not (BASE/'results/exploration-r47-stage001-seed0').exists()
    names=['task_fastdvd_r47.py','train_task_fastdvd_r47.py','evaluate_task_fastdvd_r47.py',
           'summarize_task_fastdvd_r47.py','finalize_task_fastdvd_r47.py','mixed_anchor_r40.py',
           'dynamic_training_r34.py','native_training_r14.py','fastdvdnet_adapter_r12.py',
           'training_form_r41.py','gpu_guard.py','exploration_r2_data.py','video_metrics.py','prepare_cache_r5.py']
    frozen={n:digest(BASE/'scripts'/n) for n in names}
    write(folder/'config.json',{'started':stamp(),'seed1_guard_pid':launch['guard_pid'],
        'seed1_identity':identity,'source_sha256':frozen,
        'seed0_first_attempt':'Denied before training, zero updates; preserved.',
        'plan':'Complete seed1, CPU eval1; new idle GPU7 admission for seed0 attempt002; CPU eval0; complete summary.',
        'no_wall_cutoff':True,'no_scientific_retry':True})
    started,complete,evaluated=time.perf_counter(),False,{}
    streams,children={},{}

    def check_sources():
        assert all(digest(BASE/'scripts'/n)==value for n,value in frozen.items())

    def evaluate(seed):
        check_training(seed);check_sources()
        command=['./scripts/project-python',str(BASE/'scripts/evaluate_task_fastdvd_r47.py'),'--seed',str(seed)]
        streams[seed]=(folder/f'evaluate-seed{seed}.log').open('x')
        child=subprocess.Popen(command,cwd=ROOT,stdout=streams[seed],stderr=subprocess.STDOUT)
        children[seed]=child
        write(folder/f'evaluate-seed{seed}-launch.json',{'started':stamp(),'pid':child.pid,'command':command})

    try:
        wait_training(process/'seed1-guard.json',launch['guard_pid'],identity)
        evaluate(1)
        write(folder/'state.json',{'at':stamp(),'stage':'seed1 normal8000 / CPUeval1; seed0 waiting fresh GPU7 idle'})
        while True:
            samples=[sample(7)]
            time.sleep(1);samples.append(sample(7))
            allowed=all(not r['compute_processes'] and r['memory_mib']<=64
                        and r['utilization_percent']<=5 for r in samples)
            with (folder/'seed0-resource-probes.jsonl').open('a') as stream:
                stream.write(json.dumps({'at':stamp(),'samples':samples,'allowed':allowed})+'\n')
            if allowed:break
            time.sleep(30)
        check_sources()
        command=['./scripts/project-python',str(BASE/'scripts/gpu_guard.py'),'--gpu','7',
                 '--record',str(process/'seed0-attempt002-guard.json'),'--','./scripts/project-python',
                 str(BASE/'scripts/train_task_fastdvd_r47.py'),'--seed','0']
        with (process/'seed0-attempt002-stdout.log').open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write(process/'seed0-attempt002-launch.json',{'started':stamp(),'guard_pid':child.pid,
                  'command':command,'scope':'Previously unstarted seed0; no repeated scientific updates.'})
            status=child.wait()
        assert status==0, 'Seed0 new guard/training failed; no automatic retry.'
        check_training(0)
        evaluate(0)
        write(folder/'state.json',{'at':stamp(),'stage':'both normal8000 / two final CPU evaluations active'})
        for seed,child in children.items():
            evaluated[seed]=child.wait()
        assert len(evaluated)==2 and all(value==0 for value in evaluated.values())
        check_sources()
        with (folder/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python',str(BASE/'scripts/summarize_task_fastdvd_r47.py')],
                           cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        complete=True
    except BaseException:
        (folder/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        for stream in streams.values():stream.close()
        write(folder/'runtime.json',{'completed_normally':complete,'evaluation_returncodes':evaluated,
              'seconds':time.perf_counter()-started,'ended':stamp()})


if __name__=='__main__':
    main()
