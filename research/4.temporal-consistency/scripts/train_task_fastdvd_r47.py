"""Fixed-budget task adaptation of the unchanged published FastDVDnet."""
import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool
from task_fastdvd_r47 import initial_model, prediction, provenance, bn_counts
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    gates = [BASE/f'process/exploration-r47/{n}-gate001/gate.json' for n in ['cpu','cuda']]
    assert all(json.loads(p.read_text())['passed'] for p in gates)
    out = BASE/f'results/exploration-r47-stage001-seed{args.seed}'
    out.mkdir(exist_ok=False)
    start, complete, updates, markers = time.perf_counter(), False, 0, {}
    try:
        model = initial_model(args.seed)
        pool = MixedPool()
        first_low, first_gt, first_context = pool.batch(args.seed, 0)
        old_windows = BASE/f'results/exploration-r41-stage002-seed{args.seed}/windows.jsonl'
        with old_windows.open() as stream:
            old_first = json.loads(stream.readline())
        assert first_context == old_first['windows']
        write(out/'window-control.json', {'first_context_matches_R41': True,
              'scope': 'Same deterministic MixedPool function/seed/data_step; not Adam replay.'})
        admission = cuda_headroom()
        write(out/'pre_cuda_guard.json', admission)
        assert admission['allowed'], 'Card changed during full CPU data preparation; no CUDA initialized.'
        model = model.cuda().train()
        torch.cuda.reset_peak_memory_stats()
        optimizer = optimizer_for(model)
        cache = ROOT/f'.cache/checkpoints/exploration-r47/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True, exist_ok=False)
        sources = ['task_fastdvd_r47.py','train_task_fastdvd_r47.py','mixed_anchor_r40.py',
                   'dynamic_training_r34.py','native_training_r14.py','fastdvdnet_adapter_r12.py',
                   'training_form_r41.py','gpu_guard.py']
        write(out/'config.json', {'at': datetime.now().astimezone().isoformat(), 'seed': args.seed,
             'steps': 8000, 'data_steps': [0,7999], 'batch': 4, 'frames': 4, 'crop': 128,
             'initialization': provenance(), 'data': pool.info, 'objective': 'MSE all four centers',
             'BN': 'Flatten B*T=16 centers, shared temp1 called three times per update',
             'optimizer': 'fresh Adam initially and after update4000', 'lr': 1e-4,
             'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': 0, 'clip': 1,
             'FP32': True, 'TF32': False, 'selection': 'final8000 only',
             'protocol_sha256': digest(BASE/'docs/exploration-r47-task-adapted-video-protocol.md'),
             'gates': {str(p.relative_to(ROOT)): digest(p) for p in gates},
             'source_sha256': {n:digest(BASE/'scripts'/n) for n in sources},
             'revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'scope': 'Published task-adapted reference; no novelty, equal FLOPs, full-convergence or phone claim.'})
        counts = bn_counts(model); first_bn, last_bn = next(iter(counts)), next(reversed(counts))
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(8000):
                if step == 4000:
                    optimizer = optimizer_for(model)
                low, gt, context = ((first_low, first_gt, first_context) if step == 0
                                     else pool.batch(args.seed, step))
                windows.write(json.dumps({'step':step+1,'data_step':step,'windows':context})+'\n')
                low, gt = low.cuda(), gt.cuda()
                optimizer.zero_grad(set_to_none=True)
                output = prediction(model, low)
                error = (output-gt).square(); loss = error.mean()
                assert torch.isfinite(loss)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step(); updates += 1
                counts = bn_counts(model)
                row = {'step':updates,'data_step':step,'MSE':float(loss.detach()),
                       'synthetic_MSE':float(error[:2].detach().mean()),
                       'real_MSE':float(error[2:].detach().mean()),'gradient_norm_before_clip':norm,
                       'Adam_reset':step in [0,4000],
                       'BN_first':counts[first_bn],'BN_last':counts[last_bn]}
                trace.write(json.dumps(row,allow_nan=False)+'\n')
                if updates % 100 == 0:
                    trace.flush(); windows.flush()
                    print(json.dumps({**row,'seconds':time.perf_counter()-start}),flush=True)
                if updates % 1000 == 0:
                    path = cache/f'fastdvd-step{updates:04d}.pt'
                    torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                          'seed':args.seed,'step':updates,'data_step':step,
                          'task':'R47 direct lowlight FastDVDnet','optimizer_phase':1 if step<4000 else 2},path)
                    marker = {'path':str(path.relative_to(ROOT)),'sha256':digest(path),'step':updates,
                              'seed':args.seed,'parameters':2479096,'BN_counts':counts,
                              'Adam_phase':1 if step<4000 else 2,'loss_last_window':row,
                              'final_quality_eligible':updates==8000}
                    write(out/f'fastdvd-step{updates:04d}.json',marker)
                    markers[str(updates)] = marker
        assert updates == 8000 and markers['8000']['final_quality_eligible']
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',{'completed_normally':complete,'steps':updates,
             'checkpoints':markers,'seconds':time.perf_counter()-start,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
             'ended':datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
