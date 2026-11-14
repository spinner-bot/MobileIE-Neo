"""Fixed-budget task adaptation with unchanged targets and source-valid windows."""
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

from check_source_context_r53 import check_one
from dynamic_training_r34 import BASE, ROOT, digest
from source_context_r53 import SourceContextPool, prediction
from task_fastdvd_r47 import initial_model, provenance, bn_counts
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0,1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    configure_precision(); assert not torch.cuda.is_initialized()
    gate = BASE/'process/exploration-r53/cpu-gate001/gate.json'
    assert json.loads(gate.read_text())['passed']
    out = BASE/f'results/exploration-r53-stage001-seed{args.seed}'; out.mkdir(exist_ok=False)
    start, complete, updates, markers = time.perf_counter(), False, 0, {}
    try:
        pool = SourceContextPool(); first = pool.batch(args.seed, 0)
        admission = cuda_headroom(); write(out/'pre_cuda_guard.json', admission)
        assert admission['allowed']
        model = initial_model(args.seed).cuda()
        cuda_gate = check_one(model, first[0].cuda(), first[1].cuda())
        write(out/'cuda-gate.json', cuda_gate)
        del model
        # Fresh module and all original BN buffers; discarded SGD is not inherited.
        model = initial_model(args.seed).cuda().train()
        optimizer = optimizer_for(model); torch.cuda.reset_peak_memory_stats()
        cache = ROOT/f'.cache/checkpoints/exploration-r53/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True, exist_ok=False)
        names = ['train_source_context_r53.py','check_source_context_r53.py','source_context_r53.py',
                 'task_fastdvd_r47.py','mixed_anchor_r40.py','dynamic_training_r34.py',
                 'fastdvdnet_adapter_r12.py','training_form_r41.py','gpu_guard.py']
        write(out/'config.json', {'at': datetime.now().astimezone().isoformat(), 'seed': args.seed,
              'steps': 8000, 'data_steps': [0,7999], 'batch': 4, 'target_frames': 4, 'input_source_frames': 8,
              'crop': 128, 'initialization': provenance(), 'data': pool.info, 'objective': 'unclipped MSE four original targets',
              'extra_context': 'Two source observations on each side; reflection only at available source boundaries.',
              'BN': 'B*T=16 targets, shared temp1 called three times',
              'optimizer': 'fresh Adam initially and after4000', 'lr': 1e-4, 'betas': [.9,.999],
              'eps': 1e-8, 'weight_decay': 0, 'clip': 1, 'FP32': True, 'TF32': False,
              'selection': 'final8000 only; no sigma/LR/length/seed search',
              'CPU_gate_sha256': digest(gate), 'CUDA_gate': cuda_gate,
              'protocol_sha256': digest(BASE/'docs/exploration-r53-source-context-protocol.md'),
              'source_sha256': {name:digest(BASE/'scripts'/name) for name in names},
              'revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
              'scope': 'Source-context correction of a published reference; extra input information, not novelty or full convergence.'})
        context_count = repeated = 0
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(8000):
                if step == 4000: optimizer = optimizer_for(model)
                source8, gt, identities, context, low = first if step == 0 else pool.batch(args.seed, step)
                assert torch.equal(source8[:,2:6], low)
                # Parent manifests and video+indices identify source files; no repeated long path strings.
                compact_context = [{k:v for k,v in row.items() if k != 'source_inputs'} for row in context]
                windows.write(json.dumps({'step':step+1,'data_step':step,'windows':identities,
                                          'source_context':compact_context})+'\n')
                context_count += 16
                repeated += sum(row['windows_with_duplicate_source_index'] for row in context)
                source8, gt = source8.cuda(), gt.cuda()
                optimizer.zero_grad(set_to_none=True)
                error = (prediction(model, source8)-gt).square(); loss = error.mean()
                assert torch.isfinite(loss); loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step(); updates += 1
                counts = bn_counts(model)
                row = {'step': updates, 'data_step': step, 'MSE': float(loss.detach()),
                       'synthetic_MSE': float(error[:2].detach().mean()), 'real_MSE': float(error[2:].detach().mean()),
                       'gradient_norm_before_clip': norm, 'Adam_reset': step in [0,4000],
                       'BN_first': next(iter(counts.values())), 'BN_last': next(reversed(counts.values())),
                       'windows_with_duplicate_source_index': sum(r['windows_with_duplicate_source_index'] for r in context)}
                trace.write(json.dumps(row,allow_nan=False)+'\n')
                if updates % 100 == 0:
                    trace.flush(); windows.flush()
                    print(json.dumps({**row,'seconds':time.perf_counter()-start}), flush=True)
                if updates % 1000 == 0:
                    path = cache/f'fastdvd-step{updates:04d}.pt'
                    torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'seed':args.seed,
                                'step':updates,'data_step':step,'task':'R53 source-context lowlight FastDVDnet',
                                'optimizer_phase':1 if step < 4000 else 2},path)
                    marker = {'path':str(path.relative_to(ROOT)),'sha256':digest(path),'step':updates,'seed':args.seed,
                              'parameters':2479096,'BN_counts':counts,'Adam_phase':1 if step<4000 else 2,
                              'loss_last_window':row,'final_quality_eligible':updates==8000}
                    write(out/f'fastdvd-step{updates:04d}.json',marker); markers[str(updates)] = marker
        write(out/'context-statistics.json', {'target_windows':context_count,'windows_with_repeated_source_index':repeated,
              'fraction':repeated/context_count,'short_context_R47_fraction':1.,
              'scope':'Duplicates here come from available source boundaries; real source clips have16 frames.'})
        assert updates == 8000 and markers['8000']['final_quality_eligible']; complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', {'completed_normally':complete,'steps':updates,'checkpoints':markers,
              'seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()
              if torch.cuda.is_initialized() else None,'ended':datetime.now().astimezone().isoformat()})


if __name__ == '__main__': main()
