"""Fixed 4000-update FST order adaptation; all ten independent runs share only teacher fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from loss_controls_r26 import compare_state
from native_training_r14 import load_training, real_pair
from train_exploration_r2 import new_model
from train_mixture_r4b import mixture_loss
from training_context_r13 import synthetic_pair_with_context
from training_fields_r27 import task_fields
from transport_fst_r27 import MODES, TransportFST

BASE = ROOT/'research/4.temporal-consistency'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    # These complete gates are prerequisites, not advisory status checks.
    gates = [BASE/'process/exploration-r27'/name/'audit.json' for name in ['fixtures-cuda', 'training-fields-cuda']]
    for path in gates:
        assert json.loads(path.read_text())['passed']
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    data, integrity = load_training()
    images = load_images('train')
    teacher = new_model(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    save_json(out/'source_integrity.json', integrity)
    original = BASE/'results'/f'exploration-r26-seed{args.seed}'
    references = [json.loads(line) for line in (original/'shared_context.jsonl').read_text().splitlines()]
    assert len(references) == 4000
    runs, pointers = [], set()
    shared_path = out/'shared_context.jsonl'
    for mode in MODES:
        for objective in ['charb', 'mse']:
            torch.manual_seed(args.seed)
            net = TransportFST(new_model(device), mode).train()
            params = list(net.parameters())
            own = {p.data_ptr() for p in params}
            assert not own & pointers
            pointers |= own
            assert sum(p.numel() for p in params) == 4075
            optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0)
            label = mode+'-'+objective
            folder = out/label
            folder.mkdir()
            checkpoints = ROOT/'.cache/checkpoints/exploration-r28'/f'{label}-seed{args.seed}'
            checkpoints.mkdir(parents=True, exist_ok=False)
            cfg = {'mode': mode, 'objective': objective, 'seed': args.seed, 'steps': 4000, 'parameters': 4075,
                   'optimizer': 'Adam', 'lr': 1e-3, 'weight_decay': 0, 'betas': [.9, .999], 'eps': 1e-8,
                   'global_clip': 1., 'loss_scale': 1., 'loss': 'sqrt(e²+1e-6).mean' if objective == 'charb' else 'e².mean',
                   'revision': revision, 'source_sha256': sources, 'protocol_sha256': sha256(BASE/'docs/exploration-r28-protocol.md'),
                   'R27_gate_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in gates},
                   'source_manifest_sha256': integrity['manifest_sha256'],
                   'reference_shared_context_sha256': sha256(original/'shared_context.jsonl'),
                   'official_weights_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                   'shared_context_path': str(shared_path.relative_to(ROOT)), 'history_gradients': 'full past encoder, fixed flow detached',
                   'torch_version': torch.__version__, 'opencv_version': cv2.__version__,
                   'deterministic_algorithms': True, 'cudnn_deterministic': True,
                   'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32, 'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
                   'cublas_workspace': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'resume_supported': False,
                   'scope': 'Matched structural adaptation, not official convergence or established innovation.'}
            save_json(folder/'config.json', cfg)
            trace = [json.loads(s) for s in (original/('plain-'+objective)/'training.jsonl').read_text().splitlines()] if mode == 'copy' else None
            runs.append({'mode': mode, 'objective': objective, 'label': label, 'net': net, 'params': params,
                         'optimizer': optimizer, 'folder': folder, 'checkpoints': checkpoints, 'config': cfg, 'old': trace})
    initial = {}
    start = time.perf_counter()
    total_hashes = 0
    for step in range(4000):
        rx, ry, rp, identity, real_context = real_pair(data, args.seed, step, 'native_crop')
        sx, sy, meta, sp, synthetic_context = synthetic_pair_with_context(images, args.seed, step)
        reference = references[step]
        assert [identity, meta] == reference['sample_identity']
        assert real_context == reference['real_context'] and synthetic_context == reference['synthetic_context']
        low, target = torch.stack([rx, sx]).to(device), torch.stack([ry, sy]).to(device)
        previous = torch.stack([rp, torch.stack([sp, sx[0]])]).to(device)
        align_start = time.perf_counter()
        aligned, flow, mask, details = task_fields(low, previous, teacher, device, identity['scene'], identity['frame_pair'])
        align_seconds = time.perf_counter()-align_start
        legacy = [{key: d[key] for key in old} for d, old in zip(details, reference['alignment'])]
        assert len(legacy) == 4 and legacy == reference['alignment']
        total_hashes += 4
        context = {'step': step+1, 'sample_identity': [identity, meta], 'real_context': real_context,
                   'synthetic_context': synthetic_context, 'alignment': legacy, 'old_R14_history_hash_matches': [True]*4}
        legacy_digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        assert legacy_digest == reference['context_canonical_sha256']
        context.update({'R26_context_canonical_sha256': legacy_digest, 'field_details': details})
        digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        with shared_path.open('a') as stream:
            stream.write(json.dumps({**context, 'context_canonical_sha256': digest, 'alignment_wall_seconds': align_seconds}, allow_nan=False)+'\n')
        current, prior = low.flatten(0, 1), previous.flatten(0, 1)
        last = []
        for run in runs:
            net, optimizer, mode, objective = run['net'], run['optimizer'], run['mode'], run['objective']
            if step == 0:
                with torch.no_grad():
                    maximum = float((net.transform(current)-teacher(current)).abs().max())
                save_json(run['folder']/'initial_current_limit.json', {'max_abs_normalized': maximum})
                assert maximum <= 2e-5
            prediction = net.transform(current, prior, flow, mask).reshape_as(low)
            charb = mixture_loss(prediction, target)
            mse = (prediction-target).square().mean()
            loss = charb if objective == 'charb' else mse
            assert bool(torch.isfinite(loss))
            if step == 0:
                if objective == 'charb':
                    initial[mode] = prediction.detach().clone()
                else:
                    equal = torch.equal(initial[mode], prediction.detach())
                    save_json(run['folder']/'initial_objective_output.json', {'same_architecture_Charb_MSE_exact': equal})
                    assert equal
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(run['params'], 1.))
            assert np.isfinite(norm)
            optimizer.step()
            assert all(bool(torch.isfinite(param).all()) for param in run['params'])
            error = (prediction-target).detach().double()
            mu = error.mean((-2, -1), keepdim=True)
            full_mse = float(error.square().mean())
            bias = float(mu.square().mean())
            centered = float((error-mu).square().mean())
            assert abs(full_mse-bias-centered) <= 1e-12
            task = float(loss.detach())
            row = {'step': step+1, 'task': task, 'charb_task': float(charb.detach()), 'mse_task': float(mse.detach()),
                   'gradient_norm_all': norm, 'global_clip_triggered': norm > 1., 'global_RGB_bias_mse': bias,
                   'centered_error_mse': centered, 'MSE_double': full_mse, 'MSE_closure_abs': abs(full_mse-bias-centered),
                   'old_copy_task_delta': task-run['old'][step]['task'] if mode == 'copy' else None,
                   'shared_context_sha256': digest}
            with (run['folder']/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            last.append({'label': run['label'], 'task': task, 'gradient_norm': norm, 'old_copy_task_delta': row['old_copy_task_delta']})
            del prediction, charb, mse, loss, error, mu
        if step == 0:
            initial.clear()
        if (step+1) % 100 == 0:
            print(json.dumps({'seed': args.seed, 'step': step+1, 'models': last, 'old_history_hashes_exact': total_hashes}), flush=True)
        if (step+1) % 1000 == 0:
            prefix = shared_path.read_bytes()
            for run in runs:
                path = run['checkpoints']/f'step{step+1:04d}.pth'
                saved = {'model': run['net'].state_dict(), 'optimizer': run['optimizer'].state_dict(), 'step': step+1, 'config': run['config']}
                torch.save(saved, path)
                save_json(run['folder']/f'checkpoint-step{step+1:04d}.json', {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'step': step+1})
                trace_bytes = (run['folder']/'training.jsonl').read_bytes()
                save_json(run['folder']/f'prefix-step{step+1:04d}.json', {'step': step+1,
                          'training': {'bytes': len(trace_bytes), 'sha256': hashlib.sha256(trace_bytes).hexdigest()},
                          'shared_context': {'bytes': len(prefix), 'sha256': hashlib.sha256(prefix).hexdigest()}})
                if run['mode'] == 'copy':
                    marker = json.loads((original/('plain-'+run['objective'])/f'checkpoint-step{step+1:04d}.json').read_text())
                    assert sha256(ROOT/marker['path']) == marker['sha256']
                    old = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
                    model_audit = compare_state(saved['model'], old['model'])
                    adam_audit = compare_state(saved['optimizer'], old['optimizer'])
                    save_json(run['folder']/f'copy-replay-step{step+1:04d}.json', {'model': model_audit, 'Adam': adam_audit,
                              'reference_path': marker['path'], 'reference_sha256': marker['sha256']})
                    assert model_audit['exact'] and adam_audit['exact']
    assert total_hashes == 16000
    for run in runs:
        save_json(run['folder']/'runtime.json', {'final_step': 4000, 'new_updates': 4000,
                  'checkpoint': str((run['checkpoints']/'step4000.pth').relative_to(ROOT)),
                  'checkpoint_sha256': sha256(run['checkpoints']/'step4000.pth'), 'shared_seed_wall_seconds': time.perf_counter()-start,
                  'scope': 'Concurrent logical models, sequential updates with shared fields; not standalone pipeline time.'})
    save_json(out/'runtime.json', {'seed': args.seed, 'final_step': 4000, 'independent_models': 10, 'new_updates': 40000,
              'old_history_hashes_exact': total_hashes, 'wall_seconds': time.perf_counter()-start, 'completed_normally': True})


if __name__ == '__main__':
    main()
