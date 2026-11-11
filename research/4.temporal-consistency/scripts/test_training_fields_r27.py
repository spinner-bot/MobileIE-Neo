"""Same-backend original-history parity and exact copy Adam controls on actual training ROIs."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import cv2
import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from loss_controls_r26 import compare_state
from native_training_r14 import load_training, real_pair
from train_exploration_r2 import new_model
from train_mixture_r4b import mixture_loss
from train_native_r14 import align_tasks
from training_context_r13 import synthetic_pair_with_context
from training_fields_r27 import task_fields
from transport_fst_r27 import TransportFST, gather_features

BASE = ROOT/'research/4.temporal-consistency'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--out', required=True)
    p.add_argument('--first-case-only', action='store_true')
    p.add_argument('--clone-gradients-before-clip', action='store_true')
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    save_json(out/'config.json', {'device': str(device), 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
              'protocol_sha256': sha256(BASE/'docs/exploration-r27-protocol.md'),
              'steps': [0] if args.first_case_only else [0, 1, 3999], 'seeds': [0] if args.first_case_only else [0, 1],
              'gradient_clone_before_optimizer': args.clone_gradients_before_clip,
              'scope': 'Implementation checks on training data, not quality experiments.'})
    data, integrity = load_training()
    images = load_images('train')
    save_json(out/'source_integrity.json', integrity)
    teacher = new_model(device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    checks = []
    for seed in ([0] if args.first_case_only else [0, 1]):
        path = BASE/'results'/f'exploration-r26-seed{seed}'/'shared_context.jsonl'
        with path.open() as stream:
            references = {i: json.loads(line) for i, line in enumerate(stream) if i in [0, 1, 3999]}
        for step in ([0] if args.first_case_only else [0, 1, 3999]):
            rx, ry, rp, identity, real_context = real_pair(data, seed, step, 'native_crop')
            sx, sy, meta, sp, synthetic_context = synthetic_pair_with_context(images, seed, step)
            reference = references[step]
            assert [identity, meta] == reference['sample_identity']
            assert real_context == reference['real_context'] and synthetic_context == reference['synthetic_context']
            low, target = torch.stack([rx, sx]).to(device), torch.stack([ry, sy]).to(device)
            previous = torch.stack([rp, torch.stack([sp, sx[0]])]).to(device)
            aligned, flow, mask, details = task_fields(low, previous, teacher, device, identity['scene'], identity['frame_pair'])
            old, old_details = align_tasks(low, previous, teacher, device, 'native_crop', identity['scene'], identity['frame_pair'], {})
            assert torch.equal(aligned, old)
            assert all(a['history_array_sha256'] == b['history_array_sha256'] for a, b in zip(details, old_details))
            warped, _ = gather_features(previous.flatten(0, 1), low.flatten(0, 1), flow, mask)
            warp_error = float((warped-aligned.flatten(0, 1)).abs().max())*255
            assert warp_error <= 1e-3
            baseline = CausalFST(copy.deepcopy(teacher), 'plain').train()
            candidate = TransportFST(copy.deepcopy(teacher), 'copy').train()
            for model in [baseline, candidate]:
                for param in model.parameters():
                    param.requires_grad_(True)
            optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0) for model in [baseline, candidate]]
            a = baseline.forward_pair(low)
            b = candidate.transform(low.flatten(0, 1), previous.flatten(0, 1), flow, mask).reshape_as(low)
            assert torch.equal(a, b)
            losses = [mixture_loss(a, target), mixture_loss(b, target)]
            assert torch.equal(losses[0], losses[1])
            gradient_states = []
            for loss, model, optimizer in zip(losses, [baseline, candidate], optimizers):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.clone_gradients_before_clip:
                    gradient_states.append({name: value.grad.detach().clone() for name, value in model.named_parameters()})
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                assert bool(torch.isfinite(norm))
                optimizer.step()
            model_audit = compare_state(candidate.state_dict(), baseline.state_dict())
            adam_audit = compare_state(optimizers[1].state_dict(), optimizers[0].state_dict())
            if not args.clone_gradients_before_clip:
                gradient_states = [{name: value.grad.detach().clone() for name, value in model.named_parameters()} for model in [baseline, candidate]]
            gradient_audit = compare_state(gradient_states[1], gradient_states[0])
            save_json(out/f'copy-step{step}-seed{seed}-diagnostic.json', {'model': model_audit, 'Adam': adam_audit, 'gradient': gradient_audit,
                'per_parameter_gradient_max_abs': {name: float((gradient_states[0][name]-gradient_states[1][name]).abs().max()) for name in gradient_states[0]}})
            assert model_audit['exact'] and adam_audit['exact']
            row = {'seed': seed, 'step_index': step, 'same_backend_RGB_history_exact': True, 'history_hashes': 4,
                   'raw_transport_max_abs_gray': warp_error, 'same_CPU_or_CUDA_current_context_exact': True,
                   'copy_model_after_Adam': model_audit, 'copy_Adam_state': adam_audit,
                   'historical_R26_GPU_history_hash_matches_diagnostic_only': [a['history_array_sha256'] == b['history_array_sha256'] for a, b in zip(details, reference['alignment'])]}
            checks.append(row)
            print(json.dumps(row), flush=True)
    save_json(out/'audit.json', {'passed': True, 'checks': checks, 'implementation_updates_copy': len(checks),
                               'scope': 'Same-backend parity only. CPU teacher versus historical CUDA teacher hashes are diagnostic, not a replay requirement. No retained quality checkpoint.'})


if __name__ == '__main__':
    main()
