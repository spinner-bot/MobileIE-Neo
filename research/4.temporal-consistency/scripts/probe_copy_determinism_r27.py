"""Diagnose the failed exact-Adam gate against an identical-code CPU control."""
import copy
import json
from pathlib import Path
import subprocess
import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from loss_controls_r26 import compare_state
from native_training_r14 import load_training, real_pair
from train_exploration_r2 import new_model
from train_mixture_r4b import mixture_loss
from training_context_r13 import synthetic_pair_with_context
from transport_fst_r27 import TransportFST


def main():
    out = ROOT/'research/4.temporal-consistency/process/exploration-r27/copy-CPU-determinism'
    out.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(270027)
    data, integrity = load_training()
    rx, ry, _, identity, _ = real_pair(data, 0, 0, 'native_crop')
    sx, sy, _, _, _ = synthetic_pair_with_context(load_images('train'), 0, 0)
    low, target = torch.stack([rx, sx]), torch.stack([ry, sy])
    save_json(out/'config.json', {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': sha256(Path(__file__)), 'identity': identity, 'source_integrity': integrity,
              'scope': 'Original failed CPU exact gate diagnosis, no quality model retained; no extra gradient clone before optimizer.'})
    original = new_model('cpu')
    rows = []
    for threads in [2, 1]:
        torch.set_num_threads(threads)
        for mkldnn in [True, False]:
            torch.backends.mkldnn.enabled = mkldnn
            for control in ['identical_code', 'transport_copy']:
                for repeat in range(3):
                    a = CausalFST(copy.deepcopy(original), 'plain').train()
                    b = CausalFST(copy.deepcopy(original), 'plain').train() if control == 'identical_code' else TransportFST(copy.deepcopy(original), 'copy').train()
                    optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in [a, b]]
                    predictions = [a.forward_pair(low), b.forward_pair(low) if control == 'identical_code' else b.transform(low.flatten(0, 1)).reshape_as(low)]
                    output_equal = torch.equal(*predictions)
                    # Keep the original failed test's update order and allocation pattern.
                    for prediction, net, optimizer in zip(predictions, [a, b], optimizers):
                        optimizer.zero_grad(set_to_none=True)
                        mixture_loss(prediction, target).backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                        optimizer.step()
                    model = compare_state(a.state_dict(), b.state_dict())
                    adam = compare_state(optimizers[0].state_dict(), optimizers[1].state_dict())
                    gradients = compare_state({n: p.grad for n, p in a.named_parameters()}, {n: p.grad for n, p in b.named_parameters()})
                    row = {'threads': threads, 'mkldnn': mkldnn, 'control': control, 'repeat': repeat,
                           'outputs_exact': output_equal, 'model': model, 'Adam': adam, 'clipped_gradients': gradients}
                    rows.append(row)
                    with (out/'rows.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
    save_json(out/'audit.json', {'comparisons': rows, 'pairs': len(rows), 'implementation_updates': len(rows)*2,
              'nonexact_pairs': sum(not (r['outputs_exact'] and r['model']['exact'] and r['Adam']['exact']) for r in rows),
              'scope': 'Control on the fixed first training task. Does not erase earlier failure or establish CUDA/full training reproducibility.'})


if __name__ == '__main__':
    main()
