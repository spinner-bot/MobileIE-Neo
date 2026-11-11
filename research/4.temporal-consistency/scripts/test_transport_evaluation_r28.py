"""Verify causal prefix/reset behavior and independently account the fixed evaluation grid."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from train_exploration_r2 import new_model
from transport_evaluation_r28 import EXTERNAL, infer, output_keys, load_models
from transport_fst_r27 import MODES, TransportFST


def main():
    base = ROOT/'research/4.temporal-consistency'
    out = base/'process/exploration-r28/evaluation-fixtures-focused-cpu'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(280028)
    inputs = torch.rand(5, 3, 13, 17)
    changed = inputs.clone()
    changed[3:] = 0
    flow = np.zeros((4, 13, 17, 2), np.float32)
    flow[:] = [.375, -.625]
    mask = np.ones((4, 13, 17), bool)
    data = {'half_flow': flow, 'half_mask': mask}
    subdata = {'half_flow': flow[3:], 'half_mask': mask[3:]}
    reset = np.array([False, False, False, True, False])
    backbone = new_model('cpu')
    with torch.inference_mode():
        reference = torch.stack([backbone(x[None])[0] for x in inputs])
    checks = []
    for mode in [*MODES, EXTERNAL]:
        net = copy.deepcopy(backbone) if mode == EXTERNAL else TransportFST(copy.deepcopy(backbone), mode).eval()
        current = infer(net, inputs, data, mode, 'current', None)
        assert torch.equal(current, reference)
        actual = infer(net, inputs, data, mode, 'causal', None)
        future_changed = infer(net, changed, data, mode, 'causal', None)
        assert torch.equal(actual[:3], future_changed[:3])
        assert torch.equal(actual[0], reference[0])
        interrupted = infer(net, inputs, data, mode, 'causal', reset)
        assert torch.equal(interrupted[3], reference[3])
        fresh = infer(net, inputs[3:], subdata, mode, 'causal', None)
        assert torch.equal(interrupted[3:], fresh)
        checks.append({'mode': mode, 'current_exact': True, 'future_prefix_exact': True,
                       'first_exact': True, 'reset_current_exact': True, 'reset_fresh_subsequence_exact': True})
    events_path = base/'results/exploration-r26-eval-plain/input_events.json'
    events = json.loads(events_path.read_text())
    assert len(events) == 46 and sum(bool(r['reset_frames']) for r in events) == 24
    counts = []
    for group in ['copy', 'post_transport']:
        logical = metrics = aliases = neural = 0
        nodes = [(group, 4)]+([(EXTERNAL, 2)] if group == 'copy' else [])
        for mode, weights in nodes:
            for scene in events:
                for _ in range(weights):
                    spatial = mode in ['copy', EXTERNAL]
                    for history in (['causal'] if spatial else ['causal', 'current']):
                        metric_keys, model_keys = set(), set()
                        outputs = ['raw', 'half01', 'postwarp01', 'postwarp05'] if spatial or history == 'current' else ['raw', 'half01']
                        for reset_flag in [False, True]:
                            for fusion in outputs:
                                model_key, key = output_keys(mode, history, reset_flag, bool(scene['reset_frames']), fusion)
                                logical += 1
                                if key in metric_keys:
                                    aliases += 1
                                else:
                                    metric_keys.add(key)
                                    metrics += 1
                                model_keys.add(model_key)
                        neural += len(model_keys)
        expected = (2208, 1536, 672, 276) if group == 'copy' else (2208, 1584, 624, 464)
        assert (logical, metrics, aliases, neural) == expected
        counts.append({'group': group, 'logical_rows': logical, 'metrics': metrics, 'aliases': aliases, 'neural_sequences': neural})
    assert [sum(r[k] for r in counts) for k in ['logical_rows', 'metrics', 'aliases', 'neural_sequences']] == [4416, 3120, 1296, 740]
    reused_models, reused_markers = load_models('copy', torch.device('cpu'))
    assert len(reused_models) == len(reused_markers) == 6
    assert sum(bool(r.get('reused_R26_training')) for r in reused_markers) == 4
    save_json(out/'audit.json', {'passed': True, 'causal_checks': checks, 'grid_counts': counts,
              'input_events_sha256': sha256(events_path),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), Path(__file__).with_name('transport_evaluation_r28.py')]},
              'reused_spatial_model_markers': reused_markers,
              'scope': 'Focused-grid interface/budget and reused-weight loading checks; no R28 trained quality results.'})
    print(json.dumps({'passed': True, 'causal_nodes': len(checks), 'logical_rows': 4416}), flush=True)


if __name__ == '__main__':
    main()
