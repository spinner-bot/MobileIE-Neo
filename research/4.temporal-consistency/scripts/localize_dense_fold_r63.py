"""One fixed failed-case diagnosis; does not rerun or promote the old gate."""
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import torch

from dense_joint_state_r63 import folded, make_models, prediction
from dynamic_training_r34 import BASE, digest
from training_form_r41 import LAYERS


def main():
    out = BASE/'process/exploration-r63/fold-localization001.json'
    assert not out.exists()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    started = time.perf_counter()
    result = {'created_at': datetime.now().astimezone().isoformat(), 'completed_normally': False,
              'formal_updates': 0, 'real_images': 0, 'source_sha256': {
                  name: digest(Path(__file__).parent/name) for name in
                  ['localize_dense_fold_r63.py', 'dense_joint_state_r63.py', 'budget_causal_r33.py']},
              'protocol_sha256': digest(BASE/'docs/exploration-r63-fold-localization-protocol.md'),
              'old_gate_unchanged': False, 'local_layers': []}
    gate = BASE/'process/exploration-r63/cpu-gate002.json'
    gate_digest = digest(gate)
    try:
        nets, info = make_models(1)
        result['initialization'] = info
        torch.manual_seed(63611)
        clip, _ = torch.rand(2, 4, 3, 16, 20), torch.rand(2, 4, 3, 16, 20)
        # BN state updates match gate002. Backward itself never mutates BN buffers.
        with torch.no_grad():
            for training in [True, False]:
                for net in nets.values():
                    prediction(net.train(training), clip)
            for name in ['spatial', 'causal']:
                net = nets[name].train()
                prediction(net, clip)
                generator = torch.Generator().manual_seed(63901)
                net.readout.weight.copy_(1e-3*torch.randn(net.readout.weight.shape, generator=generator))
                prediction(net, clip)
                prediction(net.eval(), clip)
            net = nets['spatial'].eval()
            deploy = folded(net)
            expanded, slim = prediction(net, clip), prediction(deploy, clip)
            absolute = (expanded-slim).abs()
            clipped = (expanded.clamp(0, 1)-slim.clamp(0, 1)).abs()
            flat_index = int(absolute.flatten().argmax())
            expected = json.loads(gate.read_text())['seeds'][1]['cases'][1]['export']['spatial']['max_abs_delta']
            actual = float(absolute.max())
            result['output'] = {'raw_max_abs_delta': actual,
                'original_gate_raw_max_abs_delta': expected,
                'raw_reproduced_exactly': actual == expected,
                'expanded_raw_min': float(expanded.min()), 'expanded_raw_max': float(expanded.max()),
                'deploy_raw_min': float(slim.min()), 'deploy_raw_max': float(slim.max()),
                'expanded_outside_0_1_fraction': float(((expanded < 0) | (expanded > 1)).float().mean()),
                'raw_worst_flat_index': flat_index,
                'raw_worst_expanded_value': float(expanded.flatten()[flat_index]),
                'raw_worst_deploy_value': float(slim.flatten()[flat_index]),
                'clipped_max_abs_delta': float(clipped.max()),
                'clipped_RMS_delta': float(clipped.square().mean().sqrt()),
                'clipped_nonzero_fraction': float((clipped != 0).float().mean())}
            assert actual == expected, 'Diagnosis did not reproduce the fixed original failure'
            hooks = []
            for name in LAYERS:
                def compare(module, args, output, layer_name=name):
                    paired = deploy.backbone.get_submodule(layer_name)(args[0])
                    result['local_layers'].append({'layer': layer_name,
                        'same_input_shape': list(args[0].shape),
                        'output_abs_max': float(output.abs().max()),
                        'local_fold_max_abs_delta': float((paired-output).abs().max())})
                hooks.append(net.backbone.get_submodule(name).register_forward_hook(compare))
            try:
                prediction(net, clip)
            finally:
                for hook in hooks:
                    hook.remove()
            assert len(result['local_layers']) == 24
            result['completed_normally'] = True
    except Exception:
        result['failure'] = traceback.format_exc()
        raise
    finally:
        result.update(wall_seconds=time.perf_counter()-started,
                      old_gate_unchanged=digest(gate) == gate_digest,
                      cuda_initialized=torch.cuda.is_initialized(),
                      finished_at=datetime.now().astimezone().isoformat())
        out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'completed_normally': result['completed_normally'],
                          'wall_seconds': result['wall_seconds'], 'output': result.get('output')}), flush=True)


if __name__ == '__main__':
    main()
