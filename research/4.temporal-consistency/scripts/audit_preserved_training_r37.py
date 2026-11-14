"""Verify fixed updates, shared training windows and the frozen spatial control."""
from collections import Counter
import json
import math

import torch

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    result = []
    for seed in [0, 1]:
        folder = BASE/f'results/exploration-r37-seed{seed}'
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['total_updates'] == 8000
        trace = list(map(json.loads, (folder/'training.jsonl').read_text().splitlines()))
        windows = list(map(json.loads, (folder/'windows.jsonl').read_text().splitlines()))
        assert len(trace) == 8000 and len(windows) == 4000
        assert [r['step'] for r in windows] == list(range(1, 4001))
        previous_windows = BASE/f'results/exploration-r34-seed{seed}/windows.jsonl'
        assert windows == list(map(json.loads, previous_windows.read_text().splitlines()))
        config = json.loads((folder/'config.json').read_text())
        anchor = torch.load(ROOT/config['anchor']['path'], map_location='cpu', weights_only=False)['state_dict']
        frozen = {}
        for name in ['learned_alignment', 'zero_alignment']:
            values = [r for r in trace if r['model'] == name]
            assert [r['step'] for r in values] == list(range(1, 4001))
            assert all(math.isfinite(r[key]) for r in values for key in ['mse', 'gradient_norm_before_clip', 'amplitude'])
            marker = json.loads((folder/f'{name}-step4000.json').read_text())
            assert digest(ROOT/marker['path']) == marker['sha256']
            state = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)['state_dict']
            frozen[name] = all(torch.equal(value, state['backbone.'+key]) for key, value in anchor.items())
            assert frozen[name]
        counts = Counter(window['video'] for step in windows for window in step['windows'])
        assert len(counts) == 57 and not set(counts) & {'bear', 'lucia', 'walking'}
        result.append({'seed': seed, 'total_updates': 8000, 'models_each_final4000': True,
                       'sampled_windows': sum(counts.values()), 'sampled_training_videos': len(counts),
                       'windows_equal_original_R34_seed': True, 'final_spatial_anchor_unchanged': frozen,
                       'wall_seconds': runtime['wall_seconds']})
    out = BASE/'process/exploration-r37/training-completion.json'
    assert not out.exists()
    out.write_text(json.dumps({'passed': True, 'total_updates': 16000, 'seeds': result,
          'scope': 'Required frozen-anchor/shared-data controls only. No optimizer replay, quality or speed inference.'}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'total_updates': 16000}))


if __name__ == '__main__':
    main()
