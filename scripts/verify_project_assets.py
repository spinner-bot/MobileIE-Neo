"""Verify independent package ownership, local links and one bounded CPU replay."""
import argparse
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import site
import sys

from project_assets import ASSETS, CACHE, ENVIRONMENT, OWNER, REPO, stamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--packages-only', action='store_true')
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    assert prefix == ENVIRONMENT.resolve() and prefix.is_relative_to(OWNER)
    assert 'include-system-site-packages = false' in (prefix/'pyvenv.cfg').read_text()
    assert site.ENABLE_USER_SITE is False
    modules = {}
    for name in ['torch', 'numpy', 'cv2', 'PIL', 'matplotlib', 'scipy', 'einops', 'yaml', 'pandas', 'tqdm', 'pypdf']:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(prefix), (name, str(path))
        modules[name] = {'path': str(path), 'version': getattr(module, '__version__', None)}
    for value in sys.path:
        path = Path(value or '.').resolve()
        if 'site-packages' in path.parts or 'dist-packages' in path.parts:
            assert path.is_relative_to(prefix), str(path)
        if str(path).startswith(('/data/Users/', '/root/')):
            assert path.is_relative_to(OWNER), str(path)
    import torch
    import numpy as np
    import cv2
    assert torch.__version__ == '2.9.1+cu126' and torch.version.cuda == '12.6'
    assert np.__version__ == '2.2.6' and cv2.__version__ == '4.11.0'
    assert not torch.cuda.is_initialized()
    report = {'observed_at': stamp(), 'prefix': str(prefix), 'system_site_packages': False,
              'user_site_enabled': site.ENABLE_USER_SITE, 'modules': modules,
              'sys_path': sys.path, 'cuda_initialized': False, 'packages_only': args.packages_only}
    if not args.packages_only:
        assert (REPO/'.venv').is_symlink() and (REPO/'.venv').resolve() == ENVIRONMENT.resolve()
        assert (REPO/'.cache').is_symlink() and (REPO/'.cache').resolve() == CACHE.resolve()
        report['links'] = {name: str((REPO/name).resolve()) for name in ['.venv', '.cache']}
        sys.path.insert(0, str(REPO/'research/4.temporal-consistency/scripts'))
        import json
        from continuation_r14 import create_model
        from exploration_r2_data import evaluate
        from prepare_cache_r5 import load_one
        from prepare_native_r10 import INDEX
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        torch.use_deterministic_algorithms(True)
        base = REPO/'research/4.temporal-consistency'
        meta = next(r for r in json.loads(INDEX.read_text())['scenes'] if r['domain'] == 'native_crop')
        data = load_one(meta)
        marker = json.loads((base/'results/exploration-r26-seed0/plain-charb/checkpoint-step4000.json').read_text())
        checkpoint = REPO/marker['path']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == marker['sha256']
        net = create_model('plain', torch.device('cpu')).eval()
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
        inputs = torch.from_numpy((data['low']/255.).transpose(0, 3, 1, 2).copy())
        with torch.inference_mode():
            output = torch.stack([net(frame[None])[0] for frame in inputs]).clamp(0, 1)
        rgb = np.ascontiguousarray(output.permute(0, 2, 3, 1).numpy()*255.)
        metrics, _ = evaluate(rgb, data['gt'], data['reference_flow'], data['reference_mask'])
        reference = next(json.loads(line) for line in (base/'results/exploration-r27-frozen-cpu/per_sequence.jsonl').read_text().splitlines()
                         if (lambda r: r['scene'] == meta['scene'] and r['weight'] == 'charb-seed0' and r['mode'] == 'copy'
                             and r['history'] == 'causal' and r['fusion'] == 'raw')(json.loads(line)))
        deltas = {key: abs(metrics[key]-reference[key]) for key in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']}
        assert max(deltas.values()) <= 1e-10, deltas
        assert not torch.cuda.is_initialized()
        report['bounded_CPU_replay'] = {'scene': meta['scene'], 'frames': len(inputs), 'max_abs_metric': max(deltas.values()),
                                         'metrics': metrics, 'checkpoint_sha256': marker['sha256']}
    report['passed'] = True
    write_json(args.out, report)
    print('Owned environment verification passed: '+str(prefix), flush=True)


if __name__ == '__main__':
    main()
