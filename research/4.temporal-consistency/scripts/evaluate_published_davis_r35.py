"""Matched-base FastDVDnet transfer on R30, CPU only and no parameter sweep."""
import json
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from fastdvdnet_adapter_r12 import MANIFEST, denoise, load_model, window_indices


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    out = BASE/'results/exploration-r35-published'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    smoke_path = BASE/'results/exploration-r12-smoke/smoke.json'
    smoke = json.loads(smoke_path.read_text())
    assert smoke['finite'] and smoke['psnr_gain'] >= 1 and smoke['official_normalized_max_abs_difference'] <= 1e-6
    sources = BASE/'results/exploration-r30-eval-references'
    prediction_manifest = sources/'prediction_manifest.jsonl'
    predictions = [json.loads(line) for line in prediction_manifest.read_text().splitlines()]
    predictions = [row for row in predictions if row['objective'] == 'mse']
    assert len(predictions) == 12
    references = [json.loads(line) for line in (sources/'per_sequence.jsonl').read_text().splitlines()]
    references = {(r['scene'], r['train_seed']): r for r in references if r['mode'] == 'copy'
                  and r['objective'] == 'mse' and r['fusion'] == 'raw' and not r['input_reset']}
    index_path = BASE/'results/exploration-r30-cache/index.json'
    index = {row['scene']: row for row in json.loads(index_path.read_text())['scenes']}
    model = load_model(torch.device('cpu'))
    write(out/'config.json', {'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
          'sigma': 15, 'modes': ['raw', 'window', 'five_current'], 'noise_added': False,
          'base': 'R26 plain MSE final4000, both seeds; cached R30 RGB',
          'parameters_denoiser_only': sum(p.numel() for p in model.parameters()),
          'prediction_manifest_sha256': digest(prediction_manifest), 'cache_index_sha256': digest(index_path),
          'official_manifest_sha256': digest(MANIFEST), 'smoke_sha256': digest(smoke_path),
          'protocol_sha256': digest(BASE/'docs/exploration-r35-published-transfer-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['evaluate_published_davis_r35.py', 'fastdvdnet_adapter_r12.py', 'exploration_r2_data.py']},
          'torch_version': torch.__version__, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Published clipped-AWGN denoiser transfer on custom DEV lowlight; window has two-frame future wait, no original-paper benchmark or phone timing.'})
    started, rows, denoised, completed = time.perf_counter(), 0, 0, False
    try:
        for source in predictions:
            meta = index[source['scene']]
            assert source['input_cache_sha256'] == meta['sha256']
            assert digest(ROOT/meta['path']) == meta['sha256']
            assert digest(ROOT/source['path']) == source['sha256']
            raw = np.load(ROOT/source['path'], allow_pickle=False)
            assert raw.shape == (64, 480, 854, 3)
            sequence = torch.from_numpy((raw/255).transpose(0, 3, 1, 2).copy())
            with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
                gt, flow, mask = archive['gt'], archive['reference_flow'], archive['reference_mask']
            for mode in ['raw', 'window', 'five_current']:
                begin = time.perf_counter()
                if mode == 'raw':
                    output, seconds = raw, 0.
                else:
                    result = denoise(model, sequence, 15, mode=mode)
                    seconds = time.perf_counter()-begin
                    assert torch.isfinite(result).all()
                    output = result.permute(0, 2, 3, 1).contiguous().numpy()*255
                    del result
                    denoised += 1
                metrics, frames = evaluate(output, gt, flow, mask)
                identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                            'base_objective': 'mse', 'train_seed': source['train_seed'], 'mode': mode, 'sigma': None if mode == 'raw' else 15}
                if mode == 'raw':
                    previous = references[(meta['scene'], source['train_seed'])]
                    delta = {key: abs(metrics[key]-previous[key]) for key in
                             ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error', 'flicker']}
                    assert max(delta.values()) <= 1e-6
                    with (out/'raw_replay.jsonl').open('a') as stream:
                        stream.write(json.dumps({**identity, 'metric_deltas': delta})+'\n')
                with (out/'per_sequence.jsonl').open('a') as stream:
                    stream.write(json.dumps({**identity, **metrics}, allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as stream:
                    for frame in frames:
                        stream.write(json.dumps({**identity, **frame}, allow_nan=False)+'\n')
                with (out/'stage_cost.jsonl').open('a') as stream:
                    stream.write(json.dumps({**identity, 'denoiser_only_seconds': seconds if mode != 'raw' else None,
                                            'scope': 'CPU stage excludes base, IO and metrics; shared server, not phone.'})+'\n')
                if mode != 'raw':
                    with (out/'context_index.jsonl').open('a') as stream:
                        stream.write(json.dumps({**identity, 'indices': window_indices(64, mode).tolist(),
                                                 'future_wait_frames_max': 2 if mode == 'window' else 0})+'\n')
                rows += 1
                print(json.dumps({**identity, 'completed_rows': rows, 'denoiser_seconds': seconds, **metrics}), flush=True)
        assert rows == 36 and denoised == 24 and not torch.cuda.is_initialized()
        completed = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': completed, 'rows': rows, 'denoised_sequences': denoised,
                                  'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
