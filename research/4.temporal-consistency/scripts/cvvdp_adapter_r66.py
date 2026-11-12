"""Unmodified official SDR-array metric, isolated optional packages and fixed display."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'
TARGET = ROOT/'.cache/tools/exploration-r66/cvvdp057'
FPS = 30


def load_metric(device):
    completed = BASE/'process/exploration-r66/package002/completed.json'
    manifest = json.loads(completed.read_text()); assert manifest['completed']
    assert TARGET.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    if str(TARGET) not in sys.path: sys.path.insert(0, str(TARGET))
    import pycvvdp
    import torchvision
    assert importlib.metadata.version('cvvdp') == '0.5.7'
    assert torchvision.__version__ == '0.24.1+cu126' and torch.__version__ == '2.9.1+cu126'
    assert Path(pycvvdp.__file__).resolve().is_relative_to(TARGET.resolve())
    assert Path(torchvision.__file__).resolve().is_relative_to(TARGET.resolve())
    for module in [torch, np]:
        assert Path(module.__file__).resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo/.venv')
    metric = pycvvdp.cvvdp(display_name='standard_fhd', device=torch.device(device), quiet=True,
                          temp_padding='replicate', heatmap=None, gpu_mem=2)
    package = Path(pycvvdp.__file__).parent
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in package.rglob('*') if path.is_file() and path.suffix in ('.py', '.json', '.mat')}
    info = dict(package_version='0.5.7', metric_info=metric.get_info_string(), display='standard_fhd',
                standard_fhd=json.loads((package/'vvdp_data/display_models.json').read_text())['standard_fhd'],
                playback_fps=FPS, temp_padding='replicate', heatmap=None, input='SDR float32 THWC [0,1]; original dimensions',
                device=str(device), package_install_manifest_sha256=hashlib.sha256(completed.read_bytes()).hexdigest(),
                package_sources=sources, Torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                scope='Nominal standard-display playback, not actual phone/capture calibration or human MOS/flicker rating.')
    return metric, info


def predict(metric, test, reference):
    assert test.shape == reference.shape and test.ndim == 4 and test.shape[-1] == 3
    assert test.dtype == reference.dtype == np.float32
    assert np.isfinite(test).all() and np.isfinite(reference).all()
    assert min(test.min(), reference.min()) >= 0 and max(test.max(), reference.max()) <= 1
    with torch.inference_mode():
        score, statistics = metric.predict(test, reference, dim_order='FHWC', frames_per_second=FPS)
    value = float(score.detach().cpu().reshape(-1)[0]); assert np.isfinite(value)
    assert statistics['N_frames'] == len(test) and statistics['frames_per_second'] == FPS
    assert statistics['height'] == test.shape[1] and statistics['width'] == test.shape[2]
    return value, statistics


def artificial_inputs():
    y, x = np.mgrid[0:64, 0:64].astype(np.float32)/63
    image = np.stack((.25+.3*x, .3+.3*y, .35+.2*x*y), -1)
    reference = np.repeat(image[None], 16, axis=0).astype(np.float32)
    alternate = np.where(np.arange(16) % 2, 1, -1).astype(np.float32)[:, None, None, None]
    return reference, dict(identity=reference.copy(), small_static=reference+np.float32(.02),
                           large_static=reference+np.float32(.08), alternating=reference+np.float32(.02)*alternate)


def qualify(device, cpu_scores=None):
    metric, info = load_metric(device)
    reference, tests = artificial_inputs()
    scores = {name: predict(metric, values, reference)[0] for name, values in tests.items()}
    assert abs(scores['identity']-10) <= 1e-4
    assert scores['large_static'] <= scores['small_static'] < scores['identity']
    differences = None
    if cpu_scores is not None:
        differences = {name: abs(value-cpu_scores[name]) for name, value in scores.items()}
        assert max(differences.values()) <= .01
    return dict(passed=True, scores=scores, CPU_CUDA_absolute_differences=differences, source=info,
                artificial_shape=[16, 64, 64, 3], new_enhancement_CNN=0, new_SGD=0,
                scope='Interface/layout/identity/finite synthetic qualification only; no real quality/MOS claim.')
