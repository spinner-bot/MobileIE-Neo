"""Bounded author-code qualification; artificial CPU inputs, no SGD or quality scores."""
import argparse
from datetime import datetime
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from project_assets import ASSETS, mounted_data, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise SystemExit('Use a new versioned project result path; refuse overwrite.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('Launch with CUDA_VISIBLE_DEVICES empty before any import.')
    mounted_data()
    started = time.monotonic()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    spec = importlib.util.spec_from_file_location('released_rebotnet', ROOT / 'third_party/rebotnet/archs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(upscale=1, img_size=[2, 384, 384], window_size=[6, 8, 8],
                  depths=[4, 3, 3, 4], indep_reconsts=[9, 10], embed_dims=[8, 12, 20, 32],
                  num_heads=[6] * 13, pa_frames=2, deformable_groups=12, mlp_dim=64,
                  bottle_depth=4, bottle_dim=32, dropout=0.1, patch_size=1)
    model = module.rebotnet(**config).cpu().eval()
    work = {'conv_macs': 0, 'linear_macs': 0}
    active = set()

    def count(layer, inputs, output):
        if isinstance(layer, torch.nn.Conv2d):
            work['conv_macs'] += output.numel() * (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            work['linear_macs'] += output.numel() * layer.in_features
        active.update(id(p) for p in layer.parameters(recurse=False))

    handles = [m.register_forward_hook(count) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    x0 = torch.rand(1, 3, 384, 384, requires_grad=True)
    y0 = model(torch.stack((x0, x0), dim=1))
    for handle in handles:
        handle.remove()
    x1 = torch.rand_like(x0)
    y1 = model(torch.stack((y0, x1), dim=1))
    grad_prediction, grad_first_input = torch.autograd.grad(y1.square().mean(), (y0, x0))
    checks = dict(valid_pair_shape=list(y0.shape), valid_finite=bool(torch.isfinite(y0).all()),
                  history_prediction_gradient_l1=float(grad_prediction.abs().sum()),
                  initial_input_gradient_l1=float(grad_first_input.abs().sum()))
    cases = []
    del y0, y1, x0, x1, grad_prediction, grad_first_input
    for name, shape in [('upstream_example_4d', (1, 3, 384, 384)),
                        ('project_crop128', (1, 2, 3, 128, 128)),
                        ('arbitrary_rectangle', (1, 2, 3, 320, 640)),
                        ('same_token_rectangle', (1, 2, 3, 256, 576))]:
        try:
            with torch.no_grad():
                result = model(torch.zeros(shape))
            cases.append(dict(name=name, input_shape=list(shape), success=True,
                              output_shape=list(result.shape), finite=bool(torch.isfinite(result).all())))
        except (RuntimeError, ValueError) as error:
            cases.append(dict(name=name, input_shape=list(shape), success=False,
                              error_type=type(error).__name__, error=str(error)))
    versions = {name: metadata.version(name) for name in
                ['torch', 'torchvision', 'timm', 'einops', 'huggingface-hub', 'safetensors']}
    package_roots = {name: str(Path(__import__(name).__file__).resolve()) for name in
                     ['torch', 'torchvision', 'timm', 'einops']}
    if not all(Path(p).is_relative_to(ASSETS) for p in package_roots.values()):
        raise RuntimeError('An imported dependency is outside project assets.')
    expected = checks['valid_finite'] and checks['valid_pair_shape'] == [1, 3, 384, 384]
    expected = expected and checks['history_prediction_gradient_l1'] > 0 and checks['initial_input_gradient_l1'] > 0
    expected = expected and [c['success'] for c in cases] == [False, False, False, True]
    report = dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
                  scope='Original released XS architecture, eval-mode artificial CPU inputs; backward proves differentiable history, not temporal quality or optimizer convergence.',
                  config=config, versions=versions, package_roots=package_roots,
                  registered_parameters=sum(p.numel() for p in model.parameters()),
                  called_conv_linear_parameters=sum(p.numel() for p in model.parameters() if id(p) in active),
                  single_384_pair_conv_linear_macs=work,
                  excluded_from_macs=['norm', 'GELU', 'pooling', 'rearrangement/copies', 'addition', 'stochastic depth', 'tile fusion', 'backward'],
                  checks=checks, shape_cases=cases, cpu_threads=2,
                  elapsed_seconds=time.monotonic() - started,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  gpu_initial_visible_mask='', cuda_initialized=bool(torch.cuda.is_initialized()),
                  new_video_frames=0, quality_rows=0, training_updates=0,
                  expected_qualification_passed=bool(expected),
                  limits='No learned weights, data, GPU/phone timings, arbitrary-native-image support or published performance reproduced. Recurrence self-start here is an artificial functional example, not the paper GT-first evaluation.')
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not expected or report['cuda_initialized']:
        raise SystemExit('Qualification failed; preserve this result and do not auto-retry.')


if __name__ == '__main__':
    main()
