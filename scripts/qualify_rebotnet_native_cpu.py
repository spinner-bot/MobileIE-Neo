"""Analytic image/recurrence checks for the explicit causal reference tiler."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

from project_assets import mounted_data, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT) or out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('Require a new project result path and initially empty CUDA mask.')
    mounted_data()
    import torch
    sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
    from rebotnet_reference import new_model, native_pair, native_predictions, tile_layout
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()

    class CurrentOnly(torch.nn.Module):
        def forward(self, pair):
            return pair[:, 1]

    class HalfHistory(torch.nn.Module):
        def forward(self, pair):
            return pair.mean(dim=1)

    spatial_checks = []
    for height, width in [(1, 1), (128, 183), (384, 384), (519, 773), (1080, 1920)]:
        current = torch.rand(1, 3, height, width)
        value = native_pair(CurrentOnly(), torch.zeros_like(current), current)
        spatial_checks.append(dict(height=height, width=width,
            tiles=len(tile_layout(height, width)), output_shape=list(value.shape),
            finite=bool(torch.isfinite(value).all()), max_identity_error=float((value-current).abs().max())))
    frames = [torch.rand(1, 3, 519, 773) for _ in range(3)]
    values = list(native_predictions(HalfHistory(), frames))
    analytic = [frames[0], .5*frames[0]+.5*frames[1], .25*frames[0]+.25*frames[1]+.5*frames[2]]
    recurrence_error = max(float((x-y).abs().max()) for x, y in zip(values, analytic))
    model = new_model(0).eval()
    previous = torch.rand(1, 3, 384, 384)
    current = torch.rand_like(previous)
    with torch.no_grad():
        direct = model(torch.stack((previous, current), dim=1))
    single_error = float((native_pair(model, previous, current)-direct).abs().max())
    irregular = native_pair(model, torch.zeros_like(frames[0]), frames[0])
    passed = (all(x['finite'] and x['max_identity_error'] < 1e-6 for x in spatial_checks)
              and recurrence_error < 1e-6 and single_error < 1e-6
              and list(irregular.shape) == [1, 3, 519, 773] and bool(torch.isfinite(irregular).all()))
    report = dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), passed=passed,
                  spatial_checks=spatial_checks, self_start_history_max_error=recurrence_error,
                  original_384_single_tile_max_error=single_error,
                  original_XS_irregular_output_shape=list(irregular.shape),
                  original_XS_irregular_finite=bool(torch.isfinite(irregular).all()),
                  native_1080p_tiles=len(tile_layout(1080,1920)),
                  native_1080p_conv_linear_macs=784862208*len(tile_layout(1080,1920)),
                  elapsed_seconds=time.monotonic()-started, cuda_initialized=torch.cuda.is_initialized(),
                  training_updates=0, quality_rows=0, new_video_pixels=0,
                  limits='Artificial CPU inputs and untrained author XS; no trained temporal or seam quality, no GPU/phone latency, no original-paper performance reproduction.')
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed or report['cuda_initialized']:
        raise SystemExit('Preserve failed qualification; no automatic retry.')


if __name__ == '__main__':
    main()
