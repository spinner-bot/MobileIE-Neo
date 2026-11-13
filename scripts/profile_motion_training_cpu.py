"""Two fixed CPU training batches solely for R74 execution-resource feasibility."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research/4.temporal-consistency/scripts'))
import cv2
import torch
from motion_transport_r74 import BASE, make_models, load_spatial, frozen_raw, teacher_proxy, sequence, proxy_loss
from mixed_anchor_r40 import MixedPool
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r74/cpu-throughput001'; out.mkdir(exist_ok=False)
    start, complete, updates, measurements = time.perf_counter(), False, 0, []
    frozen_path = BASE/'process/exploration-r74/finalize001/config.json'
    frozen = json.loads(frozen_path.read_text())['source_sha256']
    try:
        assert all(digest(ROOT/n) == sha for n, sha in frozen.items())
        began = time.perf_counter(); pool = MixedPool(); spatial, source = load_spatial(0)
        nets = make_models(0); low, gt, context = pool.batch(0, 8000)
        optimizers = {name: torch.optim.Adam(net.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
                      for name, net in nets.items()}
        setup_seconds = time.perf_counter()-began
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=0, data_step=8000,
              repeated_first_window=True, context=context, spatial_source=source, torch_threads=2, OpenCV_threads=1,
              discarded_updates_expected=4, protocol_sha256=digest(BASE/'docs/exploration-r74-training-resource-probe.md'),
              original_frozen_config_sha256=digest(frozen_path), core_sha256=digest(BASE/'scripts/motion_transport_r74.py'),
              scope='Training throughput only, not another numerical qualification or inference/phone cost. No weight inherited.'))
        for step in range(2):
            began = time.perf_counter(); target = teacher_proxy(gt); raw = frozen_raw(spatial, low)
            diagnostics = []
            for name, net in nets.items():
                optimizer = optimizers[name]; optimizer.zero_grad(set_to_none=True)
                output, fields = sequence(net, raw)
                reconstruction = (output.clamp(0, 1)-gt).square().mean()
                proxy, stats = proxy_loss(fields, target)
                loss = reconstruction if name == 'task' else proxy
                loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
                optimizer.step(); updates += 1
                diagnostics.append(dict(model=name, loss=float(loss.detach()), reconstruction_MSE=float(reconstruction.detach()),
                      gradient_norm_before_clip=norm,
                      native_flow_abs_mean=float(torch.stack([p[0].detach().abs().mean() for p in fields]).mean()),
                      history_coefficient_mean=float(.5*torch.stack([p[1].detach().sigmoid().mean() for p in fields]).mean()), **stats))
            measurements.append(dict(batch_iteration=step+1, complete_batch_seconds=time.perf_counter()-began, diagnostics=diagnostics))
        assert updates == 4 and not torch.cuda.is_initialized()
        maximum = max(r['complete_batch_seconds'] for r in measurements)
        write(out/'result.json', dict(completed=True, setup_seconds=setup_seconds, measurements=measurements,
              max_batch_seconds=maximum, CPU_execution_initially_feasible=maximum <= .5,
              estimated8000_batches_seconds=8000*maximum, no_wall_cutoff=True,
              limitations='Only two repeats of first training window; excludes resampling, checkpoints, setup and load variation. Not guaranteed whole-run time.',
              original41_sources_unchanged=all(digest(ROOT/n) == sha for n, sha in frozen.items())))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, discarded_motion_updates=updates,
              frozen_spatial_center_predictions=16*len(measurements), formal_updates=0, DEV_rows=0,
              CUDA_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
