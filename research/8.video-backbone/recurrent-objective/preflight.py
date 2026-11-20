"""Two actual training batches, native-forward and history-gradient checks only."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Registered qualification is CPU-only'

import time
import traceback
import torch
from study_io import ROOT, BASE, PROCESS, precision, verify, write, append, stamp, identity
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from recurrence import recursive


def main():
    precision()
    os.nice(5)
    frozen = verify(PROCESS / 'preflight-source-freeze001.json', 'preflight-protocol.md')
    out = PROCESS / 'preflight001'
    out.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    counts = dict(batches=0, backbone_images=0, DIS_pairs=0,
                  raw_tensor_gradient_evaluations=0, parameter_gradient_evaluations=0, optimizer_updates=0)
    write(out / 'config.json', dict(at=stamp(), process=identity(os.getpid()), source_freeze=frozen,
          seed=0, data_steps=[8000, 8001], B=4, T=4, crop=128, device='cpu',
          frozen_fields_only=True, full_DIS_derivative_claimed=False,
          independent_final_opened=False, new_DEV=False, wall_clock_cutoff=False))
    try:
        pool = MixedPool()
        model, provenance = load_spatial(0)
        model.train().requires_grad_(True)
        parameters = tuple(model.parameters())
        assert sum(p.numel() for p in parameters) == 4075
        write(out / 'data.json', pool.info)
        write(out / 'model.json', provenance)
        for step in (8000, 8001):
            low, gt, windows = pool.batch(0, step)
            assert low.shape == gt.shape == (4, 4, 3, 128, 128)
            raw = model(low.flatten(0, 1)).clamp(0, 1).unflatten(0, (4, 4))
            fused, details = recursive(raw, reference=True)
            assert details['DIS_pairs'] == 12 and details['fixed_field_native_max_abs_difference'] <= 2e-6
            assert torch.equal(fused[:, 0], raw[:, 0])
            past = {}
            for name, prediction in [('clip_gt', raw), ('rec_gt', fused)]:
                last_loss = (prediction[:, -1] - gt[:, -1]).square().mean()
                gradient = torch.autograd.grad(last_loss, raw, retain_graph=True)[0]
                assert torch.isfinite(gradient).all()
                past[name] = float(gradient[:, :-1].abs().sum())
                counts['raw_tensor_gradient_evaluations'] += 1
                loss = (prediction - gt).square().mean()
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
                assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
                counts['parameter_gradient_evaluations'] += 1
                append(out / 'objectives.jsonl', dict(at=stamp(), data_step=step, objective=name,
                       loss=float(loss.detach()), parameter_gradients_all_finite=True,
                       parameter_gradient_absolute_sum=sum(float(g.abs().sum()) for g in gradients),
                       last_frame_loss_past_raw_gradient_absolute_sum=past[name]))
            assert past['clip_gt'] == 0 and past['rec_gt'] > 0
            counts['batches'] += 1
            counts['backbone_images'] += 16
            counts['DIS_pairs'] += details['DIS_pairs']
            append(out / 'windows.jsonl', dict(data_step=step, windows=windows))
            append(out / 'native_and_history.jsonl', dict(data_step=step, **details,
                   first_frame_identical=True, raw_last_loss_past_gradient=past['clip_gt'],
                   rec_last_loss_past_gradient=past['rec_gt']))
            print(dict(data_step=step, **counts), flush=True)
            del raw, fused, loss, gradient, gradients, prediction, last_loss
        assert counts == dict(batches=2, backbone_images=32, DIS_pairs=24,
                  raw_tensor_gradient_evaluations=4, parameter_gradient_evaluations=4, optimizer_updates=0)
        verify(PROCESS / 'preflight-source-freeze001.json', 'preflight-protocol.md')
        assert not torch.cuda.is_initialized()
        write(out / 'summary.json', dict(at=stamp(), passed=True, **counts,
              allowed_next='Separate complete training/evaluation registration and qualifications only',
              performance_evidence=False, innovation_evidence=False, full_DIS_derivative_claimed=False))
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), process=identity(os.getpid()),
              completed_normally=complete, seconds=time.perf_counter()-started, counts=counts,
              CUDA_initialized=torch.cuda.is_initialized()))


if __name__ == '__main__':
    main()
