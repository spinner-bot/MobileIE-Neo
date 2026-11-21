"""Finite CPU T8 checks of the two published simple levels; updates are discarded."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import time
import traceback

import torch

from simple_io import BASE, identity, stamp, verify_preflight, write
from simple_core import make_models, objective, sequence
from motion_transport_r74 import load_spatial


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    os.nice(5)
    frozen = verify_preflight()
    out = BASE / 'process/simple-preflight001'
    out.mkdir(parents=True, exist_ok=False)
    began, complete = time.perf_counter(), False
    models = {}
    counts = dict(original_F0_images=0, adapted_backbone_images=0, discarded_optimizer_updates=0,
                  real_frames_loaded=0, flow_calls=0, GPU_used=False)
    try:
        torch.manual_seed(9300)
        original, provenance = load_spatial(0)
        original.eval()
        frames = torch.rand(1, 8, 3, 24, 32)
        target = torch.rand_like(frames)
        with torch.no_grad():
            raw = torch.stack([original(frames[:, t]) for t in range(8)], dim=1)
        counts['original_F0_images'] = 8
        models = make_models(original)
        result = {}
        for level, model in models.items():
            before = {n: p.detach().clone() for n, p in model.original_parameters().items()}
            with torch.no_grad():
                initial = sequence(model, frames)
                first = float((initial[:, 0] - raw[:, 0]).abs().max())
                replay = float((sequence(model, frames) - initial).abs().max())
                future = frames.clone()
                future[:, 4:] = 1 - future[:, 4:]
                prefix = float((sequence(model, future)[:, :4] - initial[:, :4]).abs().max())
                model.enable(False)
                disabled = float((sequence(model, frames) - raw).abs().max())
                model.enable()
                repeated = sequence(model, frames[:, :1].repeat(1, 8, 1, 1, 1))
                repeated_error = float((repeated - raw[:, :1]).abs().max())
                assert torch.allclose(repeated, raw[:, :1].expand_as(repeated), atol=1e-5, rtol=1e-5)
            assert (first, replay, prefix, disabled) == (0., 0., 0., 0.)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=1e-3)
            start = [p.detach().clone() for p in trainable]
            updates = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(sequence(model, frames), target)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
                assert any((p.grad != 0).any() for p in trainable)
                assert all(p.grad is None for p in model.original_parameters().values())
                optimizer.step()
                counts['discarded_optimizer_updates'] += 1
                assert all(torch.isfinite(p).all() for p in model.parameters())
                updates.append(float(loss.detach()))
            assert all(torch.equal(before[n], p) for n, p in model.original_parameters().items())
            changed = sum(not torch.equal(a, b) for a, b in zip(start, trainable))
            assert changed > 0
            with torch.no_grad():
                sequence(model, frames)
            result[level] = dict(passed=True, first_frame_max_difference=first,
                disabled_max_difference=disabled, reset_max_difference=replay,
                future_change_prefix_max_difference=prefix, repeated_current_max_difference=repeated_error,
                original_parameters_unchanged=True, original_gradients_none=True,
                updated_parameter_tensors=changed, discarded_losses=updates,
                budget=model.state_inventory(), actual_adapted_frame_calls=model.frame_calls)
            model.reset()
        verify_preflight()
        write(out / 'qualification.json', dict(at=stamp(), passed=True, source_freeze=frozen,
              original_F0_provenance=provenance, variants=result, temporal_length=8,
              all_weights_and_Adam_discarded=True, formal_training_started=False,
              real_video_quality_and_phone_speed_claimed=False))
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        counts['adapted_backbone_images'] = sum(m.frame_calls for m in models.values())
        write(out / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter()-began, **counts))


if __name__ == '__main__':
    main()
