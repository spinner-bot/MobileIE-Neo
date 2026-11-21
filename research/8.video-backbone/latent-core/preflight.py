"""One registered CPU qualification; discard every update and do not export weights."""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import time
import traceback

from l1_io import BASE, append, identity, stamp, verify, write
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from core import Streaming, models, optimizer, sequence
import cv2
import torch

OUT = BASE / "process/preflight001"


def gradient_norm(module):
    return sum(float(p.grad.square().sum()) for p in module.parameters()) ** .5


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    os.nice(5)
    frozen = verify()
    OUT.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    counts = dict(F0_images=0, student_learning_images=0, student_check_images=0,
                  discarded_optimizer_updates=0, formal_optimizer_updates=0)
    write(OUT / "config.json", dict(at=stamp(), process=identity(), source_freeze=frozen,
          device="cpu", seed_steps=[[0, 8000], [1, 8000]], B=4, T=4, crop=128,
          formal_training=False, all_qualification_updates_discarded=True,
          new_DEV=False, independent_final_opened=False, wall_clock_cutoff=False))
    try:
        pool = MixedPool()
        write(OUT / "data.json", pool.info)
        checks = []
        for seed in (0, 1):
            low, gt, metadata = pool.batch(seed, 8000)
            append(OUT / "windows.jsonl", dict(seed=seed, data_step=8000, windows=metadata))
            original, provenance = load_spatial(seed)
            with torch.inference_mode():
                anchor = torch.stack([original(frame) for frame in low.unbind(1)], 1)
            counts["F0_images"] += 16
            students = models(original, seed)
            assert all(torch.equal(v, students["causal"].state_dict()[k])
                       for k, v in students["spatial"].state_dict().items())
            for arm, model in students.items():
                route = "current" if arm == "spatial" else "causal"
                model.train()
                opt = optimizer(model)
                assert sum(p.numel() for p in model.parameters()) == 12431
                new_kernel_weights = sum(m.weight.numel() for name, m in model.named_modules()
                                         if not name.startswith("backbone") and isinstance(m, torch.nn.Conv2d))
                assert new_kernel_weights == 8064
                for step in range(2):
                    opt.zero_grad(set_to_none=True)
                    raw, state = sequence(model, low, route)
                    counts["student_learning_images"] += 16
                    if step == 0:
                        error = float((raw.detach() - anchor).abs().max())
                        assert error <= 1e-5
                    loss = (raw - gt).square().mean()
                    assert torch.isfinite(raw).all() and torch.isfinite(loss)
                    assert torch.isfinite(state).all() and float(state.detach().abs().max()) <= 1 + 1e-6
                    loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    norms = {name: gradient_norm(model.get_submodule(name)) for name in
                             ["backbone.head.block1", "project", "gates", "candidate", "readout"]}
                    if step == 1:
                        assert all(v > 0 for v in norms.values())
                    total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
                    opt.step()
                    counts["discarded_optimizer_updates"] += 1
                    assert all(torch.isfinite(p).all() for p in model.parameters())
                    append(OUT / "learning.jsonl", dict(at=stamp(), seed=seed, arm=arm,
                           discarded_update=step + 1, GT_MSE=float(loss.detach()),
                           gradient_norms=norms, total_norm_before_clip=total_norm,
                           initial_F0_max_difference=error, counts=counts.copy()))
                model.eval()
                clip = low[:1]
                with torch.inference_mode():
                    wanted, state = sequence(model, clip, route)
                    counts["student_check_images"] += 4
                    stream = Streaming(model, route)
                    streamed = torch.stack([stream.step(x) for x in clip.unbind(1)], 1)
                    counts["student_check_images"] += 4
                    stream_error = float((wanted - streamed).abs().max())
                    assert stream_error <= 2e-6
                    assert stream.state_bytes() == 131072
                    assert stream.state.is_contiguous() and stream.state.untyped_storage().data_ptr() != clip.untyped_storage().data_ptr()
                    assert torch.isfinite(stream.state).all() and float(stream.state.abs().max()) <= 1 + 1e-6
                    current, _ = sequence(model, clip, "current")
                    counts["student_check_images"] += 4
                    assert torch.equal(wanted[:, 0], current[:, 0])
                    history_effect = float((wanted[:, -1] - current[:, -1]).abs().max())
                    if arm == "causal":
                        assert history_effect > 1e-8
                    else:
                        assert history_effect == 0
                    stream.reset()
                    odd = stream.step(clip[:, 0, :, :11, :15])
                    counts["student_check_images"] += 1
                    assert odd.shape == (1, 3, 11, 15) and torch.isfinite(odd).all()
                    assert stream.state.shape == (1, 32, 3, 4) and stream.state_bytes() == 1536
                inputs = clip.detach().clone().requires_grad_(True)
                final, _ = sequence(model, inputs, route)
                counts["student_check_images"] += 4
                gradient, = torch.autograd.grad(final[:, -1].sum(), inputs)
                past = float(gradient[:, :-1].abs().sum())
                assert torch.isfinite(gradient).all()
                assert past > 1e-8 if arm == "causal" else past == 0
                checked = dict(seed=seed, arm=arm, source=provenance, initial_F0_max_difference=error,
                               functional_stream_max_difference=stream_error, state_bytes_128=131072,
                               odd_state_bytes=1536, current_substitution_max_difference=history_effect,
                               last_output_past_input_gradient_L1=past, passed=True)
                checks.append(checked)
                append(OUT / "checks.jsonl", checked)
                print(dict(seed=seed, arm=arm, counts=counts), flush=True)
        assert counts == dict(F0_images=32, student_learning_images=128, student_check_images=68,
                              discarded_optimizer_updates=8, formal_optimizer_updates=0)
        assert not torch.cuda.is_initialized()
        verify()
        write(OUT / "qualification.json", dict(at=stamp(), passed=True, checks=checks, counts=counts,
              parameters=12431, analytical_1080p_Conv_GMAC=(3828*1080*1920 + 8064*270*480 + 144)/1e9,
              analytical_1080p_persistent_state_bytes=4*32*270*480,
              no_qualification_weights_exported=True, allowed_formal_protocol_design=True,
              established_method_or_innovation=False))
        complete = True
    except BaseException:
        (OUT / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        write(OUT / "runtime.json", dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter()-started, counts=counts, CUDA_initialized=torch.cuda.is_initialized(),
              qualification_updates_discarded=True, weights_exported=False))


if __name__ == "__main__":
    main()
