"""M1 semantic qualification and four discarded CPU optimizer updates only."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'M1 preflight is CPU-only'
from pathlib import Path
import sys
import json
import time
import math
import itertools
import traceback
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, append, digest, stamp
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from explore_temporal_controls import MobileIELLENetS
from core import MultiaffineCore
import torch
import cv2

OUT = BASE / 'process/cpu-preflight002'


def identity():
    v = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(v[19]), state=v[0])


def verify():
    value = json.loads((BASE / 'process/preflight-source-freeze002.json').read_text())
    assert all(digest(ROOT / p) == sha for p, sha in value['source_sha256'].items())
    return value


def structure():
    torch.manual_seed(89300)
    model = MultiaffineCore().double().eval()
    assert sum(p.numel() for p in model.parameters()) == 5340
    with torch.no_grad():
        model.first.gamma.fill_(.4)
        model.second.gamma.fill_(.3)
        images = .1 + .4 * torch.rand(1, 6, 3, 16, 16, dtype=torch.float64)
        stream_difference, sources = 0., []
        for t in range(6):
            actual, origin = model.step(images[:, t])
            expected = model(images[:, t]) if t < 3 else model.four(images[:, t-3:t+1])
            stream_difference = max(stream_difference, float((actual - expected).abs().max()))
            sources.append(dict(index=t, **origin))
        assert stream_difference <= 2e-10
        assert model.state_storage_bytes() == 24576
        mean_input = images[:, :4]
        delta = .25 * (2 * torch.rand_like(mean_input) - 1)
        sums = {False: torch.zeros(1, 3, 16, 16, dtype=torch.float64),
                True: torch.zeros(1, 3, 16, 16, dtype=torch.float64)}
        clipped_sum = torch.zeros_like(sums[False])
        for signs in itertools.product((-1., 1.), repeat=4):
            perturbation = torch.tensor(signs, dtype=torch.float64).reshape(1, 4, 1, 1, 1)
            value = mean_input + perturbation * delta
            for overlap in (False, True):
                output = model.four(value, overlap=overlap)
                sums[overlap] += output / 16
                if not overlap:
                    clipped_sum += output.clamp(0, 1) / 16
        base = model.four(mean_input)
        disjoint_defect = float((sums[False] - base).abs().max())
        overlap_defect = float((sums[True] - model.four(mean_input, overlap=True)).abs().max())
        clipped_defect = float((clipped_sum - base.clamp(0, 1)).abs().max())
        assert disjoint_defect <= 2e-10 and overlap_defect > 1e-8
    return dict(passed=True, fixture_seed=89300, image_size=[16, 16], dtype='float64',
        nonzero_gamma=[.4, .3], stream_steps=6, stream_max_abs_difference=stream_difference,
        source_sets=sources, actual_steady_state_bytes=24576,
        Rademacher_corners=16, disjoint_raw_expectation_max_abs_defect=disjoint_defect,
        overlapping_raw_expectation_max_abs_defect=overlap_defect,
        final_clipped_expectation_max_abs_defect=clipped_defect,
        structural_calls=dict(four_slot=37, direct_diagonal=3, stream_steps=6),
        B1_FP32_budget_128=model.budget(128, 128), B1_FP32_budget_1080=model.budget(1080, 1920),
        pretrained_function_inherited=False, quality_or_motion_evidence=False,
        scope='Two fixed semantic checks, not Monte Carlo generalization or bit-level optimizer replay')


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    os.nice(5)
    frozen = verify()
    OUT.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    counts = dict(teacher_images=0, student_images=0, discarded_optimizer_updates=0)
    write(OUT / 'config.json', dict(at=stamp(), process=identity(), source_freeze=frozen,
        device='cpu', seed=0, data_step=8000, B=4, T=4, crop=128,
        independent_final_opened=False, new_DEV=False, new_GPU=False, wall_clock_cutoff=False))
    try:
        write(OUT / 'structure.json', structure())
        pool = MixedPool()
        low, gt, windows = pool.batch(0, 8000)
        write(OUT / 'data.json', dict(info=pool.info, windows=windows, GT_not_in_loss=True))
        teacher, provenance = load_spatial(0)
        with torch.inference_mode():
            target = teacher(low.flatten(0, 1)).clamp(0, 1)
        # A normal detached tensor can be saved by autograd's MSE backward.
        target = target.clone()
        counts['teacher_images'] = 16
        write(OUT / 'teacher.json', provenance)
        torch.manual_seed(8930)
        polynomial = MultiaffineCore()
        torch.manual_seed(8931)
        original = MobileIELLENetS(12)
        models = dict(multiaffine=polynomial, original_scratch=original)
        assert [sum(p.numel() for p in m.parameters()) for m in models.values()] == [5340, 4075]
        for name, model in models.items():
            model.train().requires_grad_(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(low.flatten(0, 1))
                loss = (prediction - target).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                parameters = tuple(model.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1))
                assert math.isfinite(norm)
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in parameters)
                counts['student_images'] += 16
                counts['discarded_optimizer_updates'] += 1
                append(OUT / 'discarded-learning.jsonl', dict(at=stamp(), arm=name, step=step+1,
                    parameters=sum(p.numel() for p in parameters), raw_student_to_clipped_teacher_MSE=float(loss.detach()),
                    gradient_norm_before_clip=norm, all_gradients_and_parameters_finite=True))
        assert counts == dict(teacher_images=16, student_images=64, discarded_optimizer_updates=4)
        assert not any(p.requires_grad for p in teacher.parameters())
        assert not torch.cuda.is_initialized()
        verify()
        write(OUT / 'qualification.json', dict(at=stamp(), passed=True, all_learning_updates_discarded=True,
            model_weights_exported=False, quality_evidence=False, established_innovation=False, **counts,
            allowed_next='Separate spatial expressivity/recovery protocol and necessary GPU qualification only; no automatic video training'))
        complete = True
    except BaseException:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(OUT / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
            seconds=time.perf_counter() - started, CUDA_initialized=torch.cuda.is_initialized(), **counts))


if __name__ == '__main__':
    main()
