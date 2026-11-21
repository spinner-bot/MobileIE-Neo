"""Complete fixed L1 training-scene draws, including full-network same-frame self4."""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import argparse
import hashlib
import time
import traceback
from training_io import BASE, ROOT, append, digest, identity, owned_cache, read, stamp, verify_training, write
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from half_motion_r4 import half_forward_backward
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from exploration_r2_data import evaluate
from core import models, sequence
import cv2
import numpy as np
import torch

ARMS = ("F0", "F0_FB", "F0_DIS", "spatial_current", "spatial_FB", "spatial_DIS",
        "causal", "causal_current", "causal_self4", "causal_current_FB", "causal_current_DIS")


def finals(seed):
    runs = sorted((BASE / "process").glob(f"train-seed{seed}-attempt*/runtime.json"))
    assert runs and read(runs[-1])["completed_normally"]
    assert read(runs[-1])["counts"]["common_steps"] == 8000
    result = {}
    for arm in ("spatial", "causal"):
        marker = read(runs[-1].parent / (arm + "-final.json"))
        assert marker["seed"] == seed and marker["updates"] == 8000 and marker["unique_final"] and not marker["qualification"]
        path = ROOT / marker["path"]
        assert path.resolve().is_relative_to("/data/Users/zrh/data/MobileIE-Neo") and digest(path) == marker["sha256"]
        result[arm] = marker
    return result


def as_image(tensor):
    return tensor.clamp(0, 1).numpy().transpose(0, 2, 3, 1).copy() * 255


def main(seed):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    os.nice(5)
    frozen, markers = verify_training(), finals(seed)
    out, cache = BASE / f"results/held-seed{seed}", owned_cache(f"held-seed{seed}")
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    began, complete = time.perf_counter(), False
    counts = dict(F0_images=0, student_images=0, windows=0, output_rows=0, frame_rows=0,
                  classical_deployment_bidirectional_pairs=0, GT_metric_FB_pairs=0)
    write(out / "config.json", dict(at=stamp(), process=identity(), seed=seed, source_freeze=frozen,
          unique_finals=markers, sampling_seed=9150, steps=list(range(20000, 20008)), arms=ARMS,
          RGB_PSNR_ceiling_dB=120, held_out_draws_not_scenes=True, new_DEV=False, independent_final_opened=False,
          self4_full_backbone_calls_per_frame=4, candidate_causal_full_backbone_calls_per_frame=1))
    try:
        original, source = load_spatial(seed)
        students = models(original, seed)
        for arm, model in students.items():
            model.load_state_dict(torch.load(ROOT / markers[arm]["path"], map_location="cpu", weights_only=True), strict=True)
            model.eval().requires_grad_(False)
        write(out / "F0.json", source)
        pool = MixedPool()
        write(out / "data.json", pool.info)
        with torch.inference_mode():
            for step in range(20000, 20008):
                low, gt, metadata = pool.batch(9150, step)
                append(out / "windows.jsonl", dict(data_step=step, windows=metadata))
                f0 = original(low.flatten(0, 1)).unflatten(0, (4, 4))
                counts["F0_images"] += 16
                spatial, _ = sequence(students["spatial"], low, "current")
                causal, _ = sequence(students["causal"], low, "causal")
                current, _ = sequence(students["causal"], low, "current")
                counts["student_images"] += 48
                refined = []
                for image in low.unbind(1):
                    state = None
                    for _ in range(4):
                        value, state = students["causal"](image, state, "causal")
                        counts["student_images"] += 4
                    refined.append(value)
                self4 = torch.stack(refined, 1)
                assert torch.equal(causal[:, 0], current[:, 0])
                for index in range(4):
                    key, draw_key = f"seed{seed}-step{step}-window{index}", f"step{step}-window{index}"
                    domain = "SYN" if index < 2 else "ND"
                    raw = dict(F0=f0[index], spatial_current=spatial[index], causal=causal[index],
                               causal_current=current[index], causal_self4=self4[index])
                    values = {arm: as_image(value) for arm, value in raw.items()}
                    for direct, prefix in [("F0", "F0"), ("spatial_current", "spatial"), ("causal_current", "causal_current")]:
                        for suffix, estimator in [("FB", half_forward_backward), ("DIS", FixedDIS())]:
                            pipeline = RecursivePipeline(estimator)
                            values[prefix + "_" + suffix] = np.stack([pipeline.step(frame)[0] for frame in values[direct]])
                            counts["classical_deployment_bidirectional_pairs"] += 3
                            assert np.array_equal(values[prefix + "_" + suffix][0], values[direct][0])
                    assert set(values) == set(ARMS)
                    target = gt[index].numpy().transpose(0, 2, 3, 1).copy() * 255
                    reference = [half_forward_backward(target[t - 1], target[t]) for t in range(1, 4)]
                    counts["GT_metric_FB_pairs"] += 3
                    path = cache / (key + ".npz")
                    np.savez_compressed(path, gt=target, low=low[index].numpy().transpose(0, 2, 3, 1) * 255, **values)
                    append(out / "prediction_manifest.jsonl", dict(key=key, draw_key=draw_key, seed=seed, domain=domain,
                           path=str(path.relative_to(ROOT)), sha256=digest(path),
                           low_sha256=hashlib.sha256(low[index].numpy().tobytes()).hexdigest(),
                           GT_sha256=hashlib.sha256(gt[index].numpy().tobytes()).hexdigest()))
                    append(out / "raw-ranges.jsonl", dict(key=key,
                           values={arm: dict(min=float(v.min()), max=float(v.max()),
                                   outside_fraction=float(((v < 0) | (v > 1)).float().mean())) for arm, v in raw.items()}))
                    for arm in ARMS:
                        output = values[arm]
                        assert np.isfinite(output).all()
                        metrics, frames = evaluate(output, target, [p[0] for p in reference], [p[1] for p in reference])
                        assert all(r["temporal_error"] is not None for r in frames[1:]), "No empty-domain omissions"
                        for frame in frames:
                            frame["PSNR_ceiling_applied"] = frame["psnr_rgb"] >= 120
                            frame["psnr_rgb"] = min(frame["psnr_rgb"], 120.)
                        metrics["psnr_rgb"] = float(np.mean([r["psnr_rgb"] for r in frames]))
                        assert all(np.isfinite(v) for v in metrics.values())
                        append(out / "per_sequence.jsonl", dict(key=key, seed=seed, domain=domain, arm=arm, **metrics))
                        for frame in frames:
                            append(out / "per_frame.jsonl", dict(key=key, seed=seed, domain=domain, arm=arm, **frame))
                        counts["output_rows"] += 1
                        counts["frame_rows"] += 4
                    counts["windows"] += 1
                print(dict(seed=seed, data_step=step, **counts), flush=True)
        assert counts == dict(F0_images=128, student_images=896, windows=32, output_rows=352, frame_rows=1408,
                              classical_deployment_bidirectional_pairs=576, GT_metric_FB_pairs=96)
        verify_training()
        assert not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        write(out / "runtime.json", dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter() - began, CUDA_initialized=torch.cuda.is_initialized(), **counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    main(parser.parse_args().seed)
