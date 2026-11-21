"""Registered A1 cached-F0 necessary test; no CNN, Torch, or training is loaded."""
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "2"
import hashlib
import time
import traceback
from a1_io import BASE, ROOT, SOURCE, append, digest, identity, lines, owned_cache, read, stamp, verify, write
import cv2
import numpy as np
from core import AffineStream, fit, render, sequence
from half_motion_r4 import half_forward_backward
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from exploration_r2_data import evaluate

ARMS = ("F0", "F0_FB", "F0_DIS", "current", "current_FB", "current_DIS",
        "cached", "cached_FB", "cached_DIS", "hold", "hold_FB", "hold_DIS")
METRICS = ("psnr_rgb", "ssim_rgb", "edge_mae", "flicker", "temporal_error")


def qualify():
    rng = np.random.default_rng(8200)
    low = rng.uniform(0, .2, (4, 128, 128, 3)).astype(np.float32)
    target = low * 2 + .1
    identity_field = np.zeros((32, 32, 12), np.float32)
    identity_field[..., :9] = np.eye(3, dtype=np.float32).ravel()
    identity_error = float(np.max(np.abs(render(low[0], identity_field) - low[0])))
    constant = np.full_like(low[0], .1)
    constant_error = float(np.max(np.abs(render(constant, fit(constant, constant * 2 + .1)) - .3)))
    assert identity_error <= 1e-6 and constant_error <= 1e-6
    outputs, routes, states = {}, {}, {}
    for mode in ("current", "cached", "hold"):
        outputs[mode], routes[mode], states[mode] = sequence(low, target, mode)
        assert routes[mode] == ([0, 1, 2, 3] if mode == "current" else [0, 2])
    assert all(np.array_equal(outputs["current"][0], v[0]) for v in outputs.values())
    probe = AffineStream("cached", lambda i: target[i])
    probe.step(low[0])
    assert probe.field.flags.owndata and probe.field.flags.c_contiguous
    assert probe.state_bytes() == states["cached"] == 49152
    difference = float(np.max(np.abs(outputs["cached"][1:] - outputs["hold"][1:])))
    assert difference > 0
    return dict(passed=True, artificial_seed=8200, identity_render_max_error=identity_error,
                constant_fit_render_max_error=constant_error, routes=routes, state_bytes=states,
                cached_field_owns_memory=True, cached_field_C_contiguous=True,
                cached_hold_later_max_difference=difference, CNN_calls=0, optimization_steps=0)


def summarize(rows, out):
    means, comparisons = [], []
    for seed in (0, 1):
        for domain in ("SYN", "ND"):
            group = {}
            for arm in ARMS:
                selected = [r for r in rows if (r["seed"], r["domain"], r["arm"]) == (seed, domain, arm)]
                group[arm] = {r["key"]: r for r in selected}
                assert len(selected) == len(group[arm]) == 16
                means.append(dict(seed=seed, domain=domain, arm=arm, windows=16,
                                  **{m: float(np.mean([r[m] for r in selected])) for m in METRICS}))
            current = {r["arm"]: r for r in means if (r["seed"], r["domain"]) == (seed, domain)}
            scheduled = [("representation", "current", "F0", None)]
            scheduled += [("strong", "cached_" + post, ref + "_" + post, 1.02)
                          for post in ("FB", "DIS") for ref in ("F0", "current")]
            for category, candidate, reference, limit in scheduled:
                assert group[candidate].keys() == group[reference].keys()
                delta = current[candidate]["psnr_rgb"] - current[reference]["psnr_rgb"]
                worst = min(group[candidate][k]["psnr_rgb"] - group[reference][k]["psnr_rgb"] for k in group[candidate])
                ratio = current[candidate]["temporal_error"] / current[reference]["temporal_error"]
                comparisons.append(dict(seed=seed, domain=domain, category=category,
                    candidate=candidate, reference=reference, PSNR_delta=delta,
                    worst_window_PSNR_delta=worst, Etemp_ratio=ratio, Etemp_limit=limit,
                    mean_quality_passed=delta >= -.05, worst_quality_passed=worst >= -.5,
                    passed=delta >= -.05 and worst >= -.5 and (limit is None or ratio <= limit)))
    assert len(means) == 48 and len(comparisons) == 20
    passed = all(r["passed"] for r in comparisons)
    write(out / "means.json", means)
    write(out / "gates.json", dict(at=stamp(), comparisons=comparisons, complete=passed,
          allowed_separate_cost_registration=passed, allowed_DEV_or_external=False,
          actual_new_CNN_calls=0, optimization_steps=0, established_method_or_innovation=False,
          held_out_draws_not_scenes=True, independent_final=False))
    text = ["# A1 全部缓存筛查结果", "", "32唯一训练场景抽样窗×两个原F0 seed；0新CNN/优化，非部署成本。", "",
            "|域|seed|输出|PSNR|SSIM|edge|Etemp|", "|---|---:|---|---:|---:|---:|---:|"]
    for r in means:
        text.append(f"|{r['domain']}|{r['seed']}|{r['arm']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['edge_mae']:.6f}|{r['temporal_error']:.6f}|")
    text += ["", "|域|seed|门|候选|参考|PSNR差|最差窗差|Etemp比|通过|", "|---|---:|---|---|---|---:|---:|---:|---|"]
    for r in comparisons:
        text.append(f"|{r['domain']}|{r['seed']}|{r['category']}|{r['candidate']}|{r['reference']}|{r['PSNR_delta']:.6f}|{r['worst_window_PSNR_delta']:.6f}|{r['Etemp_ratio']:.6f}|{r['passed']}|")
    text += ["", f"完整通过={passed}；只可能允许另登记真实整链成本，未建立创新/外部/手机结论。"]
    (out / "tables.md").write_text("\n".join(text) + "\n")


def main():
    cv2.setNumThreads(1)
    os.nice(5)
    frozen = verify()
    out, cache = BASE / "results/screen001", owned_cache()
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    began, complete = time.perf_counter(), False
    counts = dict(windows=0, output_rows=0, frame_rows=0, classical_deployment_bidirectional_pairs=0,
                  GT_metric_FB_pairs=0, actual_new_CNN_calls=0, optimization_steps=0,
                  logical_current_F0_requests=0, logical_cached_F0_requests=0, logical_hold_F0_requests=0)
    write(out / "config.json", dict(at=stamp(), process=identity(), source_freeze=frozen, arms=ARMS,
          RGB_PSNR_ceiling_dB=120, CPU_threads=2, OpenCV_threads=1, GPU_used=False,
          actual_new_CNN_calls=0, logical_budget_is_not_measured_cost=True))
    try:
        write(out / "qualification.json", qualify())
        manifests, original = [], {}
        for seed in (0, 1):
            source = SOURCE / f"results/held-seed{seed}"
            assert read(source / "runtime.json")["completed_normally"]
            manifest = lines(source / "prediction_manifest.jsonl")
            assert len(manifest) == 32 and all(r["seed"] == seed for r in manifest)
            manifests.append(manifest)
            for r in lines(source / "per_sequence.jsonl"):
                if r["arm"] in ("F0", "F0_FB", "F0_DIS"):
                    original[r["key"], r["arm"]] = r
        identities = [{r["draw_key"]: (r["domain"], r["low_sha256"], r["GT_sha256"]) for r in m} for m in manifests]
        assert identities[0] == identities[1] and len(identities[0]) == 32 and len(original) == 192
        gt_fields, source_checks, rows, alias_max = {}, [], [], 0.
        for manifest in manifests:
            for item in manifest:
                path = ROOT / item["path"]
                assert path.resolve().is_relative_to("/data/Users/zrh/data/MobileIE-Neo")
                assert digest(path) == item["sha256"]
                with np.load(path, allow_pickle=False) as bundle:
                    low, target = bundle["low"], bundle["gt"]
                    values = {arm: bundle[arm] for arm in ("F0", "F0_FB", "F0_DIS")}
                assert low.shape == target.shape == values["F0"].shape == (4, 128, 128, 3)
                assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in (low, target, *values.values()))
                draw = item["draw_key"]
                target_hash = hashlib.sha256(target.tobytes()).hexdigest()
                if draw not in gt_fields:
                    reference = [half_forward_backward(target[t-1], target[t]) for t in range(1, 4)]
                    gt_fields[draw] = target_hash, reference
                    counts["GT_metric_FB_pairs"] += 3
                else:
                    assert gt_fields[draw][0] == target_hash
                    reference = gt_fields[draw][1]
                ranges, routes, states = {}, {}, {}
                for mode in ("current", "cached", "hold"):
                    raw, requests, state = sequence(low / 255, values["F0"] / 255, mode)
                    assert requests == ([0, 1, 2, 3] if mode == "current" else [0, 2])
                    routes[mode], states[mode] = requests, state
                    counts["logical_" + mode + "_F0_requests"] += len(requests)
                    ranges[mode] = dict(min=float(raw.min()), max=float(raw.max()),
                                        outside_fraction=float(((raw < 0) | (raw > 1)).mean()))
                    values[mode] = np.clip(raw, 0, 1) * 255
                    for suffix, estimator in (("FB", half_forward_backward), ("DIS", FixedDIS())):
                        pipeline = RecursivePipeline(estimator)
                        values[mode + "_" + suffix] = np.stack([pipeline.step(frame)[0] for frame in values[mode]])
                        counts["classical_deployment_bidirectional_pairs"] += 3
                        assert np.array_equal(values[mode + "_" + suffix][0], values[mode][0])
                assert all(np.array_equal(values["current"][0], values[a][0]) for a in ("cached", "hold"))
                assert set(values) == set(ARMS)
                saved = cache / (item["key"] + ".npz")
                np.savez_compressed(saved, low=low, gt=target, **values)
                append(out / "prediction_manifest.jsonl", dict(**{k: item[k] for k in ("key", "draw_key", "seed", "domain")},
                      path=str(saved.relative_to(ROOT)), sha256=digest(saved), source_path=item["path"], source_sha256=item["sha256"]))
                append(out / "routes.jsonl", dict(key=item["key"], logical_F0_requests=routes, persistent_state_bytes=states))
                append(out / "raw-ranges.jsonl", dict(key=item["key"], values=ranges))
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
                    row = dict(**{k: item[k] for k in ("key", "seed", "domain")}, arm=arm, **metrics)
                    rows.append(row)
                    append(out / "per_sequence.jsonl", row)
                    for frame in frames:
                        append(out / "per_frame.jsonl", dict(**{k: item[k] for k in ("key", "seed", "domain")}, arm=arm, **frame))
                    if arm.startswith("F0"):
                        difference = max(abs(metrics[m] - original[item["key"], arm][m]) for m in METRICS)
                        alias_max = max(alias_max, difference)
                        assert difference <= 1e-6
                    counts["output_rows"] += 1
                    counts["frame_rows"] += 4
                source_checks.append(dict(key=item["key"], source_sha256=item["sha256"], cached_GT_byte_sha256=target_hash))
                counts["windows"] += 1
                if counts["windows"] % 8 == 0:
                    print(dict(at=stamp(), seconds=time.perf_counter()-began, **counts), flush=True)
        assert counts == dict(windows=64, output_rows=768, frame_rows=3072,
            classical_deployment_bidirectional_pairs=1152, GT_metric_FB_pairs=96,
            actual_new_CNN_calls=0, optimization_steps=0, logical_current_F0_requests=256,
            logical_cached_F0_requests=128, logical_hold_F0_requests=128)
        write(out / "source-checks.json", dict(rows=source_checks, F0_baseline_rows=192,
              maximum_metric_difference=alias_max, same_32_GT_bytes_across_seeds=True,
              L1_student_arrays_accessed=False, final_or_new_DEV_opened=False))
        summarize(rows, out)
        verify()
        complete = True
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        write(out / "runtime.json", dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter()-began, GPU_used=False, **counts))


if __name__ == "__main__":
    main()
