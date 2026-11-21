"""All 44 means and preregistered unique comparisons; no new model inference."""
import json
from training_io import BASE, read, stamp, verify_training, write
from evaluate import ARMS


def main():
    verify_training()
    rows, manifests = [], []
    for seed in (0, 1):
        source = BASE / f"results/held-seed{seed}"
        assert read(source / "runtime.json")["completed_normally"]
        rows += [json.loads(line) for line in (source / "per_sequence.jsonl").read_text().splitlines()]
        manifests.append([json.loads(line) for line in (source / "prediction_manifest.jsonl").read_text().splitlines()])
    assert len(rows) == 704 and all(len(m) == 32 for m in manifests)
    identities = [{r["draw_key"]: (r["domain"], r["low_sha256"], r["GT_sha256"]) for r in m} for m in manifests]
    assert identities[0] == identities[1] and len(identities[0]) == 32
    assert len({r[1] for r in identities[0].values()}) == 32
    assert (BASE / "results/held-seed0/windows.jsonl").read_text() == (BASE / "results/held-seed1/windows.jsonl").read_text()
    means, comparisons = [], []
    for seed in (0, 1):
        for domain in ("SYN", "ND"):
            grouped = {}
            for arm in ARMS:
                selected = [r for r in rows if (r["seed"], r["domain"], r["arm"]) == (seed, domain, arm)]
                assert len(selected) == 16
                grouped[arm] = {r["key"]: r for r in selected}
                assert len(grouped[arm]) == 16
                means.append(dict(seed=seed, domain=domain, arm=arm, windows=16,
                             **{m: sum(r[m] for r in selected)/16 for m in ("psnr_rgb", "ssim_rgb", "edge_mae", "temporal_error")}))
            current = {r["arm"]: r for r in means if (r["seed"], r["domain"]) == (seed, domain)}
            scheduled = [("space", arm, "F0", None, None) for arm in ("spatial_current", "causal_current")]
            scheduled += [("history", "causal", ref, limit, limit) for ref, limit in
                          [("spatial_current", .95), ("causal_current", .95), ("causal_self4", .97)]]
            scheduled += [("strong", "causal", ref, .90, 1.02) for ref in
                          ("F0_FB", "F0_DIS", "spatial_FB", "spatial_DIS", "causal_current_FB", "causal_current_DIS")]
            for category, arm, ref, limit_a, limit_b in scheduled:
                assert set(grouped[arm]) == set(grouped[ref])
                delta = current[arm]["psnr_rgb"] - current[ref]["psnr_rgb"]
                worst = min(grouped[arm][k]["psnr_rgb"] - grouped[ref][k]["psnr_rgb"] for k in grouped[arm])
                ratio = current[arm]["temporal_error"] / current[ref]["temporal_error"]
                quality = delta >= -.05 and worst >= -.5
                comparisons.append(dict(seed=seed, domain=domain, category=category, candidate=arm, reference=ref,
                    PSNR_delta=delta, worst_window_PSNR_delta=worst, Etemp_ratio=ratio,
                    Etemp_limit_A=limit_a, Etemp_limit_B=limit_b, mean_quality_passed=delta >= -.05,
                    worst_quality_passed=worst >= -.5,
                    passed_A=quality and (limit_a is None or ratio <= limit_a),
                    passed_B=quality and (limit_b is None or ratio <= limit_b)))
    assert len(means) == len(comparisons) == 44
    complete_a, complete_b = all(r["passed_A"] for r in comparisons), all(r["passed_B"] for r in comparisons)
    out = BASE / "report/joint"
    out.mkdir(parents=True, exist_ok=False)
    write(out / "means.json", means)
    write(out / "gates.json", dict(at=stamp(), comparisons=comparisons, complete_A=complete_a, complete_B=complete_b,
          allowed_separate_cost_registration=complete_a or complete_b, allowed_DEV_or_external=False,
          held_out_draws_not_scenes=True, same_32_input_and_GT_draws_across_seeds_verified=True,
          self4_is_four_times_backbone_not_equal_compute=True, established_method_or_innovation=False, independent_final=False))
    text = ["# L1完整保留训练窗比较", "", "训练场景32窗/两seed相同抽样；唯一8000最终，不选最好checkpoint。", "",
            "|域|seed|输出|PSNR|SSIM|edge|Etemp|", "|---|---:|---|---:|---:|---:|---:|"]
    for r in means:
        text.append(f"|{r['domain']}|{r['seed']}|{r['arm']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['edge_mae']:.6f}|{r['temporal_error']:.6f}|")
    text += ["", "|域|seed|门|候选|参考|PSNR差|最差窗口|Etemp比|A过|B过|",
             "|---|---:|---|---|---|---:|---:|---:|---|---|"]
    for r in comparisons:
        text.append(f"|{r['domain']}|{r['seed']}|{r['category']}|{r['candidate']}|{r['reference']}|{r['PSNR_delta']:.6f}|{r['worst_window_PSNR_delta']:.6f}|{r['Etemp_ratio']:.6f}|{r['passed_A']}|{r['passed_B']}|")
    text += ["", f"完整A={complete_a}；B={complete_b}。质量过门只允许另登记成本，尚无DEV/外部/手机/创新或独立最终结论。"]
    (out / "tables.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
