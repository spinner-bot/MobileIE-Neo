"""Audit saved exploration rows and export tables/standalone research figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(directory):
    return [json.loads(line) for line in (directory / "per_sequence.jsonl").read_text().splitlines()]


def select(rows, condition="noise_jpeg", pan="all", method=None):
    return [r for r in rows if r["condition"] == condition and
            (pan == "all" or r["pan"] == pan) and (method is None or r["method"] == method)]


def average(rows, key):
    return float(np.mean([row[key] for row in rows]))


def table(rows, condition, methods):
    lines = ["| Method | PSNR | SSIM | Core residual | Edge error | GT-relative temporal error |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        chosen = select(rows, condition=condition, method=method)
        values = [average(chosen, key) for key in ["psnr_rgb", "ssim_rgb_valid", "flicker_core", "edge_mae", "reference_temporal_error"]]
        lines.append("| " + method + " | " + " | ".join(f"{x:.4f}" for x in values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.results_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = {"controls": read_rows(root / "exploration-r1-v1"),
            "state": read_rows(root / "exploration-r1b-state"),
            "step": read_rows(root / "exploration-r1b-light-step")}
    checks = {}
    for name, rows in runs.items():
        expected = 351 if name == "controls" else 270
        if len(rows) != expected:
            raise AssertionError(f"{name}: expected {expected} rows, got {len(rows)}")
        keys = [(r["scene"], r["pan"], r["condition"], r["seed"], r["method"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise AssertionError(f"duplicate rows in {name}")
        for row in rows:
            for key in ["psnr_rgb", "ssim_rgb_valid", "flicker_core", "reference_temporal_error"]:
                if not np.isfinite(row[key]):
                    raise AssertionError(f"non-finite {key}")
        checks[name] = {"rows": len(rows), "unique_sequences": len(set(key[:4] for key in keys))}
    first = {(r["scene"], r["pan"], r["condition"], r["seed"]): r for r in runs["controls"] if r["method"] == "mobileie"}
    second = {(r["scene"], r["pan"], r["condition"], r["seed"]): r for r in runs["state"] if r["method"] == "mobileie"}
    max_baseline_difference = max(abs(first[key][metric] - second[key][metric]) for key in first
                                  for metric in ["psnr_rgb", "flicker_core", "ssim_rgb_valid"])
    if max_baseline_difference > 1e-4:
        raise AssertionError("baseline drift between repeated no-step matrices")
    checks["repeated_baseline_max_abs_difference"] = max_baseline_difference
    checks["total_method_sequence_evaluations"] = sum(len(rows) for rows in runs.values())
    checks["distinct_input_conditions"] = len({(r["scene"], r["pan"], r["condition"], r["seed"], r.get("target_step", False)) for rows in runs.values() for r in rows})
    features = json.loads((root / "exploration-r1-feature-noise-v2/layer_response.json").read_text())
    identities = json.loads((root / "exploration-r1-feature-noise-v2/fst_identity.json").read_text())
    checks["feature_probe_conditions"] = len(features) // 7
    checks["fst_max_scaled_identity_error"] = max(r["max_scaled_identity_error"] for r in identities)
    if checks["fst_max_scaled_identity_error"] > 1e-5:
        raise AssertionError("FST identity exceeds float32 relative tolerance")
    (out / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")

    methods = ["mobileie", "frozen_global", "input_bilateral", "output_gaussian",
               "ema_07", "ema_09", "flow_ema_07", "flow_ema_09", "oracle_ema_07", "oracle_ema_09",
               "bilateral_flow_ema_07", "bilateral_oracle_ema_07"]
    lines = ["# Exploration-r1 generated tables", "", "Synthetic pilot; averages over sequences, not independent test-set claims.",
             "", "## Noisy/compressed controls", "", table(runs["controls"], "noise_jpeg", methods),
             "", "## Clean controls", "", table(runs["controls"], "clean", methods)]
    combo_methods = ["mobileie", "frozen_global", "attention_ema_09", "flow_ema_07",
                     "frozen_flow_ema_07", "attention_flow_ema_07", "oracle_ema_07",
                     "frozen_oracle_ema_07", "attention_oracle_ema_07"]
    for label in ["state", "step"]:
        for condition in ["clean", "noise_jpeg"]:
            lines.extend(["", f"## {label}: {condition}", "", table(runs[label], condition, combo_methods)])
    (out / "tables.md").write_text("\n".join(lines) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2})
    controls = runs["controls"]
    plot_methods = ["mobileie", "frozen_global", "input_bilateral", "output_gaussian", "ema_09", "flow_ema_09", "oracle_ema_09"]
    colors = dict(zip(plot_methods, plt.get_cmap("tab10").colors))
    labels = {"mobileie": "MobileIE", "frozen_global": "frozen gate",
              "input_bilateral": "input bilateral", "output_gaussian": "output Gaussian",
              "ema_09": "EMA(.9)", "flow_ema_09": "flow EMA(.9)", "oracle_ema_09": "oracle EMA(.9)",
              "attention_ema_09": "gate EMA", "flow_ema_07": "flow EMA",
              "attention_flow_ema_07": "gate + flow", "attention_oracle_ema_07": "gate + oracle"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, pan in zip(axes, [0, 4, 12]):
        for method in plot_methods:
            rows = select(controls, pan=pan, method=method)
            x, y = average(rows, "flicker_core"), average(rows, "psnr_rgb")
            ax.scatter(x, y, color=colors[method], marker="x" if "oracle" in method else "o")
            ax.annotate(labels[method], (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.margins(x=.2, y=.15)
        ax.set(title=f"Noise + JPEG; pan={pan}px/frame", xlabel="Core aligned residual (0-255)", ylabel="RGB PSNR (dB)")
    fig.suptitle("Fixed-control pilot: oracle motion is an analysis reference")
    fig.tight_layout()
    fig.savefig(out / "controls_tradeoff.png", dpi=160)
    fig.savefig(out / "controls_tradeoff.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for method in ["mobileie", "frozen_global", "ema_09", "flow_ema_09", "oracle_ema_09"]:
        values = [average(select(controls, condition="clean", pan=p, method=method), "flicker_core") for p in [0, 4, 12]]
        ax.plot([0, 4, 12], values, "o-", label=method)
    ax.set(xlabel="Pan (pixels/frame)", ylabel="Core aligned residual (0-255)", title="Clean input: global mapping changes vs spatial motion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "clean_mechanism.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    layers = ["head.block1", "head", "body.block1", "body", "att", "att1", "tail"]
    for jpeg in [False, True]:
        values = [np.mean([r["relative_perturbation"] for r in features if r["photons"] == 40 and r["jpeg"] == jpeg and r["layer"] == layer]) for layer in layers]
        ax.plot(layers, values, "o-", label=f"40 photons; JPEG={jpeg}")
    ax.set(yscale="log", ylabel="Perturbation RMS / clean-feature RMS", title="Layer response (diagnostic, not causal contribution)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "feature_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    chosen_methods = ["mobileie", "frozen_global", "attention_ema_09", "flow_ema_07", "attention_flow_ema_07", "attention_oracle_ema_07"]
    for ax, name in zip(axes, ["state", "step"]):
        for method in chosen_methods:
            rows = select(runs[name], method=method)
            x, y = average(rows, "reference_temporal_error"), average(rows, "psnr_rgb")
            ax.scatter(x, y, marker="x" if "oracle" in method else "o")
            ax.annotate(labels[method], (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.margins(x=.2, y=.15)
        ax.set(title="Noisy inputs: " + ("fixed lighting" if name == "state" else "real lighting transition"),
               xlabel="GT-relative temporal error (0-255)", ylabel="RGB PSNR (dB)")
    fig.tight_layout()
    fig.savefig(out / "state_combination_tradeoff.png", dpi=160)
    fig.savefig(out / "state_combination_tradeoff.svg")
    plt.close(fig)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
