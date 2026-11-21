"""Posthoc complete O1 tables and preregistered contact sheet; no model imports.

Run from repository root with CUDA_VISIBLE_DEVICES='' ./scripts/project-python
research/8.video-backbone/operator-core/render.py. Original scoring stays frozen.
"""
import csv
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "report/analysis001"
GROUPS = [(0, "SYN"), (0, "ND"), (1, "SYN"), (1, "ND")]
ARMS = ["F0", "F0_FB", "F0_DIS"] + [
    f"{a}_{m}" for a in ["coordinate", "response"]
    for m in ["direct", "FB", "DIS", "causal", "zero"]
]
LABELS = {"F0": "F0", "F0_FB": "F0 FB", "F0_DIS": "F0 DIS"}
LABELS.update({a: a.replace("coordinate", "C").replace("response", "R").replace("_", " ")
               for a in ARMS if a not in LABELS})


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [HERE / "report/joint/means.json", HERE / "report/joint/gates.json"]
    rows, frames, manifests, ranges = [], [], [], []
    for seed in [0, 1]:
        base = HERE / f"results/held-seed{seed}"
        for name, target in [("per_sequence", rows), ("per_frame", frames),
                             ("prediction_manifest", manifests), ("raw-ranges", ranges)]:
            path = base / f"{name}.jsonl"
            inputs.append(path)
            target.extend(lines(path))
    assert (len(rows), len(frames), len(manifests), len(ranges)) == (416, 1664, 32, 32)
    means, gates = read(inputs[0]), read(inputs[1])
    assert len(means) == 52 and len(gates["comparisons"]) == 36
    index = {(r["key"], r["arm"]): r for r in rows}
    assert len(index) == 416
    comparisons = {
        "response_causal": ["response_direct", "response_zero", "coordinate_causal", "F0",
                            "F0_FB", "F0_DIS", "response_FB", "response_DIS"],
        "coordinate_causal": ["coordinate_direct", "coordinate_zero", "F0", "F0_FB", "F0_DIS",
                              "coordinate_FB", "coordinate_DIS"],
    }
    paired = []
    for manifest in manifests:
        for candidate, refs in comparisons.items():
            c = index[manifest["key"], candidate]
            for ref in refs:
                r = index[manifest["key"], ref]
                paired.append({"key": c["key"], "seed": c["seed"], "domain": c["domain"],
                               "candidate": candidate, "reference": ref,
                               "PSNR_delta": c["psnr_rgb"] - r["psnr_rgb"],
                               "SSIM_delta": c["ssim_rgb"] - r["ssim_rgb"],
                               "edge_delta": c["edge_mae"] - r["edge_mae"],
                               "Etemp_ratio": c["temporal_error"] / r["temporal_error"]})
    assert len(paired) == 480
    with (OUT / "paired-windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summaries = []
    for candidate, refs in comparisons.items():
        for ref in refs:
            for seed, domain in GROUPS:
                p = [r for r in paired if (r["candidate"], r["reference"], r["seed"], r["domain"])
                     == (candidate, ref, seed, domain)]
                assert len(p) == 8
                c = next(m for m in means if (m["arm"], m["seed"], m["domain"]) == (candidate, seed, domain))
                r = next(m for m in means if (m["arm"], m["seed"], m["domain"]) == (ref, seed, domain))
                summaries.append({"candidate": candidate, "reference": ref, "seed": seed, "domain": domain,
                                  "windows": 8, "mean_PSNR_delta": c["psnr_rgb"] - r["psnr_rgb"],
                                  "worst_PSNR_delta": min(x["PSNR_delta"] for x in p),
                                  "ratio_of_mean_Etemp": c["temporal_error"] / r["temporal_error"],
                                  "mean_SSIM_delta": c["ssim_rgb"] - r["ssim_rgb"],
                                  "mean_edge_delta": c["edge_mae"] - r["edge_mae"],
                                  "PSNR_positive_windows": sum(x["PSNR_delta"] > 0 for x in p),
                                  "Etemp_positive_windows": sum(x["Etemp_ratio"] < 1 for x in p),
                                  "joint_positive_windows": sum(x["PSNR_delta"] > 0 and x["Etemp_ratio"] < 1 for x in p)})
    assert len(summaries) == 60
    write("paired-summary.json", {"scope": "Descriptive paired windows, not independent-scene confidence intervals",
                                   "groups": summaries})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    handles = []
    for ax, (seed, domain) in zip(axes.flat, GROUPS):
        for arm in ARMS:
            m = next(m for m in means if (m["arm"], m["seed"], m["domain"]) == (arm, seed, domain))
            family = "coordinate" if arm.startswith("coordinate") else "response" if arm.startswith("response") else "F0"
            color = {"F0": "#555555", "coordinate": "#2474b5", "response": "#c45312"}[family]
            mode = arm.rsplit("_", 1)[-1] if arm != "F0" else "direct"
            marker = {"direct": "o", "FB": "s", "DIS": "^", "causal": "D", "zero": "x"}[mode]
            ax.scatter(m["psnr_rgb"], m["temporal_error"], color=color, marker=marker, s=62,
                       facecolors="none" if marker != "x" else color, linewidths=1.5)
            if ax is axes.flat[0]:
                handles.append(Line2D([], [], color=color, marker=marker, linestyle="none",
                                      markerfacecolor="none", label=LABELS[arm]))
        ax.set(title=f"{domain} / seed {seed} (8 windows)", xlabel="Mean PSNR (dB; higher better)",
               ylabel="Temporal residual MAE (lower better)")
        ax.grid(alpha=.22)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(.5, .01))
    fig.suptitle("O1: all 52 domain / seed / output means; overlapping points retained")
    fig.tight_layout(rect=(0, .16, 1, .96))
    save(fig, "quality-temporal")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, ref in zip(axes, ["response_direct", "response_zero"]):
        for (seed, domain), color in zip(GROUPS, ["#2474b5", "#c45312", "#268543", "#8a419d"]):
            p = [r for r in paired if (r["candidate"], r["reference"], r["seed"], r["domain"])
                 == ("response_causal", ref, seed, domain)]
            ax.scatter([r["PSNR_delta"] for r in p], [r["Etemp_ratio"] for r in p],
                       label=f"{domain} seed{seed}", color=color, s=35, alpha=.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(title=f"R causal vs {LABELS[ref]} (all 32 windows)",
               xlabel="Paired PSNR difference (dB)", ylabel="Paired Etemp ratio (<1 improves)")
        ax.grid(alpha=.22)
    axes[1].legend(fontsize=9)
    fig.suptitle("Window variation; dashed lines indicate equality, not mean-level gates")
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, "paired-history")

    labels = [(x["category"], x["candidate"], x["reference"]) for x in gates["comparisons"][:9]]
    matrix, annotations = np.zeros((9, 4)), []
    for y, key in enumerate(labels):
        for x, (seed, domain) in enumerate(GROUPS):
            g = next(g for g in gates["comparisons"]
                     if (g["category"], g["candidate"], g["reference"]) == key and (g["seed"], g["domain"]) == (seed, domain))
            matrix[y, x] = g["passed"]
            annotations.append((x, y, f"{'PASS' if g['passed'] else 'FAIL'}  {g['PSNR_delta']:+.3f} dB\n"
                                f"worst {g['worst_window_PSNR_delta']:+.3f}; E {g['Etemp_ratio']:.3f}"))
    fig, ax = plt.subplots(figsize=(12, 8))
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(["#f1d6d1", "#d4e8d5"]), vmin=0, vmax=1, aspect="auto")
    for x, y, value in annotations:
        ax.text(x, y, value, ha="center", va="center", fontsize=9)
    ax.set_xticks(range(4), [f"{d} / seed {s}" for s, d in GROUPS])
    ax.set_yticks(range(9), [f"{k}: {LABELS[c]} / {LABELS[r]}" for k, c, r in labels])
    ax.set_title("All 36 preregistered comparisons: 14 pass, complete gate FAIL\n"
                 "space 8/8; history 0/8; objective 2/4; strong 4/16")
    fig.tight_layout()
    save(fig, "gates")

    # Fixed before evaluation: first draw, first window per domain, last frame.
    contact_keys = [f"seed{s}-step16000-window{w}" for s in [0, 1] for w in [0, 2]]
    columns = ["gt", "low"] + ARMS
    sheet = Image.new("RGB", (128 * 15, 160 * 4 + 36), "white")
    draw = ImageDraw.Draw(sheet)
    contact_sources = []
    for row, key in enumerate(contact_keys):
        m = next(m for m in manifests if m["key"] == key)
        path = ROOT / m["path"]
        assert path.resolve().is_relative_to(Path("/data/Users/zrh/data/MobileIE-Neo"))
        assert sha(path) == m["sha256"]
        contact_sources.append(m)
        with np.load(path, allow_pickle=False) as z:
            for col, arm in enumerate(columns):
                a = z[arm]
                assert a.shape == (4, 128, 128, 3) and np.isfinite(a).all()
                image = Image.fromarray(np.rint(np.clip(a[3], 0, 255)).astype(np.uint8))
                x, y = 128 * col, 36 + 160 * row
                sheet.paste(image, (x, y + 28))
                draw.text((x + 3, y + 12), LABELS.get(arm, arm), fill="black")
        draw.text((3, 36 + 160 * row), f"{key} / {m['domain']} / frame 3", fill="black")
    draw.text((3, 4), "Fixed first windows / last frame; all 13 outputs. Static RGB cannot establish flicker, MOS or phone performance.", fill="black")
    sheet.save(OUT / "contact.png")
    sheet.save(OUT / "contact.pdf", resolution=140)
    write("contact-sources.json", {"selection": "Preregistered step16000, window0/2 per seed, frame3, all outputs",
                                   "sources": contact_sources})
    write("raw-range-summary.json", {
        arm: {"windows": 32,
              "minimum_raw": min(r["values"][arm]["min"] for r in ranges),
              "maximum_raw": max(r["values"][arm]["max"] for r in ranges),
              "mean_outside_fraction": float(np.mean([r["values"][arm]["outside_fraction"] for r in ranges])),
              "maximum_outside_fraction": max(r["values"][arm]["outside_fraction"] for r in ranges)}
        for arm in ["F0", "coordinate_direct", "response_direct"]})
    write("render-record.json", {
        "at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "author": "Codex  (sqs_A100)", "renderer_sha256": sha(Path(__file__)),
        "input_sha256": {str(f.relative_to(ROOT)): sha(f) for f in inputs},
        "counts": {"output_rows": len(rows), "frame_rows": len(frames), "unique_windows": len(manifests),
                   "means": len(means), "original_gates": 36, "paired_windows": len(paired),
                   "paired_means": len(summaries), "PSNR_ceiling_applied": sum(r["PSNR_ceiling_applied"] for r in frames)},
        "gate_pass_counts": {category: sum(g["passed"] for g in gates["comparisons"] if g["category"] == category)
                             for category in ["space", "history", "objective", "strong"]},
        "new_training_or_inference": False, "actual_viewed": False,
        "scope": "Complete stored scores only; same training scenes, no independent final/DEV or phone. No stored flow/logits for causal diagnoses."})
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "pairs": len(paired),
                      "gate_categories": dict(Counter(g['category'] for g in gates['comparisons']))}))


if __name__ == "__main__":
    main()
