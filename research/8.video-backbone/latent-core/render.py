"""Present complete stored L1 scores; no model/flow inference or partial ranking."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "report/analysis001"
GROUPS = [(0, "SYN"), (0, "ND"), (1, "SYN"), (1, "ND")]
LABELS = dict(F0="F0", F0_FB="F0 FB", F0_DIS="F0 DIS", spatial_current="S current",
              spatial_FB="S FB", spatial_DIS="S DIS", causal="L causal", causal_current="L current",
              causal_self4="L self4", causal_current_FB="L current FB", causal_current_DIS="L current DIS")


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
    assert read(HERE / "process/chain002/runtime.json")["completed_normally"]
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [HERE / "report/joint/means.json", HERE / "report/joint/gates.json"]
    means, gates = read(inputs[0]), read(inputs[1])
    rows, frames, manifests, ranges = [], [], [], []
    for seed in [0, 1]:
        base = HERE / f"results/held-seed{seed}"
        assert read(base / "runtime.json")["completed_normally"]
        for name, target in [("per_sequence", rows), ("per_frame", frames),
                             ("prediction_manifest", manifests), ("raw-ranges", ranges)]:
            file = base / f"{name}.jsonl"
            inputs.append(file)
            target.extend(lines(file))
    assert (len(means), len(gates["comparisons"]), len(rows), len(frames), len(manifests)) == (44, 44, 704, 2816, 64)
    index = {(r["key"], r["arm"]): r for r in rows}
    assert len(index) == 704
    paired, summaries = [], []
    for ref in [a for a in LABELS if a != "causal"]:
        for seed, domain in GROUPS:
            selected = [r for r in rows if (r["arm"], r["seed"], r["domain"]) == ("causal", seed, domain)]
            assert len(selected) == 16
            for c in selected:
                r = index[c["key"], ref]
                paired.append(dict(key=c["key"], seed=seed, domain=domain, reference=ref,
                    PSNR_delta=c["psnr_rgb"]-r["psnr_rgb"], SSIM_delta=c["ssim_rgb"]-r["ssim_rgb"],
                    edge_delta=c["edge_mae"]-r["edge_mae"], Etemp_ratio=c["temporal_error"]/r["temporal_error"]))
            p = paired[-16:]
            c = next(m for m in means if (m["arm"], m["seed"], m["domain"]) == ("causal", seed, domain))
            r = next(m for m in means if (m["arm"], m["seed"], m["domain"]) == (ref, seed, domain))
            summaries.append(dict(seed=seed, domain=domain, reference=ref, windows=16,
                mean_PSNR_delta=c["psnr_rgb"]-r["psnr_rgb"], worst_PSNR_delta=min(x["PSNR_delta"] for x in p),
                mean_SSIM_delta=c["ssim_rgb"]-r["ssim_rgb"], mean_edge_delta=c["edge_mae"]-r["edge_mae"],
                ratio_of_mean_Etemp=c["temporal_error"]/r["temporal_error"],
                PSNR_positive_windows=sum(x["PSNR_delta"] > 0 for x in p),
                Etemp_positive_windows=sum(x["Etemp_ratio"] < 1 for x in p),
                joint_positive_windows=sum(x["PSNR_delta"] > 0 and x["Etemp_ratio"] < 1 for x in p)))
    assert len(paired) == 640 and len(summaries) == 40
    with (OUT / "paired-windows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(paired[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    write("paired-summary.json", dict(scope="Descriptive windows on training scenes, not independent confidence intervals", groups=summaries))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    handles = []
    for ax, (seed, domain) in zip(axes.flat, GROUPS):
        for arm, label in LABELS.items():
            row = next(m for m in means if (m["arm"], m["seed"], m["domain"]) == (arm, seed, domain))
            color = "#555555" if arm.startswith("F0") else "#2474b5" if arm.startswith("spatial") else "#c45312"
            marker = "s" if arm.endswith("FB") else "^" if arm.endswith("DIS") else "*" if arm.endswith("self4") else "D" if arm == "causal" else "o"
            ax.scatter(row["psnr_rgb"], row["temporal_error"], color=color, marker=marker,
                       facecolors="none", s=70, linewidths=1.5)
            if ax is axes.flat[0]:
                handles.append(Line2D([], [], marker=marker, color=color, markerfacecolor="none", linestyle="none", label=label))
        ax.set(title=f"{domain} / seed {seed} (16 windows)", xlabel="Mean PSNR (dB; higher better)",
               ylabel="Temporal residual MAE (lower better)")
        ax.grid(alpha=.22)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(.5, .01))
    fig.suptitle("L1: all 44 means; same input draws across seeds; overlapping points retained")
    fig.tight_layout(rect=(0, .16, 1, .96))
    save(fig, "quality-temporal")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, ref in zip(axes, ["spatial_current", "causal_current", "causal_self4"]):
        for (seed, domain), color in zip(GROUPS, ["#2474b5", "#c45312", "#268543", "#8a419d"]):
            p = [r for r in paired if (r["reference"], r["seed"], r["domain"]) == (ref, seed, domain)]
            ax.scatter([r["PSNR_delta"] for r in p], [r["Etemp_ratio"] for r in p], color=color,
                       label=f"{domain} seed{seed}", s=28, alpha=.75)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(title=f"L causal vs {LABELS[ref]} (64 pairs)", xlabel="Paired PSNR difference (dB)", ylabel="Paired Etemp ratio (<1 improves)")
        ax.grid(alpha=.22)
    axes[-1].legend(fontsize=8)
    fig.suptitle("All windows; dashed equality lines are not mean-level gates; self4 executes four backbones/frame")
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, "paired-history")

    keys = [(g["category"], g["candidate"], g["reference"]) for g in gates["comparisons"][:11]]
    matrix = np.zeros((11, 4))
    fig, ax = plt.subplots(figsize=(13, 10))
    for y, key in enumerate(keys):
        for x, (seed, domain) in enumerate(GROUPS):
            g = next(g for g in gates["comparisons"] if (g["category"], g["candidate"], g["reference"]) == key
                     and (g["seed"], g["domain"]) == (seed, domain))
            matrix[y, x] = 2 if g["passed_A"] else 1 if g["passed_B"] else 0
            ax.text(x, y, f"A:{'P' if g['passed_A'] else 'F'} B:{'P' if g['passed_B'] else 'F'}  {g['PSNR_delta']:+.3f} dB\n"
                    f"worst {g['worst_window_PSNR_delta']:+.3f}; E {g['Etemp_ratio']:.3f}", ha="center", va="center", fontsize=8.5)
    ax.imshow(matrix, cmap=ListedColormap(["#f1d6d1", "#fff2bf", "#d4e8d5"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(4), [f"{d} / seed{s}" for s, d in GROUPS])
    ax.set_yticks(range(11), [f"{k}: {LABELS[c]} / {LABELS[r]}" for k, c, r in keys])
    ax.set_title(f"All44 preregistered comparisons: complete A={gates['complete_A']}; B={gates['complete_B']}\nGreen A+B pass; yellow B only; red both fail")
    fig.tight_layout()
    save(fig, "gates")

    columns = ["gt", "low", *LABELS]
    sheet = Image.new("RGB", (128 * 13, 676), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((3, 4), "Fixed first SYN/ND windows, frame3; same two inputs repeated for two model seeds. Static RGB is not flicker/MOS/phone evidence.", fill="black")
    contacts = []
    for row, (seed, window) in enumerate([(0, 0), (0, 2), (1, 0), (1, 2)]):
        key = f"seed{seed}-step20000-window{window}"
        source = next(m for m in manifests if m["key"] == key)
        path = ROOT / source["path"]
        assert path.resolve().is_relative_to(Path("/data/Users/zrh/data/MobileIE-Neo")) and sha(path) == source["sha256"]
        contacts.append(source)
        y = 36 + 160 * row
        draw.text((3, y), f"{key} / {source['domain']}", fill="black")
        with np.load(path, allow_pickle=False) as z:
            for col, arm in enumerate(columns):
                a = z[arm]
                assert a.shape == (4, 128, 128, 3) and np.isfinite(a).all()
                x = 128 * col
                draw.text((x + 3, y + 12), LABELS.get(arm, arm), fill="black")
                sheet.paste(Image.fromarray(np.rint(np.clip(a[3], 0, 255)).astype(np.uint8)), (x, y + 28))
    sheet.save(OUT / "contact.png")
    sheet.save(OUT / "contact.pdf", resolution=140)
    write("contact-sources.json", dict(selection="Preregistered step20000, windows0/2, frame3 per seed; no best/worst search", sources=contacts))
    write("raw-range-summary.json", {arm: dict(windows=64,
          minimum_raw=min(r["values"][arm]["min"] for r in ranges), maximum_raw=max(r["values"][arm]["max"] for r in ranges),
          mean_outside_fraction=float(np.mean([r["values"][arm]["outside_fraction"] for r in ranges])))
          for arm in ["F0", "spatial_current", "causal", "causal_current", "causal_self4"]})
    write("render-record.json", dict(at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(), author="Codex  (sqs_A100)",
          renderer_sha256=sha(Path(__file__)), input_sha256={str(f.relative_to(ROOT)): sha(f) for f in inputs},
          counts=dict(output_rows=704, frame_rows=2816, window_seed_rows=64, unique_input_draws=32, means=44,
                      gates=44, paired_windows=640, paired_means=40, PSNR_ceiling_applied=sum(r["PSNR_ceiling_applied"] for r in frames)),
          actual_viewed=False, new_model_or_flow_inference=False,
          scope="Complete stored training-scene scores only; no independent scenes, final-test, MOS or phone evidence"))
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)), paired_windows=640, means=44, actual_viewed=False)))


if __name__ == "__main__":
    main()
