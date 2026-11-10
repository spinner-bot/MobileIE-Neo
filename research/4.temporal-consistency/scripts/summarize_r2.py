"""Audit controlled training and broader state probes; export comparable evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    results = root/"research/4.temporal-consistency/results"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    directories = ["exploration-r2-training-plain", "exploration-r2-training-bounded", "exploration-r2-lambda05", "exploration-r2-lambda10"]
    all_models, seed_summaries, histories = {}, [], {}
    baseline = read(results/directories[0]/"pretrained_validation.json")
    all_models["pretrained"] = baseline
    keys = ["psnr_rgb", "ssim_rgb", "edge_mae", "flicker", "temporal_error"]
    first_task = []
    for directory in directories:
        repeated = read(results/directory/"pretrained_validation.json")
        assert len(repeated) == 48
        assert max(abs(a[k]-b[k]) for a,b in zip(baseline, repeated) for k in keys) < 1e-4
        for run in sorted((results/directory).glob("*-seed*")):
            config = read(run/"config.json")
            values = read(run/"validation.json")
            assert len(values) == 48
            history = [json.loads(line) for line in (run/"training.jsonl").read_text().splitlines()]
            assert [r["step"] for r in history] == list(range(1, 401))
            assert all(np.isfinite(r["loss"]) for r in history)
            first_task.append(history[0]["task"])
            runtime = read(run/"runtime.json")
            checkpoint = root/runtime["checkpoint"]
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == runtime["checkpoint_sha256"]
            label = config["variant"]
            if label == "task_temporal":
                label = f"task_T{config['temporal_lambda']:g}"
            elif label == "bounded_temporal":
                label = "bounded_T0.1"
            all_models.setdefault(label, []).extend(values)
            seed_summaries.append({"method": label, "seed": config["seed"], **{k:mean(values,k) for k in keys},
                "gamma_head": history[-1]["gamma_head"], "gamma_body": history[-1]["gamma_body"]})
            histories[f"{label}-seed{config['seed']}"] = history
    assert max(first_task)-min(first_task) < 1e-6
    assert len(seed_summaries) == 12
    state = [json.loads(line) for line in (results/"exploration-r2-state/per_sequence.jsonl").read_text().splitlines()]
    traces = [json.loads(line) for line in (results/"exploration-r2-state/per_frame.jsonl").read_text().splitlines()]
    assert len(state) == 384 and len(traces) == 384*24
    identity = [(r["scene"],r["case"],r["input_seed"],r["method"]) for r in state]
    assert len(set(identity)) == len(identity)
    source = read(root/"research/4.temporal-consistency/data/exploration-r2/kodak_manifest.json")
    train_hash = {r["sha256"] for r in source["images"] if r["split"] == "train"}
    val_hash = {r["sha256"] for r in source["images"] if r["split"] == "exploratory_validation"}
    assert len(train_hash) == 6 and len(val_hash) == 6 and not train_hash & val_hash
    audit = {"trained_runs": len(seed_summaries), "updates_per_run": 400,
             "trained_validation_evaluations": sum(len(r) for k,r in all_models.items() if k != "pretrained"),
             "state_evaluations": len(state), "state_frame_records": len(traces),
             "initial_task_loss_max_difference": max(first_task)-min(first_task),
             "train_validation_image_hashes_disjoint": True, "all_checkpoint_hashes_match": True}
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    summary = {name:{k:mean(rows,k) for k in keys} for name,rows in all_models.items()}
    (out/"model_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (out/"training_seed_summary.json").write_text(json.dumps(seed_summaries,indent=2)+"\n")
    lines = ["# R2 generated evidence", "", "All videos are synthetic; image versions and split are in the manifest.", "",
             "## Training (mean over exploratory validation sequences and two training seeds)", "",
             "| Method | PSNR | SSIM | Edge MAE | GT-relative temporal error |", "| --- | ---: | ---: | ---: | ---: |"]
    for name,row in summary.items():
        lines.append(f"| {name} | {row['psnr_rgb']:.4f} | {row['ssim_rgb']:.4f} | {row['edge_mae']:.4f} | {row['temporal_error']:.4f} |")
    state_summary = []
    for case in ["pan","light_step","occlusion","cut"]:
        lines.extend(["",f"## State controls: {case}","", "| Method | PSNR | SSIM | Temporal error |", "| --- | ---: | ---: | ---: |"])
        for method in sorted({r['method'] for r in state}):
            rows = [r for r in state if r['case']==case and r['method']==method]
            item = {"case":case,"method":method, **{k:mean(rows,k) for k in keys}, "mean_detector_events":mean(rows,"reset_events")}
            state_summary.append(item)
            lines.append(f"| {method} | {item['psnr_rgb']:.4f} | {item['ssim_rgb']:.4f} | {item['temporal_error']:.4f} |")
    (out/"state_summary.json").write_text(json.dumps(state_summary,indent=2)+"\n")
    (out/"tables.md").write_text("\n".join(lines)+"\n")
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.2})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,case in zip(axes.flat,["pan","light_step","occlusion","cut"]):
        for name,rows in all_models.items():
            chosen=[r for r in rows if r['case']==case]
            x,y=mean(chosen,'temporal_error'),mean(chosen,'psnr_rgb')
            ax.scatter(x,y)
            ax.annotate(name,(x,y),xytext=(3,3),textcoords='offset points',fontsize=7)
        ax.set(title=case,xlabel='GT-relative temporal error (0-255)',ylabel='RGB PSNR (dB)')
        ax.margins(x=.3,y=.2)
    fig.tight_layout()
    fig.savefig(out/'depth_tradeoff.png',dpi=150)
    fig.savefig(out/'depth_tradeoff.svg')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,case in zip(axes,['light_step','cut']):
        for method in ['mobileie','gate_ema','gate_reset','flow_ema','fb_ema','gate_reset_fb']:
            values=[mean([r for r in traces if r['case']==case and r['method']==method and r['frame']==t],'psnr_rgb') for t in range(24)]
            ax.plot(range(24),values,label=method)
        ax.axvline(12,color='black',ls='--',alpha=.5)
        ax.set(title=case,xlabel='Frame',ylabel='Mean RGB PSNR (dB)')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/'transition_response.png',dpi=150)
    plt.close(fig)
    cost=read(results/'exploration-r2-bounded-cost.json')
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    names=['ordinary','bounded']
    axes[0].bar(names,[cost['models'][n]['median_ms'] for n in names])
    axes[0].set(ylabel='Median ms',title='A100 FP32 eager, 400x600')
    axes[1].bar(names,[max(cost['models'][n]['peak_incremental_bytes'])/1024**2 for n in names])
    axes[1].set(ylabel='MiB',title='Incremental peak allocations')
    fig.tight_layout()
    fig.savefig(out/'architecture_cost.png',dpi=150)
    plt.close(fig)
    print(json.dumps(audit,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
