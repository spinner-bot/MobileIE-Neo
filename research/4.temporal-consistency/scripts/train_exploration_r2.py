"""Matched-budget exploratory fine-tuning; not an official MobileIE reproduction."""
import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bounded_fst import apply_bounded, calibrate_limits, project_mixtures
from exploration_r2_data import MANIFEST, ROOT, degrade, evaluate, load_images, make_sequence
from explore_temporal_controls import MobileIELLENetS, inference, save_json, sha256

VARIANTS = ["task", "task_temporal", "bounded", "bounded_temporal"]


def training_batch(images, seed, step, batch_size=4, patch=128):
    rng = np.random.default_rng(seed*1000000+step)
    source = list(images.values())
    inputs, targets, offsets = [], [], []
    for _ in range(batch_size):
        image = source[int(rng.integers(len(source)))]
        x = int(rng.integers(4, image.shape[1]-patch-3))
        y = int(rng.integers(4, image.shape[0]-patch-3))
        dx, dy = [int(v) for v in rng.integers(-4, 5, size=2)]
        brightness = float(rng.choice([.6, .8, 1.]))
        pair = [image[y:y+patch, x:x+patch]*brightness,
                image[y+dy:y+dy+patch, x+dx:x+dx+patch]*brightness]
        photons, jpeg = int(rng.choice([20, 40, 80])), bool(rng.integers(2))
        inputs.append(np.stack([degrade(gt, rng, photons, jpeg) for gt in pair]))
        targets.append(np.stack(pair))
        offsets.append((dx, dy))
    x = np.stack(inputs).transpose(0, 1, 4, 2, 3).copy()/255
    y = np.stack(targets).transpose(0, 1, 4, 2, 3).copy()/255
    return torch.from_numpy(x), torch.from_numpy(y), offsets


def temporal_penalty(prediction, offsets):
    losses = []
    for pair, (dx, dy) in zip(prediction, offsets):
        prev, cur = pair[..., 8:-8, 8:-8]
        h, w = prev.shape[-2:]
        px, py, cx, cy = max(dx, 0), max(dy, 0), max(-dx, 0), max(-dy, 0)
        difference = prev[:, py:py+h-abs(dy), px:px+w-abs(dx)]-cur[:, cy:cy+h-abs(dy), cx:cx+w-abs(dx)]
        losses.append(torch.sqrt(difference**2+1e-6).mean())
    return torch.stack(losses).mean()


def new_model(device):
    model = MobileIELLENetS(12).to(device).eval()
    model.load_state_dict(torch.load(ROOT/"MobileIE/pretrain/lolv1_best_slim.pkl", map_location=device, weights_only=True))
    return model


def validate(model, images, device):
    rows = []
    names = list(images)
    model.eval()
    for index, name in enumerate(names):
        for case in ["pan", "light_step", "occlusion", "cut"]:
            for seed in [0, 1]:
                gt, low, flow, mask = make_sequence(images[name], images[names[(index+1)%len(names)]], case, seed)
                output, _ = inference(model, low, device, 4)
                metrics, _ = evaluate(output, gt, flow, mask)
                rows.append({"scene": name, "case": case, "input_seed": seed, **metrics})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--temporal-weight", type=float, default=.1)
    parser.add_argument("--checkpoint-root", default=".cache/checkpoints/exploration-r2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if args.steps < 1 or args.temporal_weight < 0 or any(v not in VARIANTS for v in variants):
        parser.error("positive steps and known variants required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    train, validation = load_images("train"), load_images("exploratory_validation")
    limits = calibrate_limits(new_model(device), train, device)
    save_json(out/"limits.json", limits)
    baseline = validate(new_model(device), validation, device)
    save_json(out/"pretrained_validation.json", baseline)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    for variant in variants:
        for seed in [0, 1]:
            folder = out/f"{variant}-seed{seed}"
            folder.mkdir(exist_ok=args.resume)
            cache = ROOT/args.checkpoint_root/f"{variant}-seed{seed}"
            cache.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(seed)
            net = new_model(device)
            bounded = variant.startswith("bounded")
            if bounded:
                apply_bounded(net, limits)
            main_parameters = [p for n, p in net.named_parameters() if not n.endswith(".gamma")]
            groups = [{"params": main_parameters, "lr": 1e-5}]
            if bounded:
                groups.append({"params": [net.head.gamma, net.body.gamma], "lr": 1e-3})
            optimizer = torch.optim.Adam(groups, weight_decay=0)
            checkpoint = cache/"last.pth"
            config = {"variant": variant, "seed": seed, "steps": args.steps, "batch_pairs": 4, "patch": 128,
                      "network_lr": 1e-5, "gamma_lr": 1e-3 if bounded else None,
                      "temporal_lambda": args.temporal_weight if "temporal" in variant else 0,
                      "grad_clip": 1., "limits": limits, "manifest_sha256": sha256(MANIFEST),
                      "pretrained_sha256": sha256(ROOT/"MobileIE/pretrain/lolv1_best_slim.pkl"),
                      "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
            start_step = 0
            if args.resume and checkpoint.exists():
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                if saved["config"] != config:
                    # Code revision can advance while the numerical experiment remains unchanged.
                    a, b = dict(saved["config"]), dict(config)
                    a.pop("git_revision"); b.pop("git_revision")
                    if a != b:
                        raise ValueError("resume configuration mismatch")
                net.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                start_step = saved["step"]
            elif checkpoint.exists():
                raise FileExistsError(f"checkpoint exists: {checkpoint}")
            save_json(folder/"config.json", config)
            start = time.perf_counter()
            net.train()
            for step in range(start_step, args.steps):
                low, target, offsets = training_batch(train, seed, step)
                low, target = low.to(device), target.to(device)
                pred = net(low.flatten(0, 1)).reshape_as(target)
                task = torch.sqrt((pred-target)**2+1e-6).mean()
                temporal = temporal_penalty(pred, offsets)
                loss = task+config["temporal_lambda"]*temporal
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite loss: {variant}, seed {seed}, step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.))
                optimizer.step()
                project_mixtures(net)
                row = {"step": step+1, "task": float(task.detach()), "temporal": float(temporal.detach()),
                       "loss": float(loss.detach()), "gradient_norm": norm,
                       "gamma_head": float(net.head.gamma.detach()) if bounded else None,
                       "gamma_body": float(net.body.gamma.detach()) if bounded else None}
                with (folder/"training.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                if (step+1)%50 == 0 or step+1 == args.steps or stop[0]:
                    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                                "step": step+1, "config": config}, checkpoint)
                    print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
                if stop[0]:
                    raise SystemExit("stopped with resumable checkpoint")
            rows = validate(net, validation, device)
            save_json(folder/"validation.json", rows)
            save_json(folder/"runtime.json", {"wall_seconds": time.perf_counter()-start,
                      "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
                      "trainable_parameters": sum(p.numel() for p in net.parameters())})


if __name__ == "__main__":
    main()
