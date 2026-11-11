"""Fixed, inference-only exploration matrix; see exploration-r1-protocol.md.

Run with an explicitly assigned CUDA_VISIBLE_DEVICES. Historical scripts and
results are not modified. Float arrays are used for metrics, PNG for previews.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from video_metrics import (aligned_difference, backward_flow, edge_error, psnr,
                           ssim, temporal_error, translation_warp, warp_previous)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "MobileIE"))
from model.lle import MobileIELLENetS  # noqa: E402

SCENES = {
    "source": ROOT / "research/3.video-flicker/assets/source_scene.jpg",
    "lol1": ROOT / "MobileIE/datasets/lle/gt/1.png",
    "lol22": ROOT / "MobileIE/datasets/lle/gt/22.png",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def generate_sequence(image, pan, seed, noisy, frames, width, height, target_step=False):
    rng = np.random.default_rng(seed)
    targets, inputs = [], []
    y0 = (image.shape[0] - height) // 2
    for t in range(frames):
        gt = image[y0:y0 + height, pan*t:pan*t + width].copy()
        if gt.shape[:2] != (height, width):
            raise ValueError("trajectory leaves source image")
        if target_step and t < frames // 2:
            gt *= .6  # A real target lighting change, also present in the input.
        low = (gt / 255.0) ** 2.2 * .35
        if noisy:
            low = rng.poisson(low * 40) / 40 + rng.normal(0, .008, low.shape)
        low = (np.clip(low, 0, 1) * 255).astype(np.uint8)
        if noisy:
            bgr = cv2.cvtColor(low, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            low = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        targets.append(gt.astype(np.float32))
        inputs.append(low.astype(np.float32))
    return targets, inputs


class FrozenAttention(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.cache = None

    def forward(self, value):
        if self.cache is None:
            self.cache = self.inner(value[:1]).detach()
        return self.cache.expand(value.shape[0], -1, -1, -1)


class TemporalAttention(torch.nn.Module):
    """Baseline state in the existing C-dimensional channel gate (not novelty)."""
    def __init__(self, inner, alpha=.9):
        super().__init__()
        self.inner, self.alpha, self.cache = inner, alpha, None

    def forward(self, value):
        gates = self.inner(value)
        outputs = []
        for gate in gates.split(1, dim=0):
            self.cache = gate if self.cache is None else self.alpha*self.cache + (1-self.alpha)*gate
            outputs.append(self.cache)
        return torch.cat(outputs, dim=0)


def inference(net, images, device, batch_size):
    output, gpu_ms = [], 0.0
    with torch.inference_mode():
        for offset in range(0, len(images), batch_size):
            array = np.stack(images[offset:offset + batch_size]) / 255.0
            tensor = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(device)
            if device.type == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            pred = net(tensor).clamp(0, 1)
            if device.type == "cuda":
                end.record()
                end.synchronize()
                gpu_ms += begin.elapsed_time(end)
            output.extend((pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.float32))
    return output, gpu_ms


def smooth(outputs, alpha, flows=None, pan=None):
    sequence = [outputs[0].copy()]
    for t in range(1, len(outputs)):
        if flows is not None:
            previous, valid = warp_previous(sequence[-1], flows[t-1])
        elif pan is not None:
            previous, valid = translation_warp(sequence[-1], pan)
        else:
            previous, valid = sequence[-1], np.ones(outputs[t].shape[:2], dtype=bool)
        weight = alpha * valid[..., None]
        sequence.append((weight * previous + (1-weight) * outputs[t]).astype(np.float32))
    return sequence


def evaluate(outputs, targets, inputs, pan, margin=8):
    core = lambda value: value[margin:-margin, margin:-margin]
    flicker, core_flicker, reference_error, old_flicker = [], [], [], []
    for t in range(1, len(outputs)):
        flicker.append(aligned_difference(outputs[t-1], outputs[t], pan))
        core_flicker.append(aligned_difference(core(outputs[t-1]), core(outputs[t]), pan))
        reference_error.append(temporal_error(outputs[t-1], outputs[t], targets[t-1], targets[t], pan))
        if pan:
            wrong = np.roll(outputs[t], pan, axis=1)[:, :-pan]
            old_flicker.append(float(np.abs(wrong - outputs[t-1][:, :-pan]).mean()))
        else:
            old_flicker.append(flicker[-1])
    input_flicker = np.mean([aligned_difference(inputs[t-1], inputs[t], pan)
                             for t in range(1, len(inputs))])
    return {
        "psnr_rgb": float(np.mean([psnr(o, g) for o, g in zip(outputs, targets)])),
        "ssim_rgb_valid": float(np.mean([ssim(o, g) for o, g in zip(outputs, targets)])),
        "edge_mae": float(np.mean([edge_error(o, g) for o, g in zip(outputs, targets)])),
        "flicker": float(np.mean(flicker)), "flicker_core": float(np.mean(core_flicker)),
        "reference_temporal_error": float(np.mean(reference_error)),
        "legacy_wrapped_flicker": float(np.mean(old_flicker)),
        "input_flicker": float(input_flicker),
        "amplification": float(np.mean(flicker) / input_flicker) if input_flicker > 1e-8 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--family", choices=["controls", "attention_state"], default="controls")
    parser.add_argument("--target-step", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    weights = ROOT / "MobileIE/pretrain/lolv1_best_slim.pkl"
    config = {**vars(args), "experiment_id": "exploration-r1-v1" if args.family == "controls" else "exploration-r1b-attention-state", "pans": [0, 4, 12],
              "noise_seeds": [0, 1], "torch": torch.__version__, "opencv": cv2.__version__,
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "weights_sha256": sha256(weights),
              "scene_sha256": {key: sha256(value) for key, value in SCENES.items()},
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "notes": "exploratory synthetic videos; no training; no claim of independent generalization"}
    save_json(out_dir / "config.json", config)
    net = MobileIELLENetS(12).eval().to(device)
    net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    rows, start, gpu_ms = [], time.perf_counter(), 0.0
    for scene_name, path in SCENES.items():
        image = np.asarray(Image.open(path).convert("RGB").resize((1024, 768), Image.Resampling.LANCZOS), np.float32)
        for pan in [0, 4, 12]:
            for noisy in [False, True]:
                for seed in ([0, 1] if noisy else [0]):
                    sequence_start = time.perf_counter()
                    targets, inputs = generate_sequence(image, pan, seed, noisy, args.frames, args.width, args.height, args.target_step)
                    raw, ms = inference(net, inputs, device, args.batch_size)
                    gpu_ms += ms
                    original_attention = net.att
                    try:
                        net.att = FrozenAttention(original_attention)
                        frozen, ms = inference(net, inputs, device, args.batch_size)
                        gpu_ms += ms
                    finally:
                        net.att = original_attention
                    filtered = [cv2.bilateralFilter(frame, 5, 15, 3) for frame in inputs]
                    denoised, ms = inference(net, filtered, device, args.batch_size)
                    gpu_ms += ms
                    flows = [backward_flow(inputs[t-1], inputs[t]) for t in range(1, len(inputs))]
                    filtered_flows = [backward_flow(filtered[t-1], filtered[t]) for t in range(1, len(filtered))]
                    flow_epe = float(np.mean([np.linalg.norm(f[8:-8, 8:-8] - np.array([pan, 0]), axis=2).mean() for f in flows]))
                    methods = {"input": inputs, "mobileie": raw, "frozen_global": frozen,
                               "input_bilateral": denoised,
                               "output_gaussian": [cv2.GaussianBlur(frame, (5, 5), 1) for frame in raw]}
                    for alpha in [.7, .9]:
                        suffix = str(alpha).replace(".", "")
                        methods[f"ema_{suffix}"] = smooth(raw, alpha)
                        methods[f"flow_ema_{suffix}"] = smooth(raw, alpha, flows=flows)
                        methods[f"oracle_ema_{suffix}"] = smooth(raw, alpha, pan=pan)
                    methods["bilateral_flow_ema_07"] = smooth(denoised, .7, flows=filtered_flows)
                    methods["bilateral_oracle_ema_07"] = smooth(denoised, .7, pan=pan)
                    if args.family == "attention_state":
                        try:
                            net.att = TemporalAttention(original_attention, .9)
                            gate_state, ms = inference(net, inputs, device, args.batch_size)
                            gpu_ms += ms
                        finally:
                            net.att = original_attention
                        methods = {"input": inputs, "mobileie": raw, "frozen_global": frozen,
                                   "attention_ema_09": gate_state,
                                   "flow_ema_07": smooth(raw, .7, flows=flows),
                                   "oracle_ema_07": smooth(raw, .7, pan=pan),
                                   "frozen_flow_ema_07": smooth(frozen, .7, flows=flows),
                                   "frozen_oracle_ema_07": smooth(frozen, .7, pan=pan),
                                   "attention_flow_ema_07": smooth(gate_state, .7, flows=flows),
                                   "attention_oracle_ema_07": smooth(gate_state, .7, pan=pan)}
                    for name, values in methods.items():
                        row = {"scene": scene_name, "pan": pan, "condition": "noise_jpeg" if noisy else "clean",
                               "target_step": args.target_step,
                               "seed": seed, "method": name, "flow_epe_px": flow_epe,
                               **evaluate(values, targets, inputs, pan)}
                        rows.append(row)
                        with (out_dir / "per_sequence.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if pan == 4 and seed == 0 and noisy:
                        names = ["input", "mobileie", "input_bilateral", "flow_ema_09", "bilateral_flow_ema_07", "oracle_ema_09"]
                        if args.family == "attention_state":
                            names = ["input", "mobileie", "attention_ema_09", "flow_ema_07", "attention_flow_ema_07", "attention_oracle_ema_07"]
                        preview = np.concatenate([methods[name][-1] for name in names] + [targets[-1]], axis=1)
                        Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8)).save(out_dir / f"preview_{scene_name}.png")
                        save_json(out_dir / "preview_columns.json", names + ["ground_truth"])
                    print(json.dumps({"scene": scene_name, "pan": pan, "condition": "noise_jpeg" if noisy else "clean",
                                      "seed": seed, "rows": len(rows), "sequence_seconds": round(time.perf_counter()-sequence_start, 3)}), flush=True)
    with (out_dir / "per_sequence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for condition in ["clean", "noise_jpeg"]:
        for pan in [0, 4, 12, "all"]:
            for method in sorted({row["method"] for row in rows}):
                selected = [r for r in rows if r["condition"] == condition and r["method"] == method and (pan == "all" or r["pan"] == pan)]
                item = {"condition": condition, "pan": pan, "method": method, "sequences": len(selected)}
                for key in ["psnr_rgb", "ssim_rgb_valid", "edge_mae", "flicker", "flicker_core", "reference_temporal_error", "legacy_wrapped_flicker", "flow_epe_px"]:
                    values = [row[key] for row in selected]
                    item[key] = float(np.mean(values))
                    item[key + "_std"] = float(np.std(values))
                summary.append(item)
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "runtime.json", {"wall_seconds": time.perf_counter()-start,
              "forward_cuda_event_ms": gpu_ms, "physical_gpus_requested": config["visible_devices"],
              "timing_scope": "wall covers matrix after model load; CUDA events cover model forwards only"})
    print(f"completed {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
