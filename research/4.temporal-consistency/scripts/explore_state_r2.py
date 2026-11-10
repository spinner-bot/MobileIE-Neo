"""State/reset/flow/NLM controls on new scenes, occlusions and transitions."""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from exploration_r2_data import MANIFEST, ROOT, detect_changes, evaluate, load_images, make_sequence
from explore_temporal_controls import MobileIELLENetS, TemporalAttention, save_json, sha256
from video_metrics import backward_flow, warp_previous


def forward_backward(previous, current):
    back = backward_flow(previous, current)
    forward = backward_flow(current, previous)
    sampled, valid = warp_previous(forward, back)
    disagreement = np.linalg.norm(back+sampled, axis=2)
    threshold = 1+.05*(np.linalg.norm(back, axis=2)+np.linalg.norm(sampled, axis=2))
    return back, valid & (disagreement < threshold)


def infer_gates(net, frames, device, alpha=None, resets=None):
    original, captured = net.att, []
    if alpha is not None:
        net.att = TemporalAttention(original, alpha)
    handle = net.att.register_forward_hook(lambda module, inputs, value: captured.append(value.detach().flatten().cpu().tolist()))
    outputs = []
    try:
        with torch.inference_mode():
            for t, frame in enumerate(frames):
                if alpha is not None and resets is not None and resets[t]:
                    net.att.cache = None
                value = torch.from_numpy((frame/255).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                pred = net(value).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()*255
                outputs.append(pred.astype(np.float32))
    finally:
        handle.remove()
        net.att = original
    return outputs, captured


def fuse(outputs, flows, confidence=None, resets=None, alpha=.7):
    result = [outputs[0].copy()]
    for t in range(1, len(outputs)):
        previous, valid = warp_previous(result[-1], flows[t-1])
        if confidence is not None:
            valid &= confidence[t-1]
        if resets is not None and resets[t]:
            valid[:] = False
        weight = alpha*valid[..., None]
        result.append((weight*previous+(1-weight)*outputs[t]).astype(np.float32))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device("cuda")
    net = MobileIELLENetS(12).eval().to(device)
    weights = ROOT/"MobileIE/pretrain/lolv1_best_slim.pkl"
    net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    images = load_images("exploratory_validation")
    names = list(images)
    save_json(out/"config.json", {"experiment_id": "exploration-r2-state", "frames": 24,
        "manifest_sha256": sha256(MANIFEST), "weights_sha256": sha256(weights),
        "gate_alpha": .9, "fusion_alpha": .7, "reset_brightness_ratio": 1.35,
        "reset_thumbnail_threshold": .5, "nlm_h": 3,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    all_rows, start = [], time.perf_counter()
    for index, name in enumerate(names):
        for case in ["pan", "light_step", "occlusion", "cut"]:
            for seed in [0, 1]:
                gt, low, true_flow, true_mask = make_sequence(images[name], images[names[(index+1)%len(names)]], case, seed)
                events, details = detect_changes(low)
                raw, raw_gates = infer_gates(net, low, device)
                state, state_gates = infer_gates(net, low, device, .9)
                reset, reset_gates = infer_gates(net, low, device, .9, events)
                pairs = [forward_backward(low[t-1], low[t]) for t in range(1, len(low))]
                flows, confidence = [p[0] for p in pairs], [p[1] for p in pairs]
                denoised = [cv2.fastNlMeansDenoisingColored(np.clip(frame, 0, 255).astype(np.uint8), None, 3, 3, 7, 21).astype(np.float32) for frame in low]
                nlm, _ = infer_gates(net, denoised, device)
                methods = {"mobileie": raw, "gate_ema": state, "gate_reset": reset,
                    "flow_ema": fuse(raw, flows), "fb_ema": fuse(raw, flows, confidence),
                    "gate_reset_fb": fuse(reset, flows, confidence, events), "input_nlm": nlm,
                    "oracle_ema": fuse(raw, true_flow, true_mask)}
                flow_errors = [np.linalg.norm(a-b, axis=2)[valid].mean() for a, b, valid in zip(flows, true_flow, true_mask) if valid.any()]
                for method, outputs in methods.items():
                    summary, frames = evaluate(outputs, gt, true_flow, true_mask)
                    row = {"scene": name, "case": case, "input_seed": seed, "method": method,
                           "reset_events": sum(events), "flow_epe": float(np.mean(flow_errors)), **summary}
                    all_rows.append(row)
                    with (out/"per_sequence.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    for frame in frames:
                        t = frame["frame"]
                        gates = {"mobileie": raw_gates, "gate_ema": state_gates,
                                 "gate_reset": reset_gates}.get(method)
                        trace = {"scene": name, "case": case, "input_seed": seed, "method": method,
                                 **frame, "detector_event": events[t], "applied_reset": events[t] and method in ["gate_reset", "gate_reset_fb"],
                                 **details[t], "gate": gates[t] if gates is not None else None}
                        with (out/"per_frame.jsonl").open("a") as stream:
                            stream.write(json.dumps(trace, allow_nan=False)+"\n")
                print(json.dumps({"scene": name, "case": case, "input_seed": seed, "rows": len(all_rows), "resets": sum(events)}), flush=True)
    with (out/"per_sequence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    save_json(out/"runtime.json", {"wall_seconds": time.perf_counter()-start})


if __name__ == "__main__":
    main()
