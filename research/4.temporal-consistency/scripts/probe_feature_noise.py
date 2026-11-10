"""Fixed-scene Monte Carlo probes of pretrained FST/HDPA noise response.

This is a diagnostic, not a trained ablation: raw feature scales cannot by
themselves establish which layer causes final output flicker.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from explore_temporal_controls import ROOT, SCENES, MobileIELLENetS, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draws", type=int, default=16)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("at least two draws required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    weights = ROOT / "MobileIE/pretrain/lolv1_best_slim.pkl"
    net = MobileIELLENetS(12).eval().to(device)
    net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    layers = ["head.block1", "head", "body.block1", "body", "att", "att1", "tail"]
    captured = {}
    handles = [net.get_submodule(name).register_forward_hook(
        lambda module, inp, value, key=name: captured.__setitem__(key, value.detach())) for name in layers]
    rows, fst_rows, start = [], [], time.perf_counter()
    def tensor(images):
        array = np.stack(images).astype(np.float32) / 255
        return torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(device)
    def core(value):
        return value[..., 8:-8, 8:-8] if min(value.shape[-2:]) > 16 else value
    with torch.inference_mode():
        for index, (scene, path) in enumerate(SCENES.items()):
            image = np.asarray(Image.open(path).convert("RGB").resize((1024, 768), Image.Resampling.LANCZOS), np.float32)
            gt = image[264:504, :320]
            low = (gt / 255) ** 2.2 * .35
            clean = (low * 255).astype(np.uint8)
            net(tensor([clean]))
            reference = {key: value.clone() for key, value in captured.items()}
            for photons in [20, 40, 80]:
                rng = np.random.default_rng(20261110 + index * 100 + photons)
                noisy = [np.clip((rng.poisson(low * photons) / photons + rng.normal(0, .008, low.shape)) * 255, 0, 255).astype(np.uint8) for _ in range(args.draws)]
                for jpeg in [False, True]:
                    images = noisy
                    if jpeg:
                        images = []
                        for frame in noisy:
                            _, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                            images.append(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                    sums, squares, fst = {}, {}, {}
                    for offset in range(0, args.draws, 4):
                        net(tensor(images[offset:offset+4]))
                        for name in layers:
                            delta = core(captured[name] - reference[name])
                            ds, ss = delta.sum(dim=0), (delta ** 2).sum(dim=0)
                            sums[name] = sums.get(name, 0) + ds
                            squares[name] = squares.get(name, 0) + ss
                        for name in ["head", "body"]:
                            u = reference[name + ".block1"]
                            du = captured[name + ".block1"] - u
                            module = net.get_submodule(name)
                            a = module.weight1 * module.weight2
                            linear, quadratic = 2*a*u*du, a*du*du
                            observed = captured[name] - reference[name]
                            error = float((observed-linear-quadratic).abs().max())
                            scale = max(float(observed.abs().max()), float(linear.abs().max()), float(quadratic.abs().max()), 1e-8)
                            previous = fst.setdefault(name, {"max_identity_error": 0., "max_scaled_identity_error": 0., "linear_energy": 0., "quadratic_energy": 0., "cross_energy": 0., "observed_energy": 0.})
                            previous["max_identity_error"] = max(previous["max_identity_error"], error)
                            previous["max_scaled_identity_error"] = max(previous["max_scaled_identity_error"], error / scale)
                            size = du.shape[0]
                            previous["linear_energy"] += float((core(linear)**2).mean()) * size
                            previous["quadratic_energy"] += float((core(quadratic)**2).mean()) * size
                            previous["cross_energy"] += float((2*core(linear)*core(quadratic)).mean()) * size
                            previous["observed_energy"] += float((core(observed)**2).mean()) * size
                    for name in layers:
                        second = squares[name] / args.draws
                        bias = sums[name] / args.draws
                        signal = float((core(reference[name])**2).mean().sqrt())
                        perturbation = float(second.mean().sqrt())
                        rows.append({"scene": scene, "photons": photons, "jpeg": jpeg, "layer": name,
                                     "signal_rms": signal, "perturbation_rms": perturbation,
                                     "relative_perturbation": perturbation / max(signal, 1e-8),
                                     "bias_rms": float((bias**2).mean().sqrt()),
                                     "random_std_rms": float((second-bias**2).clamp_min(0).mean().sqrt())})
                    for name, values in fst.items():
                        for key in list(values):
                            if not key.startswith("max_"):
                                values[key] /= args.draws
                        fst_rows.append({"scene": scene, "photons": photons, "jpeg": jpeg, "layer": name, **values})
                    print(json.dumps({"scene": scene, "photons": photons, "jpeg": jpeg}), flush=True)
    for handle in handles:
        handle.remove()
    save_json(out / "layer_response.json", rows)
    save_json(out / "fst_identity.json", fst_rows)
    with (out / "layer_response.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / "config.json", {"experiment_id": "exploration-r1-feature-noise-scaled-check", "draws": args.draws,
              "seed_rule": "20261110+scene_index*100+photons", "photons": [20, 40, 80],
              "jpeg": [False, True], "weights_sha256": sha256(weights),
              "wall_seconds": time.perf_counter()-start, "note": "conditional feature response, not causal contribution attribution"})


if __name__ == "__main__":
    main()
