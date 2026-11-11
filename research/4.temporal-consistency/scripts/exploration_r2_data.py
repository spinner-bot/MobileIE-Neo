"""Disjoint-image exploratory data and exact layered motion correspondences."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from video_metrics import edge_error, psnr, ssim, warp_previous

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "research/4.temporal-consistency/data/exploration-r2/kodak_manifest.json"


def load_images(split):
    manifest = json.loads(MANIFEST.read_text())
    images = {}
    for row in manifest["images"]:
        if row["split"] != split:
            continue
        path = ROOT / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"image hash mismatch: {path}")
        with Image.open(path) as image:
            images[row["name"]] = np.asarray(ImageOps.fit(image.convert("RGB"), (768, 512), method=Image.Resampling.LANCZOS), np.float32)
    return images


def degrade(gt, rng, photons=40, jpeg=True):
    low = (np.clip(gt, 0, 255) / 255) ** 2.2 * .35
    low = rng.poisson(low * photons) / photons + rng.normal(0, .008, low.shape)
    low = (np.clip(low, 0, 1) * 255).astype(np.uint8)
    if jpeg:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(low, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        low = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return low.astype(np.float32)


def make_sequence(image, alternate, case, seed, frames=24, width=256, height=192):
    rng = np.random.default_rng(seed)
    targets, inputs, masks, flows, labels = [], [], [], [], []
    pan, side = 2, 48
    object_patch = image[160:160+side, 400:400+side]
    for t in range(frames):
        source = alternate if case == "cut" and t >= frames//2 else image
        gt = source[80:80+height, pan*t:pan*t+width].copy()
        foreground = np.zeros((height, width), dtype=bool)
        if case == "occlusion":
            left, top = 16 + 5*t, 60
            gt[top:top+side, left:left+side] = object_patch
            foreground[top:top+side, left:left+side] = True
        if case == "light_step" and t < frames//2:
            gt *= .6
        targets.append(gt)
        labels.append(foreground)
        inputs.append(degrade(gt, rng))
        if t:
            flow = np.zeros((height, width, 2), np.float32)
            flow[..., 0] = pan
            if case == "occlusion":
                flow[foreground, 0] = -5
            warped_label, valid = warp_previous(labels[t-1].astype(np.float32), flow)
            valid &= (warped_label > .5) == foreground
            if case == "cut" and t == frames//2:
                valid[:] = False
            flows.append(flow)
            masks.append(valid)
    return targets, inputs, flows, masks


def detect_changes(inputs):
    events = [False]
    details = [{"brightness_log_ratio": 0., "normalized_thumbnail_difference": 0.}]
    for prev, cur in zip(inputs[:-1], inputs[1:]):
        ratio = abs(float(np.log((cur.mean()+1)/(prev.mean()+1))))
        a = cv2.resize(prev, (32, 24)) / (prev.mean()+4)
        b = cv2.resize(cur, (32, 24)) / (cur.mean()+4)
        difference = float(np.abs(a-b).mean())
        events.append(bool(ratio > np.log(1.35) or difference > .5))
        details.append({"brightness_log_ratio": ratio, "normalized_thumbnail_difference": difference})
    return events, details


def evaluate(outputs, targets, flows, masks):
    frames = []
    for t, (output, target) in enumerate(zip(outputs, targets)):
        row = {"frame": t, "psnr_rgb": psnr(output, target), "ssim_rgb": ssim(output, target),
               "edge_mae": edge_error(output, target), "output_mean": float(output.mean()),
               "target_mean": float(target.mean()), "flicker": None, "temporal_error": None}
        if t:
            previous, valid = warp_previous(outputs[t-1], flows[t-1])
            previous_gt, _ = warp_previous(targets[t-1], flows[t-1])
            valid &= masks[t-1]
            valid[:8] = valid[-8:] = False
            valid[:, :8] = valid[:, -8:] = False
            if valid.any():
                difference = output-previous
                target_difference = target-previous_gt
                row["flicker"] = float(np.abs(difference)[valid].mean())
                row["temporal_error"] = float(np.abs(difference-target_difference)[valid].mean())
        frames.append(row)
    summary = {}
    for key in ["psnr_rgb", "ssim_rgb", "edge_mae", "flicker", "temporal_error"]:
        values = [r[key] for r in frames if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary, frames
