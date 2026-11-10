"""Shared float-domain motion/quality primitives for exploration-r1 and later.

Images are HWC RGB arrays in [0,255]. A positive translation dx means that
the current crop starts dx pixels to the right in the source scene.
"""
from __future__ import annotations

import cv2
import numpy as np


def translation_overlap(previous, current, dx, dy=0):
    if previous.shape != current.shape:
        raise ValueError("image shapes differ")
    h, w = previous.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        raise ValueError("no valid overlap")
    px, py = max(dx, 0), max(dy, 0)
    cx, cy = max(-dx, 0), max(-dy, 0)
    ow, oh = w - abs(dx), h - abs(dy)
    return (previous[py:py + oh, px:px + ow].astype(np.float32),
            current[cy:cy + oh, cx:cx + ow].astype(np.float32))


def aligned_difference(previous, current, dx, dy=0):
    p, c = translation_overlap(previous, current, dx, dy)
    return float(np.abs(c - p).mean())


def temporal_error(previous, current, previous_gt, current_gt, dx, dy=0):
    p, c = translation_overlap(previous, current, dx, dy)
    pg, cg = translation_overlap(previous_gt, current_gt, dx, dy)
    return float(np.abs((c - p) - (cg - pg)).mean())


def warp_previous(previous, backward_flow):
    """Gather previous at current-coordinate + current-to-previous flow."""
    h, w = previous.shape[:2]
    if backward_flow.shape != (h, w, 2):
        raise ValueError("flow must be HxWx2")
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy = x + backward_flow[..., 0], y + backward_flow[..., 1]
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    warped = cv2.remap(previous.astype(np.float32), sx, sy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped, valid


def translation_warp(previous, dx, dy=0):
    flow = np.empty((*previous.shape[:2], 2), np.float32)
    flow[..., 0], flow[..., 1] = dx, dy
    return warp_previous(previous, flow)


def backward_flow(previous, current):
    """Estimate current -> previous, the flow needed for gathering previous."""
    def gray(image):
        u8 = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(gray(current), gray(previous), None,
                                      0.5, 3, 15, 3, 5, 1.2, 0)


def psnr(output, reference):
    mse = float(np.mean((output.astype(np.float64) - reference.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(output, reference):
    x = output.astype(np.float32) / 255.0
    y = reference.astype(np.float32) / 255.0
    if min(x.shape[:2]) < 11:
        raise ValueError("SSIM needs at least 11x11 pixels")
    blur = lambda image: cv2.GaussianBlur(image, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    vx, vy = blur(x * x) - mx * mx, blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    score = ((2 * mx * my + .01 ** 2) * (2 * cov + .03 ** 2) /
             ((mx * mx + my * my + .01 ** 2) * (vx + vy + .03 ** 2)))
    return float(score[5:-5, 5:-5].mean())


def edge_error(output, reference):
    """Mean absolute error of x/y luminance gradients, in gray levels/pixel."""
    def gradients(image):
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        return (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8,
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8)
    ox, oy = gradients(output)
    gx, gy = gradients(reference)
    return float((np.abs(ox - gx).mean() + np.abs(oy - gy).mean()) / 2)
