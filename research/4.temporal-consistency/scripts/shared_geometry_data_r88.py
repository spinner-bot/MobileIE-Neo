"""Registered clean frame-zero geometry and independent synthetic noise pairs."""
import cv2
import numpy as np
import torch

from exploration_r2_data import degrade
from video_metrics import warp_previous


def batch(pool, seed, step):
    _, targets, identities = pool.batch(seed, 8000+step)
    previous, current, shifts, photons = [], [], [], []
    clean = targets[:, 0].numpy().transpose(0, 2, 3, 1)*255
    for b, image in enumerate(clean):
        rng = np.random.default_rng(np.random.SeedSequence([88, seed, step, b, 2]))
        shift = rng.integers(-192, 193, size=2).astype(np.float32)/16
        photon = int(rng.choice([20, 80]))
        field = np.broadcast_to(shift, (*image.shape[:2], 2)).copy()
        shifted, _ = warp_previous(image, field)
        for side, value, destination in ((0, image, previous), (1, shifted, current)):
            noise = np.random.default_rng(np.random.SeedSequence([88, seed, step, b, side]))
            destination.append(degrade(value, noise, photons=photon, jpeg=False))
        shifts.append(shift/4); photons.append(photon)
    values = torch.from_numpy(np.stack(previous+current).transpose(0, 3, 1, 2).copy()/255)
    return values, torch.from_numpy(np.stack(shifts)), dict(seed=seed, step=step,
        data_step=8000+step, identities=identities, native_shifts=(4*np.stack(shifts)).tolist(),
        photons=photons, input_shape=list(values.shape), all_noise_synthetic=True)
