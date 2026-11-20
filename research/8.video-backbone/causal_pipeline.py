"""Train/infer the same own-preview DIS and native low-light recurrence."""
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import nn

LEGACY = Path(__file__).resolve().parents[1] / '4.temporal-consistency/scripts'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
from budget_causal_r33 import transport
from fixed_dis_r86 import FixedDIS
from video_metrics import warp_previous
from packed_backbone import PackedBackbone, FusionGate


@dataclass
class CausalState:
    low: torch.Tensor
    preview_gray: np.ndarray

    def detach(self):
        return CausalState(self.low.detach(), self.preview_gray)

    def storage_bytes(self):
        return self.low.untyped_storage().nbytes() + self.preview_gray.nbytes


def preview_gray(preview):
    frames = preview.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np.stack([cv2.cvtColor(np.clip(np.rint(frame), 0, 255).astype(np.uint8),
                                  cv2.COLOR_RGB2GRAY) for frame in frames])


class PreviewDIS:
    """Same fixed R86 DIS/acceptance; retain only uint8 grayscale for motion."""
    def __init__(self):
        self.engine = FixedDIS()

    def __call__(self, previous, current):
        if previous.shape != current.shape or previous.ndim != 3 or previous.dtype != np.uint8:
            raise ValueError('Expected matching BHW uint8 preview luma arrays')
        flows, masks = [], []
        for old, new in zip(previous, current):
            back = self.engine.backward.calc(new, old, None)
            forward = self.engine.forward.calc(old, new, None)
            if not (np.isfinite(back).all() and np.isfinite(forward).all()):
                raise FloatingPointError('Nonfinite DIS field')
            moved_forward, valid = warp_previous(forward, back)
            error = np.linalg.norm(back + moved_forward, axis=2)
            threshold = 1 + .05 * (np.linalg.norm(back, axis=2) + np.linalg.norm(moved_forward, axis=2))
            flows.append(back)
            masks.append(valid & (error < threshold))
            self.engine.pairs += 1
        return np.stack(flows), np.stack(masks)


class CausalEnhancer(nn.Module):
    def __init__(self, route):
        super().__init__()
        if route not in ('spatial', 'fixed', 'learned'):
            raise ValueError('Unknown route')
        self.route = route
        self.backbone = PackedBackbone()
        self.gate = FusionGate() if route == 'learned' else None
        self.motion = PreviewDIS() if route != 'spatial' else None

    def forward(self, current, state=None, *, history=True):
        preview = self.backbone(current)
        if self.route == 'spatial':
            return preview, None, preview, dict(backbone_images=len(current), DIS_pairs=0)
        gray = preview_gray(preview)
        if state is None or not history:
            return preview, CausalState(current, gray), preview, dict(backbone_images=len(current), DIS_pairs=0)
        if state.low.shape != current.shape or state.low.device != current.device or state.low.dtype != current.dtype:
            raise ValueError('Reset state at a size/device/dtype/sequence boundary')
        flow, accepted = self.motion(state.preview_gray, gray)
        offset = torch.from_numpy(flow.transpose(0, 3, 1, 2).copy()).to(current)
        mask = torch.from_numpy(accepted[:, None].copy()).to(device=current.device)
        moved, bounds = transport(state.low, current, offset)
        valid = mask & bounds
        alpha = self.gate(current, moved) if self.gate is not None else torch.full_like(current[:, :1], .5)
        alpha = alpha * valid.to(current.dtype)
        fused = current + alpha * (moved - current)
        output = self.backbone(fused)
        details = dict(backbone_images=2 * len(current), DIS_pairs=len(current),
                       accepted_fraction=float(valid.float().mean()),
                       alpha_all_mean=float(alpha.detach().mean()))
        return output, CausalState(fused, gray), preview, details

    def window(self, low):
        if low.ndim != 5 or low.shape[1] != 4:
            raise ValueError('Registered training windows are BTCHW with T=4')
        state, outputs, previews, details = None, [], [], []
        for current in low.unbind(1):
            output, state, preview, info = self(current, state)
            outputs.append(output)
            previews.append(preview)
            details.append(info)
        return torch.stack(outputs, 1), torch.stack(previews, 1), state, details
