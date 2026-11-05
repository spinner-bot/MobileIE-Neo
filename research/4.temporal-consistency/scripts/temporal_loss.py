"""方案 A 原型：训练期时序一致性损失（推理端不增加任何模块）。

核心思想：对相邻帧的**增强输出**做运动对齐后施加一致性约束，
梯度只回传到网络参数，不改变推理图；配合 MobileIE 的重参数化，
推理端仍是 4K 参数级单分支网络。

用法（冒烟测试）：
    python research/4.temporal-consistency/scripts/temporal_loss.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _base_grid(h, w, device, dtype):
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)       # 1×H×W×2


def warp_with_flow(img, flow):
    """按光流把上一帧 warp 到当前帧视角。

    img:  B×C×H×W
    flow: B×2×H×W，flow[:,0]=x 位移（像素），flow[:,1]=y 位移；
          采样位置 = 当前像素坐标 + flow（即 flow 指向上一帧中的对应点）。
    """
    b, _, h, w = img.shape
    base = _base_grid(h, w, img.device, img.dtype)         # 1×H×W×2（归一化坐标）
    dx = flow[:, 0] * (2.0 / max(w - 1, 1))
    dy = flow[:, 1] * (2.0 / max(h - 1, 1))
    grid = torch.stack([base[..., 0] + dx, base[..., 1] + dy], dim=-1)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)


class TemporalConsistencyLoss(nn.Module):
    """对齐后的 Charbonnier 时序一致性损失。

    forward(pred, prev, flow=None, mask=None)
        pred: 当前帧增强输出 B×C×H×W
        prev: 上一帧增强输出 B×C×H×W
        flow: 可选，B×2×H×W；为 None 时按静止相机处理（直接逐像素比较）
        mask: 可选，B×1×H×W，1 表示有效像素（如去遮挡）
        detach_prev: 是否截断上一帧梯度（省显存）
    """

    def __init__(self, eps=1e-3, detach_prev=True):
        super().__init__()
        self.eps = eps
        self.detach_prev = detach_prev

    def forward(self, pred, prev, flow=None, mask=None):
        if self.detach_prev:
            prev = prev.detach()
        if flow is not None:
            prev = warp_with_flow(prev, flow)
        diff = pred - prev
        loss = torch.sqrt(diff * diff + self.eps ** 2)
        if mask is not None:
            weight = mask.expand_as(loss)
            return (loss * weight).sum() / weight.sum().clamp_min(1.0)
        return loss.mean()


class BrightnessConsistencyLoss(nn.Module):
    """全局亮度一致性（Zero-DCE 视频版的简化版），抑制整体明暗脉动。"""

    def __init__(self, detach_prev=True):
        super().__init__()
        self.detach_prev = detach_prev

    def forward(self, pred, prev):
        if self.detach_prev:
            prev = prev.detach()
        return (pred.mean(dim=(1, 2, 3)) - prev.mean(dim=(1, 2, 3))).abs().mean()


def translation_flow(batch, h, w, dx, dy, device=None, dtype=torch.float32):
    """生成整幅平移的已知光流（像素单位）：采样位置 = 当前坐标 + (dx, dy)。"""
    flow = torch.zeros(batch, 2, h, w, device=device, dtype=dtype)
    flow[:, 0] = dx
    flow[:, 1] = dy
    return flow


def _smoke():
    torch.manual_seed(0)
    b, c, h, w = 2, 3, 32, 32
    prev = torch.rand(b, c, h, w)
    # 当前帧 = 上一帧向右平移 2 像素；对应的采样光流为 x 方向 -2
    cur = torch.roll(prev, shifts=2, dims=3)
    flow = translation_flow(b, h, w, -2.0, 0.0, device=prev.device)

    loss = TemporalConsistencyLoss(eps=1e-6)
    # 左侧 2 列为越界/新内容，时序损失应将其屏蔽
    mask = torch.ones(b, 1, h, w)
    mask[:, :, :, :2] = 0
    exact = loss(cur, prev, flow, mask).item()
    no_flow = loss(cur, prev, mask=mask).item()
    assert exact < 1e-4, exact
    assert no_flow > exact, (exact, no_flow)
    print('temporal loss (masked): exact-flow %.2e, no-flow %.4f' % (exact, no_flow))
    print('temporal loss (unmasked, border effect): %.4f' % loss(cur, prev, flow).item())
    print('brightness loss: %.4f' % BrightnessConsistencyLoss()(cur, prev).item())
    print('temporal_loss OK')


if __name__ == '__main__':
    _smoke()
