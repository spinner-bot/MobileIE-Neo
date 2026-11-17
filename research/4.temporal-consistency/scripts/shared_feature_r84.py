"""Fixed hard-matching causal prototype; no learned motion network or novelty claim."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from video_metrics import warp_previous

MODES = ('signed', 'energy', 'rgb')
RADIUS = 4


def compact_feature(value):
    value = F.avg_pool2d(value.detach(), 4, 4)
    value = value-value.mean(dim=(-2, -1), keepdim=True)
    return F.normalize(value, dim=1, eps=1e-6).contiguous().clone()


def shared_forward(backbone, image):
    """Exactly one ordinary F0 forward; capture pooled features, not full tensors."""
    if image.ndim != 4 or image.shape[1] != 3 or any(s % 4 for s in image.shape[-2:]):
        raise ValueError('Expected BCHW RGB with sides divisible by four')
    captured = {}
    def hook(name):
        def record(module, inputs, output):
            if name in captured:
                raise RuntimeError('Unexpected repeated head invocation')
            captured[name] = compact_feature(output)
        return record
    handles = [backbone.head.block1.register_forward_hook(hook('signed')),
               backbone.head.register_forward_hook(hook('energy'))]
    try:
        with torch.no_grad():
            raw = backbone(image).clamp(0, 1)
            captured['rgb'] = compact_feature(raw)
    finally:
        for handle in handles:
            handle.remove()
    assert set(captured) == set(MODES)
    return raw, captured


def match(current, previous):
    """Quarter-grid current-to-previous maximum cosine, fixed row-major ties."""
    if current.shape != previous.shape or current.ndim != 4:
        raise ValueError('Matching feature shapes must be identical BCHW')
    batch, _, height, width = current.shape
    padded = F.pad(previous, (RADIUS, RADIUS, RADIUS, RADIUS))
    best = current.new_full((batch, height, width), -torch.inf)
    flow = current.new_zeros((batch, 2, height, width))
    y = torch.arange(height, device=current.device)[:, None]
    x = torch.arange(width, device=current.device)[None, :]
    for dy in range(-RADIUS, RADIUS+1):
        for dx in range(-RADIUS, RADIUS+1):
            other = padded[:, :, RADIUS+dy:RADIUS+dy+height, RADIUS+dx:RADIUS+dx+width]
            score = (current*other).sum(1)
            valid = (x+dx >= 0) & (x+dx < width) & (y+dy >= 0) & (y+dy < height)
            score = score.masked_fill(~valid, -torch.inf)
            better = score > best
            best = torch.where(better, score, best)
            flow[:, 0] = torch.where(better, dx, flow[:, 0])
            flow[:, 1] = torch.where(better, dy, flow[:, 1])
    assert torch.isfinite(flow).all() and torch.isfinite(best).all()
    return flow


def fields(current, previous, size):
    backward = match(current, previous)
    forward = match(previous, current)
    boundary = (backward.abs().amax(dim=1) == RADIUS).float().mean(dim=(-2, -1))
    backward = (4*F.interpolate(backward, size=size, mode='bilinear', align_corners=False))
    forward = (4*F.interpolate(forward, size=size, mode='bilinear', align_corners=False))
    result = []
    for b, f, edge in zip(backward, forward, boundary):
        b = b.permute(1, 2, 0).contiguous().numpy()
        f = f.permute(1, 2, 0).contiguous().numpy()
        sampled, valid = warp_previous(f, b)
        error = np.linalg.norm(b+sampled, axis=2)
        threshold = 1+.05*(np.linalg.norm(b, axis=2)+np.linalg.norm(sampled, axis=2))
        result.append((b, valid & (error < threshold), float(edge)))
    return result


class FeatureStream:
    """Single-sample deployment state; no previous raw RGB or full-res features."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_feature = self.previous_output = None

    def step(self, raw, feature):
        if feature.shape[0] != 1 or raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError('This reference stream handles one HWC sample')
        if self.previous_feature is None:
            output, field = raw.copy(), None
        else:
            field = fields(feature, self.previous_feature, raw.shape[:2])[0]
            flow, mask, _ = field
            moved, valid = warp_previous(self.previous_output, flow)
            alpha = .5*(mask & valid)[..., None]
            output = (raw+alpha*(moved-raw)).astype(np.float32)
        self.previous_feature = feature.detach().contiguous().clone()
        self.previous_output = output.copy()
        return output, field

    def state_bytes(self):
        if self.previous_feature is None:
            return 0
        assert self.previous_feature.untyped_storage().nbytes() == self.previous_feature.numel()*4
        assert self.previous_output.flags.owndata
        return self.previous_feature.untyped_storage().nbytes()+self.previous_output.nbytes


def qualification():
    generator = torch.Generator().manual_seed(84001)
    previous = F.normalize(torch.randn(1, 12, 16, 16, generator=generator), dim=1)
    current = torch.zeros_like(previous)
    current[:, :, 1:, :-2] = previous[:, :, :-1, 2:]
    estimate = match(current, previous)
    inside = estimate[:, :, 1:, :-2]
    expected = inside.new_tensor([2., -1.])[None, :, None, None]
    error = float((inside-expected).abs().max())
    assert error == 0.
    assert float(match(previous, previous).abs().max()) == 0.
    zero = match(torch.zeros_like(previous), torch.zeros_like(previous))
    assert torch.isfinite(zero).all()
    # Captured feature must not keep a larger parent tensor's storage alive.
    compact = compact_feature(torch.randn(1, 12, 64, 64, generator=generator))
    stream = FeatureStream()
    raw = np.zeros((64, 64, 3), dtype=np.float32)
    output, first_field = stream.step(raw, compact)
    expected_bytes = (12*16*16+3*64*64)*4
    assert first_field is None and np.array_equal(output, raw)
    assert stream.state_bytes() == expected_bytes
    return dict(passed=True, known_quarter_translation=[2, -1], max_translation_error=error,
                random_self_flow_max=0., zero_features_finite=True,
                actual_state_bytes=stream.state_bytes(), expected_state_bytes=expected_bytes,
                backbone_frames=0, optimizer_updates=0,
                scope='Fixed implementation qualification; no real-video matching assertion.')
