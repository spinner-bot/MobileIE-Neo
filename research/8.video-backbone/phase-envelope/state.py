"""Fixed P1 coarse band-energy intervention; approximate differences, not exact Riesz."""
import torch
import torch.nn.functional as F

ARMS = ('raw', 'feature_ema', 'envelope_spatial', 'envelope_temporal')


def box3(x):
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)


def finish(net, u):
    # Preserve the original two multiplications, attention and tail order.
    x0 = (net.head.weight1 * u) * (net.head.weight2 * u) + net.head.bias
    x1 = net.body(x0)
    x2 = net.att(x1)
    max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)
    x3 = net.att1(max_out)
    return net.tail(torch.mul(x3, x2) * x1).clamp(0, 1)


class Pipeline:
    def __init__(self, net, arm):
        assert arm in ARMS
        self.net, self.arm, self.state = net, arm, None

    def __call__(self, image):
        if self.arm == 'raw':
            return self.net(image).clamp(0, 1)
        u = self.net.head.block1(image)
        if self.arm == 'feature_ema':
            q = F.avg_pool2d(u, 4)
            s = q if self.state is None else .5 * q + .5 * self.state
            out = u + F.interpolate(s - q, size=u.shape[-2:], mode='bilinear', align_corners=False)
        else:
            b = u - box3(u)
            padded = F.pad(b, (1, 1, 1, 1), mode='replicate')
            dx = .5 * (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2])
            dy = .5 * (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1])
            q = F.avg_pool2d(b.square() + dx.square() + dy.square(), 4)
            if self.arm == 'envelope_spatial':
                s = box3(q)
            else:
                s = q if self.state is None else .5 * q + .5 * self.state
            gain = ((s + 1e-6) / (q + 1e-6)).clamp(.5, 2).sqrt()
            gain = F.interpolate(gain, size=u.shape[-2:], mode='bilinear', align_corners=False)
            out = u + (gain - 1) * b
        if self.arm != 'envelope_spatial':
            self.state = s.detach().clone(memory_format=torch.contiguous_format)
        return finish(self.net, out)

    def state_bytes(self):
        return 0 if self.state is None else self.state.untyped_storage().nbytes()
