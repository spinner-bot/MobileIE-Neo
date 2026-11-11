"""Exploratory bounded FST mixture; no novelty or zero-cost claim."""
import numpy as np
import torch
from torch import nn


class BoundedFST(nn.Module):
    def __init__(self, original, limit):
        super().__init__()
        self.block1 = original.block1
        self.weight1, self.weight2, self.bias = original.weight1, original.weight2, original.bias
        self.gamma = nn.Parameter(torch.zeros((), device=self.weight1.device))
        self.register_buffer("limit", torch.as_tensor(limit, dtype=self.weight1.dtype, device=self.weight1.device).reshape(1, -1, 1, 1))

    def forward(self, value):
        feature = self.block1(value)
        ordinary = (self.weight1*feature)*(self.weight2*feature)+self.bias
        bounded = self.limit * torch.tanh(feature/self.limit)
        changed = (self.weight1*bounded)*(self.weight2*bounded)+self.bias
        return ordinary + self.gamma*(changed-ordinary)


def apply_bounded(net, limits):
    for name in ["head", "body"]:
        setattr(net, name, BoundedFST(getattr(net, name), limits[name]))
    return net


def project_mixtures(net):
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, BoundedFST):
                module.gamma.clamp_(0, 1)


def calibrate_limits(net, images, device):
    captured, values = {}, {"head": [], "body": []}
    handles = [getattr(net, name).block1.register_forward_hook(
        lambda module, inp, output, key=name: captured.__setitem__(key, output.detach())) for name in values]
    rng = np.random.default_rng(901)
    try:
        with torch.inference_mode():
            for image in images.values():
                for _ in range(2):
                    x = int(rng.integers(0, image.shape[1]-128+1))
                    y = int(rng.integers(0, image.shape[0]-128+1))
                    low = (image[y:y+128, x:x+128]/255)**2.2*.35
                    tensor = torch.from_numpy(low.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                    net(tensor)
                    for name in values:
                        values[name].append(captured[name][0].abs().flatten(1).cpu().numpy())
        return {name: np.maximum(np.quantile(np.concatenate(items, axis=1), .995, axis=1), 1e-3).tolist()
                for name, items in values.items()}
    finally:
        for handle in handles:
            handle.remove()
