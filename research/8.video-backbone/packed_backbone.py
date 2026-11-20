"""Fixed topic8 spatial feasibility core; no equivariance or novelty claim."""
import torch
from torch import nn
import torch.nn.functional as F


class ResidualSquare(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.tensor(.1))
        self.weight2 = nn.Parameter(torch.tensor(0.))
        self.bias = nn.Parameter(torch.zeros(1, 64, 1, 1))

    def forward(self, value):
        return value + (self.weight1 * value) * (self.weight2 * value) + self.bias


class PackedBackbone(nn.Module):
    """Retain all input phases, then learn at quarter resolution, residual RGB."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(48, 64, 1), nn.PReLU(64),
                                  nn.Conv2d(64, 64, 3, padding=1, groups=64), ResidualSquare())
        self.body = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64),
                                  nn.Conv2d(64, 64, 1), ResidualSquare())
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(64, 64, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(1, 64, 1), nn.Sigmoid())
        self.tail = nn.Conv2d(64, 48, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 4:
            raise ValueError('Expected BCHW RGB, height and width >=4')
        height, width = image.shape[-2:]
        value = F.pixel_unshuffle(F.pad(image, (0, (-width) % 4, 0, (-height) % 4), mode='replicate'), 4)
        value = self.body(self.head(value))
        channel = self.channel(value)
        spatial = self.spatial((channel * value).max(dim=1, keepdim=True).values)
        residual = F.pixel_shuffle(self.tail(spatial * channel * value), 4)
        return image + residual[..., :height, :width]

    @staticmethod
    def convolution_macs(height, width):
        return 11456 * ((height + 3) // 4) * ((width + 3) // 4) + 4096


class FusionGate(nn.Module):
    """Uncalibrated RGB consistency gate, initially fixed half fusion."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(9, 8, 1), nn.PReLU(8),
                                    nn.Conv2d(8, 8, 3, padding=4, dilation=4, groups=8),
                                    nn.PReLU(8), nn.Conv2d(8, 1, 1), nn.Sigmoid())
        nn.init.zeros_(self.layers[-2].weight)
        nn.init.zeros_(self.layers[-2].bias)

    def forward(self, current, moved):
        return self.layers(torch.cat((current, moved, (current - moved).abs()), dim=1))
