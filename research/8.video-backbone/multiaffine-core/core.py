"""Fixed M1 prototype: raw disjoint-slot multilinearity, no motion or quality claim."""
import torch
from torch import nn
import torch.nn.functional as F


class AffineBilinear(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, first, second):
        return .5 * (first + second) + self.gamma * first * second + self.bias


class MultiaffineCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.first = AffineBilinear()
        self.body = nn.Conv2d(16, 16, 3, dilation=2, padding=2)
        self.second = AffineBilinear()
        self.decoder = nn.Conv2d(16, 12, 3, padding=1)
        self.reset()

    @staticmethod
    def check_image(image):
        assert image.ndim == 4 and image.shape[1] == 3
        assert image.shape[-2] % 2 == image.shape[-1] % 2 == 0

    def finish(self, first_pair, second_pair):
        return F.pixel_shuffle(self.decoder(self.second(first_pair, second_pair)), 2)

    def forward(self, image):
        """Single-frame diagonal spatial control. No pretrained-function inheritance."""
        self.check_image(image)
        z = self.encoder(image)
        c = self.body(self.first(z, z))
        return self.finish(c, c)

    def four(self, images, overlap=False):
        """Old-to-new four slots. Overlap is a semantic counterexample, not a method."""
        assert images.ndim == 5 and images.shape[1:3] == (4, 3)
        self.check_image(images[:, 0])
        b = images.shape[0]
        z = self.encoder(images.flatten(0, 1)).unflatten(0, (b, 4))
        old = self.first(z[:, 1], z[:, 2]) if overlap else self.first(z[:, 0], z[:, 1])
        new = self.first(z[:, 2], z[:, 3])
        return self.finish(self.body(new), self.body(old))

    def reset(self):
        self._previous_z = None
        self._pairs = []
        self._index = 0
        self._identity = None

    def step(self, image):
        """Inference state: three current-diagonal outputs, then disjoint frame pairs."""
        self.check_image(image)
        identity = (image.shape, image.device, image.dtype)
        if self._identity is not None and identity != self._identity:
            self.reset()
        self._identity = identity
        index = self._index
        z = self.encoder(image)
        previous = z if self._previous_z is None else self._previous_z
        paired = self.body(self.first(z, previous))
        source = (index, index) if index == 0 else (index - 1, index)
        if index >= 3:
            old, old_source = self._pairs[0]
            assert not set(source) & set(old_source)
            assert set(source) | set(old_source) == set(range(index - 3, index + 1))
            output = self.finish(paired, old)
            provenance = dict(route='disjoint', factors=[list(source), list(old_source)])
        else:
            diagonal = paired if index == 0 else self.body(self.first(z, z))
            output = self.finish(diagonal, diagonal)
            provenance = dict(route='startup_diagonal', frame=index)
        self._previous_z = z.detach().clone(memory_format=torch.contiguous_format)
        self._pairs.append((paired.detach().clone(memory_format=torch.contiguous_format), source))
        self._pairs = self._pairs[-2:]
        self._index += 1
        return output, provenance

    def state_storage_bytes(self):
        values = ([] if self._previous_z is None else [self._previous_z]) + [p[0] for p in self._pairs]
        return sum(value.untyped_storage().nbytes() for value in values)

    @staticmethod
    def budget(height, width, batch=1, element_bytes=4):
        assert height % 2 == width % 2 == 0
        return dict(parameters=5340, steady_convolution_MAC=1308 * height * width,
            steady_persistent_state_bytes=batch * 3 * 16 * (height // 2) * (width // 2) * element_bytes,
            extra_operations_not_counted='bilinear arithmetic, copies, pixel shuffle, startup extra body; no phone timing')
