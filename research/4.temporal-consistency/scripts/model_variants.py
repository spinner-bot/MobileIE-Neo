"""MobileIE 消融变体工厂（供 4 号工作使用，不改上游默认行为）。

变体：
    full            原模型
    no_global_att   去掉全局通道注意力（x2 = 1）
    no_local_att    去掉局部注意力（x3 = 1）
    no_att          两路注意力都去掉（x4 = x1）
    no_fst          去掉 FST/FSTS 的二次融合（只用 block1 输出）
    no_mbr          训练态直接用单分支网络（无多分支重参数化）

用法：
    python research/4.temporal-consistency/scripts/model_variants.py   # 跑等价性冒烟测试
"""
import copy
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'MobileIE'))

from model.lle import MobileIELLENet, MobileIELLENetS       # noqa: E402
from model.utils import DropBlock, FST, FSTS               # noqa: E402


# ---------------------------------------------------------------------------
# 推理态（slim）变体
# ---------------------------------------------------------------------------
class NoGlobalAttS(MobileIELLENetS):
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        max_out, _ = torch.max(x1, dim=1, keepdim=True)
        x3 = self.att1(max_out)
        return self.tail(x3 * x1)


class NoLocalAttS(MobileIELLENetS):
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        return self.tail(x2 * x1)


class NoAttS(MobileIELLENetS):
    def forward(self, x):
        return self.tail(self.body(self.head(x)))


# ---------------------------------------------------------------------------
# 训练态变体
# ---------------------------------------------------------------------------
class NoGlobalAttLLENet(MobileIELLENet):
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        max_out, _ = torch.max(x1, dim=1, keepdim=True)
        x3 = self.att1(max_out)
        return self.tail(x3 * x1)

    def slim(self):
        base = MobileIELLENet.slim(self)
        net = NoGlobalAttS(self.channels)
        net.load_state_dict(base.state_dict())
        return net


class NoLocalAttLLENet(MobileIELLENet):
    def forward(self, x):
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        return self.tail(x2 * x1)

    def slim(self):
        base = MobileIELLENet.slim(self)
        net = NoLocalAttS(self.channels)
        net.load_state_dict(base.state_dict())
        return net


class NoAttLLENet(MobileIELLENet):
    def forward(self, x):
        return self.tail(self.body(self.head(x)))

    def slim(self):
        base = MobileIELLENet.slim(self)
        net = NoAttS(self.channels)
        net.load_state_dict(base.state_dict())
        return net


class FSTPass(nn.Module):
    """去掉 FST 的二次融合，只保留其内部 block1。"""

    def __init__(self, block1):
        super().__init__()
        self.block1 = block1

    def forward(self, x):
        return self.block1(x)


def strip_fst(net):
    for name in ('head', 'body'):
        m = getattr(net, name, None)
        if isinstance(m, (FST, FSTS)):
            setattr(net, name, FSTPass(m.block1))
    return net


class NoFSTLLENet(MobileIELLENet):
    def __init__(self, channels, rep_scale=4):
        super().__init__(channels, rep_scale)
        strip_fst(self)

    def slim(self):
        base = MobileIELLENet.slim(self)
        strip_fst(base)
        net = NoFSTS(self.channels)
        net.load_state_dict(base.state_dict())
        return net


class NoFSTS(MobileIELLENetS):
    def __init__(self, channels):
        super().__init__(channels)
        self.channels = channels
        strip_fst(self)


class NoMBRNet(MobileIELLENetS):
    """单分支训练态：直接训练 slim 结构，不经过多分支重参数化。"""

    def __init__(self, channels, rep_scale=4):
        super().__init__(channels)
        self.channels = channels
        self.drop = DropBlock(3)

    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail(x)

    def slim(self):
        return copy.deepcopy(self)


_VARIANTS = {
    'full': (MobileIELLENet, MobileIELLENetS),
    'no_global_att': (NoGlobalAttLLENet, NoGlobalAttS),
    'no_local_att': (NoLocalAttLLENet, NoLocalAttS),
    'no_att': (NoAttLLENet, NoAttS),
    'no_fst': (NoFSTLLENet, NoFSTS),
    'no_mbr': (NoMBRNet, NoMBRNet),
}


def build_variant(name, training=True, channels=12):
    """返回指定变体；training=True 为多分支训练态，False 为 slim 推理态。"""
    if name not in _VARIANTS:
        raise ValueError('unknown variant: %s (choose from %s)' % (name, list(_VARIANTS)))
    train_cls, slim_cls = _VARIANTS[name]
    if training:
        return train_cls(channels)
    return slim_cls(channels)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _smoke():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    print('%-14s %8s %8s %12s' % ('variant', 'train', 'slim', 'max|diff|'))
    for name in ('full', 'no_global_att', 'no_local_att', 'no_att', 'no_fst', 'no_mbr'):
        tr = build_variant(name, training=True).eval()
        sl = tr.slim().eval()
        with torch.no_grad():
            diff = (tr(x) - sl(x)).abs().max().item()
        print('%-14s %8d %8d %12.2e' % (name, count_params(tr), count_params(sl), diff))
        assert diff < 1e-5, 're-parameterization mismatch for %s' % name
    print('all variants OK')


if __name__ == '__main__':
    _smoke()
