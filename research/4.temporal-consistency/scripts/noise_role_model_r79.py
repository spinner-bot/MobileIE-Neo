"""Fixed R16 architecture, independent current-only and two causal training controls."""
from copy import deepcopy

import torch

from multiscale_fst_r16 import MultiscaleFST
from train_exploration_r2 import new_model

MODES = ('spatial', 'balanced', 'curriculum')
STEPS = 4000


def models(seed):
    torch.manual_seed(7900+seed)
    base = MultiscaleFST(new_model(torch.device('cpu')), 'raw_history')
    nets = {name: deepcopy(base) for name in MODES}
    nets['spatial'].mode = 'spatial'
    return nets


def inputs(ordinary, current=None, history=None):
    current = ordinary if current is None else current
    history = ordinary if history is None else history
    # No independent observation exists at the first frame of a training/evaluation clip.
    actual = torch.cat([ordinary[:, :1], current[:, 1:]], dim=1)
    previous = torch.cat([ordinary[:, :1], history[:, :-1]], dim=1)
    return actual, previous


def sequence(net, ordinary, current=None, history=None, replace_history=False):
    actual, previous = inputs(ordinary, current, history)
    if replace_history:
        previous = actual
    shape = actual.shape
    output = net.transform(actual.flatten(0, 1), previous.flatten(0, 1))
    return output.reshape(shape)


def update(net, optimizer, ordinary, target, current=None, history=None):
    optimizer.zero_grad(set_to_none=True)
    output = sequence(net, ordinary, current, history)
    loss = ((output-target).square()+1e-6).sqrt().mean()
    if not torch.isfinite(output).all() or not torch.isfinite(loss):
        raise FloatingPointError('R79 nonfinite output/loss; no scientific retry.')
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    return dict(loss=float(loss.detach()), gradient_norm_before_clip=norm)


def optimizer_for(net):
    return torch.optim.Adam(net.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)


def precision():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
