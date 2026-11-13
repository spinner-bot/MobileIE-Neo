"""Registered update/checkpoint operations shared by CPU qualification and R75 training."""
import math
import torch

from rebotnet_reference import crop_predictions

STEPS = 64000


def optimizer_for(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-7)
    return optimizer, scheduler


def backward_video(model, low, target):
    output = crop_predictions(model, low)
    loss = ((output-target).square()+1e-6).sqrt().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Non-finite R75 loss; no scientific auto-retry.')
    (loss/4).backward()
    return float(loss.detach())


def finish_update(model, optimizer, scheduler):
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    if not math.isfinite(norm):
        raise FloatingPointError('Non-finite R75 gradient; no scientific auto-retry.')
    optimizer.step()
    scheduler.step()
    return norm


def rng_state(model):
    cuda = torch.cuda.get_rng_state(0) if next(model.parameters()).device.type == 'cuda' else None
    return dict(cpu=torch.get_rng_state(), cuda=cuda)


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device=0)


def checkpoint(model, optimizer, scheduler, seed, completed_steps, rng=None):
    return dict(experiment='R75', seed=seed, completed_steps=completed_steps,
                next_data_step=completed_steps, registered_steps=STEPS,
                state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=rng if rng is not None else rng_state(model))


def restore(saved, model, optimizer, scheduler, seed):
    if (saved['experiment'] != 'R75' or saved['seed'] != seed or saved['registered_steps'] != STEPS
            or saved['next_data_step'] != saved['completed_steps'] or not 0 <= saved['completed_steps'] <= STEPS):
        raise ValueError('Checkpoint seed, budget or next data step mismatch.')
    model.load_state_dict(saved['state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'])
    return saved['completed_steps']
