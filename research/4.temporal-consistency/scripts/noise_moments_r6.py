"""Exact empirical quadratic moments and explicitly oracle-only interventions."""
import torch

from causal_fst_r5 import coarse_pool, coarse_up


def quadratic(module, u, correction=None, statistics=None):
    result = (module.weight1*u)*(module.weight2*u)+module.bias
    if correction is None:
        return result
    a = module.weight1*module.weight2
    if correction == 'debias':
        return result-a*statistics['variance']
    if correction == 'linearize':
        return result-a*(u-statistics['mean'])**2
    raise ValueError(correction)


def forward_probe(model, x, intervention='raw', statistics=None):
    """Static path only; reproduce plain/spatial_post, expose pre-square features."""
    net = getattr(model, 'backbone', model)
    mode = getattr(model, 'mode', 'plain')
    if mode not in ('plain', 'spatial_post'):
        raise ValueError('only static plain/spatial_post supported')
    if intervention == 'raw':
        location, correction = None, None
    else:
        correction, location = intervention.split('_')
    u_head = net.head.block1(x)
    head = quadratic(net.head, u_head, correction if location == 'head' else None,
                     statistics['head'] if location == 'head' else None)
    u_body = net.body.block1(head)
    body = quadratic(net.body, u_body, correction if location == 'body' else None,
                     statistics['body'] if location == 'body' else None)
    if mode == 'spatial_post':
        body = body+coarse_up(model.branch(coarse_pool(body)), body.shape[-2:])
    att = net.att(body)
    maximum = torch.max(att*body, dim=1, keepdim=True)[0]
    att1 = net.att1(maximum)
    return net.tail(torch.mul(att1, att)*body), {'head': u_head, 'body': u_body}


def central_statistics(samples):
    """Population moments over repeated draws at each fixed spatial position."""
    samples = samples.double()
    mean = samples.mean(0)
    e = samples-mean
    m2, m3, m4 = e.square().mean(0), e.pow(3).mean(0), e.pow(4).mean(0)
    linear, cross, quadratic_term = 4*mean.square()*m2, 4*mean*m3, m4-m2.square()
    direct = samples.square().var(0, unbiased=False)
    return {'mean': mean, 'variance': m2, 'm3': m3, 'm4': m4, 'linear': linear,
            'cross': cross, 'quadratic': quadratic_term, 'total': direct,
            'closure': linear+cross+quadratic_term-direct}


def rgb_statistics(predictions, target, margin=8):
    """MSE = squared conditional bias + draw variance; values normalized to [0,1]."""
    p = predictions.double()[..., margin:-margin, margin:-margin]
    target = target.double()[..., margin:-margin, margin:-margin]
    mean = p.mean(0)
    variance = p.var(0, unbiased=False).mean()
    bias = (mean-target).square().mean()
    per_draw = (p-target).square().mean((1, 2, 3))
    mse = per_draw.mean()
    return {'mse_normalized': float(mse), 'bias_mse_normalized': float(bias),
        'variance_normalized': float(variance), 'closure_abs': float((mse-bias-variance).abs()),
        'noise_sd_255': float(variance.sqrt()*255),
        'mean_draw_psnr': float((-10*per_draw.clamp_min(1e-30).log10()).mean()),
        'mean_output_psnr': float(-10*bias.clamp_min(1e-30).log10()),
        'mean_brightness_signed_error_255': float((mean.mean()-target.mean())*255)}
