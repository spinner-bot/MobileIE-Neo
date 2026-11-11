"""Exact body-FST/attention algebra; correlated terms are not causal attributions."""
import torch
from torch.nn import functional as F

from causal_fst_r5 import coarse_pool, coarse_up


def body_components(model, u, gate, output, split_interaction=False):
    net = getattr(model, 'backbone', model)
    u, gate, output = u.double(), gate.double(), output.double()
    mu = u.mean(0, keepdim=True)
    e = u-mu
    m2 = e.square().mean(0, keepdim=True)
    a = net.body.weight1.double()*net.body.weight2.double()
    mean = a*(mu.square()+m2)+net.body.bias.double()
    linear, quadratic = 2*a*mu*e, a*(e.square()-m2)
    if getattr(model, 'mode', 'plain') == 'spatial_post':
        def transform(x):
            filtered = F.conv2d(coarse_pool(x), model.branch.weight.double(), padding=1, groups=12)
            return x+coarse_up(filtered, x.shape[-2:])
        mean, linear, quadratic = [transform(x) for x in [mean, linear, quadratic]]
    avg_gate = gate.mean(0, keepdim=True)
    dg = gate-avg_gate
    conv = lambda x: F.conv2d(x, net.tail.weight.double(), padding=1)
    baseline = conv(avg_gate*mean)+net.tail.bias.double().reshape(1, -1, 1, 1)
    components = {'feature_linear_mean_gate': conv(avg_gate*linear),
                  'feature_quadratic_mean_gate': conv(avg_gate*quadratic),
                  'gate_mean_feature': conv(dg*mean)}
    if split_interaction:
        components['gate_linear_interaction'] = conv(dg*linear)
        components['gate_quadratic_interaction'] = conv(dg*quadratic)
    else:
        components['gate_feature_interaction'] = conv(dg*(linear+quadratic))
    reconstructed = baseline+sum(components.values())
    components['rounding_residual'] = output-reconstructed
    return baseline, components


def covariance_summary(components, output, clipped=False, margin=8):
    values = dict(components)
    if clipped:
        values['clipping_residual'] = output.double().clamp(0, 1)-output.double()
        output = output.clamp(0, 1)
    centered = []
    for value in values.values():
        value = value.double()[..., margin:-margin, margin:-margin]
        centered.append(value-value.mean(0, keepdim=True))
    matrix = torch.stack([v.flatten() for v in centered])
    covariance = matrix@matrix.T/matrix.shape[1]
    actual = output.double()[..., margin:-margin, margin:-margin]
    actual = actual-actual.mean(0, keepdim=True)
    variance = actual.square().mean()
    reconstruction = (sum(centered)-actual).abs().max()
    names = list(values)
    return {'components': names, 'covariance': covariance.cpu().tolist(),
        'component_variance': covariance.diag().cpu().tolist(),
        'covariance_with_total_over_total_variance': (covariance.sum(1)/variance.clamp_min(1e-30)).cpu().tolist(),
        'component_variance_over_total_variance': (covariance.diag()/variance.clamp_min(1e-30)).cpu().tolist(),
        'actual_rgb_variance': float(variance),
        'variance_closure_abs': float((covariance.sum()-variance).abs()),
        'centered_reconstruction_max_abs': float(reconstruction)}
