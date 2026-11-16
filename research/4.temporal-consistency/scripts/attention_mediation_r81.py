"""Frozen-node interventions, not a deployable temporal attention method."""
import json

import numpy as np
import torch

from causal_reference_io_r75 import ROOT, BASE, digest

PROTOCOL = BASE/'docs/exploration-r81-attention-noise-protocol.md'
PROCESS = BASE/'process/exploration-r81'
LABELS = ('raw', 'clean_global', 'clean_local', 'clean_both', 'recursive05')
SOURCES = ['research/4.temporal-consistency/scripts/'+name for name in (
    'attention_mediation_r81.py', 'probe_attention_noise_r81.py', 'render_attention_noise_r81.py',
    'dynamic_training_r34.py', 'budget_causal_r33.py', 'exploration_r2_data.py',
    'motion_transport_r74.py', 'train_exploration_r2.py', 'explore_temporal_controls.py',
    'half_motion_r4.py', 'video_metrics.py', 'causal_reference_io_r75.py', 'noise_role_io_r79.py')]
SOURCES += ['MobileIE/model/lle.py', 'MobileIE/model/utils.py',
            'scripts/project-python', 'scripts/project_assets.py']


def verify():
    frozen = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert frozen['source_sha256'] == {name: digest(ROOT/name) for name in SOURCES}
    assert frozen['protocol_sha256'] == digest(PROTOCOL)
    return frozen


def parts(net, low):
    body = net.body(net.head(low))
    gate = net.att(body)
    maximum, winner = (gate*body).max(dim=1, keepdim=True)
    output = net.tail((net.att1(maximum)*gate)*body)
    return dict(body=body, gate=gate, maximum=maximum, winner=winner, raw=output)


def interventions(net, current, clean):
    body = current['body']
    global_max = (clean['gate']*body).max(dim=1, keepdim=True)[0]
    return dict(raw=current['raw'],
        clean_global=net.tail((net.att1(global_max)*clean['gate'])*body),
        clean_local=net.tail((net.att1(clean['maximum'])*current['gate'])*body),
        clean_both=net.tail((net.att1(clean['maximum'])*clean['gate'])*body))


def statistics(values, target):
    values, target = values.astype(np.float64), target.astype(np.float64)
    center = values.mean(axis=0)
    variance = float(np.mean((values-center)**2))
    squared_bias = float(np.mean((center-target)**2))
    losses = np.mean((values-target)**2, axis=(1, 2, 3))
    mse = float(losses.mean())
    assert np.isclose(mse, variance+squared_bias, rtol=1e-7, atol=1e-12)
    return dict(mean_psnr_rgb=float(np.mean(-10*np.log10(np.maximum(losses, 1e-12)))),
        MSE=mse, variance=variance, squared_bias=squared_bias,
        temporal_difference_255=float(np.mean(np.abs(np.diff(values, axis=0)))*255),
        minimum=float(values.min()), maximum=float(values.max()))
