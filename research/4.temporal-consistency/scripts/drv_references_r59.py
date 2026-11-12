"""Only existing published/final checkpoints and the existing classic/RGB rules."""
import json

import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from fastdvdnet_adapter_r12 import denoise
from half_motion_r4 import half_forward_backward
from multinex_reference_r55 import load_reference, prediction
from response_cue_r45 import load_frozen, RecursiveControl
from task_fastdvd_r47 import initial_model, SIGMA, provenance
from tiny_adapted_reference_r56 import load_adapted
from train_exploration_r2 import new_model
from video_metrics import psnr, ssim, edge_error

NETWORKS = ['mobileie_published', 'nano_published', 'mobileie_seed0', 'mobileie_seed1',
            'nano_seed0', 'nano_seed1', 'fastdvd_seed0', 'fastdvd_seed1']
PRIMARY = [2, 3, 4, 5]


def load_network(name):
    assert name in NETWORKS
    family = name.split('_')[0]; seed = None if name.endswith('published') else int(name[-1])
    control = None
    if family == 'mobileie':
        if seed is None:
            model = new_model(torch.device('cpu'))
            path = ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'
            source = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), project_updates=0)
        else:
            model, marker = load_frozen(seed)
            source = dict(checkpoint=marker, project_updates=8000)
            run = BASE/f'results/exploration-r45-stage001-seed{seed}'
            assert json.loads((run/'runtime.json').read_text())['completed_normally']
            rgb = json.loads((run/'rgb-step4000.json').read_text())
            assert rgb['new_step'] == 4000 and rgb['base_checkpoint_sha256'] == marker['sha256']
            assert digest(ROOT/rgb['path']) == rgb['sha256']
            saved = torch.load(ROOT/rgb['path'],map_location='cpu',weights_only=False)
            assert saved['control'] == 'rgb' and saved['base_checkpoint_sha256'] == marker['sha256']
            control = RecursiveControl('rgb',seed).eval().requires_grad_(False)
            control.load_state_dict(saved['state_dict'],strict=True)
            source.update(RGB_controller=rgb, RGB_extra_updates=4000)
    elif family == 'nano':
        model, source = load_reference() if seed is None else load_adapted(seed)
        source['project_updates'] = 0 if seed is None else 8000
    else:
        run = BASE/f'results/exploration-r53-stage001-seed{seed}'
        marker = json.loads((run/'fastdvd-step8000.json').read_text())
        training = json.loads((run/'runtime.json').read_text())
        assert training['completed_normally'] and training['steps'] == marker['step'] == 8000
        assert marker['final_quality_eligible'] and digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'],map_location='cpu',weights_only=False)
        assert saved['step'] == 8000 and saved['seed'] == seed
        model = initial_model(seed); model.load_state_dict(saved['state_dict'],strict=True)
        source = dict(checkpoint=marker, official=provenance(), project_updates=8000,
                      input='native only, true five-frame window, fixed sigma15/255')
    model.eval().requires_grad_(False)
    expected = {'mobileie':4075, 'nano':696, 'fastdvd':2479096}[family]
    assert sum(p.numel() for p in model.parameters()) == expected
    return model, control, dict(name=name, family=family, seed=seed, parameters=expected,
                               future_wait_max=2 if family == 'fastdvd' else 0, source=source)


@torch.inference_mode()
def infer(name, model, low):
    assert low.dtype == np.uint8 and low.shape[0] == 8 and low.shape[-1] == 3
    inputs = torch.from_numpy(low.transpose(0,3,1,2).copy()).float()/255
    if name.startswith('fastdvd'):
        value = denoise(model,inputs,SIGMA,'window')
    else:
        frames = []
        for image in inputs:
            raw = prediction(model,image[None]) if name.startswith('nano') else model(image[None]).clamp(0,1)
            frames.append(raw[0])
        value = torch.stack(frames)
    assert torch.isfinite(value).all() and value.dtype == torch.float32
    result = value.permute(0,2,3,1).contiguous().numpy()*255
    assert result.shape == low.shape
    return result


def postprocess(raw, control=None):
    recursive = [raw[0]]; rgb = [raw[0]] if control is not None else None
    previous_rgb = torch.from_numpy(raw[0].transpose(2,0,1).copy()/255)[None] if control is not None else None
    accepted, alpha_sum, alpha_count = [], 0., 0
    for t in range(1,len(raw)):
        flow, mask = half_forward_backward(raw[t-1],raw[t])
        moved, valid = align_history(recursive[-1],raw[t],flow,mask)
        recursive.append(raw[t]+.5*(moved-raw[t])); accepted.append(float(valid.mean()))
        if control is not None:
            current = torch.from_numpy(raw[t].transpose(2,0,1).copy()/255)[None]
            with torch.inference_mode():
                previous_rgb, alpha, valid_rgb = control(current,previous_rgb,
                    torch.from_numpy(flow.transpose(2,0,1).copy())[None],
                    torch.from_numpy(mask.copy())[None,None].bool(),cues=None)
            rgb.append(previous_rgb[0].permute(1,2,0).contiguous().numpy()*255)
            alpha_sum += float(alpha[valid_rgb].double().sum()); alpha_count += int(valid_rgb.sum())
    outputs = dict(raw=raw, recursive_self05=np.stack(recursive))
    if rgb is not None: outputs['rgb'] = np.stack(rgb)
    for output in outputs.values():
        assert np.isfinite(output).all() and np.array_equal(output[0],raw[0])
    return outputs, dict(own_raw_half_FB_acceptance_mean=float(np.mean(accepted)),
                         RGB_mean_accepted_alpha=alpha_sum/alpha_count if alpha_count else None)


def metrics(output, target):
    assert output.shape == target.shape and len(output) == 8
    frames = []
    for t,(x,y) in enumerate(zip(output,target)):
        score = psnr(x,y)
        frames.append(dict(frame=t, primary_target=t in PRIMARY,
            psnr_rgb=score if np.isfinite(score) else None, PSNR_infinite=not np.isfinite(score),
            ssim_rgb=ssim(x,y), edge_mae=edge_error(x,y), output_mean=float(x.mean()),
            target_mean=float(y.mean()), low_endpoint_fraction=float(np.mean(x<=0)),
            high_endpoint_fraction=float(np.mean(x>=255)),
            same_coordinate_adjacent_RGB_MAE=None if t == 0 else float(np.abs(x-output[t-1]).mean())))
    summary = dict(frames=8, primary_frames=PRIMARY)
    for prefix, selected in [('',[frames[t] for t in PRIMARY]), ('full8_',frames)]:
        for key in ['psnr_rgb','ssim_rgb','edge_mae']:
            summary[prefix+key] = None if any(r[key] is None for r in selected) else float(np.mean([r[key] for r in selected]))
    value = output.astype(np.float64)
    summary.update(mean_RGB=float(value.mean()), target_mean_RGB=float(target.mean()),
                   same_coordinate_time_std_RMS=float(np.sqrt(np.mean((value-value.mean(0))**2))),
                   same_coordinate_adjacent_RGB_MAE=float(np.abs(np.diff(value,axis=0)).mean()),
                   scope='Spatial primary frames2–5/full8; time fields are unregistered static proxies, not true flicker.')
    return summary, frames
