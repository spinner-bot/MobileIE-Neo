"""Frozen-network temporal FST interventions, matched two-observation controls."""
import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from exploration_r2_data import detect_changes, evaluate, load_images, make_sequence
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import inference, save_json
from train_exploration_r2 import new_model
from video_metrics import warp_previous


class PairFST(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.original, self.mode = original, mode
        self.previous = None
        self.grid = self.mask = None

    def forward(self, value):
        module = self.original
        current = module.block1(value)
        past = current
        if self.previous is not None and self.grid is not None:
            warped = F.grid_sample(self.previous, self.grid, align_corners=True, padding_mode='zeros')
            past = torch.where(self.mask, warped, current)
        self.previous = current.detach()
        if self.mode == 'cross':
            product = (module.weight1*current)*(module.weight2*past)
        elif self.mode == 'mean_square':
            mean = (current+past)/2
            product = (module.weight1*mean)*(module.weight2*mean)
        elif self.mode == 'square_mean':
            product = ((module.weight1*current)*(module.weight2*current)
                       +(module.weight1*past)*(module.weight2*past))/2
        else:
            raise ValueError(self.mode)
        return product+module.bias


def sampling(flow, mask, device):
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    grid = np.stack([(xx+flow[..., 0])*2/(w-1)-1, (yy+flow[..., 1])*2/(h-1)-1], axis=-1)
    return torch.from_numpy(grid)[None].to(device), torch.from_numpy(mask)[None, None].to(device)


def pair_blend(frames, flows, masks):
    result = [frames[0].copy()]
    for t in range(1, len(frames)):
        past, valid = warp_previous(frames[t-1], flows[t-1])
        valid &= masks[t-1]
        result.append(np.where(valid[..., None], (frames[t]+past)/2, frames[t]).astype(np.float32))
    return result


def intervention(base, frames, flows, masks, layer, mode, device):
    model = copy.deepcopy(base)
    changed = PairFST(getattr(model, layer), mode)
    setattr(model, layer, changed)
    outputs = []
    with torch.inference_mode():
        for t, image in enumerate(frames):
            if t:
                changed.grid, changed.mask = sampling(flows[t-1], masks[t-1], device)
            value = torch.from_numpy((image/255).transpose(2, 0, 1).copy())[None].to(device)
            pred = model(value).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()*255
            outputs.append(pred.astype(np.float32))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    base = new_model(device)
    images = load_images('exploratory_validation')
    names = list(images)
    save_json(out/'config.json', {'weights': 'official slim frozen', 'mask_erosion_radius': 8,
        'pair_weights': [.5, .5], 'output_ema_alpha': .7,
        'body_modes': ['cross', 'mean_square', 'square_mean'], 'head_modes': ['cross'],
        'alignments': ['fb', 'oracle'], 'note': 'no training; causal previous block1 feature, not recursive feature EMA'})
    start = time.perf_counter()
    count = 0
    with (out/'per_sequence.jsonl').open('w') as stream:
        for index, name in enumerate(names):
            for case in ['pan', 'light_step', 'occlusion', 'cut']:
                for seed in [0, 1]:
                    gt, low, true_flow, true_masks = make_sequence(images[name], images[names[(index+1)%len(names)]], case, seed)
                    raw, _ = inference(base, low, device, 1)
                    raw_metrics, _ = evaluate(raw, gt, true_flow, true_masks)
                    identity = {'scene': name, 'case': case, 'input_seed': seed}
                    stream.write(json.dumps({**identity, 'method': 'raw', 'coverage': 0., **raw_metrics}, allow_nan=False)+'\n')
                    count += 1
                    pairs = [forward_backward(low[t-1], low[t]) for t in range(1, len(low))]
                    events, _ = detect_changes(low)
                    for alignment, flows, masks in [('fb', [p[0] for p in pairs], [p[1] for p in pairs]),
                                                    ('oracle', true_flow, true_masks)]:
                        # Constant border explicitly invalid: erosion must not extend edge validity.
                        masks = [cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8),
                                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for m in masks]
                        methods = {'input_pair': inference(base, pair_blend(low, flows, masks), device, 1)[0],
                                   'output_pair': pair_blend(raw, flows, masks),
                                   'output_ema': fuse(raw, flows, masks)}
                        for layer, mode in [('body', 'cross'), ('body', 'mean_square'), ('body', 'square_mean'), ('head', 'cross')]:
                            methods[layer+'_'+mode] = intervention(base, low, flows, masks, layer, mode, device)
                        for method, outputs in methods.items():
                            metrics, _ = evaluate(outputs, gt, true_flow, true_masks)
                            row = {**identity, 'method': alignment+'__'+method,
                                   'coverage': float(np.mean([m.mean() for m in masks])), **metrics}
                            stream.write(json.dumps(row, allow_nan=False)+'\n')
                            count += 1
                    print(json.dumps({**identity, 'rows': count}), flush=True)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'sequence_rows': count})


if __name__ == '__main__':
    main()
