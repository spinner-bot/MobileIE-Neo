"""One preregistered CPU train-only operator support/conditioning preflight."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('An initially empty CUDA mask is required.')

import json
import time
import traceback

import cv2
import numpy as np
import torch

from causal_reference_io_r75 import ROOT, BASE, write, append, stamp, digest
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from half_motion_r4 import half_forward_backward
from state_operator_r83 import StateOperator, qualification

PROCESS = BASE/'process/exploration-r83'
PROTOCOL = BASE/'docs/exploration-r83-state-operator-preflight-protocol.md'
SOURCES = ['research/4.temporal-consistency/scripts/'+name for name in (
    'state_operator_r83.py', 'probe_state_operator_r83.py', 'causal_reference_io_r75.py',
    'mixed_anchor_r40.py', 'motion_transport_r74.py', 'half_motion_r4.py',
    'aligned_raw_history_r11.py', 'aggregation_controls_r20.py', 'video_metrics.py',
    'dynamic_training_r34.py', 'budget_causal_r33.py', 'native_training_r14.py',
    'train_mixture_r4b.py', 'train_exploration_r2.py', 'explore_temporal_controls.py',
    'exploration_r2_data.py', 'bounded_fst.py')]
SOURCES += ['MobileIE/model/lle.py', 'MobileIE/model/utils.py', 'scripts/project-python', 'scripts/project_assets.py']


def verify():
    value = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert value['source_sha256'] == {name: digest(ROOT/name) for name in SOURCES}
    assert value['protocol_sha256'] == digest(PROTOCOL)
    return value


def statistics(observed, supervised):
    common = observed.valid & supervised.valid
    union = observed.valid | supervised.valid
    numerator, denominator = observed.difference(supervised)
    return dict(pixels=common.size, observed_accepted=int(observed.valid.sum()),
        GT_derived_accepted=int(supervised.valid.sum()), common_accepted=int(common.sum()),
        union_accepted=int(union.sum()), observed_energy=float(observed.energy().sum()),
        observed_outside_radius2_energy=float(observed.outside_energy().sum()),
        shared_difference_numerator=float(numerator[common].sum()),
        shared_difference_denominator=float(denominator[common].sum()),
        full_difference_numerator=float(numerator.sum()), full_difference_denominator=float(denominator.sum()))


def main():
    out = PROCESS/'preflight001'; out.mkdir(parents=True, exist_ok=False)
    frozen = verify()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    selection = [(0, 8000), (0, 8001), (1, 8000), (1, 8001)]
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen,
        selection=selection, batch=4, frames=4, crop=128, F0_seeds=[0, 1],
        expected_backbone_batches=32, expected_backbone_image_frames=128,
        expected_GT_half_FB_pairs=48, expected_observed_half_FB_pairs=96,
        expected_pair_records=96, local_history_radius=2,
        H1=dict(min_coverage=.25, min_outside_energy_fraction=.20),
        H2=dict(min_common_coverage=.25, min_shared_difference=.10),
        scope='Only fixed training windows; no H/model updates, new DEV/final pixels or GPU.'))
    began, complete = time.perf_counter(), False
    calls = frames = gt_pairs = observed_pairs = pairs = 0
    grouped = {}
    try:
        gate = qualification(); write(out/'qualification.json', dict(at=stamp(), **gate))
        pool = MixedPool(); prepared = []
        for data_seed, step in selection:
            low, target, identities = pool.batch(data_seed, step)
            bright = target.permute(0, 1, 3, 4, 2).contiguous().numpy()*255
            fields = {}
            for b in range(4):
                for t in range(1, 4):
                    fields[b, t] = half_forward_backward(bright[b, t-1], bright[b, t]); gt_pairs += 1
            prepared.append((low, identities, fields))
            append(out/'windows.jsonl', dict(data_seed=data_seed, data_step=step,
                identities=identities, input_shape=list(low.shape), loader=pool.info))
        for model_seed in (0, 1):
            backbone, marker = load_spatial(model_seed)
            write(out/f'backbone-seed{model_seed}.json', marker)
            for (data_seed, step), (low, identities, fields) in zip(selection, prepared):
                with torch.no_grad():
                    raw = []
                    for value in low.unbind(1):
                        raw.append(backbone(value).clamp(0, 1)); calls += 1; frames += len(value)
                    raw = torch.stack(raw, 1).permute(0, 1, 3, 4, 2).contiguous().numpy()*255
                assert np.isfinite(raw).all()
                for b, identity in enumerate(identities):
                    assert identity['source'] in ('davis_synthetic', 'olives_real')
                    domain = 'SYN' if identity['source'] == 'davis_synthetic' else 'real'
                    for t in range(1, 4):
                        flow, mask = half_forward_backward(raw[b, t-1], raw[b, t]); observed_pairs += 1
                        obs = StateOperator(flow, mask)
                        target = StateOperator(*fields[b, t])
                        row = statistics(obs, target)
                        append(out/'pairs.jsonl', dict(model_seed=model_seed, domain=domain,
                            data_seed=data_seed, data_step=step, batch_member=b, frame=t,
                            identity=identity, **row))
                        key = (domain, model_seed)
                        total = grouped.setdefault(key, dict(pair_records=0, **{k: 0. for k in row}))
                        total['pair_records'] += 1
                        for name, value in row.items():
                            total[name] += value
                        pairs += 1
            assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
            print(json.dumps(dict(model_seed=model_seed, backbone_frames=frames, completed_pairs=pairs)), flush=True)
        summaries = []
        for (domain, seed), total in sorted(grouped.items()):
            def ratio(a, b):
                return total[a]/total[b] if total[b] else 0.
            coverage = ratio('observed_accepted', 'pixels')
            common = ratio('common_accepted', 'pixels')
            outside = ratio('observed_outside_radius2_energy', 'observed_energy')
            difference = ratio('shared_difference_numerator', 'shared_difference_denominator')
            assert total['pair_records'] == 24
            summaries.append(dict(domain=domain, model_seed=seed, totals=total,
                observed_coverage=coverage, GT_derived_coverage=ratio('GT_derived_accepted', 'pixels'),
                common_coverage=common, acceptance_Jaccard=ratio('common_accepted', 'union_accepted'),
                outside_radius2_energy_fraction=outside, shared_operator_difference=difference,
                full_operator_difference=ratio('full_difference_numerator', 'full_difference_denominator'),
                H1_pass=coverage >= .25 and outside >= .20,
                H2_pass=common >= .25 and difference >= .10))
        assert (calls, frames, gt_pairs, observed_pairs, pairs) == (32, 128, 48, 96, 96)
        assert len(summaries) == 4 and not torch.cuda.is_initialized()
        h1, h2 = all(r['H1_pass'] for r in summaries), all(r['H2_pass'] for r in summaries)
        write(out/'summary.json', dict(at=stamp(), groups=summaries, H1_all=h1, H2_all=h2,
            both_prerequisites=h1 and h2, scope='Fixed cropped training cases only; not real-video lower bound or failure-cause identification.'))
        verify(); complete = True
        print(json.dumps(dict(H1_all=h1, H2_all=h2, both_prerequisites=h1 and h2)), flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-began, backbone_batch_calls=calls, backbone_image_frames=frames,
            GT_half_FB_pairs=gt_pairs, observed_half_FB_pairs=observed_pairs, pair_records=pairs,
            optimizer_updates=0, new_DEV_pixels=0, CUDA_initialized=torch.cuda.is_initialized()))


if __name__ == '__main__':
    main()
