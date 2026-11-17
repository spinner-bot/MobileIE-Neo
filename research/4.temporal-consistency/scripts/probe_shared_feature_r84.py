"""One CPU train-window execution of the frozen R84 causal prototype."""
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
from state_operator_r83 import StateOperator
from shared_feature_r84 import MODES, FeatureStream, shared_forward, qualification
from video_metrics import warp_previous, psnr

PROCESS = BASE/'process/exploration-r84'
PROTOCOL = BASE/'docs/exploration-r84-shared-feature-prototype-protocol.md'


def verify():
    frozen = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert frozen['source_sha256'] == {name: digest(ROOT/name) for name in frozen['source_sha256']}
    assert frozen['protocol_sha256'] == digest(PROTOCOL)
    return frozen


def fuse(raw, history, field):
    flow, mask = field
    moved, valid = warp_previous(history, flow)
    return (raw+.5*(mask & valid)[..., None]*(moved-raw)).astype(np.float32)


def quality(sequence, target, gt_fields):
    scores = [psnr(a*255, b*255) for a, b in zip(sequence, target)]
    temporal = []
    for t in range(1, 4):
        flow, mask = gt_fields[t]
        old, valid = warp_previous(sequence[t-1], flow)
        old_gt, _ = warp_previous(target[t-1], flow)
        accepted = valid & mask
        if not accepted.any():
            raise ValueError('An empty fixed GT metric domain cannot silently be skipped')
        residual = (sequence[t]-old)-(target[t]-old_gt)
        temporal.append(float(np.abs(residual[accepted]).mean()*255))
    return dict(PSNR_nonfirst=float(np.mean(scores[1:])), PSNR_all=float(np.mean(scores)),
                temporal_GT_fixed_L1=float(np.mean(temporal)), per_frame_PSNR=scores,
                per_pair_temporal_GT_fixed_L1=temporal)


def main():
    out = PROCESS/'prototype001'; out.mkdir(parents=True, exist_ok=False)
    frozen = verify()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    selection = [(0, 8000), (0, 8001), (1, 8000), (1, 8001)]
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen,
        selection=selection, batch=4, frames=4, crop=128, F0_seeds=[0, 1],
        expected_backbone_batch_calls=33, expected_backbone_image_frames=132,
        expected_GT_half_FB_pairs=48, expected_F0_half_FB_pairs=96,
        expected_candidate_bidirectional_pairs=288, expected_operator_rows=288,
        expected_quality_rows=160, optimizer_updates=0,
        scope='Already seen cropped training cases, not DEV, independent natural-video or phone results.'))
    started = time.perf_counter(); complete = False
    calls = frames = gt_pairs = ref_pairs = candidate_pairs = quality_rows = 0
    operator_totals, quality_values = {}, {}
    try:
        write(out/'qualification.json', dict(at=stamp(), **qualification()))
        pool = MixedPool(); prepared = []
        for data_seed, step in selection:
            low, target, identities = pool.batch(data_seed, step)
            target = target.permute(0, 1, 3, 4, 2).contiguous().numpy()
            gt_fields = {}
            for b in range(4):
                for t in range(1, 4):
                    gt_fields[b, t] = half_forward_backward(target[b, t-1]*255, target[b, t]*255)
                    gt_pairs += 1
            prepared.append((low, target, identities, gt_fields))
            append(out/'windows.jsonl', dict(data_seed=data_seed, data_step=step,
                identities=identities, input_shape=list(low.shape), loader=pool.info))
        for model_seed in (0, 1):
            backbone, marker = load_spatial(model_seed)
            write(out/f'backbone-seed{model_seed}.json', marker)
            for window_index, (low, target, identities, gt_fields) in enumerate(prepared):
                data_seed, step = selection[window_index]
                raw_list, features = [], []
                for t, image in enumerate(low.unbind(1)):
                    raw, captured = shared_forward(backbone, image); calls += 1; frames += len(image)
                    if model_seed == 0 and window_index == 0 and t == 0:
                        with torch.no_grad():
                            original = backbone(image).clamp(0, 1); calls += 1; frames += len(image)
                        delta = float((raw-original).abs().max())
                        assert delta <= 1e-6
                        write(out/'original-path-qualification.json', dict(at=stamp(),
                            extra_backbone_batch_calls=1, extra_backbone_image_frames=4,
                            shared_vs_original_max_abs_difference=delta, passed=True))
                    raw_list.append(raw.permute(0, 2, 3, 1).contiguous().numpy())
                    features.append(captured)
                raw = np.stack(raw_list, axis=1)
                for b, identity in enumerate(identities):
                    assert identity['source'] in ('davis_synthetic', 'olives_real')
                    domain = 'SYN' if identity['source'] == 'davis_synthetic' else 'real'
                    outputs = {'raw': [v.copy() for v in raw[b]], 'strongrec': [raw[b, 0].copy()]}
                    streams = {mode: FeatureStream() for mode in MODES}
                    for mode in MODES:
                        outputs[mode] = []
                    for t in range(4):
                        ref = None
                        if t:
                            field = half_forward_backward(raw[b, t-1]*255, raw[b, t]*255)
                            ref_pairs += 1; ref = StateOperator(*field)
                            outputs['strongrec'].append(fuse(raw[b, t], outputs['strongrec'][-1], field))
                        for mode in MODES:
                            stream = streams[mode]
                            output, field = stream.step(raw[b, t], features[t][mode][b:b+1])
                            outputs[mode].append(output)
                            if not t:
                                append(out/'state-storage.jsonl', dict(model_seed=model_seed,
                                    window=window_index, batch_member=b, mode=mode,
                                    bytes=stream.state_bytes(), dtype='FP32', shape=[128, 128],
                                    scope='One deployed mode persistent state only; excludes workspace and backbone.'))
                                continue
                            candidate_pairs += 1
                            flow, mask, boundary = field
                            op = StateOperator(flow, mask)
                            numerator, denominator = op.difference(ref)
                            row = dict(pixels=mask.size, candidate_accepted=int(op.valid.sum()),
                                F0_accepted=int(ref.valid.sum()),
                                F0_domain_numerator=float(numerator[ref.valid].sum()),
                                F0_domain_denominator=float(denominator[ref.valid].sum()),
                                full_numerator=float(numerator.sum()), full_denominator=float(denominator.sum()),
                                quarter_boundary_fraction=boundary)
                            append(out/'operators.jsonl', dict(model_seed=model_seed, domain=domain,
                                data_seed=data_seed, data_step=step, batch_member=b, frame=t, mode=mode, **row))
                            group = operator_totals.setdefault((domain, model_seed, mode),
                                dict(pairs=0, **{key: 0. for key in row}))
                            group['pairs'] += 1
                            for key, value in row.items():
                                group[key] += value
                    for mode, sequence in outputs.items():
                        assert len(sequence) == 4
                        metrics = quality(sequence, target[b], {t: gt_fields[b, t] for t in range(1, 4)})
                        append(out/'quality.jsonl', dict(model_seed=model_seed, domain=domain,
                            data_seed=data_seed, data_step=step, batch_member=b, mode=mode,
                            identity=identity, **metrics)); quality_rows += 1
                        quality_values.setdefault((domain, model_seed, mode), []).append(metrics)
                print(json.dumps(dict(model_seed=model_seed, window=window_index,
                    backbone_image_frames=frames, candidate_bidirectional_pairs=candidate_pairs,
                    quality_rows=quality_rows)), flush=True)
            assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
        summaries = []
        for domain in ('SYN', 'real'):
            for seed in (0, 1):
                op_summary, q_summary = {}, {}
                for mode in MODES:
                    total = operator_totals[domain, seed, mode]
                    assert total['pairs'] == 24
                    op_summary[mode] = dict(totals=total,
                        candidate_coverage=total['candidate_accepted']/total['pixels'],
                        F0_coverage=total['F0_accepted']/total['pixels'],
                        F0_domain_difference=total['F0_domain_numerator']/total['F0_domain_denominator']
                            if total['F0_domain_denominator'] else 1.,
                        full_difference=total['full_numerator']/total['full_denominator']
                            if total['full_denominator'] else 1.,
                        quarter_boundary_fraction=total['quarter_boundary_fraction']/total['pairs'])
                for mode in ('raw', 'strongrec', *MODES):
                    values = quality_values[domain, seed, mode]
                    assert len(values) == 8
                    q_summary[mode] = {key: float(np.mean([v[key] for v in values]))
                        for key in ('PSNR_nonfirst', 'PSNR_all', 'temporal_GT_fixed_L1')}
                signed = op_summary['signed']
                matching = signed['candidate_coverage'] >= .25 and signed['candidate_coverage'] >= .5*signed['F0_coverage']
                matching = matching and all(signed['F0_domain_difference'] <= .9*op_summary[m]['F0_domain_difference']
                    for m in ('energy', 'rgb'))
                signed_quality = q_summary['signed']
                quality_checks = {mode: dict(
                    PSNR_delta=signed_quality['PSNR_nonfirst']-q_summary[mode]['PSNR_nonfirst'],
                    temporal_ratio=signed_quality['temporal_GT_fixed_L1']/q_summary[mode]['temporal_GT_fixed_L1'])
                    for mode in ('raw', 'strongrec')}
                output_gate = all(v['PSNR_delta'] >= -.05 and v['temporal_ratio'] <= .9 for v in quality_checks.values())
                summaries.append(dict(domain=domain, model_seed=seed, operators=op_summary,
                    quality=q_summary, signed_comparisons=quality_checks,
                    matching_gate=bool(matching), output_gate=bool(output_gate)))
        assert (calls, frames, gt_pairs, ref_pairs, candidate_pairs, quality_rows) == (33, 132, 48, 96, 288, 160)
        verify(); assert not torch.cuda.is_initialized()
        matching = all(g['matching_gate'] for g in summaries)
        output_gate = all(g['output_gate'] for g in summaries)
        write(out/'summary.json', dict(at=stamp(), groups=summaries,
            matching_all=matching, output_all=output_gate, both_prerequisites=matching and output_gate,
            scope='Finite frozen hard-matching prototype on already-seen training windows, no general impossibility/novelty/DEV conclusion.'))
        complete = True
        print(json.dumps(dict(matching_all=matching, output_all=output_gate, both_prerequisites=matching and output_gate)), flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-started, backbone_batch_calls=calls, backbone_image_frames=frames,
            GT_half_FB_pairs=gt_pairs, F0_half_FB_pairs=ref_pairs, candidate_bidirectional_pairs=candidate_pairs,
            operator_rows=candidate_pairs, quality_rows=quality_rows, optimizer_updates=0,
            new_DEV_pixels=0, CUDA_initialized=torch.cuda.is_initialized()))


if __name__ == '__main__':
    main()
