"""One fixed known-geometry training-source preflight, not video restoration training."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('An initially empty CUDA mask is required')

import json
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from causal_reference_io_r75 import ROOT, BASE, write, append, stamp, digest
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from functional_gauge_r87 import Gauge, capture, fold, synthetic_pairs, queries, concatenate, metrics, scores, TRANSLATIONS

PROCESS = BASE/'process/exploration-r87'
PROTOCOL = BASE/'docs/exploration-r87-functional-gauge-preflight-protocol.md'


def verify():
    value = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert value['source_sha256'] == {n: digest(ROOT/n) for n in value['source_sha256']}
    assert value['protocol_sha256'] == digest(PROTOCOL)
    return value


def main():
    frozen = verify(); torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    out = PROCESS/'preflight001'; out.mkdir(exist_ok=False)
    assets = ROOT/'.cache/checkpoints/exploration-r87/preflight001'
    assert assets.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assets.mkdir(parents=True, exist_ok=False)
    selection = [(0, 8000), (0, 8001), (1, 8000), (1, 8001)]
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen, windows=selection,
        source_batch_members=[0, 2], target_frame=0, photons=40, quarter_translations=TRANSLATIONS,
        expected_source_crops=8, expected_B4_calls=64, expected_backbone_image_frames=256,
        expected_metric_updates=1024, expected_query_rows=4800, backbone_updates=0,
        scope='Training-source known-motion synthetic noise only; check split is not independent DEV/video.'))
    started = time.perf_counter(); complete = False
    calls = frames = updates = query_rows = 0
    summaries, qualifications, final_scales = [], [], []
    try:
        pool = MixedPool(); prepared = []
        for data_seed, step in selection:
            _, target, identities = pool.batch(data_seed, step)
            for member in (0, 2):
                index = len(prepared)
                category = 'DAVIS_GT_source' if member == 0 else 'OLiVES_GT_source'
                part = 'calibration' if data_seed == 0 else 'check'
                clean = target[member, 0].permute(1, 2, 0).contiguous().numpy()*255
                previous, current = synthetic_pairs(clean, index)
                identity = dict(source_index=index, data_seed=data_seed, data_step=step, member=member,
                    part=part, category=category, identity=identities[member])
                prepared.append(dict(**identity, previous=previous, current=current))
                append(out/'sources.jsonl', dict(**identity, loader=pool.info,
                    noise_seed_template=[87, index, 'translation_index', 'side0_or1'],
                    all_noise_is_synthetic=True))
        assert len(prepared) == 8
        split_ids = {part: sorted({r['identity']['video'] for r in prepared if r['part'] == part})
                     for part in ('calibration', 'check')}
        write(out/'split.json', dict(video_ids=split_ids,
            video_overlap=sorted(set(split_ids['calibration']) & set(split_ids['check'])),
            independent_final_or_DEV=False, scope='Fixed crop/noise split inside training pool; identity overlap not repaired.'))
        for seed in (0, 1):
            net, marker = load_spatial(seed); write(out/f'backbone-seed{seed}.json', marker)
            cases = []
            for item in prepared:
                left, a = capture(net, item['previous']); calls += 1; frames += 4
                right, b = capture(net, item['current']); calls += 1; frames += 4
                cases.append(dict(item=item, previous_output=left, current_output=right, batch=queries(a, b)))
            calibration = concatenate([c['batch'] for c in cases if c['item']['part'] == 'calibration'])
            assert len(calibration['target']) == 400
            gauge = Gauge(); optimizer = torch.optim.Adam([gauge.v], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0)

            def record(phase, gain, selected_cases):
                nonlocal query_rows
                for part in ('calibration', 'check'):
                    for category in ('DAVIS_GT_source', 'OLiVES_GT_source'):
                        selected = [c for c in selected_cases if (c['item']['part'], c['item']['category']) == (part, category)]
                        total = concatenate([c['batch'] for c in selected]); summary, _, _ = metrics(total, gain)
                        assert summary['queries'] == 200
                        summaries.append(dict(model_seed=seed, phase=phase, part=part, category=category, **summary))
                        for case in selected:
                            _, logits, predicted = metrics(case['batch'], gain)
                            target = case['batch']['target']
                            for i in range(len(target)):
                                append(out/'queries.jsonl', dict(model_seed=seed, phase=phase, part=part,
                                    category=category, source_index=case['item']['source_index'],
                                    translation_index=i//25, query_grid_index=i % 25,
                                    target=int(target[i]), predicted=int(predicted[i]),
                                    target_logit=float(logits[i, target[i]]), predicted_logit=float(logits[i, predicted[i]])))
                                query_rows += 1

            record('baseline', torch.ones(12), cases)
            for step in range(513):
                if step:
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(scores(calibration, gauge.gain()), calibration['target'])
                    assert torch.isfinite(loss)
                    loss.backward(); assert gauge.v.grad is not None and torch.isfinite(gauge.v.grad).all()
                    norm = torch.nn.utils.clip_grad_norm_([gauge.v], 1.)
                    assert torch.isfinite(norm)
                    optimizer.step(); updates += 1
                    assert torch.isfinite(gauge.v).all()
                if step in (0, 64, 128, 256, 512):
                    summary, _, _ = metrics(calibration, gauge.gain())
                    append(out/'calibration.jsonl', dict(model_seed=seed, step=step, **summary,
                        gain=gauge.gain().detach().tolist(), v=gauge.v.detach().tolist()))
            gain = gauge.gain().detach(); final_scales.append(dict(model_seed=seed, gain=gain.tolist()))
            assert all(p.grad is None and not p.requires_grad for p in net.parameters())
            record('final', gain, cases)
            transformed = fold(net, gain)
            max_raw = max_clip = max_logits = 0.; folded_cases = []
            for case in cases:
                item = case['item']
                left, a = capture(transformed, item['previous']); calls += 1; frames += 4
                right, b = capture(transformed, item['current']); calls += 1; frames += 4
                for original, changed in [(case['previous_output'], left), (case['current_output'], right)]:
                    max_raw = max(max_raw, float((original-changed).abs().max()))
                    max_clip = max(max_clip, float((original.clamp(0, 1)-changed.clamp(0, 1)).abs().max()))
                batch = queries(a, b)
                with torch.no_grad():
                    max_logits = max(max_logits, float((scores(case['batch'], gain)-scores(batch, torch.ones(12))).abs().max()))
                folded_cases.append(dict(item=item, batch=batch))
            gate = dict(model_seed=seed, unclipped_RGB_max_abs=max_raw, clipped_RGB_max_abs=max_clip,
                folded_vs_explicit_metric_logit_max_abs=max_logits, deployment_parameters=4075,
                F0_parameters_have_gradient=False, gain=gain.tolist(),
                passed=max_raw <= 1e-4 and max_clip <= 1e-5 and max_logits <= 1e-4)
            qualifications.append(gate); write(out/f'fold-qualification-seed{seed}.json', dict(at=stamp(), **gate))
            assert gate['passed']
            record('fold_check', torch.ones(12), folded_cases)
            path = assets/f'seed{seed}-final.pth'
            torch.save(dict(experiment='R87', seed=seed, metric_updates=512, backbone_updates=0,
                gain=gain, v=gauge.v.detach(), state_dict=transformed.state_dict(), source=marker), path)
            write(out/f'final-seed{seed}.json', dict(at=stamp(), model_seed=seed, path=str(path.relative_to(ROOT)),
                sha256=digest(path), parameters=4075, metric_updates=512, backbone_updates=0,
                scope='Function-equivalent folded weights; no video deployment or independent quality claim.'))
            print(json.dumps(dict(model_seed=seed, metric_updates=updates, backbone_image_frames=frames,
                fold_qualification_passed=gate['passed'])), flush=True)
        gates = []
        for seed in (0, 1):
            for category in ('DAVIS_GT_source', 'OLiVES_GT_source'):
                def select(phase):
                    chosen = [r for r in summaries if (r['model_seed'], r['category'], r['part'], r['phase']) == (seed, category, 'check', phase)]
                    assert len(chosen) == 1; return chosen[0]
                baseline, final = select('baseline'), select('final')
                accuracy_gain = (final['correct']-baseline['correct'])/200
                ratio = final['CE']/baseline['CE']
                gates.append(dict(model_seed=seed, category=category, baseline=baseline, final=final,
                    absolute_top1_gain=accuracy_gain, CE_ratio=ratio, passed=accuracy_gain >= .05 and ratio <= .95))
        assert (calls, frames, updates, query_rows) == (64, 256, 1024, 4800)
        assert len(summaries) == 24 and not torch.cuda.is_initialized()
        verify(); write(out/'summary.json', dict(at=stamp(), group_summaries=summaries, gates=gates,
            all_matching_preconditions=all(g['passed'] for g in gates), qualifications=qualifications,
            all_fold_qualifications=all(g['passed'] for g in qualifications), final_scales=final_scales,
            new_method_established=False, real_video_evaluation=False,
            scope='Fixed synthetic known-geometry queries on training-source crops; no novelty/generalization proof.'))
        complete = True
        print(json.dumps(dict(all_matching_preconditions=all(g['passed'] for g in gates), new_method_established=False)), flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-started, B4_calls=calls, backbone_image_frames=frames,
            metric_optimizer_updates=updates, backbone_optimizer_updates=0, query_rows=query_rows,
            CUDA_initialized=torch.cuda.is_initialized(), new_DEV_pixels=0, final_pixels=0))


if __name__ == '__main__':
    main()
