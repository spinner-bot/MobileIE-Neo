"""One registered two-training-patch noise intervention, with no scientific retry."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Initially empty CUDA mask required before Torch imports.')

import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch

from attention_mediation_r81 import ROOT, BASE, PROCESS, LABELS, parts, interventions, statistics, verify
from causal_reference_io_r75 import write, append, stamp, digest
from dynamic_training_r34 import TrainingPool
from exploration_r2_data import degrade
from motion_transport_r74 import load_spatial
from noise_role_io_r79 import owned_cache


def main():
    frozen = verify()
    out = BASE/'results/exploration-r81-attention-noise001'; out.mkdir(exist_ok=False)
    cache = owned_cache('experiments/exploration-r81/attention-noise001')
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen, draws_per_condition=32,
        source_selection='First two alphabetically sorted registered training sources, frame0 center128.',
        noise_seed='SeedSequence([81, source_ordinal, photons, draw])', expected_conditions=8,
        expected_rows=40, expected_noisy_core_calls=256, expected_clean_core_calls=4,
        expected_original_comparison_calls=12, expected_total_backbone_calls=272, expected_tail_calls=1040,
        scope='Static known-clean-dark node interventions, not DEV, deployable state, novel method or phone evidence.'))
    began, complete = time.perf_counter(), False
    noisy_calls = clean_calls = comparisons = tail_calls = 0
    maximum_delta = 0.
    artifacts, results, gates = [], [], []
    try:
        torch.set_num_threads(2); cv2.setNumThreads(1)
        torch.use_deterministic_algorithms(True)
        pool = TrainingPool(); names = pool.names[:2]; assert len(names) == 2
        fixtures = []
        for ordinal, name in enumerate(names):
            row = pool.scenes[name][0]; image = pool.image(row)
            height, width = image.shape[:2]; x, y = (width-128)//2, (height-128)//2
            gt = image[y:y+128, x:x+128].astype(np.float32)
            assert gt.shape == (128, 128, 3)
            dark = ((np.clip(gt, 0, 255)/255)**2.2*.35*255).astype(np.uint8).astype(np.float32)/255
            fixtures.append((ordinal, name, gt, dark))
            path = cache/f'source{ordinal}-gt.npy'; np.save(path, gt/255)
            append(out/'inputs.jsonl', dict(source_ordinal=ordinal, source_video=name, source_frame=row['name'],
                source_path=row['path'], source_sha256=row['sha256'], crop_xy=[x, y], crop=128,
                GT_path=str(path.relative_to(ROOT)), no_new_asset_download=True,
                scope='Known RGB training GT and deterministic darkening, not real sensor clean low-light.'))
        with torch.inference_mode():
            for seed in (0, 1):
                net, checkpoint = load_spatial(seed)
                saved = {name: value.clone() for name, value in net.state_dict().items()}
                assert sum(p.numel() for p in net.parameters()) == 4075
                for ordinal, name, gt, dark in fixtures:
                    clean = parts(net, torch.from_numpy(dark.transpose(2, 0, 1).copy())[None])
                    clean_calls += 1; tail_calls += 1
                    expected = net(torch.from_numpy(dark.transpose(2, 0, 1).copy())[None])
                    comparisons += 1; tail_calls += 1
                    delta = float((clean['raw']-expected).abs().max()); maximum_delta = max(maximum_delta, delta)
                    assert delta <= 2e-5
                    for photons in (20, 80):
                        arrays = {label: [] for label in LABELS}
                        raw_arrays = {label: [] for label in LABELS[:-1]}
                        inputs, previous = [], None
                        for draw in range(32):
                            rng = np.random.default_rng(np.random.SeedSequence([81, ordinal, photons, draw]))
                            image = degrade(gt, rng, photons=photons, jpeg=False)/255
                            inputs.append(image)
                            low = torch.from_numpy(image.transpose(2, 0, 1).copy())[None]
                            current = parts(net, low); noisy_calls += 1; tail_calls += 1
                            if draw == 0:
                                expected = net(low); comparisons += 1; tail_calls += 1
                                delta = float((current['raw']-expected).abs().max()); maximum_delta = max(maximum_delta, delta)
                                assert delta <= 2e-5
                            outputs = interventions(net, current, clean); tail_calls += 3
                            for label, value in outputs.items():
                                assert bool(torch.isfinite(value).all())
                                raw_arrays[label].append(value[0].numpy().copy())
                                arrays[label].append(value[0].clamp(0, 1).numpy().copy())
                            value = torch.from_numpy(arrays['raw'][-1])
                            previous = value if previous is None else .5*value+.5*previous
                            arrays['recursive05'].append(previous.numpy().copy())
                            append(out/'node-observations.jsonl', dict(seed=seed, source_video=name,
                                source_ordinal=ordinal, photons=photons, draw=draw,
                                winner_change_fraction=float((current['winner'] != clean['winner']).float().mean()),
                                local_max_abs_delta_mean=float((current['maximum']-clean['maximum']).abs().mean()),
                                global_gate_abs_delta_mean=float((current['gate']-clean['gate']).abs().mean())))
                        target = (gt/255).transpose(2, 0, 1)
                        metrics = {}
                        for label in LABELS:
                            values = np.stack(arrays[label])
                            metric = dict(seed=seed, source_video=name, source_ordinal=ordinal, photons=photons,
                                label=label, draws=32, **statistics(values, target),
                                deployable_node_intervention=label in ('raw', 'recursive05'))
                            if label in raw_arrays:
                                metric['unclipped'] = statistics(np.stack(raw_arrays[label]), target)
                            metrics[label] = metric; results.append(metric); append(out/'per_condition.jsonl', metric)
                            for draw, mse in enumerate(np.mean((values.astype(np.float64)-target.astype(np.float64))**2, axis=(1,2,3))):
                                append(out/'per_draw.jsonl', dict(seed=seed, source_video=name, photons=photons,
                                    label=label, draw=draw, MSE=float(mse), psnr_rgb=float(-10*np.log10(max(mse, 1e-12)))))
                        raw, local = metrics['raw'], metrics['clean_local']
                        ratio = local['temporal_difference_255']/raw['temporal_difference_255']
                        difference = local['mean_psnr_rgb']-raw['mean_psnr_rgb']
                        gates.append(dict(seed=seed, source_video=name, source_ordinal=ordinal, photons=photons,
                            PSNR_delta=difference, temporal_ratio=ratio, passed=difference >= -.05 and ratio <= .90))
                        prediction = cache/f'seed{seed}-source{ordinal}-photons{photons}-outputs.npy'
                        noisy = cache/f'seed{seed}-source{ordinal}-photons{photons}-inputs.npy'
                        np.save(prediction, np.stack([np.stack(arrays[label]) for label in LABELS]))
                        np.save(noisy, np.stack(inputs))
                        artifact = dict(seed=seed, source_ordinal=ordinal, source_video=name, photons=photons,
                            labels=list(LABELS), checkpoint=checkpoint, output_path=str(prediction.relative_to(ROOT)),
                            output_sha256=digest(prediction), input_path=str(noisy.relative_to(ROOT)), input_sha256=digest(noisy),
                            GT_path=str((cache/f'source{ordinal}-gt.npy').relative_to(ROOT)),
                            output_layout='label,draw,channel,height,width, normalized float32, clamped[0,1]',
                            input_layout='draw,height,width,channel, normalized float32', draws=32)
                        artifacts.append(artifact); append(out/'prediction_manifest.jsonl', artifact)
                        print(json.dumps(dict(seed=seed, source_video=name, photons=photons,
                            completed_conditions=len(artifacts), expected=8)), flush=True)
                assert all(torch.equal(saved[name], value) for name, value in net.state_dict().items())
                assert not net.training and not any(p.requires_grad for p in net.parameters())
        assert (noisy_calls, clean_calls, comparisons, tail_calls) == (256, 4, 12, 1040)
        assert len(results) == 40 and len(gates) == len(artifacts) == 8 and not torch.cuda.is_initialized()
        verify(); write(out/'gates.json', gates)
        write(out/'audit.json', dict(completed_normally=True, conditions=8, rows=40, per_draw_rows=1280,
            all_primary_gates_pass=all(r['passed'] for r in gates), passed_primary_gates=sum(r['passed'] for r in gates),
            scope='Only triage of fixed known-clean local gate intervention on static synthetic training patches. Not a universal upper bound, real-video result or method.'))
        table = ['# R81完整训练图块静态独立噪声干预', '',
            'clean_*需要已知clean-dark训练参照，全部不可部署；recursive05为完美静止对应参考。非DEV/自然噪声/完整视频/手机证据。', '',
            '|seed|训练源|photons|路径|PSNR均值|相邻差/255尺度|方差|均值偏差平方|',
            '|---:|---|---:|---|---:|---:|---:|---:|']
        for r in results:
            table.append(f"|{r['seed']}|{r['source_video']}|{r['photons']}|{r['label']}|{r['mean_psnr_rgb']:.6f}|{r['temporal_difference_255']:.6f}|{r['variance']:.9f}|{r['squared_bias']:.9f}|")
        (out/'tables.md').write_text('\n'.join(table)+'\n'); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, pid=os.getpid(),
            seconds=time.perf_counter()-began, noisy_shared_core_calls=noisy_calls, clean_core_calls=clean_calls,
            original_comparison_calls=comparisons, total_backbone_calls=noisy_calls+clean_calls+comparisons,
            tail_calls=tail_calls, completed_conditions=len(artifacts), completed_rows=len(results),
            new_SGD=0, new_DEV_pixels=0, new_GPU=False, CUDA_initialized=torch.cuda.is_initialized(),
            actual_visual_review_pending=complete))


if __name__ == '__main__':
    main()
