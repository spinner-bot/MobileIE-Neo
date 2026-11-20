"""One registered, CPU-only, 1280-image P1 screening run, with no optimizer."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'P1 is CPU-only'

from pathlib import Path
import sys
import time
import traceback
import json

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, append, stamp, digest
from dynamic_training_r34 import TrainingPool
from exploration_r2_data import degrade
from motion_transport_r74 import load_spatial
from state import ARMS, Pipeline
import cv2
import numpy as np
import torch

ALL_ARMS = (*ARMS, 'known_motion_post')
OUT = BASE / 'results/run001'
CACHE = ROOT / '.cache/experiments/topic8-phase-envelope/run001'
FREEZE = BASE / 'process/source-freeze001.json'


def verify():
    frozen = json.loads(FREEZE.read_text())
    assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
    return frozen


def identity():
    values = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(values[19]), state=values[0])


def summarize(output, target):
    error = (output - target).astype(np.float64)
    central = error[..., 32:-32, 32:-32]
    mse = np.mean(central ** 2, axis=(1, 2, 3))
    assert np.all(mse > 0)
    psnr = -10 * np.log10(mse)
    temporal = (error[1:] - np.roll(error[:-1], 1, axis=-1))[..., 32:-32, 32:-32]
    return dict(first_PSNR=float(psnr[0]), PSNR=float(psnr[1:].mean()),
                temporal_MSE=float(np.mean(temporal ** 2)),
                frame_PSNR=psnr.tolist(),
                pair_temporal_MSE=np.mean(temporal ** 2, axis=(1, 2, 3)).tolist())


def aggregate(rows):
    groups = []
    for seed in (0, 1):
        for photons in (None, 20, 80):
            for arm in ALL_ARMS:
                chosen = [r for r in rows if r['seed'] == seed and r['photons'] == photons and r['arm'] == arm]
                assert len(chosen) == (4 if photons is None else 8)
                groups.append(dict(seed=seed, photons=photons, arm=arm, sequences=len(chosen),
                                   PSNR=float(np.mean([r['PSNR'] for r in chosen])),
                                   temporal_MSE=float(np.mean([r['temporal_MSE'] for r in chosen]))))
    comparisons = []
    for seed in (0, 1):
        for photons in (None, 20, 80):
            group = {r['arm']: r for r in groups if r['seed'] == seed and r['photons'] == photons}
            for reference in ('raw', 'envelope_spatial', 'feature_ema', 'known_motion_post'):
                candidate, ref = group['envelope_temporal'], group[reference]
                cand_rows = {r['key']: r for r in rows if r['seed'] == seed and r['photons'] == photons and r['arm'] == 'envelope_temporal'}
                ref_rows = {r['key']: r for r in rows if r['seed'] == seed and r['photons'] == photons and r['arm'] == reference}
                comparisons.append(dict(seed=seed, photons=photons, reference=reference,
                    PSNR_delta=candidate['PSNR'] - ref['PSNR'],
                    temporal_MSE_ratio=candidate['temporal_MSE'] / ref['temporal_MSE'],
                    worst_sequence_PSNR_delta=min(cand_rows[k]['PSNR'] - ref_rows[k]['PSNR'] for k in cand_rows)))
    noisy = [r for r in comparisons if r['photons'] is not None]
    clean_raw = [r for r in comparisons if r['photons'] is None and r['reference'] == 'raw']
    gates = dict(
        time_vs_raw=all(r['temporal_MSE_ratio'] <= .90 for r in noisy if r['reference'] == 'raw'),
        time_vs_spatial=all(r['temporal_MSE_ratio'] <= .95 for r in noisy if r['reference'] == 'envelope_spatial'),
        noisy_quality=all(r['PSNR_delta'] >= -.05 and r['worst_sequence_PSNR_delta'] >= -.5
                          for r in noisy if r['reference'] != 'known_motion_post'),
        clean_mapping=all(r['PSNR_delta'] >= -.05 and r['worst_sequence_PSNR_delta'] >= -.5 for r in clean_raw))
    return dict(groups=groups, comparisons=comparisons, gates=gates,
                allowed_separate_training_registration=all(gates.values()),
                established_method=False, independent_final_evidence=False)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    os.nice(5)
    frozen = verify()
    assert CACHE.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    OUT.mkdir(parents=True, exist_ok=False)
    CACHE.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    counts = dict(backbone_images=0, sequences=0, optimizer_updates=0, flow_estimations=0)
    write(OUT / 'config.json', dict(at=stamp(), process=identity(), named_source_freeze=frozen,
          device='cpu', CPU_threads=2, OpenCV_threads=1, seeds=[0, 1], arms=ALL_ARMS,
          new_DEV=False, independent_final_opened=False, wall_clock_cutoff=False,
          approximate_energy_not_exact_Riesz=True))
    try:
        pool, rng, textures, identities = TrainingPool(), np.random.default_rng(np.random.SeedSequence([891, 0])), [], []
        indices = rng.choice(len(pool.names), 4, replace=False)
        for index in indices:
            scene = pool.names[int(index)]
            records = pool.scenes[scene]
            row = records[int(rng.integers(len(records)))]
            image = pool.image(row)
            y, x = int(rng.integers(image.shape[0] - 127)), int(rng.integers(image.shape[1] - 127))
            textures.append(image[y:y+128, x:x+128].astype(np.float32))
            identities.append(dict(scene=scene, member=row['name'], path=row['path'],
                                   sha256=row['sha256'], xy=[x, y]))
        write(OUT / 'textures.json', identities)
        rows, checks, provenance = [], [], []
        for seed in (0, 1):
            net, source = load_spatial(seed)
            provenance.append(dict(seed=seed, **source))
            assert not net.training and not any(p.requires_grad for p in net.parameters())
            assert sum(p.numel() for p in net.parameters()) == 4075
            with torch.inference_mode():
                for texture, base in enumerate(textures):
                    gt_hwc = np.stack([np.roll(base, t, axis=1) for t in range(8)])
                    gt = gt_hwc.transpose(0, 3, 1, 2).copy() / 255
                    conditions = [(None, None), (20, 0), (20, 1), (80, 0), (80, 1)]
                    for photons, rep in conditions:
                        key = f'texture{texture}-p{photons}-rep{rep}'
                        if photons is None:
                            low = ((gt_hwc / 255) ** 2.2 * .35 * 255).astype(np.uint8).astype(np.float32)
                        else:
                            noise = np.random.default_rng(np.random.SeedSequence([891, texture, photons, rep]))
                            low = np.stack([degrade(frame, noise, photons=photons, jpeg=False) for frame in gt_hwc])
                        inputs = torch.from_numpy(low.transpose(0, 3, 1, 2).copy() / 255)
                        outputs, state_bytes = {}, {}
                        for arm in ARMS:
                            pipe = Pipeline(net, arm)
                            outputs[arm] = torch.cat([pipe(frame[None]) for frame in inputs]).numpy().copy()
                            assert outputs[arm].shape == gt.shape and np.isfinite(outputs[arm]).all()
                            state_bytes[arm] = pipe.state_bytes()
                            counts['backbone_images'] += 8
                        raw = outputs['raw']
                        for arm in ('feature_ema', 'envelope_temporal'):
                            difference = float(np.max(np.abs(outputs[arm][0] - raw[0])))
                            assert difference <= 2e-6
                            assert state_bytes[arm] == 12 * 32 * 32 * 4
                            checks.append(dict(seed=seed, key=key, arm=arm, first_frame_and_manual_recomposition_max_abs_difference=difference,
                                               persistent_state_bytes=state_bytes[arm]))
                        post = [raw[0]]
                        for frame in raw[1:]:
                            post.append(.5 * frame + .5 * np.roll(post[-1], 1, axis=-1))
                        outputs['known_motion_post'] = np.stack(post)
                        target_path = CACHE / f'seed{seed}-{key}.npz'
                        np.savez_compressed(target_path, gt=gt, low=inputs.numpy(), **outputs)
                        append(OUT / 'prediction_manifest.jsonl', dict(seed=seed, key=key,
                            path=str(target_path.relative_to(ROOT)), sha256=digest(target_path)))
                        for arm, output in outputs.items():
                            record = dict(seed=seed, texture=texture, photons=photons, noise_rep=rep,
                                          key=key, arm=arm, **summarize(output, gt))
                            rows.append(record)
                            append(OUT / 'per_sequence.jsonl', record)
                        counts['sequences'] += 1
                        write(OUT / 'live.json', dict(at=stamp(), **counts,
                              seconds=time.perf_counter() - started, last=dict(seed=seed, key=key)))
                        print(dict(seed=seed, key=key, **counts), flush=True)
        assert counts == dict(backbone_images=1280, sequences=40, optimizer_updates=0, flow_estimations=0)
        assert len(rows) == 200 and len(checks) == 80
        write(OUT / 'model_sources.json', provenance)
        write(OUT / 'implementation_checks.json', dict(all_passed=True, checks=checks,
            recomposition_uses_already_counted_first_frame_routes=True, maximum_difference=max(r['first_frame_and_manual_recomposition_max_abs_difference'] for r in checks)))
        write(OUT / 'summary.json', dict(at=stamp(), **counts, **aggregate(rows)))
        verify()
        assert not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(OUT / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter() - started, CUDA_initialized=torch.cuda.is_initialized(), **counts))


if __name__ == '__main__':
    main()
