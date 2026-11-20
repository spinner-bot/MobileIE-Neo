"""One N1 CPU calibration run: 2048 reference + 9216 probe model images."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'N1 is CPU-only'
from pathlib import Path
import json
import sys
import time
import traceback
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, append, stamp, digest
from motion_transport_r74 import load_spatial
from exploration_r2_data import degrade
from triple_statistics import PATTERNS, triple_inputs, describe
import cv2
import numpy as np
import torch

OUT = BASE / 'results/run001'
CACHE = ROOT / '.cache/experiments/topic8-triple-variance/run001'
TEXTURES = BASE.parent / 'phase-envelope/results/run001/textures.json'


def verify():
    frozen = json.loads((BASE / 'process/source-freeze001.json').read_text())
    assert all(digest(ROOT / p) == s for p, s in frozen['source_sha256'].items())
    return frozen


def identity():
    v = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(v[19]), state=v[0])


def predict(net, images, counts, kind):
    assert images.ndim == 4 and images.shape[1:] == (128, 128, 3) and len(images) % 8 == 0
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            batch = torch.from_numpy(images[start:start+8].transpose(0, 3, 1, 2).copy() / 255)
            value = net(batch).clamp(0, 1).numpy().copy()
            assert np.isfinite(value).all()
            outputs.append(value)
            counts[kind + '_images'] += len(value)
    return np.concatenate(outputs)


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
    counts = dict(reference_images=0, probe_images=0, reference_conditions=0, probe_conditions=0,
                  optimizer_updates=0, flow_estimations=0)
    write(OUT / 'config.json', dict(at=stamp(), process=identity(), named_source_freeze=frozen,
        device='cpu', CPU_threads=2, OpenCV_threads=1, NN_batch=8, reference_draws=128,
        probe_triplets=64, patterns=PATTERNS, common_fractions=[0, .5], seeds=[0, 1],
        source_textures=str(TEXTURES.relative_to(ROOT)), new_DEV=False,
        independent_final_opened=False, wall_clock_cutoff=False,
        innovation_or_video_improvement_claimed=False))
    try:
        textures = json.loads(TEXTURES.read_text())
        assert len(textures) == 4
        data = []
        for row in textures:
            p = ROOT / row['path']
            assert p.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(p) == row['sha256']
            image = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            x, y = row['xy']
            data.append(image[y:y+128, x:x+128].astype(np.float32))
            assert data[-1].shape == (128, 128, 3)
        write(OUT / 'textures.json', textures)
        rows, sources = [], []
        for seed in (0, 1):
            net, source = load_spatial(seed)
            sources.append(dict(seed=seed, **source))
            assert not net.training and not any(p.requires_grad for p in net.parameters())
            for texture, gt in enumerate(data):
                references = {}
                for photons in (20, 80):
                    rng = np.random.default_rng(np.random.SeedSequence([892, texture, photons, 999]))
                    inputs = np.stack([degrade(gt, rng, photons=photons, jpeg=False) for _ in range(128)])
                    prediction = predict(net, inputs, counts, 'reference').astype(np.float64)
                    references[photons] = dict(mean=prediction.mean(0), variance=prediction.var(0, ddof=1))
                    path = CACHE / f'reference-seed{seed}-texture{texture}-p{photons}.npz'
                    np.savez_compressed(path, **references[photons])
                    append(OUT / 'reference_manifest.jsonl', dict(seed=seed, texture=texture, photons=photons,
                        independent_draws=128, path=str(path.relative_to(ROOT)), sha256=digest(path),
                        global_output_variance=float(references[photons]['variance'][:, 16:-16, 16:-16].mean())))
                    counts['reference_conditions'] += 1
                    del prediction, inputs
                for pattern_index, photons in enumerate(PATTERNS):
                    for fraction in (0, .5):
                        inputs = np.stack([triple_inputs(gt, photons, fraction,
                            np.random.default_rng(np.random.SeedSequence([892, texture, pattern_index, int(100 * fraction), draw])))
                            for draw in range(64)])
                        prediction = predict(net, inputs.reshape(192, 128, 128, 3), counts, 'probe').reshape(64, 3, 3, 128, 128)
                        metrics, maps = describe(prediction, references, photons)
                        key = f'seed{seed}-texture{texture}-pattern{pattern_index}-common{int(100*fraction)}'
                        path = CACHE / (key + '.npz')
                        np.savez_compressed(path, **maps)
                        row = dict(key=key, seed=seed, texture=texture, photons=list(photons),
                                   common_fraction=fraction, main_independent_group=fraction == 0, **metrics)
                        rows.append(row)
                        append(OUT / 'per_condition.jsonl', row)
                        for draw, (error, values) in enumerate(zip(maps['per_triplet_relative_L1'], maps['signed_tiles'])):
                            append(OUT / 'per_triplet.jsonl', dict(key=key, draw=draw,
                                signed_global_variance=float(values.mean()), negative_tile_fraction=float((values < 0).mean()),
                                local_positive_relative_L1=float(error)))
                        append(OUT / 'probe_manifest.jsonl', dict(key=key, path=str(path.relative_to(ROOT)), sha256=digest(path)))
                        counts['probe_conditions'] += 1
                        write(OUT / 'live.json', dict(at=stamp(), last=key, seconds=time.perf_counter() - started, **counts))
                        print(dict(last=key, **counts), flush=True)
                        del inputs, prediction, maps
        assert counts == dict(reference_images=2048, probe_images=9216, reference_conditions=16,
                             probe_conditions=48, optimizer_updates=0, flow_estimations=0)
        main_rows = [r for r in rows if r['main_independent_group']]
        assert len(main_rows) == 24
        write(OUT / 'models.json', sources)
        write(OUT / 'summary.json', dict(at=stamp(), **counts, total_backbone_images=11264,
            main_groups=24, correlated_diagnostic_groups=24,
            gate_pass_counts={g: sum(r['gates'][g] for r in main_rows) for g in main_rows[0]['gates']},
            all_gates_passed_groups=sum(r['all_gates_passed'] for r in main_rows),
            allowed_separate_native_calibration_registration=all(r['all_gates_passed'] for r in main_rows),
            training_DEV_phone_allowed=False, established_innovation=False,
            limitations='Static training crops, finite independent MC reference, repeated noise; not sensor calibration or video improvement'))
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
