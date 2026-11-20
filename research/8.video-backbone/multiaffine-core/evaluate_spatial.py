"""Unique 8K finals on fixed held-out draws of training scenes, never DEV/final."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'M1 spatial quality is CPU-only'
import argparse
import time
import traceback
import numpy as np
import torch
from study_io import BASE, ROOT, PROCESS, ARMS, PARAMETERS, read, write, append, digest, stamp, verify, precision, models, owned_cache, identity
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from video_metrics import psnr, ssim, edge_error


def markers(seed):
    runs = sorted(PROCESS.glob(f'spatial-seed{seed}-attempt*/runtime.json'))
    assert runs and read(runs[-1])['completed_normally']
    assert read(runs[-1])['counts'] == dict(common_steps=8000, optimizer_updates=16000, student_images=256000)
    result = {}
    for arm in ARMS:
        marker = read(runs[-1].parent / (arm + '-final.json'))
        assert (marker['seed'], marker['updates'], marker['parameters']) == (seed, 8000, PARAMETERS[arm])
        assert marker['unique_final'] and marker['all_reloaded_finite'] and not marker['qualification']
        p = ROOT / marker['path']
        assert p.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(p) == marker['sha256']
        result[arm] = marker
    return result


def main(seed):
    precision(); os.nice(5)
    frozen, finals = verify(), markers(seed)
    out = BASE / f'results/spatial-held-seed{seed}'
    assets = owned_cache(out.name)
    out.mkdir(parents=True, exist_ok=False); assets.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    counts = dict(backbone_images=0, quality_rows=0, frame_rows=0, windows=0)
    write(out / 'config.json', dict(at=stamp(), process=identity(os.getpid()), seed=seed,
        source_freeze=frozen, unique_finals=finals, sampling_seed=8940+seed, data_steps=list(range(12000, 12016)),
        held_out_draws_not_scenes=True, new_DEV=False, independent_final_opened=False, wall_clock_cutoff=False))
    try:
        students = models(seed)
        for arm in ARMS:
            students[arm].load_state_dict(torch.load(ROOT / finals[arm]['path'], map_location='cpu', weights_only=True), strict=True)
        f0, provenance = load_spatial(seed)
        students['F0'] = f0
        write(out / 'F0.json', provenance)
        for student in students.values():
            student.eval().requires_grad_(False)
        pool = MixedPool()
        write(out / 'data.json', pool.info)
        with torch.inference_mode():
            for step in range(12000, 12016):
                low, gt, windows = pool.batch(8940+seed, step)
                outputs = {arm: model(low.flatten(0, 1)).unflatten(0, (4, 4)).numpy().copy() for arm, model in students.items()}
                outputs['F0'] = outputs['F0'].clip(0, 1)
                counts['backbone_images'] += 48
                append(out / 'windows.jsonl', dict(data_step=step, windows=windows))
                for index in range(4):
                    key = f'seed{seed}-step{step}-window{index}'
                    domain = 'SYN' if index < 2 else 'ND'
                    target = gt[index].numpy().transpose(0, 2, 3, 1) * 255
                    path = assets / (key + '.npz')
                    np.savez_compressed(path, low=low[index].numpy(), gt=gt[index].numpy(), **{k: v[index] for k, v in outputs.items()})
                    append(out / 'prediction_manifest.jsonl', dict(key=key, domain=domain, path=str(path.relative_to(ROOT)), sha256=digest(path)))
                    for arm, value in outputs.items():
                        raw = value[index]
                        assert np.isfinite(raw).all()
                        display = raw.clip(0, 1).transpose(0, 2, 3, 1) * 255
                        details = []
                        for frame in range(4):
                            row = dict(key=key, domain=domain, seed=seed, arm=arm, frame=frame,
                                psnr_rgb=min(psnr(display[frame], target[frame]), 120.),
                                psnr_120dB_ceiling_applied=psnr(display[frame], target[frame]) >= 120.,
                                ssim_rgb=ssim(display[frame], target[frame]),
                                edge_mae=edge_error(display[frame], target[frame]))
                            append(out / 'per_frame.jsonl', row); details.append(row)
                        fidelity = float(np.mean((raw.astype(np.float64) - outputs['F0'][index]) ** 2))
                        append(out / 'per_sequence.jsonl', dict(key=key, domain=domain, seed=seed, arm=arm,
                            parameters=4075 if arm == 'F0' else PARAMETERS[arm],
                            **{m: float(np.mean([r[m] for r in details])) for m in ('psnr_rgb', 'ssim_rgb', 'edge_mae')},
                            raw_fidelity_MSE_to_F0=fidelity,
                            raw_fidelity_PSNR_to_F0=None if fidelity == 0 else float(-10*np.log10(fidelity)),
                            raw_min=float(raw.min()), raw_max=float(raw.max()),
                            outside_display_range_fraction=float(((raw < 0) | (raw > 1)).mean())))
                        counts['quality_rows'] += 1; counts['frame_rows'] += 4
                    counts['windows'] += 1
                print(dict(seed=seed, data_step=step, **counts), flush=True)
        assert counts == dict(backbone_images=768, quality_rows=192, frame_rows=768, windows=64)
        verify(); assert not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), process=identity(os.getpid()), completed_normally=complete,
            seconds=time.perf_counter()-started, **counts, CUDA_initialized=torch.cuda.is_initialized()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    main(parser.parse_args().seed)
