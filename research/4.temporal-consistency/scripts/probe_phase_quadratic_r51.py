"""One bounded periodic sandbox of a finite FST operator, not DEV video quality."""
import json
import time
import traceback
from datetime import datetime
from statistics import mean

import cv2
import numpy as np
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool
from phase_quadratic_r51 import MODES, periodic_variant, translate
from train_exploration_r2 import new_model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    gate = BASE/'process/exploration-r51/cpu-gate001/gate.json'
    assert json.loads(gate.read_text())['passed'] and not torch.cuda.is_initialized()
    out = BASE/'results/exploration-r51-phase-sandbox001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r51/phase-sandbox001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    start, complete, calls, rows, sources = time.perf_counter(), False, 0, [], {}
    try:
        pool = MixedPool(); _, gt, context = pool.batch(50, 0)
        phases = [(t/16, t/32) for t in range(16)]
        ff = np.fft.fftfreq(127)
        keep = (abs(ff[:, None]) < .45)&(abs(ff[None, :]) < .45)
        inputs, normalizations = [], []
        for crop in range(4):
            base = gt[crop, 0, :, :127, :127].numpy().astype(np.float64)**2.2*.35
            low = np.fft.ifft2(np.fft.fft2(base, axes=(-2,-1))*keep, axes=(-2,-1)).real
            samples = np.stack([translate(low, dx, dy) for dx, dy in phases])
            lo, hi = float(samples.min()), float(samples.max()); assert hi > lo
            # One affine transform per fixed crop, shared by every phase/model/seed.
            scale, offset = .34/(hi-lo), .01-.34*lo/(hi-lo)
            normalized = samples*scale+offset
            assert normalized.min() >= .01-1e-12 and normalized.max() <= .35+1e-12
            inputs.append(normalized.astype(np.float32))
            normalizations.append({'crop': crop, 'minimum_before': lo, 'maximum_before': hi,
                                   'scale': scale, 'offset': offset})
        inputs = np.stack(inputs); path = cache/'inputs.npy'; np.save(path, inputs, allow_pickle=False)
        write(out/'config.json', {'at': datetime.now().astimezone().isoformat(), 'device': 'cpu',
              'torch_threads': 2, 'opencv_threads': 1, 'modes': MODES, 'phases': phases,
              'inputs': str(path.relative_to(ROOT)), 'input_sha256': digest(path),
              'normalizations': normalizations, 'training_context': context, 'data': pool.info,
              'protocol_sha256': digest(BASE/'docs/exploration-r51-phase-prototype-protocol.md'),
              'source_sha256': {n: digest(BASE/'scripts'/n) for n in ['probe_phase_quadratic_r51.py',
                                   'phase_quadratic_r51.py', 'mixed_anchor_r40.py']},
              'scope': 'Artificial noise-free periodic training proxies; circularized CNN; not original video PSNR or held-out quality.'})
        per_phase = []
        for seed in [0, 1]:
            marker = json.loads((BASE/f'results/exploration-r44-stage001-seed{seed}/square_mse-step4000.json').read_text())
            assert marker['final_export_gate_passed'] and marker['new_step'] == marker['inherited_updates'] == 4000
            assert digest(ROOT/marker['path']) == marker['sha256']
            saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
            original = new_model(torch.device('cpu')); original.load_state_dict(saved['state_dict'], strict=True)
            sources[str(seed)] = marker
            base_outputs = {}
            for mode in MODES:
                net = periodic_variant(original, mode)
                assert all(torch.equal(value, net.state_dict()[name]) for name, value in original.state_dict().items())
                for crop in range(4):
                    with torch.inference_mode():
                        output = np.stack([net(torch.from_numpy(image[None]))[0].numpy() for image in inputs[crop]])
                        calls += 16
                        integer_input = torch.from_numpy(np.roll(inputs[crop, 0], 1, axis=-1).copy()[None])
                        integer_output = net(integer_input)[0].numpy(); calls += 1
                    assert np.isfinite(output).all()
                    integer_delta = float(np.max(np.abs(integer_output-np.roll(output[0], 1, axis=-1))))
                    assert integer_delta <= 2e-5
                    if mode == 'raw': base_outputs[crop] = output[0].copy()
                    baseline = base_outputs[crop]
                    mapping_mse = float(np.mean((output[0].astype(np.float64)-baseline.astype(np.float64))**2))
                    errors, clipped_errors = [], []
                    for t, (dx, dy) in enumerate(phases):
                        error = float(np.mean((output[t].astype(np.float64)-translate(output[0], dx, dy))**2))
                        clipped = np.clip(output[t], 0., 1.)
                        clip_error = float(np.mean((clipped.astype(np.float64)-translate(np.clip(output[0], 0., 1.), dx, dy))**2))
                        errors.append(error); clipped_errors.append(clip_error)
                        per_phase.append({'seed': seed, 'mode': mode, 'crop': crop, 'phase': t,
                                          'dx': dx, 'dy': dy, 'phase_MSE': error, 'clipped_phase_MSE': clip_error})
                    p = cache/f'seed{seed}-{mode}-crop{crop}.npy'; np.save(p, output, allow_pickle=False)
                    rows.append({'seed': seed, 'mode': mode, 'crop': crop,
                        'phase_MSE': mean(errors[1:]), 'clipped_phase_MSE': mean(clipped_errors[1:]),
                        'mapping_MSE_vs_raw_phase0': mapping_mse, 'integer_max_abs': integer_delta,
                        'unclipped_outside_fraction': float(((output<0)|(output>1)).mean()),
                        'path': str(p.relative_to(ROOT)), 'sha256': digest(p)})
                    print(json.dumps({'seed': seed, 'mode': mode, 'crop': crop, 'CNN_calls': calls}), flush=True)
        assert calls == 544 and len(rows) == 32 and len(per_phase) == 512
        groups, gates = [], {}
        for seed in [0, 1]:
            for mode in MODES:
                selected = [r for r in rows if r['seed'] == seed and r['mode'] == mode]
                groups.append({'seed': seed, 'mode': mode, **{field: mean(r[field] for r in selected)
                    for field in ['phase_MSE', 'clipped_phase_MSE', 'mapping_MSE_vs_raw_phase0']}})
            by_mode = {r['mode']: r for r in groups if r['seed'] == seed}
            raw, q = by_mode['raw'], by_mode['quadrature']
            smooth = [by_mode[n] for n in ['preblur', 'postblur']]
            nonworse = sum(next(r for r in rows if (r['seed'], r['mode'], r['crop']) ==
                              (seed, 'quadrature', c))['phase_MSE'] <=
                           next(r for r in rows if (r['seed'], r['mode'], r['crop']) ==
                              (seed, 'raw', c))['phase_MSE'] for c in range(4))
            checks = {'nontrivial_raw_phase_error': raw['phase_MSE'] >= (.5/255)**2,
                      'phase_vs_raw_080': q['phase_MSE'] <= .8*raw['phase_MSE'],
                      'phase_vs_best_smoothing_090': q['phase_MSE'] <= .9*min(r['phase_MSE'] for r in smooth),
                      'mapping_vs_best_smoothing_105': q['mapping_MSE_vs_raw_phase0'] <= 1.05*min(r['mapping_MSE_vs_raw_phase0'] for r in smooth),
                      'nonworse_crops_at_least_3': nonworse >= 3}
            gates[str(seed)] = {**checks, 'eligible': all(checks.values()), 'nonworse_crops': int(nonworse)}
        write(out/'per_condition.json', rows); write(out/'per_phase.json', per_phase)
        write(out/'summary.json', groups); write(out/'sources.json', sources)
        write(out/'audit.json', {'processing_complete': True, 'CNN_calls': calls, 'phase_CNN': 512,
              'integer_control_CNN': 32, 'SGD_updates': 0, 'parameters': 4075, 'gates': gates,
              'continuation_eligible': all(r['eligible'] for r in gates.values()),
              'scope': 'Periodic artificial proxies only, no GT quality or measured phone benefit.'})
        text = ['# R51 周期机制sandbox完整表', '', '0SGD；544CNN=512分数相位+32整数控制。非真实视频质量。', '',
                '|seed|模式|相位MSE|相位RMS255|clip后相位MSE|相对raw映射MSE|映射RMS255|',
                '|---:|---|---:|---:|---:|---:|---:|']
        for r in groups:
            text.append(f"|{r['seed']}|{r['mode']}|{r['phase_MSE']:.10g}|{np.sqrt(r['phase_MSE'])*255:.6f}|{r['clipped_phase_MSE']:.10g}|{r['mapping_MSE_vs_raw_phase0']:.10g}|{np.sqrt(r['mapping_MSE_vs_raw_phase0'])*255:.6f}|")
        text += ['', '```json', json.dumps(gates, indent=2), '```']
        (out/'tables.md').write_text('\n'.join(text)+'\n')
        assert not torch.cuda.is_initialized(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'CNN_calls': calls,
              'seconds': time.perf_counter()-start, 'cuda_initialized': torch.cuda.is_initialized(),
              'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__': main()
