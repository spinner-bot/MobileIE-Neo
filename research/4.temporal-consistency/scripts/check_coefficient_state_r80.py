"""Train-only actual-image qualification; no DEV or formal readout state inherited."""
import json
import os
from pathlib import Path
import time
import traceback

from causal_reference_io_r75 import ROOT, BASE, digest, write, stamp


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('Initially empty CUDA mask required before Torch imports.')
    out = BASE/'process/exploration-r80/cpu-gate001'
    out.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    import torch
    from coefficient_state_r80 import ConditionalReadout, PrefixRidge, PrefixTeacher, rgb_basis, rgb_predict
    from mixed_anchor_r40 import MixedPool
    from motion_transport_r74 import load_spatial
    cv2.setNumThreads(1); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    began, complete, updates = time.perf_counter(), False, 0
    protocol = BASE/'docs/exploration-r80-coefficient-state-protocol.md'
    source_paths = ['research/4.temporal-consistency/scripts/'+name for name in
        ('coefficient_state_r80.py', 'check_coefficient_state_r80.py', 'mixed_anchor_r40.py',
         'motion_transport_r74.py', 'dynamic_training_r34.py', 'native_training_r14.py',
         'train_exploration_r2.py', 'explore_temporal_controls.py', 'half_motion_r4.py',
         'video_metrics.py', 'causal_reference_io_r75.py')]
    source_paths += ['MobileIE/model/lle.py', 'MobileIE/model/utils.py', 'scripts/project-python', 'scripts/project_assets.py']
    sources = {name: digest(ROOT/name) for name in source_paths}
    try:
        pool = MixedPool(); row = pool.real[0]
        assert row['scene'] not in {'video_15', 'video_47', 'video_91', 'video_121'}
        height, width = row['native_input'].shape[1:3]
        x, y = (width-128)//2, (height-128)//2
        low = row['native_input'][:9, y:y+128, x:x+128]
        gt = row['native_gt'][:9, y:y+128, x:x+128]
        low = torch.from_numpy(low.astype(np.float32).transpose(0, 3, 1, 2).copy()/255)
        gt = torch.from_numpy(gt.astype(np.float32).transpose(0, 3, 1, 2).copy()/255)
        assert low.shape == gt.shape == (9, 3, 128, 128)
        model, marker = load_spatial(0); reader = ConditionalReadout(model)
        teacher = PrefixTeacher()
        states = dict(FST_prefix=PrefixRidge(13), RGB_prefix=PrefixRidge(12),
                      FST_GT_diagnostic=PrefixRidge(13), current_replacement=PrefixRidge(13))
        write(out/'config.json', dict(at=stamp(), pid=os.getpid(), source_sha256=sources,
            protocol_sha256=digest(protocol), seed=0, backbone=marker, data=pool.info,
            source_video=row['scene'], indices=list(range(9)), crop_xy=[x, y], crop=128,
            source_input_files=[r['path'] for r in row['source_files']['input'][:9]],
            source_target_files=[r['path'] for r in row['source_files']['gt'][:9]],
            scope='One actual training crop/9 frames, all states discarded, no DEV or publication outcomes.'))
        max_initial = max_linear = max_rgb = max_self = 0.
        probe = torch.linspace(-.5, .5, 13, dtype=torch.float32)
        stats = []
        with torch.inference_mode():
            for t in range(9):
                parts = reader.parts(low[t:t+1])
                expected = model(low[t:t+1])
                max_initial = max(max_initial, float((parts['base']-expected).abs().max()))
                bank = reader.basis(parts)
                normal = bank.double().square().mean((1, 2, 3)).sqrt().clamp_min(1e-6)
                physical = (probe.double()/(255*normal)).float()
                expanded = parts['base']+torch.einsum('p,pchw->chw', physical, bank)[None]
                max_linear = max(max_linear, float((reader.predict(parts, physical)-expanded).abs().max()))
                rgb_bank = rgb_basis(parts['base'])
                rgb_probe = torch.linspace(-.001, .001, 12)
                rgb_expanded = parts['base']+torch.einsum('p,pchw->chw', rgb_probe, rgb_bank)[None]
                max_rgb = max(max_rgb, float((rgb_predict(parts['base'], rgb_probe)-rgb_expanded).abs().max()))
                self_output = reader.predict(parts, states['current_replacement'].physical)
                max_self = max(max_self, float((self_output-parts['base']).abs().max()))
                if t < 8:
                    target = teacher.step(parts['base'])
                    states['FST_prefix'].update(bank, parts['base'], target)
                    states['RGB_prefix'].update(rgb_bank, parts['base'], target)
                    states['FST_GT_diagnostic'].update(bank, parts['base'], gt[t:t+1])
                    states['current_replacement'].update(bank, parts['base'], parts['base'])
                    updates += 4
                else:
                    for state in states.values():
                        before = state.physical.clone()
                        assert state.count == 8 and state.gram is state.rhs is state.normalizer is None
                        assert torch.equal(before, state.physical)
                    refused = False
                    try:
                        teacher.step(parts['base'])
                    except RuntimeError:
                        refused = True
                    assert refused and teacher.storage_bytes() == 0
                stats.append(dict(frame=t, teacher_calls=teacher.count, teacher_flow_pairs=teacher.flow_pairs,
                    coefficient_norms={name: float(state.physical.norm()) for name, state in states.items()},
                    state_bytes={name: state.storage_bytes() for name, state in states.items()}))
        assert max_initial <= 2e-5 and max_linear <= 1e-5 and max_rgb <= 1e-5 and max_self == 0
        assert torch.equal(states['current_replacement'].physical, torch.zeros(13))
        assert states['FST_prefix'].storage_bytes() == states['FST_GT_diagnostic'].storage_bytes() == 52
        assert states['RGB_prefix'].storage_bytes() == 48 and teacher.flow_pairs == 7 and updates == 32
        assert all(not p.requires_grad for p in model.parameters()) and not torch.cuda.is_initialized()
        assert sources == {name: digest(ROOT/name) for name in source_paths}
        complete = True
        write(out/'gate.json', dict(at=stamp(), passed=True, discarded_ridge_updates=updates,
            formal_SGD_updates=0, initial_function_max_delta=max_initial, linear_basis_max_delta=max_linear,
            RGB_affine_expansion_max_delta=max_rgb, current_replacement_max_delta=max_self,
            warm_teacher_calls=teacher.count, warm_flow_pairs=teacher.flow_pairs, postwarm_teacher_bytes=0,
            postwarm_state_bytes={name: state.storage_bytes() for name, state in states.items()},
            train_video=row['scene'], actual_train_frames=9, shared_core_calls=9, baseline_comparison_calls=9,
            frame_records=stats, CUDA_initialized=False, new_DEV_pixels=0,
            scope='Actual-image function/state/causal API qualification only; not video quality or novel method.'))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), passed=complete, discarded_ridge_updates=updates,
            formal_SGD_updates=0, pid=os.getpid(), CUDA_initialized=torch.cuda.is_initialized(),
            seconds=time.perf_counter()-began))


if __name__ == '__main__':
    main()
