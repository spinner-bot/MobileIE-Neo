"""Actual fixed-window CPU qualification; discarded updates are never inherited."""
import json
from datetime import datetime
import time
import traceback

import cv2
import torch

from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool, prediction
from spatial_l1_r69 import initial_model, verify_old_reference, check_one
from training_form_r41 import configure_precision


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    configure_precision(); assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r69/cpu-gate001'
    out.mkdir(parents=True, exist_ok=False)
    start, complete, discarded = time.perf_counter(), False, 0
    try:
        refs = {seed: verify_old_reference(seed) for seed in [0, 1]}
        write(out/'MSE-reference-check.json', dict(passed=True, seeds=refs, new_MSE_training=0))
        pool = MixedPool(); low, target, windows = pool.batch(0, 0)
        model, info = initial_model(0)
        model.eval()
        with torch.inference_mode():
            folded = model.slim().eval()
            initial_fold = float((prediction(model, low)-prediction(folded, low)).abs().max())
        assert initial_fold <= 1e-4
        del folded, model
        models = {objective: initial_model(0)[0] for objective in ['L1', 'MSE']}
        with torch.no_grad():
            predictions = {name: prediction(model.train(), low) for name, model in models.items()}
            delta = float((predictions['L1']-predictions['MSE']).abs().max())
        assert delta <= 1e-6
        del predictions
        source = {name: digest(BASE/'scripts'/name) for name in
                  ['spatial_l1_r69.py', 'check_spatial_l1_r69.py', 'training_form_r41.py',
                   'mixed_anchor_r40.py', 'dynamic_training_r34.py', 'native_training_r14.py']}
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), device='cpu',
              data=pool.info, initialization=info, windows=windows, source_sha256=source,
              initial_eval_fold_max_abs=initial_fold, same_initial_prediction_max_abs=delta,
              protocol_sha256=digest(BASE/'docs/exploration-r69-matched-spatial-L1-protocol.md'),
              scope='One real train window only; not arbitrary-input fold, GPU qualification or quality.'))
        results = {}
        for objective, model in models.items():
            results[objective] = check_one(model, low, target, objective)
            discarded += 1
            write(out/f'{objective}-discarded-update.json', results[objective])
        assert not torch.cuda.is_initialized()
        write(out/'gate.json', dict(passed=True, objectives=results, discarded_SGD_updates=discarded,
              initial_eval_fold_max_abs=initial_fold, same_initial_prediction_max_abs=delta,
              new_formal_updates=0, cuda_initialized=False, new_DEV_observations=0))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              discarded_SGD_updates=discarded, new_formal_updates=0,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
