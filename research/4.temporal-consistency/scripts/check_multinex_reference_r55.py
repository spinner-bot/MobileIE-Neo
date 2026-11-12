"""Strict original checkpoint and six synthetic CPU inputs; no quality pixels."""
import json
import time
import traceback
from datetime import datetime

import torch

from dynamic_training_r34 import BASE, digest
from multinex_reference_r55 import load_reference, prediction


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r55/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    start, complete, count = time.perf_counter(), False, 0
    try:
        model, source = load_reference()
        generator = torch.Generator().manual_seed(55)
        images = {'black': torch.zeros(1, 3, 37, 53), 'white': torch.ones(1, 3, 37, 53)}
        for c, name in enumerate(['red', 'green', 'blue']):
            image = torch.zeros(1, 3, 37, 53); image[:, c] = 1; images[name] = image
        images['random'] = torch.rand(1, 3, 37, 53, generator=generator)
        rows = []
        with torch.inference_mode():
            for name, image in images.items():
                output = prediction(model, image, clamp=False); count += 1
                assert output.shape == image.shape and torch.isfinite(output).all()
                clipped = output.clamp(0, 1)
                assert clipped.min() >= 0 and clipped.max() <= 1
                rows.append(dict(input=name, shape=list(output.shape), preclip_min=float(output.min()),
                                 preclip_max=float(output.max()), mean_clipped_RGB=clipped.mean((0, 2, 3)).tolist()))
        assert count == 6 and not torch.cuda.is_initialized()
        write(out/'gate.json', dict(at=datetime.now().astimezone().isoformat(), passed=True, source=source,
              synthetic_CNN_images=count, SGD_updates=0, cuda_initialized=False, inputs=rows,
              source_sha256={name: digest(BASE/'scripts'/name) for name in
                             ['multinex_reference_r55.py', 'check_multinex_reference_r55.py']},
              protocol_sha256=digest(BASE/'docs/exploration-r55-tiny-published-reference-protocol.md'),
              scope='Strict source/weights and numerical input handling only; no dataset quality, GPU parity, training or mobile cost claim.'))
        complete = True
        print(json.dumps(dict(passed=True, parameters=source['parameters'], synthetic_CNN_images=count)))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, synthetic_CNN_images=count,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
