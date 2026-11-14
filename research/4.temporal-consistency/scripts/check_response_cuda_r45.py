"""One seed0 CUDA backend gate after fresh external and post-data admission."""
from datetime import datetime
import json
import os
import subprocess
import time
import traceback

import cv2
import torch

from check_response_cue_r45 import numpy_reference
from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool, prediction
from response_cue_r45 import FrozenResponses, RecursiveControl, load_frozen
from response_sequence_r45 import motion_for, move_fields, rollout
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert json.loads((BASE/'process/exploration-r45/cpu-attempt002/gate.json').read_text())['passed']
    out = BASE/'process/exploration-r45/cuda-attempt001'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True;torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    started, complete, updates = time.perf_counter(), False, 0
    try:
        pool = MixedPool()
        low, gt, context = pool.batch(0, 8000)
        backbone, marker = load_frozen(0)
        before = {n: p.clone() for n, p in backbone.state_dict().items()}
        admission = cuda_headroom()
        write(out/'pre_cuda_guard.json', admission)
        assert admission['allowed'], 'Headroom changed while loading CPU training data'
        names = ['check_response_cuda_r45.py', 'response_cue_r45.py', 'response_sequence_r45.py',
                 'budget_causal_r33.py', 'mixed_anchor_r40.py', 'half_motion_r4.py']
        write(out/'config.json', {'started_at': datetime.now().astimezone().isoformat(),
              'seed': 0, 'context': context, 'checkpoint': marker,
              'protocol_sha256': digest(BASE/'docs/exploration-r45-controlled-recursion-training-protocol.md'),
              'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Six discarded SGD updates, CUDA backend implementation only; no formal quality.'})
        backbone = backbone.cuda().eval();low = low.cuda();gt = gt.cuda()
        with torch.no_grad():
            original = prediction(backbone, low)
        encoder = FrozenResponses(backbone)
        values = [encoder(frame) for frame in low.unbind(1)]
        unclipped = torch.stack([v[0] for v in values], dim=1)
        cues = torch.stack([v[1] for v in values], dim=1)
        parity = float((unclipped-original).abs().max())
        assert parity <= 1e-6 and bool(torch.isfinite(cues).all())
        raw = unclipped.clamp(0, 1)
        fields = motion_for(raw)
        reference = numpy_reference(raw.cpu(), fields).cuda()
        fields = move_fields(fields, 'cuda')
        initial, gradients = {}, {}
        for name in ['fixed', 'scalar', 'rgb', 'response']:
            control = RecursiveControl(name, 0).cuda()
            output = rollout(control, raw, cues, fields)
            delta = float((output-reference).abs().max().detach())
            assert delta <= 1e-5 and torch.equal(output[:, 0], raw[:, 0])
            flow, mask = fields[0]
            rejected = control(raw[:, 1], raw[:, 0], flow, torch.zeros_like(mask), cues[:, 1])[0]
            assert torch.equal(rejected, raw[:, 1])
            initial[name] = delta
            if name == 'fixed':
                continue
            optimizer = torch.optim.SGD(control.parameters(), lr=.01)
            values = []
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = (rollout(control, raw, cues, fields)-gt).square().mean()
                assert bool(torch.isfinite(loss));loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in control.parameters())
                norms = {n: float(p.grad.norm()) for n, p in control.named_parameters()}
                assert sum(norms.values()) > 0
                if name in ['rgb', 'response'] and step == 1:
                    assert norms['gate.0.weight'] > 0
                values.append(norms);optimizer.step();updates += 1
            gradients[name] = values
        assert updates == 6 and all(torch.equal(before[n], t.cpu()) for n, t in backbone.state_dict().items())
        write(out/'gate.json', {'passed': True, 'raw_parity': parity, 'initial_recursive_deltas': initial,
              'gradient_norms': gradients, 'frozen_backbone_unchanged': True,
              'discarded_SGD_updates': updates, 'formal_updates': 0,
              'scope': 'CUDA implementation gate only, not quality or independent cue evidence.'})
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'discarded_SGD_updates': updates,
              'wall_seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat()})
    print(json.dumps({'passed': True, 'discarded_updates': updates}), flush=True)


if __name__ == '__main__':
    main()
