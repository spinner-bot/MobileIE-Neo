"""One-forward precision localization after the actual initial-function failure."""
import json
from pathlib import Path

import torch

from dynamic_training_r34 import BASE, digest
from train_exploration_r2 import new_model
from training_form_r41 import create_model, cuda_headroom


def main():
    out = BASE/'process/exploration-r41/forward-precision-localization.json'
    assert not out.exists()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    admission = cuda_headroom()
    assert admission['allowed']
    full, _ = create_model(0)
    reference = new_model(torch.device('cpu'))
    full, reference = full.cuda().eval(), reference.cuda().eval()
    image = torch.rand(2, 3, 32, 36, device='cuda')
    names = ['head.block1.0', 'head.block1.2', 'head', 'body', 'att', 'att1', 'tail']
    rows = []
    with torch.inference_mode():
        for allowed in [True, False]:
            torch.backends.cudnn.allow_tf32 = allowed
            captures, handles = {}, []
            for key, net in [('full', full), ('slim', reference)]:
                for name in names:
                    def hook(module, inputs, output, key=key, name=name):
                        captures[(key, name)] = output.detach().clone()
                    handles.append(net.get_submodule(name).register_forward_hook(hook))
            try:
                full_value, slim_value = full(image), reference(image)
            finally:
                for handle in handles:
                    handle.remove()
            rows.append({'cudnn_tf32_allowed': allowed,
                         'max_abs_output_delta': float((full_value-slim_value).abs().max()),
                         'layer_max_abs_deltas': {name: float((captures[('full', name)]-captures[('slim', name)]).abs().max())
                                                  for name in names}})
    result = {'admission': admission, 'comparisons': rows, 'optimizer_updates': 0,
              'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                                ['localize_training_form_precision_r41.py', 'training_form_r41.py']},
              'scope': 'Single input, same initial weights, two forward-only precision controls. No optimizer replay, quality or training result.'}
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
