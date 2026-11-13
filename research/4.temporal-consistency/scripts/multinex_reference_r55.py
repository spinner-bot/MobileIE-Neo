"""Original single published Nano checkpoint, with the official inference wrapper."""
import importlib.util
import json

import torch
import torch.nn.functional as F
import yaml

from dynamic_training_r34 import BASE, ROOT, digest


def load_reference():
    run = BASE/'process/external-multinex/acquisition001'
    assert json.loads((run/'runtime.json').read_text())['completed_normally']
    manifest = json.loads((run/'manifest.json').read_text())
    files = {r['source']: r for r in manifest['files']}
    names = ['basicsr/models/archs/MultinexNano_arch.py', 'Options/Multinex-Nano_LOL-v1.yaml',
             'pretrained_weights/Multinex-Nano_LOL-v1.pth']
    for name in names:
        assert digest(ROOT/files[name]['path']) == files[name]['sha256']
    path = ROOT/files[names[0]]['path']
    spec = importlib.util.spec_from_file_location('mobileie_multinex_nano_upstream', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config = yaml.safe_load((ROOT/files[names[1]]['path']).read_text())['network_g'].copy()
    assert config.pop('type') == 'MultinexNano'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5500)
        model = module.MultinexNano(**config)
    saved = torch.load(ROOT/files[names[2]]['path'], map_location='cpu', weights_only=True)
    assert isinstance(saved['params'], dict)
    model.load_state_dict(saved['params'], strict=True)
    model.eval().requires_grad_(False)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    info = dict(repository=manifest['repository'], commit=manifest['commit'],
                source_manifest_sha256=digest(run/'manifest.json'), files={name: files[name] for name in names},
                network_kwargs=config, parameters=sum(p.numel() for p in model.parameters()),
                buffer_elements=sum(b.numel() for b in model.buffers()), state_entries=len(model.state_dict()),
                strict_load=True, input='RGB FP32 [0,1]', inference='Official even reflection pad, unpad and clip; no GT mean, ensemble or large-image splitting.',
                training='Published LOL-v1 weights; no project task adaptation or seed search; complete original training not reproduced.')
    return model, info


def prediction(model, image, clamp=True):
    assert image.ndim == 4 and image.shape[1] == 3
    h, w = image.shape[-2:]
    output = model(F.pad(image, (0, (-w) % 2, 0, (-h) % 2), mode='reflect'))[..., :h, :w]
    return output.clamp(0, 1) if clamp else output
