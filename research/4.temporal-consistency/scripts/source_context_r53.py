"""Keep four original targets, supplying five-frame windows from the source clip."""
import numpy as np
import torch
import torch.nn.functional as F

from exploration_r2_data import degrade
from mixed_anchor_r40 import MixedPool
from task_fastdvd_r47 import SIGMA


def reflect(index, length):
    assert length >= 2
    q = index % (2*length-2)
    return q if q < length else 2*length-2-q


def crop(image, context):
    x, y = context['xy']
    value = image[y:y+128, x:x+128]
    if context['flip_horizontal']: value = value[:, ::-1]
    if context['flip_vertical']: value = value[::-1]
    assert value.shape == (128, 128, 3)
    return value.astype(np.float32)


class SourceContextPool:
    def __init__(self):
        self.original = MixedPool()
        self.info = self.original.info
        self.real = {row['scene']: row for row in self.original.real}

    def batch(self, seed, step):
        low, target, identities = self.original.batch(seed, step)
        context8 = low.new_empty((4, 8, 3, 128, 128))
        context8[:, 2:6] = low
        provenance = []
        for b, identity in enumerate(identities):
            if identity['source'] == 'davis_synthetic':
                records = self.original.synthetic.scenes[identity['video']]
                names = [r['name'] for r in records]
                start = names.index(identity['members'][0]); length = len(records)
                assert names[start:start+4] == identity['members']
                def load(index):
                    image = crop(self.original.synthetic.image(records[index]), identity)
                    rng = np.random.default_rng(np.random.SeedSequence([53, seed, step, b, index]))
                    value = degrade(image, rng, photons=identity['photons'], jpeg=False)
                    return torch.from_numpy(value.transpose(2,0,1).copy()/255)
                def source_name(index): return records[index]['path']
            else:
                assert identity['source'] == 'olives_real'
                row = self.real[identity['video']]
                start = identity['indices'][0]; length = len(row['native_input'])
                assert identity['indices'] == list(range(start, start+4)) and length == 16
                def load(index):
                    value = crop(row['native_input'][index], identity)
                    return torch.from_numpy(value.transpose(2,0,1).copy()/255)
                def source_name(index): return row['source_files']['input'][index]['path']
            indices = [reflect(start+j-2, length) for j in range(8)]
            observations = {start+j: low[b,j] for j in range(4)}
            for j in [0,1,6,7]:
                index = indices[j]
                if index not in observations: observations[index] = load(index)
                context8[b,j] = observations[index]
            assert indices[2:6] == list(range(start, start+4))
            windows = [indices[j:j+5] for j in range(4)]
            provenance.append({'source': identity['source'], 'video': identity['video'],
                'available_source_length': length, 'target_start': start, 'indices': indices,
                'source_inputs': [source_name(index) for index in indices], 'target_windows': windows,
                'windows_with_duplicate_source_index': sum(len(set(w)) < 5 for w in windows),
                'reflects_available_source_boundary': start < 2 or start+5 >= length})
        assert torch.equal(context8[:,2:6], low)
        return context8, target, identities, provenance, low


def prediction(model, source8, clamp=False):
    b, t, c, h, w = source8.shape
    assert t == 8 and c == 3
    indices = torch.arange(4, device=source8.device)[:,None]+torch.arange(5, device=source8.device)[None,:]
    windows = source8[:,indices].reshape(b*4, 15, h, w)
    padding = (0, (-w)%4, 0, (-h)%4)
    condition = torch.full((b*4, 1, h, w), SIGMA/255, dtype=source8.dtype, device=source8.device)
    output = model(F.pad(windows, padding, mode='reflect'),
                   F.pad(condition, padding, mode='reflect'))[..., :h, :w]
    if clamp: output = output.clamp(0,1)
    return output.reshape(b,4,3,h,w)
