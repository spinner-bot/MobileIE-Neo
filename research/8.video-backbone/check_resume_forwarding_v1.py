"""Regress the observed resource-continuation failure using original metadata."""
import copy
import json
import os
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
from run_v1 import ROOT, PROCESS, read, resource_resume, write, stamp


def rejected(attempts, previous):
    try:
        resource_resume(attempts, previous)
    except AssertionError:
        return True
    return False


def main():
    first = PROCESS / 'train-seed0-attempt001'
    second = PROCESS / 'train-seed0-attempt002'
    saved, refused = read(first / 'runtime.json'), read(second / 'runtime.json')
    assert saved['counts']['updates']['warm'] == refused['counts']['updates']['warm'] == 241
    original = ROOT / saved['latest_checkpoint']['path']
    direct, direct_record = resource_resume([first], saved)
    forwarded, forwarded_record = resource_resume([first, second], refused)
    assert direct == forwarded == original
    assert not direct_record['forwarded_without_new_update']
    assert forwarded_record['forwarded_without_new_update']
    advanced = copy.deepcopy(refused)
    advanced['counts']['updates']['warm'] += 1
    assert rejected([first, second], advanced), 'Must not silently lose an additional update'
    wrong_seed = copy.deepcopy(refused)
    wrong_seed['seed'] = 1
    assert rejected([first, second], wrong_seed), 'Must not transfer another seed checkpoint'
    assert 'torch' not in sys.modules
    result = dict(at=stamp(), passed=True, original_saved_step=241,
                  direct_checkpoint_retained=True, observed_zero_update_refusal_forwarded=True,
                  uncheckpointed_extra_update_rejected=True, different_seed_rejected=True,
                  checkpoint=str(original.relative_to(ROOT)),
                  checkpoint_sha256=saved['latest_checkpoint']['sha256'],
                  optimizer_updates=0, model_forwards=0, new_video_pixels=0, CUDA_imported=False,
                  scope='Metadata/actual file identity regression only; original runtime and weights untouched.')
    path = PROCESS / 'resource-forwarding-check001.json'
    assert not path.exists()
    write(path, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
