"""Owned R86 artifact paths and fixed scientific source checks."""
import json
import numpy as np

from causal_reference_io_r75 import ROOT, BASE, write, append, stamp, digest

PROCESS = BASE/'process/exploration-r86'
PROTOCOL = BASE/'docs/exploration-r86-fixed-dis-reference-protocol.md'
QUALITY = BASE/'results/exploration-r86-eval001'
COST = BASE/'results/exploration-r86-cost001'
METRIC = BASE/'results/exploration-r86-perceptual001'
LABEL = 'DIS_recursive_self05'


def verify():
    value = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert value['source_sha256'] == {n: digest(ROOT/n) for n in value['source_sha256']}
    assert value['protocol_sha256'] == digest(PROTOCOL)
    return value


def cached(entry):
    path = ROOT/entry['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == entry['sha256']
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    assert list(value.shape) == entry['shape']
    return value


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]
