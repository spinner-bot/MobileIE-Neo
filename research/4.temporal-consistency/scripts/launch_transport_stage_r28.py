"""Continue a single admitted GPU stage from its remaining gate into fixed-seed training."""
import argparse
import json
from pathlib import Path
import runpy
import sys

from exploration_r2_data import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=[0, 1], required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    base = ROOT/'research/4.temporal-consistency'
    fields = base/'process/exploration-r27/training-fields-cuda/audit.json'
    assert json.loads(fields.read_text())['passed']
    gate_dir = base/'process/exploration-r27/fixtures-cuda'
    if not (gate_dir/'audit.json').exists():
        # A partly created directory signals an earlier failed attempt, never overwrite it.
        assert not gate_dir.exists()
        sys.argv = ['test_transport_fst_r27.py', '--device', 'cuda', '--out', str(gate_dir)]
        runpy.run_path(str(Path(__file__).with_name('test_transport_fst_r27.py')), run_name='__main__')
    assert json.loads((gate_dir/'audit.json').read_text())['passed']
    # runpy retains the admitted CUDA process/context. No new GPU is acquired.
    # The training entry point independently verifies both gate source hashes.
    sys.argv = ['train_transport_fst_r28.py', '--seed', str(args.seed), '--out', args.out]
    runpy.run_path(str(Path(__file__).with_name('train_transport_fst_r28.py')), run_name='__main__')


if __name__ == '__main__':
    main()
