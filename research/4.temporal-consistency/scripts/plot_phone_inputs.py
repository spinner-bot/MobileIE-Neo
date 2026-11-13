"""Show first/middle/last registered phone frames without brightness alteration."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from explore_temporal_controls import ROOT, sha256, save_json


def main():
    base = ROOT/'research/4.temporal-consistency'
    source = base/'results/exploration-r31-data/index.json'
    rows = json.loads(source.read_text())['scenes']
    out = base/'report/exploration-r31-data'
    out.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(3, 3, figsize=(12, 7.3), constrained_layout=True)
    sources = []
    for axis_row, row in zip(axes, rows):
        path = ROOT/row['path']
        assert sha256(path) == row['sha256']
        with np.load(path, allow_pickle=False) as data:
            values = data['low']
        for column, index in enumerate([0, 32, 63]):
            axis_row[column].imshow(values[index])
            axis_row[column].set_title(f"{row['phone']} | frame {index} | {row['decoded_timestamp_ms'][index]/1000:.3f} s", fontsize=9)
            axis_row[column].axis('off')
        sources.append({'scene': row['scene'], 'cache_sha256': row['sha256'], 'frames': [0, 32, 63]})
    figure.suptitle('Fixed real-phone development inputs: original brightness, no paired reference', fontsize=12)
    figure.savefig(out/'phone_inputs.png', dpi=150)
    plt.close(figure)
    save_json(out/'figure_sources.json', {'index_sha256': sha256(source), 'sources': sources,
               'scope': 'No enhancement, gamma correction or per-panel contrast adjustment; fixed frame0/32/63.'})


if __name__ == '__main__':
    main()
