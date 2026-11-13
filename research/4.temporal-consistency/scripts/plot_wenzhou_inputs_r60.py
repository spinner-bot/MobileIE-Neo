"""Show every fixed Night source at local frames4/64/123 without brightness edits."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT


def main():
    source = BASE/'results/exploration-r60-wenzhou-data-attempt001'
    assert json.loads((source/'runtime.json').read_text())['completed_normally']
    index = json.loads((source/'index.json').read_text()); assert len(index['scenes']) == 6
    out = BASE/'report/exploration-r60-wenzhou'; out.mkdir(exist_ok=True)
    target = out/'input-contacts.png'; assert not target.exists()
    fig, axes = plt.subplots(6, 3, figsize=(14, 16), constrained_layout=True)
    for row, meta in enumerate(index['scenes']):
        with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
            low = archive['low']
        for column, frame in enumerate([4, 64, 123]):
            ax = axes[row, column]; ax.imshow(low[frame], vmin=0, vmax=255); ax.axis('off')
            ax.set_title(f"{meta['scene']} | local{frame} / source{int(meta['source_frame_positions'][frame])}", fontsize=10)
    fig.suptitle('All six author backup Night DEV clips | fixed midpoint excerpts | original brightness, no CNN', fontsize=13)
    fig.savefig(target, dpi=130); plt.close(fig)


if __name__ == '__main__': main()
