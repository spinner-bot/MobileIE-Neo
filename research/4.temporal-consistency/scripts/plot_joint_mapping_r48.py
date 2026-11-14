"""Post-hoc mapping diagnosis: all four ND sources, both seeds, same frame8."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/4.temporal-consistency'


def lines(path):return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    out=BASE/'report/exploration-r48-joint-recursion'
    audit=json.loads((out/'audit.json').read_text());assert audit['processing_complete']
    assert not (out/'raw_mapping_sources.json').exists()
    metas=[r for r in json.loads((BASE/'results/exploration-r10-cache/index.json').read_text())['scenes']
           if r['domain']=='native_crop']
    assert len(metas)==4
    sources=[]
    for seed in [0,1]:
        old={r['scene']:r for r in lines(BASE/f'results/exploration-r44-stage001-eval-seed{seed}/prediction_manifest.jsonl')
             if r['model']=='square_mse'}
        new={(r['scene'],r['model']):r for r in lines(BASE/f'results/exploration-r48-stage001-eval-seed{seed}/prediction_manifest.jsonl')}
        fig,axes=plt.subplots(4,4,figsize=(13.5,9))
        for row,meta in enumerate(metas):
            scene=meta['scene']
            with np.load(ROOT/meta['path'],allow_pickle=False) as archive:gt=archive['gt'][8]
            records=[old[scene],new[scene,'plain'],new[scene,'joint']]
            values=[gt]+[np.load(ROOT/r['path'],mmap_mode='r',allow_pickle=False)[8] for r in records]
            for col,value in enumerate(values):
                ax=axes[row,col];ax.imshow(np.clip(value/255,0,1));ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_ylabel(scene)
                if row==0:ax.set_title(['GT','R44 raw / 8000','plain raw / 12000','joint raw / 12000'][col])
            sources.append({'scene':scene,'seed':seed,'frame':8,'GT_cache':meta['path'],
                            'raw_records':records,'scope':'Unfused spatial mapping only; no new CNN or metric.'})
        fig.suptitle(f'Paired ND DEV / seed {seed} / fixed frame8 raw before fusion / not temporal assessment')
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/f'raw_mapping_seed{seed}.png',dpi=150);plt.close(fig)
    (out/'raw_mapping_sources.json').write_text(json.dumps(sources,indent=2)+'\n')


if __name__=='__main__':main()
