"""Fixed uniformly spaced frame-index input contacts, original brightness and no model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/4.temporal-consistency'


def main():
    folder=BASE/'results/exploration-r49-full-phone-attempt001'
    runtime=json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['source_frames_decoded']==952
    index=json.loads((folder/'index.json').read_text())
    out=BASE/'report/exploration-r49-full-phone';out.mkdir(exist_ok=False)
    sources=[]
    for row in index['scenes']:
        with np.load(ROOT/row['path'],allow_pickle=False) as archive:
            low=archive['low'];times=archive['decoded_timestamp_ms']
        assert low.dtype==np.uint8 and len(low)==row['source_frames_decoded']
        indices=np.rint(np.linspace(0,len(low)-1,12)).astype(int)
        fig,axes=plt.subplots(3,4,figsize=(12,6.5))
        for ax,frame in zip(axes.flat,indices):
            ax.imshow(low[frame]);ax.axis('off')
            ax.set_title(f'frame {frame} / decoder {times[frame]/1000:.2f}s',fontsize=8)
        fig.suptitle(row['phone']+' / original brightness / full phone DEV / no GT',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.95])
        path=out/(row['scene']+'.png');fig.savefig(path,dpi=150);plt.close(fig)
        sources.append({'scene':row['scene'],'path':str(path.relative_to(ROOT)),
                        'cache_path':row['path'],'cache_sha256':row['sha256'],
                        'indices':indices.tolist(),'decoder_timestamps_ms':[float(times[t]) for t in indices],
                        'brightness_transform':None,'model_calls':0})
    (out/'contact_sources.json').write_text(json.dumps(sources,indent=2)+'\n')


if __name__=='__main__':main()
