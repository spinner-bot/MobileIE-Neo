"""All preregistered frame0 candidates and matched-field smoothing effects."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    folder=Path(p.parse_args().report)
    index=ROOT/'research/4.temporal-consistency/results/exploration-r10-eval/preview_index.json'
    for row in json.loads(index.read_text()):
        path=ROOT/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('preview cache changed')
        with np.load(path,allow_pickle=False) as archive:images={k:archive[k] for k in archive.files}
        fig,axes=plt.subplots(3,5,figsize=(12.8,5.5),constrained_layout=True)
        for ax in axes.flat:ax.set_axis_off()
        for ax,(name,array) in zip(axes.flat,images.items()):
            ax.imshow(np.clip(array/255.,0,1));ax.set_title(name,fontsize=8)
        fig.suptitle(row['scene']+' / '+row['preparation']+' / fixed frame0 / raw outputs',fontsize=11)
        fig.savefig(folder/(row['scene']+'_'+row['preparation']+'.png'),dpi=160)
        plt.close(fig)
    deltas=json.loads((folder/'matched_smoothing_deltas.json').read_text())
    variants=['pretrained','plain','spatial_copy','head_lr1e3','full_lr1e4','full_lr1e3']
    labels=['Official','Plain\n1e-5','Foldable\ncopy','Fast\nConv5','Full\n1e-4','Full\n1e-3']
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,key,title in zip(axes,['psnr_rgb','ssim_rgb','temporal_error'],['PSNR change, dB (higher)','SSIM change (higher)','Temporal error change (lower)']):
        values=[next(r for r in deltas if r['variant']==v and r['fusion']=='raw')['summary'][key]['mean'] for v in variants]
        ax.bar(range(6),values,color='#518aab');ax.axhline(0,color='black',lw=.8)
        ax.set(xticks=range(6),xticklabels=labels,title=title);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Smooth input minus native input: same crop, GT and pixel grid\nFour project-dev clips, all candidates; smoothing changes noise AND detail',fontsize=11)
    fig.savefig(folder/'matched_smoothing.png',dpi=180);fig.savefig(folder/'matched_smoothing.pdf');plt.close(fig)


if __name__=='__main__':main()
