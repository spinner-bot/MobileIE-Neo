"""Summarize signed MSE components without equating mean MSE and mean PSNR rankings."""
import collections
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import jsonl

BASE=ROOT/'research/4.temporal-consistency'
KEYS=['mse_normalized','global_RGB_bias_mse','common_channel_bias_mse','opponent_channel_bias_mse','centered_error_mse']


def main():
    folder=BASE/'results/exploration-r23';out=BASE/'report/exploration-r23';out.mkdir(parents=True,exist_ok=False)
    rows=jsonl(folder/'per_sequence.jsonl');frames=jsonl(folder/'per_frame.jsonl');deltas=jsonl(folder/'paired_deltas.jsonl')
    runtime=json.loads((folder/'runtime.json').read_text());assert len(rows)==44 and len(frames)==704 and len(deltas)==64
    assert runtime['formal_frame_PSNR_replay_max_abs_dB']<=1e-8 and runtime['MSE_closure_max_abs']<=1e-12 and not runtime['cuda_initialized']
    groups=collections.defaultdict(list)
    for r in rows:groups[r['panel']].append(r)
    summary=[{'panel':panel,'scenes':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in [*KEYS,
        'mean_frame_psnr_rgb','persistent_global_RGB_bias_mse','temporal_global_RGB_error_fluctuation_mse']}}
        for panel,rr in sorted(groups.items())]
    groups=collections.defaultdict(list)
    for r in deltas:groups[(r['contrast'],r['panel'])].append(r)
    contrasts=[]
    for (contrast,panel),rr in sorted(groups.items()):
        avg={k:float(np.mean([r[k] for r in rr])) for k in KEYS}
        contrasts.append({'contrast':contrast,'panel':panel,'scenes':len(rr),**avg,
            'ratio_of_mean_global_bias_excess_to_total_excess':avg['global_RGB_bias_mse']/avg['mse_normalized'] if abs(avg['mse_normalized'])>1e-15 else None})
    save_json(out/'domain_summary.json',summary);save_json(out/'contrast_summary.json',contrasts)
    save_json(out/'audit.json',{'sequences':44,'frames':704,'contrasts':64,'source_arrays':52,'seed':0,
        'formal_PSNR_replay_max_abs_dB':runtime['formal_frame_PSNR_replay_max_abs_dB'],'MSE_closure_max_abs':runtime['MSE_closure_max_abs'],
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir()},
        'scope':'All fixed seed0 native clips; mean MSE weights high-error scenes more than mean PSNR. Components are algebra, not causal shares or perceptual luminance/chroma.'})
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    pieces=[('common_channel_bias_mse','Global common-channel bias'),('opponent_channel_bias_mse','Global opponent-channel bias'),('centered_error_mse','Centered spatial error')]
    for ax,contrast,panels,title in [(axes[0],'actual_history',[m+' / causal' for m in MODES],'Actual history minus same-model current replacement'),
            (axes[1],'vs_single',[m+' / '+h for m in MODES for h in ['current','causal']],'Candidate minus R14 single-frame baseline')]:
        selected=[next(r for r in contrasts if r['contrast']==contrast and r['panel']==p) for p in panels];x=np.arange(len(panels))
        for j,(key,label) in enumerate(pieces):ax.bar(x+(j-1)*.22,[r[key]*65025 for r in selected],width=.21,label=label)
        ax.scatter(x,[r['mse_normalized']*65025 for r in selected],color='black',marker='D',s=30,label='Total MSE delta',zorder=4)
        ax.set_xticks(x,[p.replace(' / ','\n') for p in panels],fontsize=8);ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2)
        ax.set_title(title);ax.set_ylabel('Signed delta MSE (8-bit RGB units²)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Full-frame exact error decomposition; four native exploratory clips, fixed seed0\nNegative is lower MSE; this aggregate need not match mean-PSNR ranking. No enhancement or causal attribution.',fontsize=11)
    for suffix in ['png','pdf']:fig.savefig(out/f'rgb_error_components.{suffix}',dpi=170)
    plt.close(fig)
    lines=['# Fixed native RGB error decomposition','','All fixed seed0 clips; 8-bit MSE units squared. Mean frame PSNR and mean MSE may rank methods differently.','',
        '| Scene | Panel | Mean frame PSNR | MSE | Global bias | Common | Opponent | Centered |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['scene']}|{r['panel']}|{r['mean_frame_psnr_rgb']:.6f}|"+'|'.join(f'{r[k]*65025:.6f}' for k in KEYS)+'|')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
