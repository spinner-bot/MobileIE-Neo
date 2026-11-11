"""Audit a frozen readout-node intervention and all paired quality/error effects."""
import collections
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import KEYS,jsonl,means,paired

BASE=ROOT/'research/4.temporal-consistency'
ERRORS=['mse_normalized','global_RGB_bias_mse','centered_error_mse']


def main():
    folder=BASE/'results/exploration-r24';out=BASE/'report/exploration-r24';out.mkdir(parents=True,exist_ok=False)
    rows=jsonl(folder/'per_sequence.jsonl');frames=jsonl(folder/'per_frame.jsonl');runtime=json.loads((folder/'runtime.json').read_text())
    assert len(rows)==96 and len(frames)==1536 and runtime['body_pixel_equal_checks']==640 and not runtime['cuda_initialized']
    assert len({(r['scene'],r['variant'],r['train_seed'],r['path']) for r in rows})==96
    bodies=json.loads((folder/'body_invariance.json').read_text());assert len(bodies)==640
    assert all(r['pixel_equal'] and r['original_sha256']==r['intervened_sha256'] for r in bodies)
    assert len({(r['scene'],r['mode'],r['seed'],r['frame'],r['branch']) for r in bodies})==640
    replays=json.loads((folder/'baseline_replays.json').read_text());assert len(replays)==64
    assert max(r['sequence_and_frame_max_abs_metric'] for r in replays)<=1e-10
    first=json.loads((folder/'first_frame_audit.json').read_text());assert len(first)==96 and max(r['max_abs_pixel'] for r in first)<=1e-4
    gates=json.loads((folder/'gates.json').read_text());assert len(gates)==512
    assert all(len(r['current_global_gate'])==12 and all(0<=v<=1 for v in r['current_global_gate']) for r in gates)
    refs=[];sources={str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir()}
    for seed in [0,1]:
        path=BASE/'results'/f'exploration-r21-reference-post-seed{seed}-cpu'/'per_sequence.jsonl';sources[str(path.relative_to(ROOT))]=sha256(path)
        for r in jsonl(path):
            if r['domain']=='dev_native' and not r['input_reset'] and r['method'] in ['raw','postwarp01','postwarp05']:
                refs.append({**r,'variant':'R20_'+r['method'],'path':'reference'})
    assert len(refs)==24;combined=rows+refs;groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in combined:
        groups[(r['variant'],r['path'])].append(r);clips[(r['scene'],r['variant'],r['path'])].append(r)
    summary=[{'variant':v,'path':p,'rows':len(rr),**means(rr),
        **({k:float(np.mean([r[k] for r in rr])) for k in ERRORS} if p!='reference' else {})}
        for (v,p),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':s,'variant':v,'path':p,**means(rr)} for (s,v,p),rr in sorted(clips.items())])
    combined=[{**r,'method':r['variant']+'__'+r['path']} for r in combined]
    contrasts=[]
    for mode in MODES:
        for kind,name in [('intervention','causal'),('current_limit','current')]:
            contrasts.append({**paired(combined,mode+'__current_gate',mode+'__'+name,'dev_native'),'contrast':kind})
        contrasts.append({**paired(combined,mode+'__current_gate','R20_postwarp05__reference','dev_native'),'contrast':'strong_post'})
    save_json(out/'paired_deltas.json',contrasts)
    lookup={(r['scene'],r['variant'],r['train_seed'],r['path']):r for r in rows};errors=[]
    for r in rows:
        if r['path']!='current_gate':continue
        for path in ['causal','current']:
            ref=lookup[(r['scene'],r['variant'],r['train_seed'],path)]
            errors.append({'scene':r['scene'],'variant':r['variant'],'train_seed':r['train_seed'],'reference':path,
                **{k:r[k]-ref[k] for k in [*KEYS,*ERRORS]}})
    save_json(out/'error_component_deltas.json',errors)
    save_json(out/'audit.json',{'rows':96,'new_intervention_rows':32,'replayed_sequence_rows':64,'replayed_frame_rows':1024,
        'baseline_sequence_and_frame_max_abs':max(r['sequence_and_frame_max_abs_metric'] for r in replays),
        'body_arrays_pixel_equal':640,'first_frame_max_abs_pixel':max(r['max_abs_pixel'] for r in first),
        'MSE_closure_max_abs':runtime['MSE_closure_max_abs'],'PSNR_identity_max_abs_dB':runtime['PSNR_identity_max_abs_dB'],
        'source_sha256':sources,'scope':'One frozen global-att node intervention; downstream spatial gates can change. No efficiency/novelty or unseen-test claim.'})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    specs=[('psnr_rgb',1,'PSNR delta (dB, higher better)'),('temporal_error',1,'Temporal-error delta (lower better)'),
        ('global_RGB_bias_mse',65025,'Global RGB bias MSE delta (8-bit²)'),('centered_error_mse',65025,'Centered error MSE delta (8-bit²)')]
    for ax,(key,scale,title) in zip(axes.flat,specs):
        for j,mode in enumerate(MODES):
            values=[]
            for scene in sorted({r['scene'] for r in rows}):values.append(float(np.mean([r[key] for r in errors if r['variant']==mode and r['reference']=='causal' and r['scene']==scene]))*scale)
            ax.bar(j,float(np.mean(values)),color=plt.get_cmap('tab10')(j));ax.scatter(np.full(4,j),values,color='black',s=17,zorder=4)
        ax.set_xticks(range(4),['fixed pre','learn0','learn.5','fixed post']);ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_title(title)
    fig.suptitle('Current global gate minus original causal output; frozen weights\nBars: four-scene/two-seed means; dots: all four scene means. Body arrays unchanged.',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(out/f'current_gate_intervention.{suffix}',dpi=170)
    plt.close(fig)
    lines=['# Frozen current-global-gate intervention','','All four modes and seeds; fixed step4000, noreset/raw, matched CPU references.','',
        '| Variant | Path | PSNR | SSIM | Temporal | Edge |','|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['variant']}|{r['path']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
