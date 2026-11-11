"""Keep mask restriction and matched-acceptance flow choice as distinct contrasts."""
import collections
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from matched_correspondence_r25 import REGIMES
from summarize_causal_r5 import jsonl,means,paired

BASE=ROOT/'research/4.temporal-consistency'
VARIANTS=(*MODES,'R14_postwarp05')


def main():
    folder=BASE/'results/exploration-r25';out=BASE/'report/exploration-r25';out.mkdir(parents=True,exist_ok=False)
    rows=jsonl(folder/'per_sequence.jsonl');frames=jsonl(folder/'per_frame.jsonl');runtime=json.loads((folder/'runtime.json').read_text())
    assert len(rows)==runtime['rows']==160 and len(frames)==2560 and not runtime['cuda_initialized']
    assert len({(r['scene'],r['variant'],r['train_seed'],r['regime']) for r in rows})==160
    assert sum(r['metric_alias_of'] is None for r in rows)==runtime['metric_computations']
    replays=json.loads((folder/'baseline_replays.json').read_text());assert len(replays)==80
    assert max(r['sequence_and_frame_max_abs_metric'] for r in replays)<=1e-10
    first=json.loads((folder/'first_frame_audit.json').read_text());assert len(first)==160 and max(r['max_abs_pixel'] for r in first)<=1e-4
    correspondence=json.loads((folder/'correspondence.json').read_text());assert len(correspondence)==60
    assert all(r['all_common_accepted_pixels_equal'] for r in correspondence)
    groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:groups[(r['variant'],r['regime'])].append(r);clips[(r['scene'],r['variant'],r['regime'])].append(r)
    summary=[{'variant':v,'regime':h,'rows':len(rr),**means(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in ['mse_normalized','global_RGB_bias_mse','centered_error_mse']}}
        for (v,h),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':s,'variant':v,'regime':h,**means(rr)} for (s,v,h),rr in sorted(clips.items())])
    rr=[{**r,'method':r['variant']+'__'+r['regime']} for r in rows];deltas=[]
    for variant in VARIANTS:
        for kind,a,b in [('mask_restriction','teacher_common','teacher_default'),('flow_choice','reference_common','teacher_common'),
                ('combined_GT_assisted_change','reference_common','teacher_default'),('actual_history','reference_common','current')]:
            deltas.append({**paired(rr,variant+'__'+a,variant+'__'+b,'dev_native'),'contrast':kind})
    save_json(out/'paired_deltas.json',deltas)
    keys=['teacher_valid_fraction','reference_valid_fraction','common_valid_fraction','flow_difference_mean_common_pixels',
        'flow_difference_p95_common_pixels','aligned_input_MAE255_common']
    scene_motion=[{'scene':scene,'pairs':15,**{k:float(np.mean([r[k] for r in correspondence if r['scene']==scene and r[k] is not None])) for k in keys}}
        for scene in sorted({r['scene'] for r in correspondence})]
    save_json(out/'correspondence_summary.json',scene_motion)
    save_json(out/'audit.json',{'rows':160,'new_diagnostic_configurations':80,'frames':2560,'baseline_replayed_sequences':80,'baseline_replayed_frames':1280,
        'baseline_max_abs_metric':max(r['sequence_and_frame_max_abs_metric'] for r in replays),'same_accepted_pixel_sets':60,
        'metric_computations':runtime['metric_computations'],'metric_aliases':runtime['metric_aliases'],
        'first_frame_max_abs_pixel':max(r['max_abs_pixel'] for r in first),'MSE_closure_max_abs':runtime['MSE_closure_max_abs'],
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir()},
        'scope':'Both shared mask and reference flow use GT-derived information; not motion truth/EPE or deployable performance. Temporal evaluator uses reference flow too.'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,key,title in [(axes[0],'psnr_rgb','PSNR change (higher better)'),(axes[1],'temporal_error','Temporal error change (lower better)')]:
        for j,(contrast,label) in enumerate([('mask_restriction','Restrict teacher mask'),('flow_choice','Change flow, same accepted pixels')]):
            values=[next(r['summary'][key]['mean'] for r in deltas if r['contrast']==contrast and r['first_minus_second'][0].startswith(v+'__')) for v in VARIANTS]
            ax.bar(np.arange(5)+(j-.5)*.32,values,width=.3,label=label)
        ax.set_xticks(range(5),['fixed pre','learn0','learn.5','fixed post','R14 post .5']);ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_title(title)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('GT-estimated correspondence sensitivity, four native clips / two seeds\nGT-derived masks/flow are diagnostic only; reference flow is not true motion.',fontsize=11)
    for suffix in ['png','pdf']:fig.savefig(out/f'matched_correspondence.{suffix}',dpi=170)
    plt.close(fig)
    lines=['# Matched-acceptance flow sensitivity','','Common masks and reference flow are GT-assisted diagnostics, not deployment candidates.','',
        '| Variant | Regime | PSNR | SSIM | Temporal | Edge |','|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['variant']}|{r['regime']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
