"""Complete F1 stored-score plots and preregistered native/detail contacts."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image,ImageDraw
from f1_io import BASE,ROOT,ARMS,AUDIT,D1,digest,owned,read,stamp,verify,write as atomic_write

SOURCE=BASE/'results/eval001'
OUT=BASE/'report/analysis001'
GROUPS=[(0,'SYN'),(0,'ND'),(1,'SYN'),(1,'ND')]
REFS=['naive_future','F0_DIS','F0_FB','D1_center3']


def lines(path):
    import json
    return [json.loads(s) for s in path.read_text().splitlines()]


def write(name,value):
    atomic_write(OUT/name,value)


def save(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(OUT/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)


def load(entry):
    path=owned(ROOT/entry['path']);assert digest(path)==entry['sha256']
    return np.load(path,mmap_mode='r',allow_pickle=False)


def main():
    verify()
    execution=read(BASE/'process/syntax-execution-freeze001.json')
    assert all(digest(ROOT/p)==h for p,h in execution['source_sha256'].items())
    assert read(SOURCE/'runtime.json')['completed_normally']
    OUT.mkdir(parents=True,exist_ok=False)
    inputs=[SOURCE/n for n in ('runtime.json','means.json','gates.json','per_sequence.jsonl','per_frame.jsonl','prediction_manifest.jsonl')]+[AUDIT,D1/'prediction_manifest.jsonl',SOURCE/'raw-ranges.jsonl']
    means,gates=read(inputs[1]),read(inputs[2])
    rows,frames,manifests=[lines(p) for p in inputs[3:6]]
    assert (len(rows),len(frames),len(means),len(gates['comparisons']),len(manifests))==(120,5016,24,16,40)
    index={(r['seed'],r['scene'],r['arm']):r for r in rows}
    paired,summaries=[],[]
    for ref in REFS:
        for seed,domain in GROUPS:
            selected=[r for r in rows if (r['seed'],r['domain'],r['arm'])==(seed,domain,'innovation_lag1')]
            part=[]
            for c in selected:
                r=index[seed,c['scene'],ref]
                part.append(dict(seed=seed,domain=domain,scene=c['scene'],reference=ref,
                    PSNR_delta=c['psnr_rgb']-r['psnr_rgb'],SSIM_delta=c['ssim_rgb']-r['ssim_rgb'],
                    edge_delta=c['edge_mae']-r['edge_mae'],Etemp_ratio=c['temporal_error']/r['temporal_error']))
            paired.extend(part)
            c=next(m for m in means if (m['seed'],m['domain'],m['arm'])==(seed,domain,'innovation_lag1'))
            r=next(m for m in means if (m['seed'],m['domain'],m['arm'])==(seed,domain,ref))
            summaries.append(dict(seed=seed,domain=domain,reference=ref,conditions=len(part),
                mean_PSNR_delta=c['psnr_rgb']-r['psnr_rgb'],worst_PSNR_delta=min(p['PSNR_delta'] for p in part),
                ratio_of_mean_Etemp=c['temporal_error']/r['temporal_error'],
                joint_positive_conditions=sum(p['PSNR_delta']>0 and p['Etemp_ratio']<1 for p in part)))
    assert len(paired)==80 and len(summaries)==16
    with (OUT/'paired-sequences.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n');w.writeheader();w.writerows(paired)
    write('paired-summary.json',dict(groups=summaries,scope='Repeated DEV, 3 SYN sources/4 ND, not independent confidence intervals'))

    fig,axes=plt.subplots(2,2,figsize=(12,8));handles=[]
    colors={'F0':'#555555','F0_FB':'#555555','F0_DIS':'#555555','naive_future':'#2474b5','innovation_lag1':'#c45312','D1_center3':'#268543'}
    markers={'F0':'o','F0_FB':'s','F0_DIS':'^','naive_future':'D','innovation_lag1':'*','D1_center3':'o'}
    for ax,(seed,domain) in zip(axes.flat,GROUPS):
        for arm in ARMS:
            r=next(m for m in means if (m['seed'],m['domain'],m['arm'])==(seed,domain,arm))
            ax.scatter(r['psnr_rgb'],r['temporal_error'],edgecolors=colors[arm],facecolors='none',marker=markers[arm],s=85)
            if ax is axes.flat[0]:handles.append(Line2D([],[],color=colors[arm],marker=markers[arm],markerfacecolor='none',linestyle='none',label=arm))
        ax.set(title=f'{domain} / seed{seed}',xlabel='Mean PSNR (dB; higher better)',ylabel='Temporal residual MAE (lower better)');ax.grid(alpha=.2)
    fig.legend(handles=handles,ncol=3,loc='lower center',fontsize=9,bbox_to_anchor=(.5,.01))
    fig.suptitle('F1: all 24 means; innovation_lag1 reads one future frame; common interior centers only')
    fig.tight_layout(rect=(0,.13,1,.96));save(fig,'quality-temporal')

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,ref in zip(axes.flat,REFS):
        for (seed,domain),color in zip(GROUPS,['#2474b5','#c45312','#268543','#8a419d']):
            p=[r for r in paired if (r['reference'],r['seed'],r['domain'])==(ref,seed,domain)]
            ax.scatter([r['PSNR_delta'] for r in p],[r['Etemp_ratio'] for r in p],color=color,s=35,alpha=.8,label=f'{domain} seed{seed}')
        ax.axvline(0,color='gray',ls='--',lw=1);ax.axhline(1,color='gray',ls='--',lw=1)
        ax.set(title=f'innovation_lag1 / {ref} (20 condition-seed pairs)',xlabel='Paired PSNR difference (dB)',ylabel='Paired Etemp ratio (<1 improves)');ax.grid(alpha=.2)
    axes.flat[-1].legend(fontsize=8)
    fig.suptitle('All pairs retained; more future information is not equal latency or measured equal compute')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'paired-innovation')

    fig,ax=plt.subplots(figsize=(12,5));matrix=np.zeros((4,4))
    for y,ref in enumerate(REFS):
        for x,(seed,domain) in enumerate(GROUPS):
            g=next(g for g in gates['comparisons'] if (g['reference'],g['seed'],g['domain'])==(ref,seed,domain))
            matrix[y,x]=g['passed']
            ax.text(x,y,f"{'PASS' if g['passed'] else 'FAIL'} {g['PSNR_delta']:+.3f} dB\nworst {g['worst_sequence_PSNR_delta']:+.3f}; E {g['Etemp_ratio']:.3f}",ha='center',va='center',fontsize=9)
    ax.imshow(matrix,cmap=ListedColormap(['#f1d6d1','#d4e8d5']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(4),[f'{d} / seed{s}' for s,d in GROUPS]);ax.set_yticks(range(4),[f'innovation_lag1 / {r}' for r in REFS])
    ax.set_title(f"All16 preregistered gates; complete={gates['complete']}; red fail / green pass")
    fig.tight_layout();save(fig,'gates')

    audit=read(AUDIT);columns=['gt','low',*ARMS]
    delayed={(r['seed'],r['scene']):r for r in lines(D1/'prediction_manifest.jsonl') if r['arm']=='center3'}
    sheet=Image.new('RGB',(224*8,1320),'white');draw=ImageDraw.Draw(sheet)
    draw.text((3,2),'Fixed bear_photons20 t32 / video15 t8; two model seeds. Full FOV thumbnail + native center128 detail. Static only, not flicker/MOS/phone.',fill='black')
    for col,label in enumerate(columns):draw.text((224*col+3,16),label,fill='black')
    contacts=[]
    for row,(seed,scene,t) in enumerate([(0,'bear_photons20',32),(0,'video_15',8),(1,'bear_photons20',32),(1,'video_15',8)]):
        item=next(v for v in audit['baseline_items'] if (v['seed'],v['scene'])==(seed,scene))
        meta=next(v for v in audit['scenes'] if v['scene']==scene)
        source=owned(ROOT/meta['path']);assert digest(source)==meta['sha256']
        with np.load(source,allow_pickle=False) as z:images={'gt':z['gt'][t],'low':z['low'][t]}
        for arm,name in [('F0','F0'),('F0_FB','FB'),('F0_DIS','DIS')]:images[arm]=load(item[name])[t]
        images['D1_center3']=load(delayed[seed,scene])[t-2]
        selected=[v for v in manifests if (v['seed'],v['scene'])==(seed,scene)]
        for v in selected:images[v['arm']]=load(v)[t-2]
        contacts.append(dict(seed=seed,scene=scene,source_frame=t,source=meta,D1_center3=delayed[seed,scene],new_predictions=selected))
        y=36+320*row;draw.text((3,y),f'seed{seed} / {scene} / t{t}: top full FOV; bottom center128 detail',fill='black')
        for col,arm in enumerate(columns):
            value=images[arm];h,w=value.shape[:2]
            img=Image.fromarray(np.rint(np.clip(value,0,255)).astype(np.uint8));img.thumbnail((224,126),Image.Resampling.LANCZOS)
            sheet.paste(img,(224*col,y+20))
            patch=value[h//2-64:h//2+64,w//2-64:w//2+64]
            assert patch.shape==(128,128,3)
            sheet.paste(Image.fromarray(np.rint(np.clip(patch,0,255)).astype(np.uint8)),(224*col+48,y+160))
    sheet.save(OUT/'contact.png');sheet.save(OUT/'contact.pdf',resolution=140)
    write('contact-sources.json',dict(selection='Registered first source per domain/model seed, fixed source-frame32/8 and center128 detail',sources=contacts))
    ranges=lines(SOURCE/'raw-ranges.jsonl')
    write('raw-range-summary.json',{a:dict(conditions=20,minimum_raw=min(r['values'][a]['min'] for r in ranges),maximum_raw=max(r['values'][a]['max'] for r in ranges),mean_sequence_outside_fraction=float(np.mean([r['values'][a]['outside_fraction'] for r in ranges]))) for a in ('naive_future','innovation_lag1')})
    write('render-record.json',dict(at=stamp(),renderer_sha256=digest(Path(__file__)),
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        counts=dict(sequence_rows=120,frame_rows=5016,means=24,gates=16,paired_sequences=80,paired_groups=16,
            PSNR_ceiling_applied=sum(r['PSNR_ceiling_applied'] for r in frames)),
        actual_viewed=False,new_model_or_flow_inference=False,candidate_future_frames=1,
        scope='Repeated DEV and common internal centers; static plots are not video MOS, phone or independent final evidence'))
    print('F1 complete plots and registered contacts exported')


if __name__=='__main__':
    main()
