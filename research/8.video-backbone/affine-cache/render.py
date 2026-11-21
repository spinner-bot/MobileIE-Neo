"""Complete A1 stored-score plots and fixed contacts; no CNN or flow inference."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw
from a1_io import BASE, ROOT, digest, lines, read, stamp, verify, write as atomic_write

OUT = BASE / 'report/analysis001'
SOURCE = BASE / 'results/screen001'
GROUPS = [(0,'SYN'),(0,'ND'),(1,'SYN'),(1,'ND')]
ARMS = ['F0','F0_FB','F0_DIS','current','current_FB','current_DIS',
        'cached','cached_FB','cached_DIS','hold','hold_FB','hold_DIS']
PAIRS = [('current','F0'),('cached','F0'),('cached','current'),('cached','hold'),
         ('cached_FB','F0_FB'),('cached_FB','current_FB'),('cached_DIS','F0_DIS'),('cached_DIS','current_DIS')]


def write(name, value):
    atomic_write(OUT/name, value)


def save(fig, name):
    for ext in ('png','pdf'):
        fig.savefig(OUT/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    verify()
    runtime = read(SOURCE/'runtime.json')
    assert runtime['completed_normally']
    OUT.mkdir(parents=True,exist_ok=False)
    inputs = [SOURCE/(n+'.json') for n in ('means','gates','runtime')]
    inputs += [SOURCE/(n+'.jsonl') for n in ('per_sequence','per_frame','prediction_manifest','raw-ranges')]
    means,gates = read(inputs[0]),read(inputs[1])
    rows,frames,manifests,ranges = [lines(p) for p in inputs[3:]]
    assert (len(means),len(gates['comparisons']),len(rows),len(frames),len(manifests)) == (48,20,768,3072,64)
    index={(r['key'],r['arm']):r for r in rows}
    paired,summaries=[],[]
    for candidate,ref in PAIRS:
        for seed,domain in GROUPS:
            p=[]
            for c in rows:
                if (c['arm'],c['seed'],c['domain']) != (candidate,seed,domain):
                    continue
                r=index[c['key'],ref]
                p.append(dict(key=c['key'],seed=seed,domain=domain,candidate=candidate,reference=ref,
                    PSNR_delta=c['psnr_rgb']-r['psnr_rgb'],SSIM_delta=c['ssim_rgb']-r['ssim_rgb'],
                    edge_delta=c['edge_mae']-r['edge_mae'],Etemp_ratio=c['temporal_error']/r['temporal_error']))
            assert len(p)==16
            paired.extend(p)
            c=next(m for m in means if (m['arm'],m['seed'],m['domain'])==(candidate,seed,domain))
            r=next(m for m in means if (m['arm'],m['seed'],m['domain'])==(ref,seed,domain))
            summaries.append(dict(seed=seed,domain=domain,candidate=candidate,reference=ref,windows=16,
                mean_PSNR_delta=c['psnr_rgb']-r['psnr_rgb'],worst_PSNR_delta=min(v['PSNR_delta'] for v in p),
                ratio_of_mean_Etemp=c['temporal_error']/r['temporal_error'],
                PSNR_positive_windows=sum(v['PSNR_delta']>0 for v in p),
                Etemp_positive_windows=sum(v['Etemp_ratio']<1 for v in p),
                joint_positive_windows=sum(v['PSNR_delta']>0 and v['Etemp_ratio']<1 for v in p)))
    assert len(paired)==512 and len(summaries)==32
    with (OUT/'paired-windows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(paired)
    write('paired-summary.json',dict(scope='Descriptive training-scene draws, not independent-scene confidence intervals',groups=summaries))

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    handles=[]
    colors={'F0':'#555555','current':'#2474b5','cached':'#c45312','hold':'#268543'}
    for ax,(seed,domain) in zip(axes.flat,GROUPS):
        for arm in ARMS:
            r=next(m for m in means if (m['arm'],m['seed'],m['domain'])==(arm,seed,domain))
            base=arm.split('_')[0]
            marker='s' if arm.endswith('FB') else '^' if arm.endswith('DIS') else 'o'
            ax.scatter(r['psnr_rgb'],r['temporal_error'],edgecolors=colors[base],facecolors='none',marker=marker,s=65)
            if ax is axes.flat[0]:
                handles.append(Line2D([],[],color=colors[base],marker=marker,markerfacecolor='none',linestyle='none',label=arm))
        ax.set(title=f'{domain} / seed{seed} (16 windows)',xlabel='Mean PSNR (dB; higher better)',ylabel='Temporal residual MAE (lower better)')
        ax.grid(alpha=.2)
    fig.legend(handles=handles,ncol=4,loc='lower center',fontsize=9,bbox_to_anchor=(.5,.01))
    fig.suptitle('A1: all 48 means; cached F0 targets, zero new CNN calls; no deployment timing')
    fig.tight_layout(rect=(0,.16,1,.96));save(fig,'quality-temporal')

    shown=[('cached','current'),('cached','hold'),('cached_FB','F0_FB'),('cached_DIS','F0_DIS')]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(candidate,ref) in zip(axes.flat,shown):
        for (seed,domain),color in zip(GROUPS,['#2474b5','#c45312','#268543','#8a419d']):
            p=[v for v in paired if (v['candidate'],v['reference'],v['seed'],v['domain'])==(candidate,ref,seed,domain)]
            ax.scatter([v['PSNR_delta'] for v in p],[v['Etemp_ratio'] for v in p],s=25,alpha=.7,color=color,label=f'{domain} seed{seed}')
        ax.axvline(0,color='gray',ls='--',lw=1);ax.axhline(1,color='gray',ls='--',lw=1)
        ax.set(title=f'{candidate} / {ref} (64 pairs)',xlabel='Paired PSNR difference (dB)',ylabel='Paired Etemp ratio (<1 improves)')
        ax.grid(alpha=.2)
    axes.flat[-1].legend(fontsize=8)
    fig.suptitle('All paired windows; equality lines are not mean-level gates; hold can trade detail for stability')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'paired-reuse')

    keys=[(g['category'],g['candidate'],g['reference']) for g in gates['comparisons'][:5]]
    fig,ax=plt.subplots(figsize=(13,6))
    matrix=np.zeros((5,4))
    for y,key in enumerate(keys):
        for x,(seed,domain) in enumerate(GROUPS):
            g=next(g for g in gates['comparisons'] if (g['category'],g['candidate'],g['reference'])==key and (g['seed'],g['domain'])==(seed,domain))
            matrix[y,x]=g['passed']
            ax.text(x,y,f"{'PASS' if g['passed'] else 'FAIL'}  {g['PSNR_delta']:+.3f} dB\nworst {g['worst_window_PSNR_delta']:+.3f}; E {g['Etemp_ratio']:.3f}",ha='center',va='center',fontsize=9)
    ax.imshow(matrix,cmap=ListedColormap(['#f1d6d1','#d4e8d5']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(4),[f'{d} / seed{s}' for s,d in GROUPS])
    ax.set_yticks(range(5),[f'{k}: {c} / {r}' for k,c,r in keys])
    ax.set_title(f"All 20 preregistered comparisons; complete={gates['complete']}; green pass / red fail")
    fig.tight_layout();save(fig,'gates')

    columns=['gt','low',*ARMS]
    sheet=Image.new('RGB',(128*len(columns),676),'white');draw=ImageDraw.Draw(sheet)
    draw.text((3,4),'Fixed first SYN/ND windows, frame3; same inputs repeated for two model seeds. Static RGB is not flicker/MOS/phone evidence.',fill='black')
    contacts=[]
    for row,(seed,window) in enumerate([(0,0),(0,2),(1,0),(1,2)]):
        key=f'seed{seed}-step20000-window{window}'
        source=next(m for m in manifests if m['key']==key)
        path=ROOT/source['path']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path)==source['sha256']
        contacts.append(source);y=36+160*row
        draw.text((3,y),f"{key} / {source['domain']}",fill='black')
        with np.load(path,allow_pickle=False) as z:
            for col,arm in enumerate(columns):
                a=z[arm];assert a.shape==(4,128,128,3) and np.isfinite(a).all()
                x=128*col;draw.text((x+3,y+12),arm,fill='black')
                sheet.paste(Image.fromarray(np.rint(np.clip(a[3],0,255)).astype(np.uint8)),(x,y+28))
    sheet.save(OUT/'contact.png');sheet.save(OUT/'contact.pdf',resolution=140)
    write('contact-sources.json',dict(selection='Registered step20000, windows0/2, frame3 per model seed; no ranking',sources=contacts))
    write('raw-range-summary.json',{arm:dict(windows=64,minimum_raw=min(r['values'][arm]['min'] for r in ranges),maximum_raw=max(r['values'][arm]['max'] for r in ranges),mean_outside_fraction=float(np.mean([r['values'][arm]['outside_fraction'] for r in ranges]))) for arm in ('current','cached','hold')})
    write('render-record.json',dict(at=stamp(),author='Codex  (sqs_A100)',renderer_sha256=digest(Path(__file__)),
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        counts=dict(output_rows=768,frame_rows=3072,window_seed_rows=64,unique_input_draws=32,means=48,gates=20,paired_windows=512,paired_means=32,PSNR_ceiling_applied=sum(r['PSNR_ceiling_applied'] for r in frames)),
        actual_viewed=False,new_model_or_flow_inference=False,scope='Complete stored training-scene scores; no independent final/MOS/phone'))
    print('A1 complete plots and fixed contacts exported')


if __name__=='__main__':
    main()
