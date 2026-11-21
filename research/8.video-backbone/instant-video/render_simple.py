"""Four complete-result I1-S charts and fixed full-FOV/native-detail contacts."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw

from simple_io import BASE, ROOT, digest, read, stamp, verify_training, write
from summarize_simple import ARMS

SOURCE = BASE/'results/simple-eval001'
OUT = BASE/'report/simple-analysis001'
AUDIT = BASE.parent/'evidence-reconciliation/process/delay-source-audit001.json'
GROUPS = [(0,'SYN'),(0,'ND'),(1,'SYN'),(1,'ND')]
CONTRASTS = [(c,r) for c in ('all','output')
             for r in ('F0','F0_FB','F0_DIS',c+'_FB',c+'_DIS')] + [('all','output')]
GROUP_COLORS = ['#2474b5','#c45312','#268543','#8a419d']


def lines(path):
    import json
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(fig, name):
    for ext in ('png','pdf'):
        fig.savefig(OUT/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)


def load(entry):
    path = ROOT/entry['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == entry['sha256']
    value = np.load(path,mmap_mode='r',allow_pickle=False)
    assert value.dtype == np.float32
    return value


def main():
    verify_training()
    assert read(SOURCE/'runtime.json')['completed_normally']
    inputs = [SOURCE/n for n in ('runtime.json','means.json','gates.json','per_sequence.jsonl',
              'per_frame.jsonl','prediction_manifest.jsonl','raw-ranges.jsonl')] + [AUDIT]
    means,gates = read(inputs[1]),read(inputs[2])
    rows,frames,manifest = [lines(p) for p in inputs[3:6]]
    assert (len(rows),len(frames),len(means),len(gates['comparisons']),len(manifest)) == (180,8064,36,44,120)
    assert all(r['future_frames'] == 0 and r['first_source_frame'] == 0 for r in manifest)
    OUT.mkdir(parents=True,exist_ok=False)
    index = {(r['seed'],r['scene'],r['arm']):r for r in rows}
    paired,summary = [],[]
    for candidate,reference in CONTRASTS:
        for seed,domain in GROUPS:
            part = []
            for c in rows:
                if (c['seed'],c['domain'],c['arm']) != (seed,domain,candidate):
                    continue
                r = index[seed,c['scene'],reference]
                part.append(dict(seed=seed,domain=domain,scene=c['scene'],candidate=candidate,reference=reference,
                     PSNR_delta=c['psnr_rgb']-r['psnr_rgb'],SSIM_delta=c['ssim_rgb']-r['ssim_rgb'],
                     edge_delta=c['edge_mae']-r['edge_mae'],Etemp_ratio=c['temporal_error']/r['temporal_error']))
            paired.extend(part)
            c = next(r for r in means if (r['seed'],r['domain'],r['arm']) == (seed,domain,candidate))
            r = next(r for r in means if (r['seed'],r['domain'],r['arm']) == (seed,domain,reference))
            summary.append(dict(seed=seed,domain=domain,candidate=candidate,reference=reference,conditions=len(part),
                mean_PSNR_delta=c['psnr_rgb']-r['psnr_rgb'],worst_PSNR_delta=min(v['PSNR_delta'] for v in part),
                ratio_of_mean_Etemp=c['temporal_error']/r['temporal_error'],
                joint_positive_conditions=sum(v['PSNR_delta']>0 and v['Etemp_ratio']<1 for v in part)))
    assert len(paired) == 220 and len(summary) == 44
    with (OUT/'paired-sequences.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(paired[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(paired)
    write(OUT/'paired-summary.json',dict(groups=summary,scope='Repeated DEV, three SYN sources and four ND; no confidence intervals or independent-final claim'))

    colors = {arm:('#555555' if arm.startswith('F0') else '#2474b5' if arm.startswith('all') else '#c45312') for arm in ARMS}
    markers = {arm:('s' if arm.endswith('_FB') else '^' if arm.endswith('_DIS') else 'o' if arm=='F0' else '*' if arm=='all' else 'D') for arm in ARMS}
    fig,axes = plt.subplots(2,2,figsize=(12,9))
    handles = []
    for ax,(seed,domain) in zip(axes.flat,GROUPS):
        for arm in ARMS:
            r = next(r for r in means if (r['seed'],r['domain'],r['arm']) == (seed,domain,arm))
            ax.scatter(r['psnr_rgb'],r['temporal_error'],edgecolors=colors[arm],facecolors='none',marker=markers[arm],s=85)
            if ax is axes.flat[0]:
                handles.append(Line2D([],[],color=colors[arm],marker=markers[arm],markerfacecolor='none',linestyle='none',label=arm))
        ax.set(title=f'{domain} / seed{seed}',xlabel='Mean PSNR (dB; higher better)',ylabel='Temporal residual MAE (lower better)')
        ax.grid(alpha=.2)
    fig.legend(handles=handles,ncol=3,loc='lower center',fontsize=9,bbox_to_anchor=(.5,.005))
    fig.suptitle('I1-S: all36 full-length means; causal / zero future; source-faithful published baseline')
    fig.tight_layout(rect=(0,.14,1,.96));save(fig,'quality-temporal')

    fig,axes = plt.subplots(3,4,figsize=(17,12))
    for ax,(candidate,reference) in zip(axes.flat,CONTRASTS):
        for (seed,domain),color in zip(GROUPS,GROUP_COLORS):
            part = [r for r in paired if (r['candidate'],r['reference'],r['seed'],r['domain']) == (candidate,reference,seed,domain)]
            ax.scatter([r['PSNR_delta'] for r in part],[r['Etemp_ratio'] for r in part],color=color,s=24,alpha=.8,label=f'{domain} seed{seed}')
        ax.axvline(0,color='gray',ls='--',lw=1);ax.axhline(1,color='gray',ls='--',lw=1)
        ax.set(title=f'{candidate} / {reference} (20 pairs)',xlabel='Paired PSNR difference (dB)',ylabel='Paired Etemp ratio')
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(handles=[Line2D([],[],color=c,marker='o',linestyle='none',label=f'{d} seed{s}') for (s,d),c in zip(GROUPS,GROUP_COLORS)],loc='center')
    fig.suptitle('All220 condition-seed pairs in all11 registered contrasts; no favorable subset')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'paired-levels')

    fig,ax = plt.subplots(figsize=(13,11));matrix = np.zeros((11,4))
    for y,(candidate,reference) in enumerate(CONTRASTS):
        for x,(seed,domain) in enumerate(GROUPS):
            g = next(g for g in gates['comparisons'] if (g['candidate'],g['reference'],g['seed'],g['domain']) == (candidate,reference,seed,domain))
            matrix[y,x] = g['passed']
            ax.text(x,y,f"{'PASS' if g['passed'] else 'FAIL'} {g['PSNR_delta']:+.3f} dB\nworst {g['worst_sequence_PSNR_delta']:+.3f}; E {g['Etemp_ratio']:.3f}",ha='center',va='center',fontsize=8)
    ax.imshow(matrix,cmap=ListedColormap(['#f1d6d1','#d4e8d5']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(4),[f'{d} / seed{s}' for s,d in GROUPS])
    ax.set_yticks(range(11),[f'{c} / {r}' for c,r in CONTRASTS])
    ax.set_title(f"All44 fixed gates; separate cost/perceptual eligibility={gates['eligible_separate_cost_and_perceptual_registration']}")
    fig.tight_layout();save(fig,'gates')

    audit = read(AUDIT);columns = ['gt','low',*ARMS];width = 176
    sheet = Image.new('RGB',(width*len(columns),1320),'white');draw = ImageDraw.Draw(sheet)
    draw.text((3,2),'Fixed bear_photons20 t32 / video15 t8, two model seeds. Full FOV + native center128. Static only; not flicker, MOS, phone or final.',fill='black')
    for col,label in enumerate(columns):
        draw.text((width*col+3,16),label,fill='black')
    contacts = []
    for row,(seed,scene,t) in enumerate([(0,'bear_photons20',32),(0,'video_15',8),(1,'bear_photons20',32),(1,'video_15',8)]):
        meta = next(r for r in audit['scenes'] if r['scene']==scene)
        source = ROOT/meta['path']
        assert source.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(source)==meta['sha256']
        with np.load(source,allow_pickle=False) as bundle:
            images = {'gt':bundle['gt'][t],'low':bundle['low'][t]}
        item = next(r for r in audit['baseline_items'] if (r['seed'],r['scene'])==(seed,scene))
        for arm,key in [('F0','F0'),('F0_FB','FB'),('F0_DIS','DIS')]:
            images[arm] = load(item[key])[t]
        selected = [r for r in manifest if (r['seed'],r['scene'])==(seed,scene)]
        assert len(selected)==6
        for entry in selected:
            images[entry['arm']] = load(entry)[t]
        contacts.append(dict(seed=seed,scene=scene,source_frame=t,source=meta,new_predictions=selected))
        y = 36+320*row
        draw.text((3,y),f'seed{seed} / {scene} / source t{t}: full FOV above, native center128 below',fill='black')
        for col,arm in enumerate(columns):
            value = images[arm];h,w = value.shape[:2]
            thumbnail = Image.fromarray(np.rint(np.clip(value,0,255)).astype(np.uint8))
            thumbnail.thumbnail((width,126),Image.Resampling.LANCZOS)
            sheet.paste(thumbnail,(width*col,y+20))
            patch = value[h//2-64:h//2+64,w//2-64:w//2+64]
            assert patch.shape == (128,128,3)
            sheet.paste(Image.fromarray(np.rint(np.clip(patch,0,255)).astype(np.uint8)),(width*col+24,y+160))
    sheet.save(OUT/'contact.png');sheet.save(OUT/'contact.pdf',resolution=140)
    write(OUT/'contact-sources.json',dict(selection='Registered first source/domain, full source indices32/8, both seeds, center128 unscaled',sources=contacts))
    ranges = lines(inputs[6]);assert len(ranges)==20
    write(OUT/'raw-range-summary.json',{level:dict(conditions=20,minimum_raw=min(r['values'][level]['min'] for r in ranges),maximum_raw=max(r['values'][level]['max'] for r in ranges),mean_sequence_outside_fraction=float(np.mean([r['values'][level]['outside_fraction'] for r in ranges]))) for level in ('all','output')})
    write(OUT/'render-record.json',dict(at=stamp(),renderer_sha256=digest(Path(__file__)),
          input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
          counts=dict(sequence_rows=180,frame_rows=8064,means=36,gates=44,paired_sequences=220,paired_groups=44,PSNR_ceiling_applied=sum(r['PSNR_ceiling_applied'] for r in frames)),
          actual_viewed=False,new_model_or_flow_inference=False,future_frames=0,
          scope='Static complete-result review only; not MOS, unique mechanism, actual phone latency or independent final'))
    print('I1-S complete plots and registered contacts exported',flush=True)


if __name__ == '__main__':
    main()
