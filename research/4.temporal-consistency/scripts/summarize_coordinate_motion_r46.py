"""Accept only full normal runs; report all domain/seed and matched-mask tests."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coordinate_state_r46 import BASE, ROOT


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    out = BASE/'report/exploration-r46-coordinate-state'
    assert out.exists() and not (out/'real_motion_audit.json').exists()
    inputs, rows, old, stats = [], [], [], []
    for seed in [0, 1]:
        run = BASE/f'results/exploration-r46-stage002-eval-seed{seed}'
        runtime = json.loads((run/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['rows'] == 30
        assert runtime['per_frame_rows'] == 1344 and runtime['neural_sequences'] == 0
        assert not runtime['cuda_initialized'] and runtime['max_reference_metric_difference'] <= 1e-5
        source = run/'per_sequence.jsonl'
        part = load_lines(source)
        assert len(part) == 30
        rows.extend(part)
        old_source = BASE/f'results/exploration-r45-stage001-eval-seed{seed}/per_sequence.jsonl'
        reference = load_lines(old_source)
        assert len(reference) == 90
        old.extend(reference)
        stats.extend(load_lines(run/'fusion_stats.jsonl'))
        inputs.extend([run/'config.json', run/'runtime.json', source,
                       run/'reference_replay.jsonl', run/'fusion_stats.jsonl', old_source])
    assert len(stats) == 876
    groups = defaultdict(list)
    for row in rows + old:
        groups[row['domain'], row['train_seed'], row['method']].append(row)
    metrics = ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error']
    means = {key: {k: float(np.mean([row[k] for row in value])) for k in metrics}
             for key, value in groups.items()}
    effects, frontiers = [], []
    domains = ['davis_train_pilot', 'paired_real_ND_DEV']
    for domain in domains:
        for seed in [0, 1]:
            candidate = means[domain, seed, 'anchor8']
            for reference in ['ordinary', 'same_mask']:
                baseline = means[domain, seed, reference]
                cand_rows = {r['scene']: r for r in groups[domain, seed, 'anchor8']}
                base_rows = {r['scene']: r for r in groups[domain, seed, reference]}
                worst = min(cand_rows[k]['psnr_rgb']-base_rows[k]['psnr_rgb'] for k in cand_rows)
                delta = candidate['psnr_rgb']-baseline['psnr_rgb']
                ratio = candidate['temporal_error']/baseline['temporal_error']
                effects.append(dict(domain=domain, seed=seed, baseline=reference,
                    psnr_delta=delta, temporal_ratio=ratio, worst_condition_psnr_delta=worst,
                    quality_passed=delta >= -.05, temporal_passed=ratio <= .90,
                    condition_passed=worst >= -.5))
            dominators = []
            for fixed in ['fixed_01', 'fixed_025', 'fixed_05', 'fixed_075', 'fixed_09']:
                ref = means[domain, seed, fixed]
                if (ref['psnr_rgb'] >= candidate['psnr_rgb']
                        and ref['temporal_error'] <= candidate['temporal_error']
                        and (ref['psnr_rgb'] > candidate['psnr_rgb']
                             or ref['temporal_error'] < candidate['temporal_error'])):
                    dominators.append(fixed)
            frontiers.append(dict(domain=domain, seed=seed, dominators=dominators,
                                  sampled_frontier_nondominated=not dominators))
    gate = (all(r[k] for r in effects for k in
                ['quality_passed', 'temporal_passed', 'condition_passed'])
            and all(r['sampled_frontier_nondominated'] for r in frontiers))
    dump(out/'real_motion_effects.json', effects)
    dump(out/'real_motion_frontier.json', frontiers)
    mean_rows = [dict(domain=d, seed=s, method=m, **value) for (d,s,m), value in means.items()]
    dump(out/'real_motion_domain_seed.json', mean_rows)
    fractions = []
    for domain in domains:
        for seed in [0, 1]:
            part = [r for r in stats if r['domain'] == domain and r['train_seed'] == seed]
            denominator = sum(r['pixels'] for r in part)
            fractions.append(dict(domain=domain, seed=seed,
                ordinary_fraction=sum(r['ordinary_accepted'] for r in part)/denominator,
                anchor_fraction=sum(r['anchor_accepted'] for r in part)/denominator))
    dump(out/'real_motion_acceptance.json', fractions)
    dump(out/'real_motion_audit.json', dict(at=datetime.now().astimezone().isoformat(),
        processing_complete=True, quality_gate_passed=gate, new_quality_rows=60,
        new_per_frame_rows=2688, new_CNN_sequences=0, GPU=False,
        local_K8_bilinear_family_frozen=not gate, cost_evaluation_allowed=gate,
        inference_history_ratio=1.75, novelty_established=False,
        final_test=False, phone_measured=False))
    methods = ['raw', 'fixed_025', 'ordinary', 'same_mask', 'anchor8', 'fixed_075', 'rgb']
    table = ['# R46实际运动完整结果', '',
        '两个seed均正常结束，60新质量行/2688逐帧行/0CNN；ordinary重算核对R45 fixed .5。', '',
        '|域|seed|方法|PSNR|SSIM|时间↓|edge↓|', '|---|---:|---|---:|---:|---:|---:|']
    for d in domains:
        for s in [0, 1]:
            for method in methods:
                v = means[d,s,method]
                table.append(f"|{d}|{s}|{method}|{v['psnr_rgb']:.6f}|{v['ssim_rgb']:.6f}|{v['temporal_error']:.6f}|{v['edge_mae']:.6f}|")
    table += ['', 'raw/fixed/RGB复用R45，其它三方法为本轮共同评价；全部相同R44最终空间底座。same_mask为额外拒绝匹配的诊断，不是免坐标成本的新方法。', '',
              '|域|seed|主对照|ΔPSNR|时间比|最差条件ΔPSNR|画质/时间/条件通过|',
              '|---|---:|---|---:|---:|---:|---|']
    for r in effects:
        table.append(f"|{r['domain']}|{r['seed']}|{r['baseline']}|{r['psnr_delta']:+.6f}|{r['temporal_ratio']:.6f}|{r['worst_condition_psnr_delta']:+.6f}|{r['quality_passed']}/{r['temporal_passed']}/{r['condition_passed']}|")
    table += ['', f'完整质量门槛：{gate}。候选持久历史比原ordinary多75%；无手机/零开销/新颖性或独立最终结论。实际motion不是GT，GT流仅用于既有评价。']
    (out/'real_motion_tables.md').write_text('\n'.join(table)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for col, domain in enumerate(domains):
        for seed in [0, 1]:
            ax = axes[seed, col]
            fixed = [means[domain,seed,m] for m in
                     ['fixed_01','fixed_025','fixed_05','fixed_075','fixed_09']]
            ax.plot([r['temporal_error'] for r in fixed], [r['psnr_rgb'] for r in fixed],
                    'o-', color='gray', label='Five fixed-alpha references')
            for method, marker, color, label in [('same_mask','s','tab:blue','Rejection-matched recursive'),
                    ('anchor8','*','tab:red','Coordinate state K=8'),
                    ('rgb','D','tab:green','R45 RGB gate')]:
                r = means[domain,seed,method]
                ax.scatter(r['temporal_error'], r['psnr_rgb'], marker=marker,
                           color=color, s=110 if method == 'anchor8' else 45, label=label)
            ax.set_title(('SYN' if col == 0 else 'Paired ND DEV')+f' / seed {seed}')
            ax.set_xlabel('GT-change temporal error (lower is better)')
            ax.set_ylabel('PSNR (dB)');ax.grid(alpha=.25)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Fixed alpha=0.5, native coordinate state: complete DEV screen')
    fig.tight_layout(rect=[0,.09,1,.95])
    fig.savefig(out/'real_motion_frontier.png', dpi=150)
    fig.savefig(out/'real_motion_frontier.pdf')
    plt.close(fig)
    dump(out/'real_motion_sources.json', {str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    print((out/'real_motion_audit.json').read_text())


if __name__ == '__main__':
    main()
