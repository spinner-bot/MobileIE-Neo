"""Final-only registered strong-post gates, retaining all failed conditions."""
from collections import defaultdict
from datetime import datetime
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest
from train_dense_history_r64 import MODELS, write


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    out = BASE/'report/exploration-r64-dense-history'
    assert not out.exists()
    rows, sources = [], []
    for seed in [0, 1]:
        train = BASE/f'results/exploration-r64-stage001-seed{seed}'
        run = BASE/f'results/exploration-r64-stage001-eval-seed{seed}'
        training = json.loads((train/'runtime.json').read_text())
        runtime = json.loads((run/'runtime.json').read_text())
        config = json.loads((run/'config.json').read_text())
        assert training['completed_normally'] and training['model_updates'] == {n: 4000 for n in MODELS}
        assert runtime['completed_normally'] and runtime['rows'] == 70 and runtime['neural_sequences'] == 40
        assert runtime['center_outputs'] == 1792 and runtime['per_frame_rows'] == 3136
        assert not runtime['cuda_initialized']
        part = lines(run/'per_sequence.jsonl')
        assert len(part) == 70 and len(lines(run/'per_frame.jsonl')) == 3136
        rows.extend([{**r, 'mapping': r['model']+':'+r['method']} for r in part])
        sources += [train/'config.json', train/'runtime.json', run/'config.json', run/'runtime.json',
                    run/'per_sequence.jsonl', run/'per_frame.jsonl', run/'prediction_manifest.jsonl',
                    run/'motion_acceptance.jsonl', run/'first_frame.jsonl']
        for old, model, mapping in [('r44', 'square_mse', 'R44_square'),
                                    ('r48', 'plain', 'R48_plain'), ('r45', None, 'R45_rgb')]:
            reference = BASE/f'results/exploration-{old}-stage001-eval-seed{seed}'
            old_config = json.loads((reference/'config.json').read_text())
            assert old_config['cache_indices'] == config['cache_indices']
            assert json.loads((reference/'runtime.json').read_text())['completed_normally']
            previous = lines(reference/'per_sequence.jsonl')
            selected = [r for r in previous if (r['model'] == model if model else r['method'] == 'rgb')]
            assert len(selected) == (30 if model else 10)
            rows.extend([{**r, 'mapping': mapping+(':'+r['method'] if model else '')} for r in selected])
            sources += [reference/'config.json', reference/'runtime.json', reference/'per_sequence.jsonl']
    assert len(rows) == 280
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain'], row['train_seed'], row['mapping']].append(row)
    metrics = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']
    means = {}
    for key, part in groups.items():
        assert len(part) == (6 if key[0] == 'davis_train_pilot' else 4)
        assert len({r['scene'] for r in part}) == len(part)
        values = {m: float(np.mean([r[m] for r in part])) for m in metrics}
        assert all(np.isfinite(v) for v in values.values())
        means[key] = values
    assert len(means) == 56
    effects, history = [], []
    domains = ['davis_train_pilot', 'paired_real_ND_DEV']
    for domain in domains:
        for seed in [0, 1]:
            candidate = means[domain, seed, 'causal:raw']
            current = means[domain, seed, 'causal:current_raw']
            history.append({'domain': domain, 'seed': seed,
                'psnr_delta': candidate['psnr_rgb']-current['psnr_rgb'],
                'temporal_ratio': candidate['temporal_error']/current['temporal_error'],
                'passed': candidate['psnr_rgb'] >= current['psnr_rgb']-.05
                          and candidate['temporal_error'] <= current['temporal_error']*.95})
            cand_rows = {r['scene']: r for r in groups[domain, seed, 'causal:raw']}
            for reference in ['plain:recursive_self05', 'spatial:recursive_self05', 'R44_square:recursive_self05']:
                ref = means[domain, seed, reference]
                ref_rows = {r['scene']: r for r in groups[domain, seed, reference]}
                assert set(ref_rows) == set(cand_rows)
                quality = candidate['psnr_rgb']-ref['psnr_rgb']
                time_ratio = candidate['temporal_error']/ref['temporal_error']
                worst = min(cand_rows[k]['psnr_rgb']-ref_rows[k]['psnr_rgb'] for k in cand_rows)
                effects.append({'domain': domain, 'seed': seed, 'reference': reference,
                    'psnr_delta': quality, 'temporal_ratio': time_ratio, 'worst_condition_psnr_delta': worst,
                    'quality_passed': quality >= -.05, 'worst_condition_passed': worst >= -.5,
                    'temporal_A_passed': time_ratio <= .90, 'temporal_B_preserved': time_ratio <= 1.02})
    history_passed = all(r['passed'] for r in history)
    common = history_passed and all(r['quality_passed'] and r['worst_condition_passed'] for r in effects)
    gate_A = common and all(r['temporal_A_passed'] for r in effects)
    gate_B = common and all(r['temporal_B_preserved'] for r in effects)
    out.mkdir()
    write(out/'all_conditions.json', rows)
    write(out/'domain_seed_summary.json', [dict(domain=d, seed=s, mapping=m, **v) for (d, s, m), v in means.items()])
    write(out/'effects.json', effects)
    write(out/'history_effects.json', history)
    write(out/'audit.json', {'at': datetime.now().astimezone().isoformat(), 'processing_complete': True,
        'new_quality_rows': 140, 'new_per_frame_rows': 6272, 'new_CNN_sequences': 80, 'new_CNN_centers': 3584,
        'new_formal_updates': 24000, 'history_gate_passed': history_passed,
        'gate_A_quality_qualified': gate_A, 'gate_B_quality_qualified': gate_B,
        'cost_stage_allowed': gate_A or gate_B, 'cost_stage_measured': False,
        'local_family_frozen': not (gate_A or gate_B), 'novelty_established': False,
        'phone_measured': False, 'independent_final': False, 'R63_stress_gate_passed': False})
    write(out/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
    table = ['# R64 完整dense骨干共同状态：最终双seed全表', '',
        '原8000起点，三组各新增4000，两seed共24000；所有组相同clipped raw MSE/资料/Adam。',
        '140新质量行/80CNN序列/3584中心/6272逐帧；SYN为3源各两个退化，ND四源，均重复使用DEV。',
        'R44/R48/R45完整旧参考条件不同；旧R63人工压力资格仍失败，实际任务资格不能推广全输入。', '',
        '|域|seed|映射|PSNR|SSIM|Etemp↓|edge↓|', '|---|---:|---|---:|---:|---:|---:|']
    for (d, s, m), value in means.items():
        table.append(f"|{d}|{s}|{m}|{value['psnr_rgb']:.6f}|{value['ssim_rgb']:.6f}|{value['temporal_error']:.6f}|{value['edge_mae']:.6f}|")
    table += ['', '|域|seed|强post参考|causal raw ΔPSNR|Etemp比|最差条件ΔPSNR|',
              '|---|---:|---|---:|---:|---:|']
    for r in effects:
        table.append(f"|{r['domain']}|{r['seed']}|{r['reference']}|{r['psnr_delta']:+.6f}|{r['temporal_ratio']:.6f}|{r['worst_condition_psnr_delta']:+.6f}|")
    table += ['', f'同权重历史贡献门槛={history_passed}；A质量资格={gate_A}；B质量资格={gate_B}。',
        'B只容许保时间的资源取舍，仍须实际完整CPU≥20%节省；A仍须实际成本≤参考1.05。',
        '这些门槛只针对causal raw，causal recursive不替代无外部flow问题；任何资格不代表创新/手机/最终验证。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, domain in enumerate(domains):
        for seed in [0, 1]:
            ax = axes[seed, col]
            for model, color in [('R44_square', 'gray'), ('plain', 'tab:blue'),
                                 ('spatial', 'tab:orange'), ('causal', 'tab:red')]:
                values = [means[domain, seed, model+':'+method] for method in ['raw', 'recursive_self05']]
                ax.plot([v['temporal_error'] for v in values], [v['psnr_rgb'] for v in values],
                        'o-', color=color, label=model+' raw/recursive')
            for mapping, marker, color in [('causal:current_raw', 'x', 'tab:red'), ('R45_rgb', 'D', 'tab:green')]:
                v = means[domain, seed, mapping]
                ax.scatter(v['temporal_error'], v['psnr_rgb'], marker=marker, color=color, label=mapping)
            ax.set_title(('SYN 3 sources / 6 conditions' if col == 0 else 'ND 4 sources')+f' / seed{seed}')
            ax.set_xlabel('GT-change temporal error (lower)')
            ax.set_ylabel('PSNR (dB)')
            ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Original dense backbone, matched joint-history controls / reused DEV / final12000')
    fig.tight_layout(rect=[0, .1, 1, .95])
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'comparison.{extension}', dpi=150)
    plt.close(fig)
    print((out/'audit.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
