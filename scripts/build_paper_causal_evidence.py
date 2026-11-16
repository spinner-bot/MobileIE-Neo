"""Add completed R74/R75/R76 evidence; no model, pixel, CUDA or metric work."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/4.temporal-consistency'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    primary_path = ROOT / 'paper/evidence-snapshot.json'
    primary = json.loads(primary_path.read_text())
    assert primary['R73_complete_state_included']
    assert all(sha(ROOT / name) == value for name, value in primary['source_sha256'].items())
    sources = {}

    def read(name):
        path = BASE / name
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    motion = read('report/exploration-r74-joint/audit.json')
    motion_gates = read('report/exploration-r74-joint/gates.json')
    review = read('report/exploration-r74-contacts/actual-review001.json')
    assert motion['completed'] and motion['perceptual_included']
    assert (motion['formal_motion_updates'], motion['new_quality_rows']) == (16000, 80)
    assert len(motion_gates) == 16 and not any(g['passed'] for g in motion_gates)
    assert review['actual_reviewed'] and review['reviewed_contacts'] == 20
    for stage in ('r75', 'r76'):
        assert read(f'process/exploration-{stage}/finalize001/runtime.json')['completed_normally']
        review = read(f'report/exploration-{stage}-contacts/actual-review001.json')
        assert review['actual_review_completed'] and review['contacts_reviewed'] == 20
        assert len(review['observations']) == 20 and all(x['actual_reviewed'] for x in review['observations'])
        assert read(f'report/exploration-{stage}-contacts/runtime.json')['completed_normally']
        for seed in (0, 1):
            runtime = read(f'results/exploration-{stage}-eval-seed{seed}/runtime.json')
            assert runtime['completed_normally'] and runtime['training_updates'] == 0
            counts = (20, 896, 4864) if stage == 'r75' else (10, 448, 9728)
            assert (runtime['quality_rows'], runtime['frame_rows'], runtime['tile_CNN_calls']) == counts
        metric = read(f'results/exploration-{stage}-perceptual001/runtime.json')
        assert metric['completed_normally']
        assert (metric['quality_rows'], metric['comparison_frames']) == ((40, 1792) if stage == 'r75' else (20, 896))
        source = read(f'results/exploration-{stage}-perceptual001/metric-source.json')
        assert source['package_version'] == '0.5.7' and source['playback_fps'] == 30
    for seed in (0, 1):
        marker = read(f'results/exploration-r75-stage001-seed{seed}/final.json')
        runtime = read(f'results/exploration-r75-stage001-seed{seed}/attempt001/runtime.json')
        assert marker['completed_normally'] and marker['final_quality_eligible']
        assert marker['completed_steps'] == marker['formal_updates'] == 64000
        assert marker['parameters'] == 890275
        assert runtime['completed_normally'] and runtime['new_steps'] == 64000
        assert runtime['inherited_steps'] == runtime['uncommitted_video_backwards'] == 0
        assert runtime['latest_checkpoint']['sha256'] == marker['sha256']
    a = read('report/exploration-r75-joint/audit.json')
    b = read('report/exploration-r76-joint/audit.json')
    assert a['completed'] and a['full_gates_evaluated'] and a['formal_updates'] == 128000
    assert (a['all_quality_rows'], a['domain_seed_means']) == (80, 16)
    assert b['completed'] and b['scores_complete'] and not b['primary_results_rewritten']
    assert (b['all_conditions'], b['domain_seed_means'], b['SGD']) == (100, 20, 0)
    original = read('report/exploration-r75-joint/all_conditions.json')
    combined = read('report/exploration-r76-joint/all_conditions.json')
    key = lambda x: (x['domain'], x['seed'], x['scene'], x['label'])
    combined_index = {key(x): x for x in combined}
    assert len(original) == 80 and len(combined_index) == 100
    assert all(combined_index[key(x)] == x for x in original)
    gates = read('report/exploration-r75-joint/gates.json')
    assert gates == read('report/exploration-r76-joint/original_R75_gates.json')
    support = read('report/exploration-r76-joint/attribution_support.json')
    assert len(gates) == 8 and not any(g['passed'] for g in gates)
    assert len(support) == 4 and sum(g['support_passed'] for g in support) == 1
    assert len(read('report/exploration-r76-joint/first_frame_effects.json')) == 20
    stop = read('process/exploration-r77/investment-decision001.json')
    assert not stop['prerequisites_satisfied'] and stop['formal_training_updates'] == stop['new_GPU_jobs'] == 0
    means = read('report/exploration-r76-joint/domain_seed_summary.json')
    index = {(x['domain'], x['seed'], x['label']): x for x in means}
    assert len(index) == 20
    methods = [('MobileIE_raw', 'MobileIE raw'), ('MobileIE_recursive_self05', 'MobileIE + rec.'),
               ('ReBotNet_causal', 'XS causal'), ('ReBotNet_current', 'XS current'), ('ReBotNet_self4', 'XS self4')]
    domains = [('davis_train_pilot', 'SYN'), ('paired_real_ND_DEV', 'ND')]
    lines = [r'\begin{table}[!htbp]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}',
        r'\caption{Completed causal-reference and fixed self-conditioning controls (R75/R76), both seeds and all domain means. MobileIE has 4,075 parameters and 8K/128 training; released ReBotNet-XS has 890,275 parameters and 64K/384 training from scratch. Causal uses the previous prediction; current substitutes the current dark input in both slots; self4 resets each real frame and calls the same weights four times. These are unequal-budget adaptations, not a matched independent spatial training experiment or new methods. SYN: six repeated conditions, $480\times854$/64 frames; ND: four repeated native crops, $288\times512$/16 frames. JOD retains nominal 30\,fps and the fixed display of Table~\ref{tab:perceptual}.}',
        r'\label{tab:causal-reference}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'Output & Seed & SYN PSNR & SYN $E_{\rm temp}$ & SYN JOD & ND PSNR & ND $E_{\rm temp}$ & ND JOD\\', r'\midrule']
    for label, title in methods:
        for seed in (0, 1):
            values = [index[domain, seed, label] for domain, _ in domains]
            cells = [f'{x[field]:.3f}' for x in values for field in ('psnr_rgb', 'temporal_error', 'JOD')]
            lines.append(f'{title} & {seed} & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tables = {'causal-reference-table.tex': '\n'.join(lines) + '\n'}
    lines = [r'\begin{table}[!htbp]', r'\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Every registered causal-versus-reference joint screen and fixed self4 support check. Differences/ratios are causal relative to the indicated reference. Quality requires $\Delta$PSNR$\geq-0.05$\,dB and $\Delta$JOD$\geq-0.02$; temporal ratios must be at most 0.90 for rec. and 0.95 for current/self4. Only one supplementary cell passes; none of the eight primary screens passes. Supplementary results do not replace the primary screens.}',
        r'\label{tab:causal-screens}', r'\begin{tabular}{lllr rrl}', r'\toprule',
        r'Domain & Seed & Reference & $\Delta$PSNR & $E_{\rm temp}$ ratio & $\Delta$JOD & Pass\\', r'\midrule']
    for domain, title in domains:
        for seed in (0, 1):
            for group in (gates, support):
                for g in group:
                    if (g['domain'], g['seed']) != (domain, seed):
                        continue
                    ref = {'MobileIE_recursive_self05': 'MobileIE + rec.', 'ReBotNet_current': 'XS current', 'ReBotNet_self4': 'XS self4'}[g['reference']]
                    passed = g['passed'] if group is gates else g['support_passed']
                    lines.append(f'{title} & {seed} & {ref} & {g["psnr_delta"]:+.3f} & {g["temporal_ratio"]:.6f} & {g["JOD_delta"]:+.3f} & {"yes" if passed else "no"}' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tables['causal-screens-table.tex'] = '\n'.join(lines) + '\n'
    if args.check_only:
        print(json.dumps(dict(ready=True, complete_sources=len(sources), quality_rows=10, screen_rows=12, outputs_written=False)))
        return
    out = ROOT / 'paper/generated'
    prior = {str(p.relative_to(ROOT)): sha(p) for p in out.glob('*.tex') if p.name not in tables}
    for name, text in tables.items():
        (out / name).write_text(text)
    record = dict(created_at=datetime.now().astimezone().isoformat(), source_sha256=sources,
        generated_tables=list(tables), generated_sha256={str((out / n).relative_to(ROOT)): sha(out / n) for n in tables},
        prior_tables_sha256=prior, R74_complete=True, R75_complete=True, R76_complete=True,
        R75_formal_updates=128000, R75_primary_gates=8, R75_passed=0, R76_support=4, R76_passed=1,
        R77_additional_training_stopped=True, quality_table_rows=10, screen_table_rows=12,
        actual_R75_contacts_reviewed=20, actual_R76_contacts_reviewed=20,
        new_method=False, handset_measured=False, independent_final=False,
        scope='Completed fixed-budget published architecture and interpretation controls; repeated DEV, unequal training, no original-paper reproduction or new GPU work.')
    child = ROOT / 'paper/causal-evidence-snapshot.json'
    child.write_text(json.dumps(record, indent=2) + '\n')
    primary['source_sha256'].update(sources)
    primary['source_sha256'][str(child.relative_to(ROOT))] = sha(child)
    primary['generated_tables'] = list(dict.fromkeys(primary['generated_tables'] + list(tables)))
    primary.update(created_at=record['created_at'], R74_complete_motion_included=True,
        R75_complete_causal_reference_included=True, R76_complete_self4_included=True,
        supplemental_causal_snapshot='paper/causal-evidence-snapshot.json')
    primary_path.write_text(json.dumps(primary, indent=2) + '\n')
    print(json.dumps(dict(tables=2, quality_rows=10, screen_rows=12, complete_sources=len(sources), new_CNN=0, new_SGD=0)))


if __name__ == '__main__':
    main()
