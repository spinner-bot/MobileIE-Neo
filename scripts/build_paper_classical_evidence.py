"""Extend the paper only with normal complete R70 quality/perception/cost evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/4.temporal-consistency'


def main():
    primary_path = ROOT / 'paper/evidence-snapshot.json'
    primary = json.loads(primary_path.read_text())
    assert primary['R69_complete_objective_included']
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
               for name, value in primary['source_sha256'].items())
    sources = {}

    def read(name):
        path = BASE / name
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    assert read('process/exploration-r70/finalize001/runtime.json')['completed_normally']
    quality = read('results/exploration-r70-classical-attempt001/runtime.json')
    assert quality['completed_normally'] and (quality['filter_sequences'], quality['filter_frames']) == (20, 896)
    assert (quality['quality_rows'], quality['frame_rows'], quality['CNN_sequences'], quality['CNN_centers']) == (80, 3584, 40, 1792)
    perception = read('results/exploration-r70-perceptual-attempt001/runtime.json')
    assert perception['completed_normally'] and (perception['quality_rows'], perception['comparison_frames']) == (80, 3584)
    metric = read('results/exploration-r70-perceptual-attempt001/metric-source.json')
    assert metric['package_version'] == '0.5.7' and metric['playback_fps'] == 30
    render = read('process/exploration-r70/render001/runtime.json')
    review = read('report/exploration-r70-classical-quality/actual-visual-review.json')
    assert render['completed_normally'] and render['contacts'] == review['contacts'] == 20 and review['actual_reviewed']
    audit = read('report/exploration-r70-classical-joint/audit.json')
    assert audit['completed'] and (audit['quality_rows'], audit['domain_seed_means'], audit['perceptual_quality_rows_joined']) == (300, 60, 280)
    assert not audit['old_acceptance_gates_rewritten']
    rows = read('report/exploration-r70-classical-joint/domain_seed_means.json')
    cost_audit = read('report/exploration-r70-classical-cost/audit.json'); assert cost_audit['completed']
    for name, sha in cost_audit['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
        sources[name] = sha
    costs = read('report/exploration-r70-classical-cost/stage_costs.json')
    index = {(r['domain'], r['seed'], r['label']): r for r in rows}
    domains = ['davis_train_pilot', 'paired_real_ND_DEV']
    lines = [r'\begin{table}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}',
        r'\caption{Fixed original-core block-matching prefilters followed by the same frozen MobileIE weights. S-only disables temporal search in the VBM3D implementation; it is not an optimized BM3D reproduction. V-default uses the original video search on a full offline clip. A single blind first-frame noise estimate is shared by both modes. Rec. adds the same own-output motion/history reference. These reused development results and fixed-display JOD predictions do not establish calibrated noise, human judgments or a new method.}',
        r'\label{tab:classical-block}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'Prefilter/output & Seed & SYN PSNR & SYN $E_{\rm temp}$ & SYN JOD & ND PSNR & ND $E_{\rm temp}$ & ND JOD\\', r'\midrule']
    count = 0
    for family, label in [('MobileIE', 'None'), ('MobileIE_spatial_only', 'S-only'), ('MobileIE_video_default', 'V-default')]:
        for method, suffix in [('raw', 'raw'), ('recursive_self05', '+ rec.')]:
            for seed in [0, 1]:
                values = [index[domain, seed, family + '_' + method] for domain in domains]
                assert all(r['parameters'] == 4075 and r['project_updates'] == 8000 for r in values)
                cells = [f'{r[field]:.3f}' for r in values for field in ['psnr_rgb', 'temporal_error', 'JOD']]
                lines.append(f'{label} {suffix} & {seed} & ' + ' & '.join(cells) + r'\\'); count += 1
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tables = {'classical-block-reference-table.tex': '\n'.join(lines) + '\n'}
    lines = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Descriptive CPU2 complete input-filter stages, including array I/O, in seconds per frame. Medians are over the registered sequence conditions; SYN has three source videos under two noise conditions. The same filtered input is reused across two enhancement seeds, so filtering is counted once. Shared-server timings have no isolated repeated latency trial and are not handset speed or an end-to-end future-wait bound.}',
        r'\label{tab:classical-cost}', r'\begin{tabular}{llrrr}',
        ]
    lines += [r'\toprule', r'Domain/size & Mode & Conditions & Median s/frame & Range\\', r'\midrule']
    cost_count = 0
    for domain, label in [('davis_train_pilot', r'SYN $480\times854$'), ('paired_real_ND_DEV', r'ND $288\times512$')]:
        for mode, name in [('spatial_only', 'S-only'), ('video_default', 'V-default')]:
            row, = [r for r in costs if r['domain'] == domain and r['mode'] == mode and r['seed'] is None]
            value = row['seconds_per_frame']['full_filter']
            lines.append(f"{label} & {name} & {row['sequence_conditions']} & {value['median']:.3f} & {value['minimum']:.3f}--{value['maximum']:.3f}" + r'\\')
            cost_count += 1
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tables['classical-filter-cost-table.tex'] = '\n'.join(lines) + '\n'
    assert (count, cost_count) == (12, 4)
    for name, text in tables.items():
        (ROOT / 'paper/generated' / name).write_text(text)
    record = dict(created_at=datetime.now().astimezone().isoformat(), source_sha256=sources,
                  generated_tables=list(tables), R70_complete=True, quality_table_rows=12,
                  cost_table_rows=4, new_method=False, handset_measured=False, independent_final=False)
    child = ROOT / 'paper/classical-evidence-snapshot.json'
    child.write_text(json.dumps(record, indent=2) + '\n')
    primary['source_sha256'].update(sources)
    primary['source_sha256'][str(child.relative_to(ROOT))] = hashlib.sha256(child.read_bytes()).hexdigest()
    primary['generated_tables'] = list(dict.fromkeys(primary['generated_tables'] + list(tables)))
    primary.update(created_at=record['created_at'], R70_complete_classical_included=True,
                   supplemental_classical_snapshot='paper/classical-evidence-snapshot.json')
    primary_path.write_text(json.dumps(primary, indent=2) + '\n')
    print(json.dumps(dict(quality_rows=count, cost_rows=cost_count, completed_sources=len(sources))))


if __name__ == '__main__':
    main()
