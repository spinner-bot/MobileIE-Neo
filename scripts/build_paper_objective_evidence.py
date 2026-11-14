"""Add the complete R69 matched-objective reference to the manuscript snapshot."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/4.temporal-consistency'


def main():
    sources = {}

    def read(name):
        path = BASE / name
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    primary_path = ROOT / 'paper/evidence-snapshot.json'
    primary = json.loads(primary_path.read_text())
    assert primary['R65_complete_quality_included'] and primary['R66_complete_perceptual_included']
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
               for name, value in primary['source_sha256'].items())
    for seed in [0, 1]:
        runtime = read(f'results/exploration-r69-stage001-seed{seed}/runtime.json')
        assert runtime['completed_normally'] and runtime['updates'] == 8000
        final = read(f'results/exploration-r69-stage001-seed{seed}/step8000.json')
        assert final['final_quality_eligible'] and final['final_fold_gate_passed']
        assert final['step'] == 8000 and final['parameters'] == 4075
        evaluation = read(f'results/exploration-r69-stage001-eval-seed{seed}/runtime.json')
        assert evaluation['completed_normally']
        assert (evaluation['quality_rows'], evaluation['frame_rows'],
                evaluation['CNN_sequences'], evaluation['CNN_centers']) == (20, 896, 10, 448)
    assert read('process/exploration-r69/finalize001/runtime.json')['completed_normally']
    metric = read('results/exploration-r69-perceptual-attempt001/runtime.json')
    assert metric['completed_normally'] and (metric['quality_rows'], metric['comparison_frames']) == (40, 1792)
    read('results/exploration-r69-perceptual-attempt001/metric-source.json')
    review = read('report/exploration-r69-spatial-L1-quality/actual-visual-review.json')
    assert review['actual_reviewed'] and review['contacts'] == 20
    audit = read('report/exploration-r69-spatial-L1-joint/audit.json')
    assert audit['completed'] and audit['all_quality_rows'] == 260
    assert audit['domain_seed_means'] == 52 and audit['perceptual_quality_rows_joined'] == 240
    assert not audit['old_acceptance_gates_rewritten']
    rows = read('report/exploration-r69-spatial-L1-joint/domain_seed_means.json')
    index = {(r['domain'], r['seed'], r['label']): r for r in rows}
    lines = [r'\begin{table}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}',
             r'\caption{Matched original MobileIE objective settings, final 8,000 updates only. Both objectives use the same source-window schedule, expanded training parameterization and two-stage Adam budget; L1/MSE gradient scale interacts with clipping. Every row deploys 4,075 parameters. Rec. adds the same classical motion/history reference. SYN comprises three source videos under two noise conditions; ND comprises four reused development clips. JOD uses the fixed assumptions in Table~\ref{tab:perceptual}. No objective search, human study, handset or independent final-test claim.}',
             r'\label{tab:matched-objective}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Objective/output & Seed & SYN PSNR & SYN $E_{\rm temp}$ & SYN JOD & ND PSNR & ND $E_{\rm temp}$ & ND JOD\\',
             r'\midrule']
    count = 0
    for family, objective in [('MobileIE', 'MSE'), ('MobileIE_L1', 'L1')]:
        for method, output in [('raw', 'raw'), ('recursive_self05', '+ rec.')]:
            for seed in [0, 1]:
                values = [index[domain, seed, family + '_' + method]
                          for domain in ['davis_train_pilot', 'paired_real_ND_DEV']]
                assert all(r['parameters'] == 4075 and r['project_updates'] == 8000 for r in values)
                cells = [f'{r[field]:.3f}' for r in values
                         for field in ['psnr_rgb', 'temporal_error', 'JOD']]
                lines.append(f'{objective} {output} & {seed} & ' + ' & '.join(cells) + r'\\')
                count += 1
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    table = 'matched-objective-table.tex'
    (ROOT / 'paper/generated' / table).write_text('\n'.join(lines) + '\n')
    record = dict(created_at=datetime.now().astimezone().isoformat(), source_sha256=sources,
                  generated_tables=[table], R69_complete=True, table_rows=count,
                  new_method=False, handset_measured=False, independent_final=False)
    child = ROOT / 'paper/objective-evidence-snapshot.json'
    child.write_text(json.dumps(record, indent=2) + '\n')
    primary['source_sha256'].update(sources)
    primary['source_sha256'][str(child.relative_to(ROOT))] = hashlib.sha256(child.read_bytes()).hexdigest()
    primary['generated_tables'] = list(dict.fromkeys(primary['generated_tables'] + [table]))
    primary.update(created_at=record['created_at'], R69_complete_objective_included=True,
                   supplemental_objective_snapshot='paper/objective-evidence-snapshot.json')
    primary_path.write_text(json.dumps(primary, indent=2) + '\n')
    print(json.dumps(dict(table_rows=count, completed_sources=len(sources))))


if __name__ == '__main__':
    main()
