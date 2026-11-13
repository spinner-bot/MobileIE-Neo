"""Add only the completed R73 state study, retaining every registered reference."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/4.temporal-consistency'


def main():
    primary_path = ROOT/'paper/evidence-snapshot.json'
    primary = json.loads(primary_path.read_text())
    assert primary['R72_complete_external_development_included']
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == v for n, v in primary['source_sha256'].items())
    sources = {}

    def read(name):
        path = BASE/name
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    # Attempt001 was an execution failure; only the actual normal002 qualifies.
    assert read('process/exploration-r73/finalize002/runtime.json')['completed_normally']
    for seed in [0, 1]:
        folder = f'results/exploration-r73-stage001-seed{seed}'
        train, config = read(folder+'/runtime.json'), read(folder+'/config.json')
        assert train['completed_normally'] and train['model_updates'] == dict(spatial=4000, mean=4000, moment=4000)
        assert config['inherited_updates_each'] == 8000 and config['new_steps_each'] == 4000
        for model in ['spatial', 'mean', 'moment']:
            marker = read(folder+f'/{model}-step4000.json')
            assert marker['final_fold_passed'] and marker['final_fold_delta'] <= 1e-4
        evaluation = read(f'results/exploration-r73-eval-seed{seed}/runtime.json')
        assert evaluation['completed_normally'] and (evaluation['quality_rows'], evaluation['frame_rows']) == (100, 4480)
    metric = read('results/exploration-r73-perceptual001/runtime.json')
    assert metric['completed_normally'] and (metric['quality_rows'], metric['comparison_frames']) == (240, 10752)
    source = read('results/exploration-r73-perceptual001/metric-source.json')
    assert source['package_version'] == '0.5.7' and source['playback_fps'] == 30
    assert read('report/exploration-r73-contacts/runtime.json')['completed_normally']
    review = read('report/exploration-r73-contacts/actual-review001.json')
    assert review['actual_reviewed'] and review['reviewed_contacts'] == 20
    audit = read('report/exploration-r73-joint/audit.json')
    assert audit['completed'] and audit['perceptual_included'] and not audit['all_gates_pass']
    assert (audit['new_quality_rows'], audit['all_quality_rows'], audit['domain_seed_means']) == (200, 280, 56)
    means = read('report/exploration-r73-joint/domain_seed_means.json')
    gates = read('report/exploration-r73-joint/gates.json')
    assert len(means) == 56 and len(gates) == 20 and not any(g['passed'] for g in gates)
    index = {(r['domain'], r['seed'], r['model'], r['history'], r['method']): r for r in means}
    assert len(index) == 56
    methods = [('spatial', 'current', 'raw', 'Spatial'), ('mean', 'causal', 'raw', 'Mean history'),
        ('moment', 'current', 'raw', 'Moments/current'), ('moment', 'causal', 'raw', 'Moments/history'),
        ('R44_plain', 'current', 'recursive_self05', 'Plain 8K + rec.'),
        ('R64_plain', 'current', 'recursive_self05', 'Plain 12K + rec.')]
    references = {tuple(row[:3]) for row in methods if row[0:3] != ('moment', 'causal', 'raw')}
    # Methods above are precisely the target and all five registered comparisons.
    for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
        for seed in [0, 1]:
            selected = [g for g in gates if (g['domain'], g['seed']) == (domain, seed)]
            assert {tuple(g['reference']) for g in selected} == references
            main = index[domain, seed, 'moment', 'causal', 'raw']
            for gate in selected:
                reference = index[(domain, seed, *gate['reference'])]
                assert abs(main['psnr_rgb']-reference['psnr_rgb']-gate['PSNR_delta']) < 1e-10
                assert abs(main['JOD']-reference['JOD']-gate['JOD_delta']) < 1e-10
                assert abs(main['temporal_error']/reference['temporal_error']-gate['Etemp_ratio']) < 1e-10
    lines = [r'\begin{table}[!htbp]', r'\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Completed native-state study (R73): target and all five registered references, both domains and seeds. New spatial/mean/moment models inherit 8K updates and receive 4K more each; the plain references retain their indicated totals. Current uses the same moment weights with history disabled. Rec. adds classical motion/RGB recursion. The 20 joint screens all fail. Reused development sources and nominal 30\,fps perceptual evaluation; no phone or final-test result.}',
        r'\label{tab:native-state}', r'\begin{tabular}{lllr rrr}', r'\toprule',
        r'Domain & Seed & Predictor/output & Params & PSNR & $E_{\rm temp}$ & JOD\\', r'\midrule']
    count = 0
    for domain, label in [('davis_train_pilot', 'SYN'), ('paired_real_ND_DEV', 'ND')]:
        for seed in [0, 1]:
            for model, history, method, title in methods:
                r = index[domain, seed, model, history, method]
                assert r['conditions'] == (6 if label == 'SYN' else 4)
                lines.append(f'{label} & {seed} & {title} & {r["parameters"]} & '
                    f'{r["psnr_rgb"]:.3f} & {r["temporal_error"]:.3f} & {r["JOD"]:.3f}'+r'\\')
                count += 1
            if not (label == 'ND' and seed == 1):
                lines.append(r'\midrule')
    assert count == 24
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    name = 'native-state-table.tex'
    (ROOT/'paper/generated'/name).write_text('\n'.join(lines)+'\n')
    record = dict(created_at=datetime.now().astimezone().isoformat(), source_sha256=sources,
        generated_tables=[name], R73_complete=True, table_rows=24, registered_gates=20, passed_gates=0,
        new_quality_rows=200, joint_quality_rows=280, new_perceptual_rows=240,
        formal_updates=24000, metric_playback_fps=30, handset_measured=False, independent_final=False,
        active_R74_included=False, scope='Complete fixed failed state study, not a new method or general impossibility result.')
    child = ROOT/'paper/state-evidence-snapshot.json'
    child.write_text(json.dumps(record, indent=2)+'\n')
    primary['source_sha256'].update(sources)
    primary['source_sha256'][str(child.relative_to(ROOT))] = hashlib.sha256(child.read_bytes()).hexdigest()
    primary['generated_tables'] = list(dict.fromkeys(primary['generated_tables']+[name]))
    primary.update(created_at=record['created_at'], R73_complete_state_included=True,
                   supplemental_state_snapshot='paper/state-evidence-snapshot.json')
    primary_path.write_text(json.dumps(primary, indent=2)+'\n')
    print(json.dumps(dict(table_rows=count, complete_sources=len(sources), new_CNN=0, new_SGD=0, active_R74_included=False)))


if __name__ == '__main__':
    main()
