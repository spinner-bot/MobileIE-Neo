"""Generate manuscript tables from completed evidence; never include a partial study."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/4.temporal-consistency'


def main():
    sources = {}
    def read(relative):
        path = BASE/relative
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    assert read('report/exploration-r41-training-form/audit.json')['new_rows'] == 120
    assert read('report/exploration-r42-tlc/audit.json')['total_rows'] == 120
    assert read('report/exploration-r36-classical/audit.json')['rows'] == 48
    assert read('report/exploration-r36-runtime/audit.json')['runs'] == 18
    anchor = read('report/exploration-r41-training-form/domain_summary.json')
    classical = read('report/exploration-r36-classical/domain_summary.json')
    costs = read('report/exploration-r36-runtime/cost.json')
    effects = read('report/exploration-r42-tlc/raw_effects.json')
    read('process/exploration-r43/cpu-gate.json')
    read('process/exploration-r43/scale-calibration.json')
    assert len(anchor) == 30 and len(classical) == 4 and len(effects) == 4
    lookup = {(r['domain'], r['anchor_family'], r['method']): r for r in anchor}
    out = ROOT/'paper/generated'
    out.mkdir(parents=True, exist_ok=True)
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Completed anchor evidence: two-seed means on custom synthetic and paired ND development data. Historical rows differ in training; only the two R41 raw forms are a matched FP32 parameterization comparison.}',
             r'\label{tab:anchors}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Anchor/output & SYN PSNR & SYN $E_{\rm temp}$ & ND PSNR & ND $E_{\rm temp}$\\',
             r'\midrule']
    selected = [('R34_original', 'raw', 'Dynamic synthetic (R34)'),
                ('R40_mixed', 'raw', 'Fixed mixed Slim (R40)'),
                ('R41_slim_fp32', 'raw', 'Matched Slim FP32 (R41)'),
                ('R41_expanded_fp32', 'raw', 'Expanded FP32 (R41)'),
                ('R41_expanded_fp32', 'finite_self05', 'Expanded + finite'),
                ('R41_expanded_fp32', 'recursive_self05', 'Expanded + recursive')]
    for family, method, label in selected:
        syn, real = [lookup[(domain, family, method)] for domain in ['davis_train_pilot', 'paired_real_ND_DEV']]
        lines.append(f"{label} & {syn['psnr_rgb']:.3f} & {syn['temporal_error']:.3f} & "
                     f"{real['psnr_rgb']:.3f} & {real['temporal_error']:.3f}"+r'\\')
    real = lookup[('paired_real_ND_DEV', 'R26_plain_MSE', 'raw')]
    lines.append(f"Earlier native training (R26) & -- & -- & {real['psnr_rgb']:.3f} & {real['temporal_error']:.3f}"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out/'anchor-table.tex').write_text('\n'.join(lines)+'\n')
    q, c = {r['method']: r for r in classical}, {r['method']: r for r in costs}
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Classical references on the same R34 weights. Quality uses the full synthetic pilot. Resident CPU timing uses only the first 16 bear/P20 frames at $480\times854$, two Torch threads and one OpenCV thread. Ranges are run-average extrema, not confidence intervals or phone timings.}',
             r'\label{tab:classical}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Output & PSNR & $E_{\rm temp}$ & History MiB & CPU ms/frame [range]\\', r'\midrule']
    for method, label in [('raw', 'Raw'), ('finite_self05', 'Finite .5'), ('recursive_self05', 'Recursive .5')]:
        a, b = q[method], c[method]
        lines.append(f"{label} & {a['psnr_rgb']:.3f} & {a['temporal_error']:.3f} & "
                     f"{b['online_history_storage_bytes']/2**20:.3f} & "
                     f"{b['median_pipeline_ms_per_frame']:.1f} "
                     f"[{b['min_pipeline_ms_per_frame']:.1f}, {b['max_pipeline_ms_per_frame']:.1f}]"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out/'classical-table.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Fixed known TLC192 versus the same R41 expanded weights. No retraining or window search.}',
             r'\label{tab:tlc}', r'\begin{tabular}{llrr}', r'\toprule',
             r'Domain & Seed & Raw $\Delta$PSNR & Raw temporal ratio\\', r'\midrule']
    for r in effects:
        domain = 'Synthetic' if r['domain'] == 'davis_train_pilot' else 'Paired ND'
        lines.append(f"{domain} & {r['seed']} & {r['delta_psnr_rgb']:.3f} & {r['temporal_ratio']:.4f}"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out/'tlc-table.tex').write_text('\n'.join(lines)+'\n')
    for name in ['exploration-r41-training-form/matched_forms.pdf', 'exploration-r42-tlc/fixed_tlc.pdf']:
        p = BASE/'report'/name
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    generated = ['anchor-table.tex', 'classical-table.tex', 'tlc-table.tex']
    closure = BASE/'process/exploration-r44/stage001/finalize001/runtime.json'
    r44_complete = closure.exists() and json.loads(closure.read_text())['completed_normally']
    if r44_complete:
        read('process/exploration-r44/stage001/finalize001/runtime.json')
        audit = read('report/exploration-r44-bounded-video/audit.json')
        assert audit['passed'] and audit['new_rows'] == 240 and audit['new_neural_sequences'] == 80
        for seed in [0, 1]:
            runtime = read(f'results/exploration-r44-stage001-eval-seed{seed}/runtime.json')
            assert runtime['completed_normally'] and runtime['rows'] == 120
        values = read('report/exploration-r44-bounded-video/domain_summary.json')
        assert len(values) == 24
        by_key = {(r['domain'], r['model'], r['method']): r for r in values}
        lines = [r'\begin{table}[t]', r'\centering\small',
                 r'\caption{Complete R44 factorial, both-seed means. All models inherit the same R41 training form and add 4,000 updates. Raw inference is single-frame; recursive outputs retain motion and history. Rational denominators remain at inference.}',
                 r'\label{tab:r44}', r'\begin{tabular}{llrrrr}', r'\toprule',
                 r'Model & Output & SYN PSNR & SYN $E_{\rm temp}$ & ND PSNR & ND $E_{\rm temp}$\\',
                 r'\midrule']
        for method, short in [('raw', 'Raw'), ('recursive_self05', 'Recursive')]:
            for model, label in [('square_mse', 'Square/MSE'), ('square_temporal', 'Square/temporal'),
                                 ('bounded_mse', 'Rational/MSE'), ('bounded_temporal', 'Rational/temporal')]:
                syn, real = [by_key[(domain, model, method)] for domain in ['davis_train_pilot', 'paired_real_ND_DEV']]
                lines.append(f"{label} & {short} & {syn['psnr_rgb']:.3f} & {syn['temporal_error']:.3f} & "
                             f"{real['psnr_rgb']:.3f} & {real['temporal_error']:.3f}"+r'\\')
            if method == 'raw':
                lines.append(r'\midrule')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        (out/'r44-table.tex').write_text('\n'.join(lines)+'\n')
        generated.append('r44-table.tex')
    record = {'created_at': datetime.now().astimezone().isoformat(), 'source_sha256': sources,
              'generated_tables': generated,
              'R44_complete_quality_included': r44_complete,
              'excluded_active_study': None if r44_complete else 'R44; no partial checkpoint or loss curve included',
              'scope': 'Completed DEV evidence only; no novelty, SOTA, phone latency or independent final-test claim.'}
    (ROOT/'paper/evidence-snapshot.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'tables': len(generated), 'completed_sources': len(sources), 'R44_complete_quality_included': r44_complete}))


if __name__ == '__main__':
    main()
