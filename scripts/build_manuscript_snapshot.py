"""Generate manuscript tables from completed evidence; never include a partial study."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/4.temporal-consistency'


def published_reference_table(read, out):
    """Keep both final seeds; equal updates do not establish equal convergence."""
    for experiment, report in [(53, 'exploration-r53-source-context-reference'),
                               (56, 'exploration-r56-tiny-task-reference')]:
        assert read(f'process/exploration-r{experiment}/finalize001/runtime.json')['completed_normally']
        audit = read(f'report/{report}/audit.json')
        assert audit['new_training_updates'] == 16000 and audit['new_quality_rows'] == 40
        for seed in [0, 1]:
            train = read(f'results/exploration-r{experiment}-stage001-seed{seed}/runtime.json')
            runtime = read(f'results/exploration-r{experiment}-stage001-eval-seed{seed}/runtime.json')
            assert train['completed_normally'] and train['steps'] == 8000
            assert runtime['completed_normally'] and not runtime['cuda_initialized']
            assert runtime.get('rows', runtime.get('quality_rows')) == 20
    nano = read('report/exploration-r56-tiny-task-reference/domain_seed_summary.json')
    video = read('report/exploration-r53-source-context-reference/domain_seed_summary.json')
    n = {(r['domain'], r['seed'], r['method']): r for r in nano}
    v = {(r['domain'], r['seed'], r['method']): r for r in video}
    domains = ['davis_train_pilot', 'paired_real_ND_DEV']
    rows = []
    clean_run = 'results/exploration-r62-stablellve-common-attempt001'
    clean_config = read(clean_run+'/config.json')
    anchor_config = read('results/exploration-r44-stage001-eval-seed0/config.json')
    assert clean_config['cache_indices'] == anchor_config['cache_indices']
    assert clean_config['project_updates'] == 0 and clean_config['claimed_seed'] is None
    assert clean_config['parameters'] == 4316259 and clean_config['future_wait_max'] == 0
    clean_audit = read('report/exploration-r62-stablellve-common/audit.json')
    assert clean_audit['completed_normally'] and clean_audit['quality_rows'] == 20
    assert clean_audit['per_frame_rows'] == 896 and not clean_audit['training_budget_matched']
    clean = read('report/exploration-r62-stablellve-common/domain-summary.json')
    assert len(clean) == 4
    c = {(r['domain'], r['method']): r for r in clean}
    for method, label in [('raw', 'StableLLVE clean raw'), ('recursive_self05', 'StableLLVE clean + rec.')]:
        values = [c[d, 'StableLLVE_clean_'+method] for d in domains]
        assert [r['rows'] for r in values] == [6, 4]
        assert all(r['project_updates'] == 0 and r['seed'] is None for r in values)
        rows.append((label, '--', 4316259, values))
    for method, label in [('raw', 'Nano published raw'), ('recursive_self05', 'Nano published + rec.')]:
        rows.append((label, '--', 696, [n[d, 'published', 'Nano_published_'+method] for d in domains]))
    methods = [('Nano_adapted_raw', 'Nano adapted raw', 696),
               ('Nano_adapted_recursive_self05', 'Nano adapted + rec.', 696),
               ('source_context_window', 'FastDVDnet window', 2479096),
               ('source_context_five_current', 'FastDVDnet current', 2479096),
               ('MobileIE_raw', 'MobileIE raw', 4075),
               ('MobileIE_fixed_05', 'MobileIE + rec.', 4075),
               ('MobileIE_rgb', 'MobileIE + RGB gate', 4116)]
    for method, label, parameters in methods:
        for seed in [0, 1]:
            values = ([n[d, f'seed{seed}', method] for d in domains] if method.startswith('Nano_')
                      else [v[d, seed, method] for d in domains])
            rows.append((label, str(seed), parameters, values))
    assert len(rows) == 18
    lines = [r'\begin{table}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}',
             r'\caption{Completed published and adapted references on reused DEV data, all evaluated with the common CPU protocol. The original StableLLVE synthetic-clean and Nano LOL-v1 checkpoints have zero project updates and no claimed seed. Adapted backbones use 8,000 updates with the same four supervised targets; the RGB gate has 4,000 additional controller updates. FastDVDnet sees surrounding inputs and two future frames. Pretraining, training parameterization, noise conditions and compute differ; these are not matched-convergence architectural rankings. Current means five copies of the current input, a diagnostic. Recursion and RGB gating add motion/history cost beyond learned parameters.}',
             r'\label{tab:published-adapted}', r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Model/output & Seed & Params & SYN PSNR & SYN $E_{\rm temp}$ & ND PSNR & ND $E_{\rm temp}$\\',
             r'\midrule']
    for index, (label, seed, parameters, (syn, real)) in enumerate(rows):
        if index == 4:
            lines.append(r'\midrule')
        lines.append(f"{label} & {seed} & {parameters} & {syn['psnr_rgb']:.3f} & "
                     f"{syn['temporal_error']:.3f} & {real['psnr_rgb']:.3f} & {real['temporal_error']:.3f}"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out/'published-adapted-table.tex').write_text('\n'.join(lines)+'\n')


def main():
    sources = {}
    def read(relative):
        path = BASE/relative
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    # Fail before writing any generated table if the new reference is incomplete.
    clean_run = read('results/exploration-r62-stablellve-common-attempt001/runtime.json')
    assert clean_run['completed_normally'] and clean_run['sequences'] == 20
    assert clean_run['CNN_center_outputs'] == 448 and not clean_run['cuda_initialized']
    assert read('process/exploration-r62/finalize001/runtime.json')['completed_normally']
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
        p = BASE/'report/exploration-r44-bounded-video/factorial.pdf'
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    closure = BASE/'process/exploration-r45/stage001/finalize002/runtime.json'
    r45_complete = closure.exists() and json.loads(closure.read_text())['completed_normally']
    if r45_complete:
        read('process/exploration-r45/stage001/finalize002/runtime.json')
        audit = read('report/exploration-r45-response-recursion/audit.json')
        assert audit['passed'] and audit['rows'] == 180 and audit['new_neural_sequences'] == 20
        for seed in [0, 1]:
            runtime = read(f'results/exploration-r45-stage001-eval-seed{seed}/runtime.json')
            assert runtime['completed_normally'] and runtime['rows'] == 90 and runtime['neural_sequences'] == 10
        values = read('report/exploration-r45-response-recursion/domain_summary.json')
        assert len(values) == 18
        lookup = {(r['domain'], r['method']): r for r in values}
        methods = [('raw', 'Raw'), ('fixed_01', r'Fixed $\alpha=.1$'),
                   ('fixed_025', r'Fixed $\alpha=.25$'), ('fixed_05', r'Fixed $\alpha=.5$'),
                   ('fixed_075', r'Fixed $\alpha=.75$'), ('fixed_09', r'Fixed $\alpha=.9$'),
                   ('scalar', 'Learned global'), ('rgb', 'RGB gate'), ('response', 'Internal cue')]
        lines = [r'\begin{table}[t]', r'\centering\small',
                 r'\caption{Complete R45 comparison on the same frozen square-MSE base. Both-seed means; all fixed alpha points retained. All outputs except raw use motion and RGB history. Parameters omit unlearned motion work; no phone or pipeline cost is inferred.}',
                 r'\label{tab:r45}', r'\begin{tabular}{lrrrrr}', r'\toprule',
                 r'Output & Params & SYN PSNR & SYN $E_{\rm temp}$ & ND PSNR & ND $E_{\rm temp}$\\', r'\midrule']
        for method, label in methods:
            syn, real = [lookup[(domain, method)] for domain in ['davis_train_pilot', 'paired_real_ND_DEV']]
            lines.append(f"{label} & {syn['parameters']} & {syn['psnr_rgb']:.3f} & {syn['temporal_error']:.3f} & "
                         f"{real['psnr_rgb']:.3f} & {real['temporal_error']:.3f}"+r'\\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        (out/'r45-table.tex').write_text('\n'.join(lines)+'\n')
        generated.append('r45-table.tex')
        for name in ['frontier.pdf', 'alpha_distribution.pdf']:
            p = BASE/'report/exploration-r45-response-recursion'/name
            sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    published_reference_table(read, out)
    generated.append('published-adapted-table.tex')
    excluded = ('R44; no partial checkpoint or loss curve included' if not r44_complete else
                'R45; no partial quality included' if not r45_complete else None)
    record = {'created_at': datetime.now().astimezone().isoformat(), 'source_sha256': sources,
              'generated_tables': generated,
              'R44_complete_quality_included': r44_complete,
              'R45_complete_quality_included': r45_complete,
              'R53_R56_complete_references_included': True,
              'R62_complete_clean_transfer_reference_included': True,
              'excluded_active_study': excluded,
              'additional_studies_not_in_this_snapshot': ['R59 static RAW pilot', 'R60 unpaired Night DEV', 'R61 scalar boundary'],
              'scope': 'Completed DEV evidence only; no novelty, SOTA, phone latency or independent final-test claim.'}
    (ROOT/'paper/evidence-snapshot.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'tables': len(generated), 'completed_sources': len(sources),
                      'R44_complete_quality_included': r44_complete, 'R45_complete_quality_included': r45_complete}))


if __name__ == '__main__':
    main()
