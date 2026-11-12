"""Paper extension from only complete R65/R66/R68; no active quality is written."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/4.temporal-consistency'
REQUIRED = [
    'process/exploration-r65/finalize001/runtime.json',
    'results/exploration-r65-stage001-eval-seed0/runtime.json',
    'results/exploration-r65-stage001-eval-seed1/runtime.json',
    'process/exploration-r66/finalize001/runtime.json',
    'results/exploration-r66-stablellve-attempt001/runtime.json',
    'results/exploration-r68-reference-budget001/runtime.json']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (BASE/name).exists()]
    unsuccessful = [name for name in REQUIRED if (BASE/name).exists() and
                    not json.loads((BASE/name).read_text()).get('completed_normally')]
    if args.check_only:
        print(json.dumps(dict(ready=not (missing or unsuccessful), missing=missing, unsuccessful=unsuccessful,
                              outputs_written=False))); return
    if missing or unsuccessful:
        raise SystemExit('Completed video evidence not available; no table written: '+str(missing+unsuccessful))
    sources = {}
    def read(name):
        path = BASE/name; sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    for name in REQUIRED: read(name)
    for seed in [0, 1]:
        runtime = read(f'results/exploration-r65-stage001-seed{seed}/runtime.json')
        assert runtime['completed_normally'] and runtime['updates'] == {'spatial_l1': 8000, 'commute20': 8000}
        evaluation = read(f'results/exploration-r65-stage001-eval-seed{seed}/runtime.json')
        assert (evaluation['quality_rows'], evaluation['frame_rows'], evaluation['CNN_sequences'], evaluation['CNN_centers']) == (40, 1792, 20, 896)
        for arm in ['spatial_l1', 'commute20']:
            marker = read(f'results/exploration-r65-stage001-seed{seed}/{arm}-step8000.json')
            assert marker['step'] == 8000 and marker['final_quality_eligible'] and marker['parameters'] == 4316259
            read(f'results/exploration-r65-stage001-eval-seed{seed}/{arm}-source.json')
    for phase, count, frames in [('reference-attempt002', 140, 6272), ('stablellve-attempt001', 80, 3584)]:
        runtime = read(f'results/exploration-r66-{phase}/runtime.json')
        assert runtime['completed_normally'] and (runtime['quality_rows'], runtime['comparison_frames']) == (count, frames)
        read(f'results/exploration-r66-{phase}/metric-source.json')
        read(f'results/exploration-r66-{phase}/config.json')
    quality_audit = read('report/exploration-r65-stablellve-task/audit.json')
    assert quality_audit['complete_paired_table'] and quality_audit['new_quality_rows'] == 80
    perceptual_audit = read('report/exploration-r66-perceptual-complete/audit.json')
    assert perceptual_audit['completed'] and perceptual_audit['perceptual_rows'] == 220
    assert perceptual_audit['domain_seed_means'] == 44 and perceptual_audit['old_quality_rows_joined'] == 200
    assert perceptual_audit['all_GT_controls_exact10'] and not perceptual_audit['old_acceptance_gates_rewritten']
    quality = read('report/exploration-r65-stablellve-task/domain_seed_means.json')
    perceptual = read('report/exploration-r66-perceptual-complete/domain_seed_means.json')
    budget = read('results/exploration-r68-reference-budget001/budget.json')
    assert len(quality) == len(perceptual) == 44 and len(budget['rows']) == 16
    q = {(r['domain'], r['seed'], r['label']): r for r in quality}
    p = {(r['domain'], r['seed'], r['family'], r['method']): r for r in perceptual}
    domains = ['davis_train_pilot', 'paired_real_ND_DEV']
    new_rows = []
    for arm, label in [('spatial_l1', 'L1'), ('commute20', 'L1 + commute20')]:
        for method, suffix in [('raw', 'raw'), ('recursive_self05', '+ rec.')]:
            for seed in [0, 1]:
                values = []
                for domain in domains:
                    a = q[domain, seed, 'StableLLVE_'+arm+'_'+method]
                    b = p[domain, seed, 'StableLLVE_'+arm, method]
                    assert a['project_updates'] == b['project_updates'] == 8000
                    assert a['parameters'] == b['parameters'] == 4316259
                    assert all(abs(a[field]-b[field]) <= 1e-10 for field in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'])
                    values.append(b)
                new_rows.append((label+' '+suffix, seed, values))
    lines = [r'\begin{table}[t]', r'\centering\scriptsize\setlength{\tabcolsep}{4pt}',
        r'\caption{Complete StableLLVE network/loss task adaptation. Every arm starts from the same public clean checkpoint and receives 8,000 updates per seed. The official L1 and weight-20 commutator form are retained, but training flow source, crop, initialization and schedule differ from the paper. Raw has no history/future input; rec. adds classical motion/history. JOD uses the fixed nominal display conditions in Table~\ref{tab:perceptual}. This is neither a new method nor a full-paper reproduction.}',
        r'\label{tab:stablellve-adapted}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'Loss/output & Seed & SYN PSNR & SYN $E_{\rm temp}$ & SYN JOD & ND PSNR & ND $E_{\rm temp}$ & ND JOD\\', r'\midrule']
    for label, seed, values in new_rows:
        cells = [f'{r[field]:.3f}' for r in values for field in ['psnr_rgb', 'temporal_error', 'JOD']]
        lines.append(f'{label} & {seed} & '+' & '.join(cells)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tables = {'stablellve-adapted-table.tex': '\n'.join(lines)+'\n'}
    lines = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Full-reference ColorVideoVDP v0.5.7 predictions for the existing references, with both adapted seeds. The original StableLLVE clean checkpoint has zero project updates and no training seed. Nominal 30\,fps, original-sized SDR arrays, standard FHD display, 37.84 pixels/degree, 200\,cd/m$^2$ peak and replicate temporal padding are assumed. These JOD predictions are not measured handset viewing, capture timing, human MOS or a separately calibrated flicker score.}',
        r'\label{tab:perceptual}', r'\begin{tabular}{llrr}', r'\toprule',
        r'Reference/output & Seed & SYN JOD & ND JOD\\', r'\midrule']
    for family, name in [('MobileIE', 'MobileIE'), ('Nano', 'Nano'), ('FastDVDnet', 'FastDVDnet'), ('StableLLVE_clean', 'StableLLVE clean')]:
        for method in (['window'] if family == 'FastDVDnet' else ['raw', 'recursive_self05']):
            for seed in ([None] if family == 'StableLLVE_clean' else [0, 1]):
                values = [p[domain, seed, family, method]['JOD'] for domain in domains]
                label = name+(' + rec.' if method == 'recursive_self05' else ' '+method)
                lines.append(f'{label} & {"--" if seed is None else seed} & {values[0]:.3f} & {values[1]:.3f}'+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tables['perceptual-reference-table.tex'] = '\n'.join(lines)+'\n'
    cost = {r['model']: r for r in budget['rows'] if (r['height'], r['width']) == (1080, 1920)}
    assert set(cost) == {'MobileIE', 'Nano', 'FastDVDnet', 'StableLLVE'}
    lines = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Reference deployment conditions at $1080\times1920$, batch one, per output frame. Conv GMAC counts only executed Conv2d multiply--accumulates; it omits bias, resizing, priors, pooling, activation, motion, warping and memory work. FastDVDnet uses the original uncached three first-stage and one second-stage calls. Rec. additionally retains two FP32 RGB frames plus motion/workspace; raw spatial networks have no temporal state. Identical StableLLVE graphs serve its public and adapted weights. No row is a measured handset latency or complete pipeline FLOP claim.}',
        r'\label{tab:reference-budget}', r'\begin{tabular}{lrrrrl}', r'\toprule',
        r'Network/output & Params & Conv GMAC & Future & Flow & Handset ms\\', r'\midrule']
    for model in ['MobileIE', 'Nano', 'FastDVDnet', 'StableLLVE']:
        row = cost[model]
        for recursive in ([False] if model == 'FastDVDnet' else [False, True]):
            label = model+(' + rec.' if recursive else ' window' if model == 'FastDVDnet' else ' raw')
            lines.append(f'{label} & {row["parameters"]} & {row["conv_MACs"]/1e9:.3f} & {row["future_wait_max"]} & '
                         f'{"yes" if recursive else "no"} & --'+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tables['reference-budget-table.tex'] = '\n'.join(lines)+'\n'
    # All required terminal records and cross-table identities are checked before the first write.
    out = ROOT/'paper/generated'
    for name, text in tables.items(): (out/name).write_text(text)
    record = dict(created_at=datetime.now().astimezone().isoformat(), source_sha256=sources,
        generated_tables=list(tables), R65_complete_quality=True, R66_complete_perceptual=True,
        R68_Conv_budget_only=True, new_method=False, handset_measured=False, independent_final=False,
        scope='Only complete reused-DEV evidence; no partial checkpoint, active score, SOTA/novelty or handset claim.')
    (ROOT/'paper/video-evidence-snapshot.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(tables=len(tables), completed_sources=len(sources), new_adapted_table_rows=len(new_rows))))


if __name__ == '__main__': main()
