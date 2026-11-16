"""All full/startup and tail evidence, with GT diagnostic excluded from24 primary gates."""
from collections import defaultdict
import json
import numpy as np

from coefficient_io_r80 import ROOT, BASE, digest, write, lines, verify, key
from causal_reference_evaluation_r75 import reference_quality


def main():
    verify(); old, sources = reference_quality()
    rows = [dict(r, scope='full', parameters=4075, deployable=True, project_updates=8000) for r in old]
    new = []
    for seed in (0, 1):
        run = BASE/f'results/exploration-r80-eval-seed{seed}'
        terminal = json.loads((run/'runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['new_quality_rows'] == 80 and terminal['ridge_updates'] == 240
        path = run/'per_sequence.jsonl'; selected = lines(path); assert len(selected) == 80
        new += selected; sources[str(path.relative_to(ROOT))] = digest(path)
    run = BASE/'results/exploration-r80-perceptual001'; terminal = json.loads((run/'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['quality_rows'] == 160 and terminal['comparison_frames'] == 6368
    path = run/'per_sequence.jsonl'; perceptual = lines(path); lookup = {key(r): r for r in perceptual}
    assert len(lookup) == len(perceptual) == 160 and set(lookup) == {key(r) for r in new}
    sources[str(path.relative_to(ROOT))] = digest(path)
    rows += [dict(r, JOD=lookup[key(r)]['JOD']) for r in new]
    assert len(rows) == len({key(r) for r in rows}) == 200
    groups = defaultdict(list)
    for r in rows:
        groups[r['scope'], r['domain'], r['seed'], r['label']].append(r)
    means = []
    for (scope, domain, seed, label), entries in groups.items():
        assert len(entries) == (6 if domain == 'davis_train_pilot' else 4)
        means.append(dict(scope=scope, domain=domain, seed=seed, label=label, conditions=len(entries),
            deployable=label != 'FST_GT_diagnostic', **{field:float(np.mean([r[field] for r in entries])) for field in
            ('psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae', 'JOD')}))
    assert len(means) == 40
    index = {(r['scope'], r['domain'], r['seed'], r['label']):r for r in means}
    gates = []
    for scope in ('full', 'tail'):
        thresholds = (1., 1., 1.10) if scope == 'full' else (.90, .95, 1.05)
        for domain in ('davis_train_pilot', 'paired_real_ND_DEV'):
            for seed in (0, 1):
                candidate = index[scope, domain, seed, 'FST_prefix']
                for label, threshold in zip(('MobileIE_raw', 'RGB_prefix', 'MobileIE_recursive_self05'), thresholds):
                    reference = index[scope, domain, seed, label]
                    psnr = candidate['psnr_rgb']-reference['psnr_rgb']; jod = candidate['JOD']-reference['JOD']
                    ratio = candidate['temporal_error']/reference['temporal_error']
                    gates.append(dict(scope=scope, domain=domain, seed=seed, reference=label,
                        PSNR_delta=psnr, JOD_delta=jod, Etemp_ratio=ratio, Etemp_max=threshold,
                        passed=psnr >= -.05 and jod >= -.02 and ratio <= threshold))
    assert len(gates) == 24
    out = BASE/'report/exploration-r80-joint'; out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'domain_seed_means.json', means); write(out/'gates.json', gates); write(out/'sources.json', sources)
    table = ['# R80完整条件读出：启动阶段和固定尾段', '',
        '原F0固定4075参数/8000更新；FST/RGB前缀拟合不使用GT，GT diagnostic只作不可部署诊断。full含启动；tail固定去前8帧，不同scope的JOD均独立计算。', '',
        '|scope|域|seed|控制|PSNR|Etemp|JOD|', '|---|---|---:|---|---:|---:|---:|']
    for r in means:
        table.append(f"|{r['scope']}|{r['domain']}|{r['seed']}|{r['label']}|{r['psnr_rgb']:.6f}|{r['temporal_error']:.6f}|{r['JOD']:.6f}|")
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    passed = all(r['passed'] for r in gates)
    write(out/'audit.json', dict(completed=True, all_quality_rows=200, new_quality_rows=160,
        reused_full_reference_rows=40, domain_seed_means=40, registered_comparisons=24,
        all_gates_pass=passed, shared_core_frames=896, adaptation_ridge_updates=480, new_global_SGD=0,
        GT_diagnostic_deployable=False, decision='cost and distinct mechanism checks allowed' if passed else 'STOP local coefficient-prefix combination',
        novel_method_established=False, phone_measured=False, independent_final=False))


if __name__ == '__main__':
    main()
