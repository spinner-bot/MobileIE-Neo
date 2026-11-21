"""All full-length I1-S means and44 fixed comparisons, without selecting a best seed."""
from simple_io import stamp, write

ARMS = ('F0', 'F0_FB', 'F0_DIS', 'all', 'all_FB', 'all_DIS', 'output', 'output_FB', 'output_DIS')


def summarize(rows, out):
    assert len(rows) == 180
    means, gates = [], []
    for seed in (0, 1):
        for domain, count in [('SYN', 6), ('ND', 4)]:
            grouped = {}
            for arm in ARMS:
                part = [r for r in rows if (r['seed'], r['domain'], r['arm']) == (seed, domain, arm)]
                assert len(part) == count
                grouped[arm] = {r['scene']: r for r in part}
                assert len(grouped[arm]) == count
                means.append(dict(seed=seed, domain=domain, arm=arm, conditions=count,
                    **{m: sum(r[m] for r in part)/count for m in ('psnr_rgb', 'ssim_rgb', 'edge_mae', 'flicker', 'temporal_error')}))
            current = {r['arm']: r for r in means if (r['seed'], r['domain']) == (seed, domain)}
            comparisons = []
            for candidate in ('all', 'output'):
                comparisons += [(candidate, ref, .90 if ref == 'F0' else 1.05, 'route')
                                for ref in ('F0', 'F0_FB', 'F0_DIS', candidate+'_FB', candidate+'_DIS')]
            comparisons.append(('all', 'output', .95, 'internal_increment'))
            for candidate, reference, limit, kind in comparisons:
                assert grouped[candidate].keys() == grouped[reference].keys()
                delta = current[candidate]['psnr_rgb'] - current[reference]['psnr_rgb']
                worst = min(grouped[candidate][s]['psnr_rgb']-grouped[reference][s]['psnr_rgb'] for s in grouped[candidate])
                ratio = current[candidate]['temporal_error']/current[reference]['temporal_error']
                gates.append(dict(seed=seed, domain=domain, candidate=candidate, reference=reference, kind=kind,
                    PSNR_delta=delta, worst_sequence_PSNR_delta=worst, Etemp_ratio=ratio, Etemp_limit=limit,
                    mean_quality_passed=delta >= -.05, worst_quality_passed=worst >= -.50,
                    passed=delta >= -.05 and worst >= -.50 and ratio <= limit))
    assert len(means) == 36 and len(gates) == 44
    route = {level: all(g['passed'] for g in gates if g['candidate'] == level and g['kind'] == 'route')
             for level in ('all', 'output')}
    internal = all(g['passed'] for g in gates if g['kind'] == 'internal_increment')
    eligible = [level for level in ('all', 'output') if route[level] and (level == 'output' or internal)]
    write(out/'means.json', means)
    write(out/'gates.json', dict(at=stamp(), comparisons=gates, route_passed=route,
          all_internal_increment_passed=internal, eligible_separate_cost_and_perceptual_registration=eligible,
          established_novelty=False, independent_final=False, phone_measured=False))
    text = ['# I1-S完整全长结果', '',
            '重复DEV十条件×两seed；零未来，完整源前缀。不是独立最终、手机或原作者HDRNet成绩复现。', '',
            '|域|seed|输出|PSNR|SSIM|edge|Etemp|', '|---|---:|---|---:|---:|---:|---:|']
    for r in means:
        text.append(f"|{r['domain']}|{r['seed']}|{r['arm']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['edge_mae']:.6f}|{r['temporal_error']:.6f}|")
    text += ['', '|域|seed|比较|PSNR差|最差序列差|Etemp比|限|过|', '|---|---:|---|---:|---:|---:|---:|---|']
    for g in gates:
        text.append(f"|{g['domain']}|{g['seed']}|{g['candidate']}/{g['reference']}|{g['PSNR_delta']:+.6f}|{g['worst_sequence_PSNR_delta']:+.6f}|{g['Etemp_ratio']:.6f}|{g['Etemp_limit']:.2f}|{g['passed']}|")
    text += ['', f'两层级完整路线={route}；内部增量={internal}；可另登记成本/感知={eligible}。']
    (out/'tables.md').write_text('\n'.join(text)+'\n')
