"""All 52 O1 means and 36 preregistered comparisons, without new inference."""
import json
from study_io import BASE, read, write, stamp, verify
from evaluate import ARMS


def main():
    verify('training-source-freeze001.json'); rows = []
    for seed in (0, 1):
        source = BASE / f'results/held-seed{seed}'
        assert read(source / 'runtime.json')['completed_normally']
        rows += [json.loads(line) for line in (source / 'per_sequence.jsonl').read_text().splitlines()]
    assert len(rows) == 416
    means, comparisons = [], []
    for seed in (0, 1):
        for domain in ('SYN', 'ND'):
            grouped = {}
            for arm in ARMS:
                selected = [r for r in rows if (r['seed'], r['domain'], r['arm']) == (seed, domain, arm)]
                assert len(selected) == 8
                grouped[arm] = {r['key']: r for r in selected}
                means.append(dict(seed=seed, domain=domain, arm=arm, windows=8,
                    **{m: sum(r[m] for r in selected)/8 for m in ('psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error')}))
            current = {r['arm']: r for r in means if (r['seed'], r['domain']) == (seed, domain)}
            scheduled = [('space', a+'_direct', 'F0', None) for a in ('coordinate', 'response')]
            scheduled += [('history', 'response_causal', 'response_direct', .90),
                          ('history', 'response_causal', 'response_zero', .95),
                          ('objective', 'response_causal', 'coordinate_causal', .95)]
            scheduled += [('strong', 'response_causal', ref, 1.05) for ref in ('F0_FB', 'F0_DIS', 'response_FB', 'response_DIS')]
            for category, arm, ref, limit in scheduled:
                delta = current[arm]['psnr_rgb']-current[ref]['psnr_rgb']
                worst = min(grouped[arm][k]['psnr_rgb']-grouped[ref][k]['psnr_rgb'] for k in grouped[arm])
                ratio = current[arm]['temporal_error']/current[ref]['temporal_error']
                comparisons.append(dict(seed=seed, domain=domain, category=category, candidate=arm, reference=ref,
                    PSNR_delta=delta, worst_window_PSNR_delta=worst, Etemp_ratio=ratio, Etemp_limit=limit,
                    mean_quality_passed=delta >= -.05, worst_quality_passed=worst >= -.5,
                    temporal_passed=limit is None or ratio <= limit,
                    passed=delta >= -.05 and worst >= -.5 and (limit is None or ratio <= limit)))
    assert len(means) == 52 and len(comparisons) == 36
    out = BASE / 'report/joint'; out.mkdir(parents=True, exist_ok=False)
    write(out / 'means.json', means)
    write(out / 'gates.json', dict(at=stamp(), comparisons=comparisons, complete_passed=all(r['passed'] for r in comparisons),
        allowed_separate_DEV_cost_registration=all(r['passed'] for r in comparisons),
        held_out_draws_not_scenes=True, established_method_or_innovation=False, independent_final=False))
    text = ['# O1完整保留训练窗比较', '', '非独立DEV/最终；两seed分开，唯一4000最终，零最好checkpoint选择。', '',
            '|域|seed|输出|PSNR|SSIM|edge|Etemp|', '|---|---:|---|---:|---:|---:|---:|']
    for r in means: text.append(f"|{r['domain']}|{r['seed']}|{r['arm']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['edge_mae']:.6f}|{r['temporal_error']:.6f}|")
    text += ['', '|域|seed|门|候选|参考|PSNR差|最差窗口|Etemp比|完整通过|', '|---|---:|---|---|---|---:|---:|---:|---|']
    for r in comparisons: text.append(f"|{r['domain']}|{r['seed']}|{r['category']}|{r['candidate']}|{r['reference']}|{r['PSNR_delta']:.6f}|{r['worst_window_PSNR_delta']:.6f}|{r['Etemp_ratio']:.6f}|{r['passed']}|")
    (out / 'tables.md').write_text('\n'.join(text)+'\n')


if __name__ == '__main__':
    main()
