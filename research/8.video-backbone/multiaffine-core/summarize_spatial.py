"""All spatial-recovery groups and comparisons; no new model inference."""
from study_io import BASE, read, write, stamp, verify


def main():
    verify(); rows = []
    import json
    for seed in (0, 1):
        root = BASE / f'results/spatial-held-seed{seed}'
        runtime = read(root / 'runtime.json')
        assert runtime['completed_normally'] and runtime['backbone_images'] == 768
        rows += [json.loads(line) for line in (root / 'per_sequence.jsonl').read_text().splitlines()]
    assert len(rows) == 384
    means, comparisons = [], []
    for seed in (0, 1):
        for domain in ('SYN', 'ND'):
            grouped = {}
            for arm in ('multiaffine', 'original_scratch', 'F0'):
                chosen = [r for r in rows if (r['seed'], r['domain'], r['arm']) == (seed, domain, arm)]
                assert len(chosen) == 32
                grouped[arm] = {r['key']: r for r in chosen}
                means.append(dict(seed=seed, domain=domain, arm=arm, windows=32,
                    **{m: sum(r[m] for r in chosen)/32 for m in
                       ('psnr_rgb', 'ssim_rgb', 'edge_mae', 'raw_fidelity_MSE_to_F0', 'outside_display_range_fraction')}))
            current = {r['arm']: r for r in means if r['seed'] == seed and r['domain'] == domain}
            for reference in ('F0', 'original_scratch'):
                delta = current['multiaffine']['psnr_rgb'] - current[reference]['psnr_rgb']
                worst = min(grouped['multiaffine'][k]['psnr_rgb'] - grouped[reference][k]['psnr_rgb'] for k in grouped[reference])
                comparisons.append(dict(seed=seed, domain=domain, reference=reference, mean_PSNR_delta=delta,
                    worst_window_PSNR_delta=worst, mean_passed=delta >= -.20, worst_passed=worst >= -.75,
                    passed=delta >= -.20 and worst >= -.75))
    out = BASE / 'report/spatial-recovery'
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'domain_seed_means.json', means)
    write(out / 'gates.json', dict(at=stamp(), comparisons=comparisons,
        complete_passed=all(r['passed'] for r in comparisons),
        allowed_separate_video_registration=all(r['passed'] for r in comparisons),
        new_innovation_or_video_quality_established=False, held_out_draws_not_scenes=True))
    content = ['# M1-S：完整空间恢复必要检验', '',
        '训练池内固定保留抽样，非独立DEV/最终；三输出预训/参数/训练形态不同，不能据此纯架构排名。', '',
        '|域|seed|输出|PSNR|SSIM|edge|原始F0差MSE|越界比例|', '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in means:
        content.append(f'|{r["domain"]}|{r["seed"]}|{r["arm"]}|{r["psnr_rgb"]:.6f}|{r["ssim_rgb"]:.6f}|{r["edge_mae"]:.6f}|{r["raw_fidelity_MSE_to_F0"]:.8f}|{r["outside_display_range_fraction"]:.6f}|')
    content += ['', '八个对照/域/seed比较全部按原平均−.20dB/最差窗口−.75dB，不按最好seed或备份择优。', '']
    (out / 'tables.md').write_text('\n'.join(content))


if __name__ == '__main__':
    main()
