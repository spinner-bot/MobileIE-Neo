"""All complete D1 domain means and original sixteen necessary comparisons."""
from d1_io import ARMS,write,stamp


def summarize(rows,out):
    assert len(rows)==120
    means,comparisons=[],[]
    for seed in (0,1):
        for domain,expected in [('SYN',6),('ND',4)]:
            grouped={}
            for arm in ARMS:
                selected=[r for r in rows if (r['seed'],r['domain'],r['arm'])==(seed,domain,arm)]
                grouped[arm]={r['scene']:r for r in selected}
                assert len(selected)==len(grouped[arm])==expected
                means.append(dict(seed=seed,domain=domain,arm=arm,conditions=expected,
                    **{m:sum(r[m] for r in selected)/expected for m in ('psnr_rgb','ssim_rgb','edge_mae','flicker','temporal_error')}))
            current={r['arm']:r for r in means if (r['seed'],r['domain'])==(seed,domain)}
            for ref,limit in [('past3',.95),('center_self',.95),('F0_FB',1.),('F0_DIS',1.)]:
                assert grouped['center3'].keys()==grouped[ref].keys()
                delta=current['center3']['psnr_rgb']-current[ref]['psnr_rgb']
                worst=min(grouped['center3'][s]['psnr_rgb']-grouped[ref][s]['psnr_rgb'] for s in grouped[ref])
                ratio=current['center3']['temporal_error']/current[ref]['temporal_error']
                comparisons.append(dict(seed=seed,domain=domain,candidate='center3',reference=ref,
                    PSNR_delta=delta,worst_sequence_PSNR_delta=worst,Etemp_ratio=ratio,Etemp_limit=limit,
                    mean_quality_passed=delta>=-.05,worst_quality_passed=worst>=-.5,
                    passed=delta>=-.05 and worst>=-.5 and ratio<=limit))
    assert len(means)==24 and len(comparisons)==16
    passed=all(r['passed'] for r in comparisons)
    write(out/'means.json',means)
    write(out/'gates.json',dict(at=stamp(),comparisons=comparisons,complete=passed,
        allowed_separate_cost_or_structure_registration=passed,allowed_training_JOD_or_external=False,
        candidate_future_frames=1,baseline_past3_future_frames=0,
        zero_added_latency=False,independent_final=False,established_method_or_innovation=False))
    text=['# D1全部内部中心结果','',
          '重复DEV六SYN条件（3源）/四ND源×两模型seed；t=2..T−2，非完整视频首尾。center3读一未来，不是零延迟或新方法。','',
          '|域|seed|输出|PSNR|SSIM|edge|Etemp|','|---|---:|---|---:|---:|---:|---:|']
    for r in means:
        text.append(f"|{r['domain']}|{r['seed']}|{r['arm']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['edge_mae']:.6f}|{r['temporal_error']:.6f}|")
    text+=['','|域|seed|候选|参照|PSNR差|最差序列差|Etemp比|门限|过|',
           '|---|---:|---|---|---:|---:|---:|---:|---|']
    for r in comparisons:
        text.append(f"|{r['domain']}|{r['seed']}|center3|{r['reference']}|{r['PSNR_delta']:+.6f}|{r['worst_sequence_PSNR_delta']:+.6f}|{r['Etemp_ratio']:.6f}|{r['Etemp_limit']:.2f}|{r['passed']}|")
    text+=['',f'完整通过={passed}；只可能允许另登记成本/结构研究，不自动训练或作为论文主方法。']
    (out/'tables.md').write_text('\n'.join(text)+'\n')
