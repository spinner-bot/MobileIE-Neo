"""All completed F1 means and sixteen original necessary contrasts."""
from f1_io import ARMS,stamp,write


def summarize(rows,out):
    assert len(rows)==120
    means,gates=[],[]
    for seed in (0,1):
        for domain,n in [('SYN',6),('ND',4)]:
            groups={}
            for arm in ARMS:
                selected=[r for r in rows if (r['seed'],r['domain'],r['arm'])==(seed,domain,arm)]
                groups[arm]={r['scene']:r for r in selected};assert len(selected)==len(groups[arm])==n
                means.append(dict(seed=seed,domain=domain,arm=arm,conditions=n,
                    **{m:sum(r[m] for r in selected)/n for m in ('psnr_rgb','ssim_rgb','edge_mae','flicker','temporal_error')}))
            current={r['arm']:r for r in means if (r['seed'],r['domain'])==(seed,domain)}
            for ref,limit in [('naive_future',.97),('F0_DIS',.95),('F0_FB',1.),('D1_center3',1.)]:
                assert groups['innovation_lag1'].keys()==groups[ref].keys()
                delta=current['innovation_lag1']['psnr_rgb']-current[ref]['psnr_rgb']
                worst=min(groups['innovation_lag1'][s]['psnr_rgb']-groups[ref][s]['psnr_rgb'] for s in groups[ref])
                ratio=current['innovation_lag1']['temporal_error']/current[ref]['temporal_error']
                gates.append(dict(seed=seed,domain=domain,candidate='innovation_lag1',reference=ref,
                    PSNR_delta=delta,worst_sequence_PSNR_delta=worst,Etemp_ratio=ratio,Etemp_limit=limit,
                    mean_quality_passed=delta>=-.05,worst_quality_passed=worst>=-.5,
                    passed=delta>=-.05 and worst>=-.5 and ratio<=limit))
    assert len(means)==24 and len(gates)==16
    passed=all(r['passed'] for r in gates)
    write(out/'means.json',means)
    write(out/'gates.json',dict(at=stamp(),comparisons=gates,complete=passed,
        allowed_separate_full_cost_and_perceptual_registration=passed,allowed_training_or_external=False,
        candidate_future_frames=1,zero_added_wait=False,calibrated_RTS_or_established_innovation=False,independent_final=False))
    text=['# F1完整内部中心结果','',
          '原十DEV条件×两F0 seed，t=2..T−2；创新与直接混合均一未来，原因果强post单列。非完整视频/独立最终/手机计时。','',
          '|域|seed|输出|PSNR|SSIM|edge|Etemp|','|---|---:|---|---:|---:|---:|---:|']
    for r in means:text.append(f"|{r['domain']}|{r['seed']}|{r['arm']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['edge_mae']:.6f}|{r['temporal_error']:.6f}|")
    text+=['','|域|seed|参考|PSNR差|最差序列差|Etemp比|门限|过|','|---|---:|---|---:|---:|---:|---:|---|']
    for r in gates:text.append(f"|{r['domain']}|{r['seed']}|{r['reference']}|{r['PSNR_delta']:+.6f}|{r['worst_sequence_PSNR_delta']:+.6f}|{r['Etemp_ratio']:.6f}|{r['Etemp_limit']:.2f}|{r['passed']}|")
    text+=['',f'完整通过={passed}；只可能允许另登记完整费用/感知，不自动训练或建立创新。']
    (out/'tables.md').write_text('\n'.join(text)+'\n')
