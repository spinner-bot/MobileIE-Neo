"""Apply preregistered necessary spatial bounds to every existing reference seed."""
from datetime import datetime
import json

import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def number(value): return '+inf' if value is None else f'{value:.6f}'


def main():
    run = BASE/'results/exploration-r57-current-ray-attempt001'
    runtime = json.loads((run/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['frames'] == 64 and runtime['CNN'] == runtime['SGD'] == 0
    values = json.loads((run/'per_sequence.json').read_text())
    frames = [json.loads(line) for line in (run/'per_frame.jsonl').read_text().splitlines()]
    assert len(values) == 4 and len(frames) == 64
    oracle = {r['scene']: r['psnr_rgb'] for r in values}
    mean = None if None in oracle.values() else float(np.mean(list(oracle.values())))
    references, conditions, means, sources = {}, [], [], [run/'runtime.json', run/'per_sequence.json', run/'config.json']
    for seed in [0, 1]:
        reference = BASE/f'results/exploration-r45-stage001-eval-seed{seed}/per_sequence.jsonl'
        assert json.loads((reference.parent/'runtime.json').read_text())['completed_normally']
        rows = [json.loads(line) for line in reference.read_text().splitlines()]
        sources.append(reference)
        for method in ['raw', 'fixed_05']:
            selected = [r for r in rows if r['domain'] == 'paired_real_ND_DEV' and r['method'] == method]
            assert {r['scene'] for r in selected} == set(oracle)
            ref_mean = float(np.mean([r['psnr_rgb'] for r in selected]))
            delta = None if mean is None else mean-ref_mean
            means.append(dict(seed=seed, method=method, reference_psnr=ref_mean,
                              oracle_psnr=mean, delta_psnr=delta, necessary_mean_pass=delta is None or delta >= -.05))
            for r in selected:
                q = oracle[r['scene']]; delta = None if q is None else q-r['psnr_rgb']
                references[r['scene'], seed, method] = r['psnr_rgb']
                conditions.append(dict(scene=r['scene'], seed=seed, method=method,
                                       reference_psnr=r['psnr_rgb'], oracle_psnr=q, delta_psnr=delta,
                                       necessary_source_pass=delta is None or delta >= -.25))
    assert len(means) == 4 and len(conditions) == 16
    eligible = all(r['necessary_mean_pass'] for r in means) and all(r['necessary_source_pass'] for r in conditions)
    out = BASE/'report/exploration-r57-current-ray'; out.mkdir(exist_ok=False)
    write(out/'conditions.json', conditions); write(out/'domain_summary.json', means)
    write(out/'decision.json', dict(at=datetime.now().astimezone().isoformat(),
          necessary_spatial_condition_passed=eligible,
          action='not excluded by this necessary spatial bound; no training/novelty claim' if eligible
                 else 'exclude strict current-ray output as quality-preserving candidate; no training or local grid',
          new_method=False, deployable=False, independent_final=False, temporal_evaluated=False,
          scope='Encoded-RGB current-pixel shared positive gain with final clip only; excludes neither neighbour denoising nor general Retinex.'))
    table = ['# R57 当前像素正增益类：不可部署空间上限', '',
             'GT逐像素求解，0CNN/0SGD/0GPU，仅既有四ND DEV的64帧；不是增强模型或时间/手机成绩。',
             '每行原尺度全RGB平均逐帧PSNR；+inf仅在实际零误差时出现。', '',
             '|源|GT oracle|raw seed0|rec .5 seed0|raw seed1|rec .5 seed1|',
             '|---|---:|---:|---:|---:|---:|']
    for scene, q in oracle.items():
        old = [references[scene, seed, method] for seed in [0, 1] for method in ['raw', 'fixed_05']]
        table.append('|'+scene+'|'+number(q)+'|'+ '|'.join(number(v) for v in old)+'|')
    table += ['', f'全部必要空间门槛通过：{eligible}。均值容差.05dB/逐源.25dB按事前协议，不由结果改写。',
              '即使通过也不证明可学习/可迁移/有用历史或创新；失败仅排除严格输出类，不否定所有颜色结构/Retinex。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
    print(json.dumps(dict(necessary_spatial_condition_passed=eligible, oracle_mean_psnr=mean,
                         comparisons=16, domain_comparisons=4)))


if __name__ == '__main__': main()
