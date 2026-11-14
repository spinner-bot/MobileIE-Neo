"""Report both source-context seeds and every preregistered reference."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from dynamic_training_r34 import BASE, ROOT


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    out = BASE/'report/exploration-r53-source-context-reference'
    out.mkdir(exist_ok=False)
    rows, sources, contexts = [], [], []
    for seed in [0, 1]:
        for experiment, prefix in [(53, 'source_context'), (47, 'short_context')]:
            train = BASE/f'results/exploration-r{experiment}-stage001-seed{seed}'
            run = BASE/f'results/exploration-r{experiment}-stage001-eval-seed{seed}'
            training = json.loads((train/'runtime.json').read_text())
            runtime = json.loads((run/'runtime.json').read_text())
            assert training['completed_normally'] and training['steps'] == 8000
            assert runtime['completed_normally'] and runtime['rows'] == runtime['neural_sequences'] == 20
            assert runtime['per_frame_rows'] == 896 and not runtime['cuda_initialized']
            new = lines(run/'per_sequence.jsonl')
            assert len(new) == 20 and len(lines(run/'per_frame.jsonl')) == 896
            rows.extend([{**r, 'method': prefix+'_'+r['method']} for r in new])
            sources.extend([train/'runtime.json', train/'fastdvd-step8000.json',
                            run/'config.json', run/'runtime.json', run/'per_sequence.jsonl'])
            if experiment == 53:
                path = train/'context-statistics.json'
                contexts.append(dict(seed=seed, **json.loads(path.read_text())))
                sources.append(path)
        reference = BASE/f'results/exploration-r45-stage001-eval-seed{seed}/per_sequence.jsonl'
        old = lines(reference)
        assert len(old) == 90
        rows.extend([{**r, 'method': 'MobileIE_'+r['method']} for r in old
                     if r['method'] in ['raw', 'fixed_025', 'fixed_05', 'rgb']])
        sources.append(reference)
    assert len(rows) == 160
    groups = defaultdict(list)
    for r in rows:
        groups[r['domain'], r['train_seed'], r['method']].append(r)
    keys = ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error']
    means = {k: {m: float(np.mean([r[m] for r in group])) for m in keys}
             for k, group in groups.items()}
    summary = [dict(domain=d, seed=s, method=m, **v) for (d, s, m), v in means.items()]
    effects = []
    for domain, seed, method in means:
        if method != 'source_context_window':
            continue
        for reference in ['short_context_window', 'source_context_five_current',
                          'MobileIE_raw', 'MobileIE_fixed_05', 'MobileIE_rgb']:
            value, ref = means[domain, seed, method], means[domain, seed, reference]
            effects.append(dict(domain=domain, seed=seed, method=method, reference=reference,
                                psnr_delta=value['psnr_rgb']-ref['psnr_rgb'],
                                temporal_ratio=value['temporal_error']/ref['temporal_error']))
        current, previous = means[domain, seed, 'source_context_five_current'], means[domain, seed, 'short_context_five_current']
        effects.append(dict(domain=domain, seed=seed, method='source_context_five_current',
                            reference='short_context_five_current',
                            psnr_delta=current['psnr_rgb']-previous['psnr_rgb'],
                            temporal_ratio=current['temporal_error']/previous['temporal_error']))
    write(out/'domain_seed_summary.json', summary)
    write(out/'paired_conditions.json', rows)
    write(out/'context_effects.json', effects)
    write(out/'training_context_statistics.json', contexts)
    table = ['# R53 源上下文适配：完整有限预算参考', '',
             '新训练双seed各8000更新；新CPU40行/40CNN/1792逐帧。全部结果保留。',
             'R53保留R47四个监督目标，但额外看到两侧源输入；不能将差异仅归因为消除重复，也不是等信息控制。',
             'FastDVDnet 2479096参数、公开AWGN预训练与三DEV源重叠、window未来2帧；MobileIE4075参数。',
             '同更新数不等于同计算量；有限适配不证明充分收敛。时间列是GT变化残差指标，不是延迟。', '',
             '|域|seed|方法|PSNR|SSIM|时间误差↓|edge↓|',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in summary:
        table.append(f"|{r['domain']}|{r['seed']}|{r['method']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    table += ['', '这是既有视频模型的训练设计控制，不建立本项目新颖性、独立最终/自然夜间泛化或手机实测结论。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'sources.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    write(out/'audit.json', dict(at=datetime.now().astimezone().isoformat(), completed_normally=True,
          new_training_updates=16000, new_quality_rows=40, new_neural_sequences=40,
          new_per_frame_rows=1792, comparison_rows=160, new_method=False,
          extra_input_observations_disclosed=True, unseen_final=False, natural_night=False, phone_measured=False))


if __name__ == '__main__':
    main()
