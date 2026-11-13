"""Complete-only published clean reference, common CPU metrics and caveats."""
import json
from datetime import datetime

import numpy as np

from dynamic_training_r34 import BASE, digest
from fetch_wenzhou_samples_r52 import write


def lines(path): return [json.loads(row) for row in path.read_text().splitlines()]


def main():
    run = BASE/'results/exploration-r62-stablellve-common-attempt001'
    terminal = json.loads((run/'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['sequences'] == 20 and terminal['CNN_center_outputs'] == 448
    rows = lines(run/'per_sequence.jsonl'); frames = lines(run/'per_frame.jsonl')
    assert len(rows) == 20 and len(frames) == 896 and len(lines(run/'prediction_manifest.jsonl')) == 20
    assert all(r['project_updates'] == 0 and r['seed'] is None for r in rows)
    keys = ['psnr_rgb', 'ssim_rgb', 'flicker', 'temporal_error', 'edge_mae']
    summary = []
    for domain, count in [('davis_train_pilot', 6), ('paired_real_ND_DEV', 4)]:
        for method in ['raw', 'recursive_self05']:
            selected = [r for r in rows if r['domain'] == domain and r['method'] == method]
            assert len(selected) == count
            summary.append(dict(domain=domain, method='StableLLVE_clean_'+method, rows=count,
                    parameters=4316259, seed=None, project_updates=0, future_wait_max=0,
                    **{key:float(np.mean([r[key] for r in selected])) for key in keys}))
    out = BASE/'report/exploration-r62-stablellve-common'; out.mkdir(exist_ok=False)
    write(out/'domain-summary.json', summary)
    write(out/'conditions.json', dict(at=datetime.now().astimezone().isoformat(),
          checkpoint='Original public synthetic-clean StableLLVE; no project adaptation or claimed seed',
          parameters=4316259, future_wait_max=0, training_updates=0, CPU_threads=2,
          observation_count=448, CNN_sequences=10, output_sequences=20, per_frame_rows=896,
          format='Original BGR float64/255->FP32, model internal odd-size handling; output clipped RGB float32',
          post='Own raw half-FB and classic recursive .5, first raw; motion/history cost additional',
          comparison='Same actual DEV arrays/reference-flow/mask/CPU metrics as R44/R53/R56; unequal training/pretraining/compute.',
          scope='Published clean transfer reference, no noisy-paper reproduction/SOTA or architecture/final/phone rank.'))
    text = ['# R62 原公开StableLLVE clean权重：共同CPU完整参考', '',
            '同当前6SYN/4ND资料、448观察；公开clean预训练，项目更新0/不虚构seed，非原论文noisy设置完整复现。',
            '与已适配模型训练条件/计算不同，不据迁移结果作一般架构或SOTA排名。经典递归含额外运动/历史成本。', '',
            '|域|输出|源数|PSNR|SSIM|时间误差|Edge|', '|---|---|---:|---:|---:|---:|---:|']
    for row in summary:
        text.append('|'+ '|'.join([row['domain'],row['method'],str(row['rows'])]+[f'{row[key]:.6f}' for key in ['psnr_rgb','ssim_rgb','temporal_error','edge_mae']])+'|')
    (out/'tables.md').write_text('\n'.join(text)+'\n')
    write(out/'audit.json', dict(at=datetime.now().astimezone().isoformat(), completed_normally=True,
          quality_rows=20, per_frame_rows=896, CNN_centers=448, summary_rows=4,
          source_sha256={name:digest(run/name) for name in ['config.json','runtime.json','per_sequence.jsonl','per_frame.jsonl','prediction_manifest.jsonl']},
          training_budget_matched=False, original_published_weights=True, independent_final=False, new_method=False))


if __name__ == '__main__': main()
