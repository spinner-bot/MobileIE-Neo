"""Complete-only unpaired Night tables; descriptive proxies never quality ranks."""
import json
from datetime import datetime

import numpy as np

from drv_references_r59 import NETWORKS
from dynamic_training_r34 import BASE
from fetch_wenzhou_samples_r52 import write


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def main():
    run = BASE/'results/exploration-r60-wenzhou-reference-attempt001'
    terminal = json.loads((run/'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['sequences'] == 96 and terminal['CNN_center_outputs'] == 6144
    rows = lines(run/'per_sequence.jsonl'); assert len(rows) == 96
    assert len(lines(run/'per_frame.jsonl')) == 12288
    assert len(lines(run/'prediction_manifest.jsonl')) == 96
    assert len(lines(run/'model_sources.jsonl')) == 8
    inputs = lines(run/'input_per_sequence.jsonl'); assert len(inputs) == 6
    scenes = {row['scene'] for row in inputs}; assert len(scenes) == 6
    for scene in scenes:
        accepted = [row['shared_input_flow_valid_fraction'] for row in rows if row['scene'] == scene]
        assert len(accepted) == 16 and max(accepted)-min(accepted) < 1e-12
    out = BASE/'report/exploration-r60-wenzhou'; out.mkdir(exist_ok=True)
    assert not (out/'domain-summary.json').exists()
    fields = ['mean_luma_0to255', 'low_endpoint_fraction', 'high_endpoint_fraction', 'sobel_energy_proxy',
              'shared_input_flow_valid_fraction', 'shared_input_flow_warp_luma_MAE_proxy', 'warp_proxy_valid_frames']
    summary = []
    for name in NETWORKS:
        for method in ['raw', 'recursive_self05']:
            selected = [row for row in rows if row['reference'] == name and row['method'] == method]
            assert len(selected) == 6 and {row['scene'] for row in selected} == scenes
            group = dict(reference=name, method=method, sources=6, parameters=selected[0]['parameters'],
                         future_wait_max=selected[0]['future_wait_max'])
            for prefix in ['', 'full128_']:
                for field in fields:
                    values = [row[prefix+field] for row in selected if row[prefix+field] is not None]
                    group[prefix+field] = float(np.mean(values)) if values else None
                    group[prefix+field+'_nonempty_sources'] = len(values)
            summary.append(group)
    write(out/'domain-summary.json', summary)
    write(out/'conditions.json', dict(at=datetime.now().astimezone().isoformat(),
          sequence_rows=96, per_frame_rows=12288, CNN_sequences=48, CNN_centers=6144,
          unique_observations=768, unique_primary_targets=720, primary_frames=list(range(4, 124)),
          GT=False, true_flicker=False, independent_final=False, phone_latency=False, new_method=False,
          aggregation='Equal six-source means; all sources/seeds retained. Zero-valid metrics remain null.',
          flow='Shared original-input half-FB for all outputs; no GT; not posterior own-output flow.',
          costs='Recursive also needs flow/history; FastDVD five inputs/future2/different training; parameters alone are not speed.'))
    text = ['# R60 六段固定真实动态夜景：完整描述表', '',
            '无配对GT、质量排名、真实闪烁或独立最终/手机结论。所有8既有实例/16输出均保留。',
            '主目标每源4..123，6源等权；原输入流/接受域对全部输出相同，暗区可失准。',
            '低变化、低Sobel或高亮度不能独立称改善。full128与逐源/逐帧见JSON/原结果。', '',
            '|参考|输出|参数|未来|亮度|低端比例|高端比例|Sobel描述|公共流变化代理↓|公共接受率|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value): return 'null' if value is None else f'{value:.4f}'
    for row in summary:
        values = [row['reference'], row['method'], str(row['parameters']), str(row['future_wait_max'])]
        values += [fmt(row[field]) for field in fields[:6]]
        # fields puts acceptance before warp; table intentionally displays warp then acceptance.
        values[-2], values[-1] = values[-1], values[-2]
        text.append('|'+ '|'.join(values)+'|')
    (out/'tables.md').write_text('\n'.join(text)+'\n')
    write(out/'summary-audit.json', dict(at=datetime.now().astimezone().isoformat(), completed_normally=True,
          all96_outputs_retained=True, shared_evaluation_acceptance_verified=True, visual_review_required=True,
          summary_rows=len(summary), no_quality_rank=True))


if __name__ == '__main__': main()
