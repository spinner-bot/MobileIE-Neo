"""Descriptive registered CPU stages; shared filtering counted once, not per seed."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/4.temporal-consistency'


def main():
    source = BASE / 'results/exploration-r70-classical-attempt001'
    runtime = json.loads((source / 'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['quality_rows'] == 80
    read = lambda name: [json.loads(x) for x in (source / name).read_text().splitlines()]
    filters = read('filtered_manifest.jsonl'); stages = read('stage_seconds.jsonl')
    assert len(filters) == len({(r['scene'], r['mode']) for r in filters}) == 20
    assert len(stages) == 40
    groups = defaultdict(list)
    for row in filters:
        value = row['filtering']; frames, height, width, _ = value['shape']
        groups[row['domain'], row['mode'], None, height, width].append(dict(
            native_filter=value['native_filter_seconds'] / frames,
            full_filter=value['full_filter_stage_seconds'] / frames))
    frame_counts = {(r['scene'], r['mode']): r['shape'][0] for r in filters}
    for row in stages:
        value = row['shared_input_filtering']; _, height, width, _ = value['shape']
        frames = frame_counts[row['scene'], row['mode']]
        groups[row['domain'], row['mode'], row['seed'], height, width].append(dict(
            neural=row['neural_seconds'] / frames, classic_post=row['classic_post_seconds'] / frames))
    rows = []
    for (domain, mode, seed, height, width), entries in groups.items():
        rows.append(dict(domain=domain, mode=mode, seed=seed, height=height, width=width,
                         sequence_conditions=len(entries), seconds_per_frame={field: dict(
                             minimum=float(np.min([r[field] for r in entries])),
                             median=float(np.median([r[field] for r in entries])),
                             maximum=float(np.max([r[field] for r in entries]))) for field in entries[0]}))
    out = BASE / 'report/exploration-r70-classical-cost'; out.mkdir(exist_ok=False)
    (out / 'stage_costs.json').write_text(json.dumps(rows, indent=2) + '\n')
    table = ['# R70 CPU2阶段描述性耗时', '', '同一过滤结果跨seed复用，过滤仅计一次。共享服务器、未独立预热重复计时；不是手机/实时、完整FLOP或因果等待界。质量/IO未归入未明确包含的阶段，原完整过滤阶段含数组IO。', '',
             '|域|mode|seed|尺寸|序列条件|阶段|秒/帧中位|最小|最大|', '|---|---|---:|---|---:|---|---:|---:|---:|']
    for r in rows:
        for field, value in r['seconds_per_frame'].items():
            table.append(f"|{r['domain']}|{r['mode']}|{r['seed']}|{r['height']}×{r['width']}|{r['sequence_conditions']}|{field}|{value['median']:.6f}|{value['minimum']:.6f}|{value['maximum']:.6f}|")
    (out / 'tables.md').write_text('\n'.join(table) + '\n')
    audit = dict(at=datetime.now().astimezone().isoformat(), completed=True, shared_filter_sequences=20,
                 neural_and_post_sequences=40, new_CNN=0, new_SGD=0, new_GPU=0, phone=False,
                 source_sha256={str((source / n).relative_to(ROOT)): hashlib.sha256((source / n).read_bytes()).hexdigest()
                                for n in ['runtime.json', 'filtered_manifest.jsonl', 'stage_seconds.jsonl']},
                 scope='Only existing registered descriptive CPU2 stages; no new timing runs or cost extrapolation.')
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(groups=len(rows), shared_filters=20, new_runs=0)))


if __name__ == '__main__':
    main()
