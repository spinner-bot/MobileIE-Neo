"""Export only normally completed R85 baseline costs, without model execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/4.temporal-consistency'
SOURCE = BASE/'process/exploration-r85/cpu001'
DEST = BASE/'report/exploration-r85-baseline-cost'


def read(name):
    return json.loads((SOURCE/name).read_text())


def rows(name):
    return [json.loads(s) for s in (SOURCE/name).read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runtime = read('runtime.json')
    assert runtime['completed_normally'] and not runtime['CUDA_initialized']
    assert (runtime['backbone_image_frames'], runtime['half_FB_pairs'], runtime['optimizer_updates']) == (3200, 1500, 0)
    warm, measured, qualified = rows('warmup.jsonl'), rows('measured.jsonl'), rows('qualification.jsonl')
    assert (len(warm), len(measured), len(qualified)) == (80, 120, 40)
    assert all(r['passed'] and r['normalized_max_abs_difference'] <= 1e-5 for r in qualified)
    assert sum(r['frames'] for r in warm+measured) == 3200
    assert sum(r['frames']-1 for r in warm+measured if r['mode'] == 'strongrec') == 1500
    assert all(r['frames'] == len(r['per_frame']) == 16 for r in warm+measured)
    assert len({(r['seed'], r['scene'], r['mode']) for r in qualified}) == 40
    assert len({(r['seed'], r['scene'], r['mode'], r['repeat']) for r in measured}) == 120
    summary = read('summary.json'); config = read('config.json')
    assert len(summary['conditions']) == 20 and len(summary['groups']) == 4
    frozen_path = BASE/'process/exploration-r85/source-freeze001.json'
    frozen = json.loads(frozen_path.read_text())
    assert all(digest(ROOT/name) == value for name, value in frozen['source_sha256'].items())
    protocol = BASE/'docs/exploration-r85-baseline-pipeline-cost-protocol.md'
    assert digest(protocol) == frozen['protocol_sha256']
    table = ['# Current F0 and strong-rec resident CPU pipeline costs', '',
        'Both frozen R44 seeds, every registered condition, first16 frames only. '
        'Each mode/condition uses two full warmups then three full measured runs, reset at each run. '
        'The primary number is the mean across conditions of each three-run median ms/frame. '
        'Sizes are native; CPU is shared and nonexclusive. No handset latency or new quality claim.', '',
        f"CPU: {config['cpu_model']}; Torch {config['torch_version']} intra2/inter1; "
        f"OpenCV {config['OpenCV_version']} thread1; FP32; batch1. "
        'Includes input/output conversion, actual backbone, post motion/warp/blend and output sink copy. '
        'Excludes model loading, file/camera/display I/O, sink allocation and qualification comparison.', '',
        '|Domain|Native H×W|Seed|Raw ms|Strong-rec ms|Mean per-condition rec/raw|Ratio range across conditions|',
        '|---|---|---:|---:|---:|---:|---|']
    tex = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Current 4,075-parameter F0 and its strong-rec resident CPU pipeline. '
        r'Two warmups and three measured sixteen-frame runs per condition, both seeds. '
        r'Values are means of per-condition median ms/frame. Torch intra2/inter1, OpenCV1, '
        r'FP32, batch1; shared nonexclusive server CPU. Includes conversions, actual backbone, '
        r'motion/warp/fusion and output copy; excludes file/camera/display I/O. '
        r'Quality tables use full sequences; these cost prefixes are not handset measurements.}',
        r'\label{tab:current-baseline-cpu}', r'\begin{tabular}{llrrr}', r'\toprule',
        r'Domain/size & Seed & Raw ms & Rec. ms & Mean paired ratio\\', r'\midrule']
    for g in summary['groups']:
        syn = g['domain'] == 'davis_train_pilot'
        label, shape = ('SYN', '480×854') if syn else ('ND', '288×512')
        raw, rec = g['mean_condition_median_raw_ms'], g['mean_condition_median_recursive_ms']
        ratio = g['mean_condition_median_ratio']
        table.append(f"|{label}|{shape}|{g['seed']}|{raw:.3f}|{rec:.3f}|{ratio:.3f}|"
                     f"{g['minimum_condition_ratio']:.3f}–{g['maximum_condition_ratio']:.3f}|")
        size_tex = r'$480\times854$' if syn else r'$288\times512$'
        tex.append(f"{label} {size_tex} & {g['seed']} & {raw:.3f} & {rec:.3f} & {ratio:.3f}"+r'\\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    table += ['', '## All condition medians and repeat extrema', '',
        'Brackets are minimum/maximum of three measured sequence means, not confidence intervals. '
        'Ratios divide matched condition medians; the mean of these ratios need not equal a ratio of domain means.', '',
        '|Seed|Condition|Raw median [min,max] ms|Rec. median [min,max] ms|Ratio|Rec. RGB state B|Shared output sink B|',
        '|---:|---|---|---|---:|---:|---:|']
    for row in summary['conditions']:
        raw, rec = row['modes']['raw'], row['modes']['strongrec']
        table.append(f"|{row['seed']}|{row['scene']}|{raw['median_ms']:.3f} "
            f"[{raw['minimum_ms']:.3f},{raw['maximum_ms']:.3f}]|{rec['median_ms']:.3f} "
            f"[{rec['minimum_ms']:.3f},{rec['maximum_ms']:.3f}]|{row['recursive_to_raw_median_ratio']:.3f}|"
            f"{rec['persistent_state_bytes']}|{rec['output_sink_bytes']}|")
    table += ['', 'Raw has zero temporal state. Rec. stable state is two native FP32 RGB frames; '
        'startup aliases the two buffers. This excludes flow/workspace/current images/CNN activations. '
        'The common sixteen-frame host sink is reported separately and is not model temporal state. '
        'All first-frame/nonfirst-frame samples and host loads remain in original JSONL.', '',
        'All 40 warmup equivalence checks pass against the exact same frozen-weight raw/recursive '
        'prediction prefixes; these checks add no CNN calls. No experimental kernel/thread/backend '
        'selection follows timing results. Existing failed-candidate STOP rules remain unchanged.']
    DEST.mkdir(exist_ok=False)
    (DEST/'tables.md').write_text('\n'.join(table)+'\n')
    (DEST/'baseline-cost-table.tex').write_text('\n'.join(tex)+'\n')
    sources = list(SOURCE.glob('*.json'))+list(SOURCE.glob('*.jsonl'))+[frozen_path, protocol, Path(__file__).resolve()]
    audit = dict(completed=True, new_CNN_calls_in_export=0, new_optimizer_updates=0,
        measured_sequences=120, warmup_sequences=80, passed_equivalence_checks=40,
        original_backbone_image_frames=3200, original_half_FB_pairs=1500,
        new_quality_metrics=0, phone_measured=False, manuscript_rebuilt=False,
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources})
    (DEST/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(exported=True, measured_sequences=120, new_CNN_calls=0)))


if __name__ == '__main__':
    main()
