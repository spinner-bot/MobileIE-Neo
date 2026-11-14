"""Extend manuscript sources using complete R72 REDS development evidence only."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/4.temporal-consistency'


def main():
    primary_path = ROOT/'paper/evidence-snapshot.json'; primary = json.loads(primary_path.read_text())
    assert primary['R70_complete_classical_included']
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha for name, sha in primary['source_sha256'].items())
    sources = {}

    def read(name):
        path = BASE/name; sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    assert read('process/exploration-r72/finalize001/runtime.json')['completed_normally']
    data = read('results/exploration-r72-data001/runtime.json')
    assert data['completed_normally'] and (data['conditions'], data['decoded_source_frames'], data['degraded_observations']) == (8, 256, 512)
    config = read('results/exploration-r72-data001/config.json')
    assert config['scenes'] == ['000', '009', '019', '029'] and config['frames_per_scene'] == 64
    assert config['photons'] == [20, 80] and config['jpeg'] is False and config['source_fps'] == 24
    data_index = read('results/exploration-r72-data001/index.json')
    assert len(data_index['scenes']) == 8
    assert all(r['gt']['shape'] == r['low']['shape'] == [64, 720, 1280, 3] for r in data_index['scenes'])
    for seed in [0, 1]:
        neural = read(f'results/exploration-r72-neural-seed{seed}/runtime.json')
        quality = read(f'results/exploration-r72-quality-seed{seed}/runtime.json')
        assert neural['completed_normally'] and (neural['CNN_sequences'], neural['CNN_centers']) == (40, 2560)
        assert quality['completed_normally'] and (quality['quality_rows'], quality['frame_rows']) == (72, 4608)
        read(f'results/exploration-r72-neural-seed{seed}/sources.json')
    perception = read('results/exploration-r72-perceptual001/runtime.json')
    assert perception['completed_normally'] and (perception['perceptual_rows'], perception['comparison_frames']) == (156, 9984)
    assert perception['source_fps'] == 24 and perception['unique_GT_controls'] == 4
    metric = read('results/exploration-r72-perceptual001/metric-source.json')
    assert metric['package_version'] == '0.5.7' and metric['playback_fps'] == 24
    review = read('report/exploration-r72-reds-contacts/actual-visual-review.json')
    assert review['actual_reviewed'] and review['contacts'] == 16
    rendered = read('report/exploration-r72-reds-contacts/runtime.json')
    assert rendered['completed_normally'] and rendered['contacts'] == 16
    audit = read('report/exploration-r72-reds-joint/audit.json')
    assert audit['completed'] and audit['perceptual_included'] and not audit['independent_final']
    assert (audit['quality_rows'], audit['condition_means']) == (152, 38)
    rows = read('report/exploration-r72-reds-joint/condition_means.json')
    assert len(rows) == 38 and all(r['source_videos'] == r['conditions'] == 4 for r in rows)
    pairs = read('report/exploration-r72-reds-joint/recursive_pairs.json')
    assert len(pairs['pairs']) == 64 and {r['family'] for r in pairs['summary']} == {'MobileIE_MSE', 'MobileIE_L1', 'Nano', 'StableLLVE_L1'}
    source_path = ROOT/pairs['source']; sources[pairs['source']] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert sources[pairs['source']] == pairs['source_sha256']
    lineage = read('process/exploration-r72/pretraining-provenance001/lineage.json')
    assert len(lineage['rows']) == 10 and lineage['local_adaptation_training']['REDS_used_by_project_adaptation'] is False
    assert not any(r['public_start_training_pixels_verified'] for r in lineage['rows'])
    for name, sha in lineage['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
        sources[name] = sha
    index = {(r['family'], r['seed'], r['method'], r['photons']): r for r in rows}
    lines = [r'\begin{table}[!htbp]', r'\centering\small\setlength{\tabcolsep}{3pt}',
        r'\caption{Cross-source synthetic development on four REDS sharp validation clips at native $720\times1280$, first 64 frames, with two fixed photon conditions. Every row averages the same four source videos; both adapted seeds are retained. Rec. adds the fixed classical motion/history reference to the corresponding raw output. ColorVideoVDP uses source-rate 24\,fps and the same standard FHD display/replicate padding; these scores are not pooled with the 30\,fps pilot tables. This is a custom low-light development protocol, not the official REDS task or independent final testing.}',
        r'\label{tab:external-reds}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'Network/output & Seed & 20: PSNR & $E_{\rm temp}$ & JOD & 80: PSNR & $E_{\rm temp}$ & JOD\\', r'\midrule']
    count = 0
    for family, label in [('Input', 'Input'), ('MobileIE_MSE', 'MobileIE MSE'), ('MobileIE_L1', 'MobileIE L1'),
                          ('Nano', 'Nano'), ('StableLLVE_L1', 'U-Net L1'), ('FastDVDnet', 'FastDVDnet')]:
        methods = ['input'] if family == 'Input' else ['window'] if family == 'FastDVDnet' else ['raw', 'recursive_self05']
        for method in methods:
            for seed in ([None] if family == 'Input' else [0, 1]):
                selected = [index[family, seed, method, photons] for photons in [20, 80]]
                cells = [f'{r[field]:.3f}' for r in selected for field in ['psnr_rgb', 'temporal_error', 'JOD']]
                output = label+(' + rec.' if method == 'recursive_self05' else ' window' if method == 'window' else '')
                lines.append(f'{output} & {"--" if seed is None else seed} & '+' & '.join(cells)+r'\\'); count += 1
    assert count == 19
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    name = 'external-reds-table.tex'; (ROOT/'paper/generated'/name).write_text('\n'.join(lines)+'\n')
    record = dict(created_at=datetime.now().astimezone().isoformat(), source_sha256=sources,
          generated_tables=[name], R72_complete=True, table_rows=19, quality_rows=152, perceptual_rows=156,
          source_videos=4, source_fps=24, public_pretraining_pixels_verified=False,
          new_method=False, handset_measured=False, independent_final=False)
    child = ROOT/'paper/external-evidence-snapshot.json'; child.write_text(json.dumps(record, indent=2)+'\n')
    primary['source_sha256'].update(sources)
    primary['source_sha256'][str(child.relative_to(ROOT))] = hashlib.sha256(child.read_bytes()).hexdigest()
    primary['generated_tables'] = list(dict.fromkeys(primary['generated_tables']+[name]))
    primary.update(created_at=record['created_at'], R72_complete_external_development_included=True,
                   supplemental_external_snapshot='paper/external-evidence-snapshot.json')
    primary_path.write_text(json.dumps(primary, indent=2)+'\n')
    print(json.dumps(dict(table_rows=count, complete_source_files=len(sources), source_fps=24, new_CNN=0, new_SGD=0)))


if __name__ == '__main__': main()
