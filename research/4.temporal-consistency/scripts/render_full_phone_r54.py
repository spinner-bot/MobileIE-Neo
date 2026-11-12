"""All fixed contacts/previews and descriptive brightness traces for complete R54."""
import json
import time
import traceback
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest
from evaluate_full_phone_r54 import METHODS


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    cv2.setNumThreads(1)
    report = BASE/'report/exploration-r54-full-phone'
    report.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r54/previews'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(exist_ok=False)
    start, complete, videos, contacts, decoded = time.perf_counter(), False, [], [], 0
    try:
        index = BASE/'results/exploration-r49-full-phone-attempt001/index.json'
        metas = json.loads(index.read_text())['scenes']
        manifests, frames, sequences, sources = {}, {}, [], [index]
        for seed in [0, 1]:
            run = BASE/f'results/exploration-r54-full-phone-seed{seed}'
            runtime = json.loads((run/'runtime.json').read_text())
            assert runtime['completed_normally'] and runtime['rows'] == 9 and runtime['neural_sequences'] == 3
            assert runtime['neural_frames'] == 952 and not runtime['cuda_initialized']
            manifests[seed] = {(r['scene'], r['method']): r for r in lines(run/'prediction_manifest.jsonl')}
            assert len(manifests[seed]) == 9
            frames[seed] = lines(run/'per_frame.jsonl')
            assert len(frames[seed]) == 2856 and len(lines(run/'input_statistics.jsonl')) == 952
            sequences += lines(run/'per_sequence.jsonl')
            sources += [run/name for name in ['runtime.json', 'config.json', 'per_sequence.jsonl',
                        'per_frame.jsonl', 'input_statistics.jsonl', 'prediction_manifest.jsonl']]
        text = ['# Full existing phone DEV / frozen references', '',
                'Huawei329 / Pixel327 / iPhone296, all EOF frames, same three existing DEV sources. Rows: seed0/seed1. Columns: input, R44 raw, recursive .5, R45 RGB. No selected best seed or frames.', '',
                'No paired GT, PSNR/SSIM, user-study, true-flicker rank, new method or phone latency. Exposure/edge/warp statistics are diagnostic; smoother/darker output need not be better. Shared raw-flow is fixed within each seed, can differ across seeds, and has no GT-change compensation.', '',
                'Contacts always show 0 / floor(T/2) / T-1. MJPEG contains all frames at constant30FPS for inspection; original R49 timestamps remain the source and are not reproduced by preview timing. FP32 arrays are the numerical source, PNG/AVI are visualizations.', '']
        for meta in metas:
            assert digest(ROOT/meta['path']) == meta['sha256']
            with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
                low = archive['low'].copy()
            n = len(low); selected = [0, n//2, n-1]
            arrays = {}
            for seed in [0, 1]:
                for method in METHODS:
                    row = manifests[seed][meta['scene'], method]
                    assert digest(ROOT/row['path']) == row['sha256']
                    arrays[seed, method] = np.load(ROOT/row['path'], mmap_mode='r', allow_pickle=False)
                    assert list(arrays[seed, method].shape) == meta['shape']
            width, height = 1280, 404
            video = cache/(meta['scene']+'.avi')
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (width, height))
            assert writer.isOpened(), 'Preview codec unavailable; preserve failure and all numerical outputs.'
            panels = []
            try:
                for t in range(n):
                    panel = np.full((height, width, 3), 245, np.uint8)
                    for seed in [0, 1]:
                        images = [low[t]]+[arrays[seed, method][t] for method in METHODS]
                        labels = [f'seed{seed}: input']+[f'seed{seed}: {method}' for method in METHODS]
                        for col, (frame, label) in enumerate(zip(images, labels)):
                            image = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
                            panel[seed*202+22:(seed+1)*202, col*320:(col+1)*320] = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
                            cv2.putText(panel, label, (col*320+5, seed*202+16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
                    writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    if t in selected:
                        band = np.full((28, width, 3), 255, np.uint8)
                        cv2.putText(band, f"{meta['phone']} | frame {t}/{n-1} | unpaired DEV", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
                        panels.extend([band, panel.copy()])
            finally:
                writer.release()
            contact = report/(meta['scene']+'.png')
            assert len(panels) == 6
            assert cv2.imwrite(str(contact), cv2.cvtColor(np.concatenate(panels), cv2.COLOR_RGB2BGR))
            capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            count = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok: break
                    assert frame.shape == (height, width, 3)
                    count += 1
            finally:
                capture.release()
            assert count == n
            decoded += count
            contacts.append(dict(scene=meta['scene'], frames=selected, path=str(contact.relative_to(ROOT)), sha256=digest(contact)))
            videos.append(dict(scene=meta['scene'], path=str(video.relative_to(ROOT)), sha256=digest(video),
                               frames=count, preview_fps=30, sensor_timing_preserved=False))
            text += [f"## {meta['phone']}", '', f'![Fixed contacts]({contact.name})', '',
                     f'[All {n} preview frames](../../../../{video.relative_to(ROOT)})', '']
            print(json.dumps(dict(scene=meta['scene'], contact_frames=selected, video_frames=count)), flush=True)
        assert decoded == 952 and len(sequences) == 18
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
        for col, meta in enumerate(metas):
            run = BASE/'results/exploration-r54-full-phone-seed0'
            source_rows = [r for r in lines(run/'input_statistics.jsonl') if r['scene'] == meta['scene']]
            for seed in [0, 1]:
                ax = axes[seed, col]
                ax.plot([r['frame'] for r in source_rows], [r['mean_luma_0to255'] for r in source_rows], color='gray', linestyle='--', label='input')
                for method in METHODS:
                    chosen = [r for r in frames[seed] if r['scene'] == meta['scene'] and r['method'] == method]
                    assert len(chosen) == meta['shape'][0]
                    ax.plot([r['frame'] for r in chosen], [r['mean_luma_0to255'] for r in chosen], label=method)
                ax.set_title(f"{meta['phone']} / seed{seed}"); ax.set_xlabel('Source frame index')
                ax.set_ylabel('Mean luma (0-255)'); ax.grid(alpha=.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
        fig.suptitle('Full existing phone DEV: exposure traces, not a flicker/quality ranking')
        fig.tight_layout(rect=[0, .06, 1, .95])
        for ext in ['png', 'pdf']: fig.savefig(report/f'brightness-traces.{ext}', dpi=150)
        plt.close(fig)
        write(report/'per_sequence.json', sequences)
        write(report/'preview_manifest.json', dict(contacts=contacts, videos=videos))
        sources += [BASE/'scripts/evaluate_full_phone_r54.py', BASE/'scripts/render_full_phone_r54.py']
        write(report/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
        write(report/'audit.json', dict(passed=True, new_CNN_sequences=6, new_CNN_frames=1904,
              output_sequences=18, output_frames=5712, full_preview_frames_decoded=952,
              contacts=3, full_previews=3, new_training=0, new_GPU=False, ground_truth=False,
              independent_new_scenes=False, phone_cost_measured=False, visual_review_still_required=True))
        text += ['![All fixed brightness traces](brightness-traces.png)', '']
        (report/'README.md').write_text('\n'.join(text)+'\n')
        complete = True
    except BaseException:
        (report/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(report/'runtime.json', dict(completed_normally=complete, preview_frames_decoded=decoded,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
