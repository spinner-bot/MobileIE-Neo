"""Fixed phone contacts, centers and full previews for one final8000 adapted seed."""
import argparse
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


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run = BASE/f'results/exploration-r56-stage001-eval-seed{args.seed}'
    runtime = json.loads((run/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['phone_rows'] == 6
    parent = BASE/'report/exploration-r56-tiny-task-reference'
    report = parent/f'seed{args.seed}'; report.mkdir(exist_ok=False)
    assert json.loads((parent/'audit.json').read_text())['complete_paired_table']
    preview = ROOT/f'.cache/experiments/exploration-r56/previews/seed{args.seed}'; preview.mkdir(parents=True, exist_ok=False)
    index = BASE/'results/exploration-r49-full-phone-attempt001/index.json'
    metas = json.loads(index.read_text())['scenes']
    records = [r for r in lines(run/'prediction_manifest.jsonl') if not r['paired']]
    lookup = {(r['scene'], r['method']): r for r in records}; assert len(lookup) == 6
    frame_rows, input_rows = lines(run/'phone_per_frame.jsonl'), lines(run/'phone_input_statistics.jsonl')
    assert len(frame_rows) == 1904 and len(input_rows) == 952
    start, complete, decoded, artifacts = time.perf_counter(), False, 0, []
    text = [f'# Seed{args.seed}: Fixed 696-parameter task-adapted reference / full phone DEV', '',
            'One final8000 adapted seed, no intermediate-checkpoint or best-seed selection. Same three existing DEV sources, full EOF. Columns: input / Nano raw / own-raw half-FB recursive .5.', '',
            'No paired GT, perceptual flicker truth or phone latency. Contacts are always 0/floor(T/2)/T-1; center regions always the final-frame central256x256. MJPEG shows all frames at constant30FPS for inspection, not original sensor-time playback. FP32 arrays and original R49 PTS remain the numeric/source evidence.', '',
            '[Prior frozen MobileIE full-phone references](../../exploration-r54-full-phone/README.md). Different training histories; neither visual comparisons nor unpaired statistics establish an equal-budget or GT-quality ranking.', '']
    try:
        for meta in metas:
            assert digest(ROOT/meta['path']) == meta['sha256']
            with np.load(ROOT/meta['path'], allow_pickle=False) as archive: low = archive['low'].copy()
            n, h, w, _ = low.shape; selected = [0, n//2, n-1]
            arrays = {}
            for method in ['raw', 'recursive_self05']:
                record = lookup[meta['scene'], method]
                assert digest(ROOT/record['path']) == record['sha256']
                arrays[method] = np.load(ROOT/record['path'], mmap_mode='r', allow_pickle=False)
                assert arrays[method].shape == low.shape
            video = preview/(meta['scene']+'.avi')
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (960, 202))
            assert writer.isOpened(); contacts = []
            try:
                for t in range(n):
                    panel = np.full((202, 960, 3), 245, np.uint8)
                    for col, (image, label) in enumerate([(low[t], 'input'), (arrays['raw'][t], 'Nano696 raw'),
                                                        (arrays['recursive_self05'][t], 'Nano696 recursive .5')]):
                        value = np.clip(np.rint(image), 0, 255).astype(np.uint8)
                        panel[22:, col*320:(col+1)*320] = cv2.resize(value, (320, 180), interpolation=cv2.INTER_AREA)
                        cv2.putText(panel, label, (col*320+4, 16), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 1, cv2.LINE_AA)
                    writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    if t in selected:
                        band = np.full((28, 960, 3), 255, np.uint8)
                        cv2.putText(band, f"{meta['phone']} | frame{t}/{n-1} | unpaired DEV", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1, cv2.LINE_AA)
                        contacts.extend([band, panel.copy()])
            finally: writer.release()
            contact = report/(meta['scene']+'.png')
            assert cv2.imwrite(str(contact), cv2.cvtColor(np.concatenate(contacts), cv2.COLOR_RGB2BGR))
            x, y = (w-256)//2, (h-256)//2
            center = np.full((280, 768, 3), 245, np.uint8)
            for col, (array, label) in enumerate([(low, 'input'), (arrays['raw'], 'Nano696 raw'),
                                                (arrays['recursive_self05'], 'Nano696 recursive .5')]):
                value = array[-1, y:y+256, x:x+256]
                center[24:, col*256:(col+1)*256] = np.clip(np.rint(value), 0, 255).astype(np.uint8)
                cv2.putText(center, label, (col*256+4, 17), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 0, 0), 1, cv2.LINE_AA)
            crop = report/(meta['scene']+'--final-center.png')
            assert cv2.imwrite(str(crop), cv2.cvtColor(center, cv2.COLOR_RGB2BGR))
            capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]); count = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok: break
                    assert frame.shape == (202, 960, 3); count += 1
            finally: capture.release()
            assert count == n; decoded += count
            artifacts.append(dict(scene=meta['scene'], contact_frames=selected,
                 contact=str(contact.relative_to(ROOT)), contact_sha256=digest(contact),
                 center=str(crop.relative_to(ROOT)), center_sha256=digest(crop), center_xywh=[x, y, 256, 256],
                 video=str(video.relative_to(ROOT)), video_sha256=digest(video), frames=count,
                 preview_fps=30, sensor_timing_preserved=False))
            text += [f"## {meta['phone']}", '', f'![Fixed contacts]({contact.name})', '',
                     f'![Fixed final center]({crop.name})', '', f'[All{n} preview frames](../../../../../{video.relative_to(ROOT)})', '']
            print(json.dumps(dict(scene=meta['scene'], preview_frames=count)), flush=True)
        assert decoded == 952
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, meta in zip(axes, metas):
            selected = [r for r in input_rows if r['scene'] == meta['scene']]
            ax.plot([r['frame'] for r in selected], [r['mean_luma_0to255'] for r in selected], '--', color='gray', label='input')
            for method in ['raw', 'recursive_self05']:
                selected = [r for r in frame_rows if r['scene'] == meta['scene'] and r['method'] == method]
                assert len(selected) == meta['shape'][0]
                ax.plot([r['frame'] for r in selected], [r['mean_luma_0to255'] for r in selected], label=method)
            ax.set_title(meta['phone']); ax.set_xlabel('Source frame index'); ax.set_ylabel('Mean luma (0-255)'); ax.grid(alpha=.25)
        handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
        fig.suptitle('One final8000 adapted micro-model / exposure traces, not quality or flicker ranks')
        fig.tight_layout(rect=[0, .1, 1, .94])
        for ext in ['png', 'pdf']: fig.savefig(report/f'phone-brightness.{ext}', dpi=150)
        plt.close(fig)
        write(report/'phone-preview-manifest.json', artifacts)
        write(report/'phone-render-audit.json', dict(passed=True, all_preview_frames_decoded=952,
              fixed_contacts=3, fixed_centers=3, new_CNN=0, visual_review_still_required=True,
              sources={str(p.relative_to(ROOT)): digest(p) for p in [index, run/'prediction_manifest.jsonl',
                        run/'runtime.json', BASE/'scripts/render_tiny_phone_r56.py']}))
        text += ['![All fixed brightness traces](phone-brightness.png)', '']
        (report/'phone-previews.md').write_text('\n'.join(text)+'\n'); complete = True
    except BaseException:
        (report/'phone-render-failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(report/'phone-render-runtime.json', dict(completed_normally=complete, preview_frames_decoded=decoded,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
