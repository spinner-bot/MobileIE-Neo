"""Complete fixed phone-reference contact sheets and all-frame preview videos."""
import json
from pathlib import Path

import cv2
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest

ANCHORS = ['official_LOL', 'R34_seed0', 'R34_seed1']
METHODS = ['raw', 'finite_self05', 'recursive_self05']


def main():
    cv2.setNumThreads(1)
    source = BASE/'results/exploration-r38-phone'
    runtime = json.loads((source/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['rows'] == 27 and runtime['neural_sequences'] == 9
    index = BASE/'results/exploration-r31-data/index.json'
    metas = json.loads(index.read_text())['scenes']
    manifest_path = source/'prediction_manifest.jsonl'
    rows = list(map(json.loads, manifest_path.read_text().splitlines()))
    lookup = {(r['scene'], r['anchor'], r['method']): r for r in rows}
    assert len(lookup) == 27
    report = BASE/'report/exploration-r38-phone'
    preview = ROOT/'.cache/experiments/exploration-r38/previews'
    report.mkdir(parents=True, exist_ok=False)
    preview.mkdir(parents=True, exist_ok=False)
    files, lines = [], ['# Fixed references on real phone input', '',
        'Three R31 DEV clips, original brightness and full field of view, no added degradation. Rows in each panel are official LOL / R34 seed0 / R34 seed1; columns repeat the input, raw, finite .5, recursive .5. Both seeds and every device are retained; no best-case selection.', '',
        'No synchronized GT, PSNR/SSIM, true-flicker, user-study or phone-latency claim. Edge/brightness/endpoint statistics are diagnostics rather than a quality rank. MJPEG previews are all 64 frames at constant 30 fps for inspection, not sensor-time reproduction; original decoder timestamp records remain in R31. PNG contact sheets always show frames 0/32/63. Float32 arrays are the numerical source; PNG/AVI alone are not numerical evidence.', '']
    for meta in metas:
        assert digest(ROOT/meta['path']) == meta['sha256']
        with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
            low = archive['low'].copy()
        predictions = {}
        for anchor in ANCHORS:
            for method in METHODS:
                row = lookup[(meta['scene'], anchor, method)]
                assert digest(ROOT/row['path']) == row['sha256']
                predictions[(anchor, method)] = np.load(ROOT/row['path'], mmap_mode='r', allow_pickle=False)
        width, height = 1280, 3*202
        video = preview/(meta['scene']+'.avi')
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (width, height))
        assert writer.isOpened(), 'Preview codec unavailable; do not replace or alter inference results'
        selected = []
        try:
            for t in range(64):
                panel = np.full((height, width, 3), 245, np.uint8)
                for i, anchor in enumerate(ANCHORS):
                    images = [low[t]]+[predictions[(anchor, method)][t] for method in METHODS]
                    labels = [f'{anchor}: input']+[f'{anchor}: {method}' for method in METHODS]
                    for j, (frame, label) in enumerate(zip(images, labels)):
                        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
                        panel[i*202+22:(i+1)*202, j*320:(j+1)*320] = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                        cv2.putText(panel, label, (j*320+5, i*202+16), cv2.FONT_HERSHEY_SIMPLEX, .41, (0, 0, 0), 1, cv2.LINE_AA)
                writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                if t in [0, 32, 63]:
                    band = np.full((28, width, 3), 255, np.uint8)
                    cv2.putText(band, f"{meta['phone']} | frame {t} | no synchronized reference GT", (6, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
                    selected += [band, panel]
        finally:
            writer.release()
        image_path = report/(meta['scene']+'.png')
        assert cv2.imwrite(str(image_path), cv2.cvtColor(np.concatenate(selected, axis=0), cv2.COLOR_RGB2BGR))
        # Decode the written artifact to verify count, without GPU/codecs outside our environment.
        capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (height, width, 3)
            count += 1
        capture.release()
        assert count == 64
        files.append({'scene': meta['scene'], 'phone': meta['phone'], 'contact_frames': [0, 32, 63],
                      'contact_sheet': str(image_path.relative_to(ROOT)), 'contact_sha256': digest(image_path),
                      'video': str(video.relative_to(ROOT)), 'video_sha256': digest(video),
                      'video_frames': count, 'preview_fps': 30, 'sensor_timing_preserved': False})
        lines += [f"## {meta['phone']}", '', f"![Fixed frames]({image_path.name})", '',
                  f"[All 64 preview frames](../../../../{video.relative_to(ROOT)})", '']
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    (report/'preview_manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    (report/'audit.json').write_text(json.dumps({'passed': True, 'reference_rows': 27,
          'videos': 3, 'all_frame_previews': True, 'fixed_contact_frames': [0, 32, 63],
          'ground_truth_available': False,
          'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                           [source/'runtime.json', source/'config.json', manifest_path, index, Path(__file__)]},
          'scope': 'Preview generation verified, not scientific quality, true temporal stability or phone runtime validation.'}, indent=2)+'\n')
    print(json.dumps({'rows': 27, 'videos': 3, 'frames_per_preview': 64}))


if __name__ == '__main__':
    main()
