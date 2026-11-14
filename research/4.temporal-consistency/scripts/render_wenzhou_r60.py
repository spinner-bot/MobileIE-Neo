"""All17 signals/all6 fixed sources, frame64 and12 times, full128 previews."""
import json
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

from drv_references_r59 import NETWORKS
from dynamic_training_r34 import BASE, ROOT
from fetch_wenzhou_samples_r52 import write
from resume_wenzhou_samples_r52 import file_digest
from summarize_wenzhou_r60 import lines


def byte_image(frame, size):
    value = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    return cv2.resize(value, size, interpolation=cv2.INTER_AREA)


def grid(signals, frame):
    canvas = np.full((1020, 1280, 3), 245, dtype=np.uint8)
    for index, (name, array) in enumerate(signals):
        row, column = divmod(index, 4); x, y = column*320, row*204
        canvas[y+24:y+204, x:x+320] = byte_image(array[frame], (320, 180))
        cv2.putText(canvas, name, (x+4, y+17), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main():
    cv2.setNumThreads(1)
    run = BASE/'results/exploration-r60-wenzhou-reference-attempt001'
    runtime = json.loads((run/'runtime.json').read_text()); assert runtime['completed_normally'] and runtime['sequences'] == 96
    index_path = BASE/'results/exploration-r60-wenzhou-data-attempt001/index.json'
    metas = json.loads(index_path.read_text())['scenes']; assert len(metas) == 6
    manifest = {(row['scene'], row['reference'], row['method']): row for row in lines(run/'prediction_manifest.jsonl')}
    assert len(manifest) == 96
    out = BASE/'report/exploration-r60-wenzhou'
    process = BASE/'process/exploration-r60/render001'; process.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r60/previews/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    times = np.rint(np.linspace(4, 123, 12)).astype(int).tolist()
    start, complete, previews, contacts, returned = time.perf_counter(), False, [], [], 0
    try:
        for meta in metas:
            assert file_digest(ROOT/meta['path']) == meta['sha256']
            with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
                low = archive['low'].copy()
            signals = [('input', low)]
            for name in NETWORKS:
                for method in ['raw', 'recursive_self05']:
                    row = manifest[meta['scene'], name, method]
                    path = ROOT/row['path']; assert file_digest(path) == row['sha256']
                    array = np.load(path, mmap_mode='r', allow_pickle=False)
                    assert array.shape == (128, 360, 640, 3) and array.dtype == np.float32
                    signals.append((name+('/raw' if method == 'raw' else '/rec.5'), array))
            assert len(signals) == 17
            contact = out/(meta['scene']+'--frame64.png')
            assert not contact.exists()
            assert cv2.imwrite(str(contact), cv2.cvtColor(grid(signals, 64), cv2.COLOR_RGB2BGR))
            strip = np.full((1900, 2150, 3), 245, dtype=np.uint8)
            for column, t in enumerate(times):
                cv2.putText(strip, f'local{t}', (230+column*160+4, 22), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 1, cv2.LINE_AA)
            for row, (name, array) in enumerate(signals):
                y = 30+row*110
                cv2.putText(strip, name, (4, y+52), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 1, cv2.LINE_AA)
                for column, t in enumerate(times):
                    x = 230+column*160
                    strip[y:y+90, x:x+160] = byte_image(array[t], (160, 90))
            temporal = out/(meta['scene']+'--fixed12.png')
            assert not temporal.exists()
            assert cv2.imwrite(str(temporal), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
            for path in [contact, temporal]:
                contacts.append(dict(scene=meta['scene'], path=str(path.relative_to(ROOT)), sha256=file_digest(path)))
            video = cache/(meta['scene']+'.avi'); fps = meta['video_metadata']['reported_fps']
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), fps, (1280, 1020))
            assert writer.isOpened()
            try:
                for t in range(128): writer.write(cv2.cvtColor(grid(signals, t), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]); decoded = 0
            try:
                assert cap.isOpened()
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    assert frame.shape == (1020, 1280, 3); decoded += 1
            finally:
                cap.release()
            assert decoded == 128; returned += decoded
            previews.append(dict(scene=meta['scene'], path=str(video.relative_to(ROOT)), sha256=file_digest(video),
                  frames=decoded, fps=fps, sensor_timing_preserved=False,
                  note='Visualization only, numeric FP32 arrays unchanged; not new source observations or manual playback.'))
            print(json.dumps(dict(scene=meta['scene'], signals=17, contacts=2, preview_frames=decoded)), flush=True)
            del signals, low
        assert len(contacts) == 12 and len(previews) == 6 and returned == 768
        write(out/'visual-manifest.json', dict(at=datetime.now().astimezone().isoformat(), contacts=contacts, previews=previews,
              fixed_full_frame=64, fixed_temporal_frames=times, preview_decoded_frames=returned,
              visual_review_required=True, human_full_video_playback=False, no_GT_or_quality_rank=True))
        complete = True
    except BaseException:
        (process/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(process/'runtime.json', dict(completed_normally=complete, contacts=len(contacts), previews=len(previews),
              preview_frames_returned=returned, seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
