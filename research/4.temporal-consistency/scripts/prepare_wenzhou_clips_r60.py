"""Prepare the preregistered midpoint clips of all six author night videos."""
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest
from fetch_wenzhou_samples_r52 import write
from resume_wenzhou_samples_r52 import file_digest


def main():
    source = BASE/'process/external-night-wenzhou/video-extraction001/manifest.json'
    terminal = json.loads((source.parent/'runtime.json').read_text())
    assert terminal['completed_normally'] and terminal['members_extracted'] == 6
    manifest = json.loads(source.read_text()); assert len(manifest['files']) == 6
    out = BASE/'results/exploration-r60-wenzhou-data-attempt001'
    cache = ROOT/'.cache/experiments/exploration-r60/data/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(exist_ok=False); cache.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), source_manifest_sha256=digest(source),
          protocol_sha256=digest(BASE/'docs/exploration-r60-wenzhou-input-identity-protocol.md'),
          script_sha256=digest(Path(__file__)), revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          opencv_version=cv2.__version__, opencv_threads=1, decoder_threads=1,
          clip='128 sequential RGB returns from floor((reported_count-128)/2); all six sources',
          primary_local_frames=list(range(4, 124)), resize='Full FOV INTER_AREA, long side640, nearest even short side',
          representation='uint8 RGB, original brightness', CNN=0, GPU=False))
    start, complete, rows = time.perf_counter(), False, []
    try:
        for meta in sorted(manifest['files'], key=lambda r: r['member']):
            path = ROOT/meta['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            assert path.stat().st_size == meta['bytes'] and file_digest(path) == meta['sha256']
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            frames, times, positions, native = [], [], [], None
            try:
                assert cap.isOpened() and cap.get(cv2.CAP_PROP_N_THREADS) == 1
                header = dict(reported_fps=cap.get(cv2.CAP_PROP_FPS),
                      reported_frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT),
                      reported_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), reported_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                      orientation_metadata_degrees=cap.get(cv2.CAP_PROP_ORIENTATION_META),
                      orientation_auto_property=cap.get(cv2.CAP_PROP_ORIENTATION_AUTO), backend=cap.getBackendName())
                count = int(header['reported_frame_count'])
                assert count == header['reported_frame_count'] and count >= 128 and header['reported_fps'] > 0
                first = (count-128)//2
                assert cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-first) < .1
                for t in range(128):
                    ok, image = cap.read(); assert ok, 'Fixed midpoint clip ended early; no replacement.'
                    position = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    assert abs(position-(first+t+1)) < .1, 'Decoder position differs from fixed target.'
                    if native is None: native = list(image.shape)
                    assert list(image.shape) == native
                    h, w = image.shape[:2]
                    size = (640, max(2, int(round(h*640/w/2))*2)) if w >= h else (max(2, int(round(w*640/h/2))*2), 640)
                    frames.append(cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                    times.append(cap.get(cv2.CAP_PROP_POS_MSEC)); positions.append(position-1)
            finally:
                cap.release()
            low, timestamps = np.stack(frames), np.asarray(times, dtype=np.float64)
            assert low.dtype == np.uint8 and len(low) == 128 and np.isfinite(timestamps).all()
            assert np.all(np.diff(timestamps) > 0)
            label = path.stem; target = cache/(label+'.npz')
            np.savez_compressed(target, low=low, decoded_timestamp_ms=timestamps,
                                source_frame_positions=np.asarray(positions, dtype=np.float64))
            row = dict(domain='Wenzhou_author_backup_unpaired_DEV', scene=label, member=meta['member'],
                  source_path=meta['path'], source_sha256=meta['sha256'], path=str(target.relative_to(ROOT)),
                  sha256=file_digest(target), shape=list(low.shape), dtype=str(low.dtype), video_metadata=header,
                  native_decoded_shape=native, requested_start=first, source_frame_positions=positions,
                  decoded_timestamp_ms=times, primary_local_frames=list(range(4, 124)),
                  primary_source_frames=list(range(first+4, first+124)), ground_truth_available=False,
                  mean_RGB_0to255=low.mean(axis=(0, 1, 2)).tolist(), CNN=0,
                  scope='Fixed middle development excerpt; no full-video/GT/final test or sensor timing claim; seek may decode additional compressed frames internally.')
            write(out/(label+'.json'), row); rows.append(row)
            print(json.dumps(dict(scene=label, shape=list(low.shape), start=first, metadata=header)), flush=True)
        assert len(rows) == 6 and sum(r['shape'][0] for r in rows) == 768
        write(out/'index.json', dict(scenes=rows, returned_RGB_observations=768, primary_targets=720,
              primary_targets_per_scene=120, source_manifest_sha256=digest(source), CNN=0, GPU=False,
              scope='All six author backup night videos, fixed midpoint excerpts, no GT or independent final.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, scenes_completed=len(rows),
              returned_RGB_observations=sum(r['shape'][0] for r in rows), CNN=0, GPU=False,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
