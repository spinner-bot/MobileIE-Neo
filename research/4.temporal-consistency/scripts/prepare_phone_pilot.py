"""Decode the registered three phone clips using the owned OpenCV backend."""
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np

from explore_temporal_controls import ROOT, save_json, sha256

BASE = ROOT/'research/4.temporal-consistency'


def main():
    protocol = BASE/'docs/exploration-r31-protocol.md'
    source = BASE/'data/external-phone/manifest.json'
    manifest = json.loads(source.read_text())
    assert sha256(protocol) == manifest['protocol_sha256']
    out = BASE/'results/exploration-r31-data'
    out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r31'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    start, rows = time.perf_counter(), []
    save_json(out/'config.json', {'opencv_version': cv2.__version__, 'source_manifest_sha256': sha256(source),
              'protocol_sha256': sha256(protocol), 'opencv_threads': 1, 'decoder_threads_requested': 1,
              'device': 'cpu', 'extra_noise': False, 'brightness_transform': None,
              'resize': 'long side640, other side nearest even, INTER_AREA, no crop', 'frames_per_video': 64})
    for video in manifest['videos']:
        path = ROOT/video['path']
        assert sha256(path) == video['sha256']
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        assert capture.isOpened(), 'Cannot decode registered video; do not substitute another clip: '+str(path)
        metadata = {'reported_fps': capture.get(cv2.CAP_PROP_FPS), 'reported_frame_count': capture.get(cv2.CAP_PROP_FRAME_COUNT),
                    'reported_width': capture.get(cv2.CAP_PROP_FRAME_WIDTH), 'reported_height': capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    'orientation_metadata_degrees': capture.get(cv2.CAP_PROP_ORIENTATION_META),
                    'orientation_auto_property': capture.get(cv2.CAP_PROP_ORIENTATION_AUTO),
                    'decoder_threads_property': capture.get(cv2.CAP_PROP_N_THREADS), 'backend': capture.getBackendName()}
        assert metadata['decoder_threads_property'] == 1
        frames, timestamps, shape = [], [], None
        try:
            for index in range(64):
                ok, image = capture.read()
                assert ok, f'Original video ended before registered frame {index}; no replacement'
                if shape is None:
                    shape = list(image.shape)
                assert list(image.shape) == shape
                height, width = image.shape[:2]
                if width >= height:
                    size = (640, max(2, int(round(height*640/width/2))*2))
                else:
                    size = (max(2, int(round(width*640/height/2))*2), 640)
                value = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                frames.append(value)
                timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        finally:
            capture.release()
        label = re.sub(r'[^A-Za-z0-9]+', '_', video['phone']+'_'+path.stem).strip('_')
        array = np.stack(frames)
        times = np.asarray(timestamps, dtype=np.float64)
        valid_times = bool(np.isfinite(times).all() and np.all(np.diff(times) > 0))
        target = cache/(label+'.npz')
        np.savez_compressed(target, low=array, decoded_timestamp_ms=times)
        row = {'domain': 'lliv_phone_pilot', 'scene': label, 'phone': video['phone'], 'source_sha256': video['sha256'],
               'path': str(target.relative_to(ROOT)), 'sha256': sha256(target), 'shape': list(array.shape),
               'decoded_display_shape': shape, 'cached_dtype': str(array.dtype), 'video_metadata': metadata,
               'decoded_timestamp_ms': times.tolist(), 'timestamps_strictly_increasing': valid_times,
               'timestamp_scope': 'OpenCV decoder-reported positions; not independently verified sensor capture time.',
               'ground_truth_available': False, 'source_frames_decoded': 64}
        rows.append(row)
        save_json(out/(label+'.json'), row)
        print(json.dumps({'scene': label, 'shape': list(array.shape), 'timestamp_valid': valid_times}), flush=True)
    assert len(rows) == 3
    save_json(out/'index.json', {'scenes': rows, 'source_manifest_sha256': sha256(source),
              'protocol_sha256': sha256(protocol), 'ground_truth_available': False,
              'scope': 'Three fixed real-phone development clips, no full-reference quality or mobile latency measurement.'})
    save_json(out/'runtime.json', {'completed_normally': True, 'videos': 3, 'source_frames_decoded': 192,
              'device': 'cpu', 'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
