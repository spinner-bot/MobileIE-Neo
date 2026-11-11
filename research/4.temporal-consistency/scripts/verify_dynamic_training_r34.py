"""Decode the complete registered training pool before any optimizer launch."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time

import cv2

from dynamic_training_r34 import BASE, MANIFEST, ROOT, digest


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest['frames']) == 3428 and manifest['videos'] == 57
    out = BASE/'results/exploration-r32-decode'
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    start = time.perf_counter()
    def verify(row):
        path = ROOT/row['path']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert path.stat().st_size == row['size'] and digest(path) == row['sha256']
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[0] == 480 and image.shape[1] >= 128 and image.shape[2] == 3
        return {'name': row['name'], 'scene': row['scene'], 'shape': list(image.shape), 'source_sha256': row['sha256']}
    # Two worker threads, one OpenCV thread each; no image retention or CUDA.
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(verify, manifest['frames']))
    (out/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    value = {'passed': True, 'frames': len(rows), 'videos': len({row['scene'] for row in rows}),
             'manifest_sha256': digest(MANIFEST), 'opencv_version': cv2.__version__,
             'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'decode_workers': 2, 'opencv_threads_per_worker': 1, 'wall_seconds': time.perf_counter()-start,
             'scope': 'Registered TRAIN files decoded and source hashes checked; no quality metrics, validation frames, training or GPU.'}
    (out/'audit.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
