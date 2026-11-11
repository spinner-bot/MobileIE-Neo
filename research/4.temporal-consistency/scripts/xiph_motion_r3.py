"""Xiph CIF prefix reader and synthetic correlated-read-noise observations."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json


def read_y4m(path, count=24):
    frames = []
    with Path(path).open('rb') as stream:
        header = stream.readline().decode('ascii').strip()
        tokens = header.split()
        if tokens[0] != 'YUV4MPEG2':
            raise ValueError('invalid Y4M signature')
        metadata = {token[0]: token[1:] for token in tokens[1:]}
        w, h = int(metadata['W']), int(metadata['H'])
        if metadata.get('C', '420jpeg') not in ['420', '420jpeg', '420mpeg2', '420paldv']:
            raise ValueError('only planar 8bit YUV420 is supported')
        for _ in range(count):
            marker = stream.readline()
            if not marker.startswith(b'FRAME'):
                raise ValueError('missing frame marker')
            payload = stream.read(w*h*3//2)
            if len(payload) != w*h*3//2:
                raise ValueError('truncated frame')
            yuv = np.frombuffer(payload, np.uint8).reshape(h*3//2, w)
            frames.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420).astype(np.float32))
    return frames, header


def corrupt(frames, seed, rho):
    rng = np.random.default_rng(seed)
    read = rng.normal(0, .008, frames[0].shape)
    result = []
    for t, gt in enumerate(frames):
        if t:
            read = rho*read+np.sqrt(1-rho*rho)*rng.normal(0, .008, gt.shape)
        clean = (gt/255.)**2.2*.35
        low = (np.clip(rng.poisson(clean*40)/40+read, 0, 1)*255).astype(np.uint8)
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(low, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        result.append(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    rows = []
    for name in ['foreman', 'coastguard']:
        url = f'https://media.xiph.org/video/derf/y4m/{name}_cif.y4m'
        path = ROOT/'.cache/data/xiph'/f'{name}-first24.y4m'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Fetch only the declared 24-frame diagnostic budget, not all 300 frames.
            with urllib.request.urlopen(url, timeout=120) as source, path.with_suffix('.partial').open('wb') as target:
                header = source.readline()
                target.write(header)
                meta = {s[:1]: s[1:] for s in header.split()[1:]}
                size = int(meta[b'W'])*int(meta[b'H'])*3//2
                for t in range(24):
                    marker = source.readline()
                    if not marker.startswith(b'FRAME'):
                        raise ValueError('invalid source frame marker')
                    target.write(marker)
                    payload = source.read(size)
                    if len(payload) != size:
                        raise ValueError('short source frame')
                    target.write(payload)
                    print(json.dumps({'scene': name, 'frame': t, 'bytes': target.tell()}), flush=True)
            path.with_suffix('.partial').rename(path)
        frames, header = read_y4m(path)
        rows.append({'scene': name, 'url': url, 'path': str(path.relative_to(ROOT)), 'frames': len(frames),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
                     'header': header, 'shape': list(frames[0].shape),
                     'scope': 'exact first 24 frames, not checksum of full 300-frame source',
                     'conversion': 'OpenCV I420 BT.601 limited-range; original pixel aspect preserved in raster'})
    save_json(args.manifest, {'source': 'https://media.xiph.org/video/derf/', 'clips': rows})


if __name__ == '__main__':
    main()
