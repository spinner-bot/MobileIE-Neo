"""Fixed final-frame center crops, numerical arrays to readable visual evidence."""
import json

import cv2
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    cv2.setNumThreads(1)
    report = BASE/'report/exploration-r54-full-phone'
    assert json.loads((report/'runtime.json').read_text())['completed_normally']
    metas = json.loads((BASE/'results/exploration-r49-full-phone-attempt001/index.json').read_text())['scenes']
    manifests = {}
    for seed in [0, 1]:
        path = BASE/f'results/exploration-r54-full-phone-seed{seed}/prediction_manifest.jsonl'
        manifests[seed] = {(r['scene'], r['method']): r for r in map(json.loads, path.read_text().splitlines())}
    rows = []
    for meta in metas:
        n, h, w, _ = meta['shape']; t = n-1
        x, y = (w-256)//2, (h-256)//2
        with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
            source = archive['low'][t, y:y+256, x:x+256].copy()
        panel = np.full((560, 1024, 3), 245, np.uint8)
        for seed in [0, 1]:
            images, labels = [source], ['input']
            for method in ['raw', 'recursive_self05', 'rgb']:
                record = manifests[seed][meta['scene'], method]
                # Whole-array hashes were verified by the completed main renderer.
                a = np.load(ROOT/record['path'], mmap_mode='r', allow_pickle=False)
                images.append(a[t, y:y+256, x:x+256]); labels.append(method)
            for col, (value, label) in enumerate(zip(images, labels)):
                panel[seed*280+24:seed*280+280, col*256:(col+1)*256] = np.clip(np.rint(value), 0, 255).astype(np.uint8)
                cv2.putText(panel, f'seed{seed}: {label}', (col*256+4, seed*280+17), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
        path = report/(meta['scene']+'--final-center.png'); assert not path.exists()
        assert cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        rows.append(dict(scene=meta['scene'], frame=t, xywh=[x, y, 256, 256],
                         path=str(path.relative_to(ROOT)), sha256=digest(path)))
    path = report/'center-crops.json'; assert not path.exists()
    path.write_text(json.dumps(dict(crops=rows, new_CNN=0, script_sha256=digest(BASE/'scripts/render_phone_centers_r54.py'),
          scope='Supplemental same final-frame center region for every phone and seed; no new source pixels, numerical correction or quality rank.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
