#!/usr/bin/env python3
"""Package the Windows/Linux Python desktop demo as a self-contained ZIP.

Deterministic: fixed entry timestamps, sorted order, so the same source tree
always yields the same bytes. Android artifacts are not touched by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[3]
DESKTOP = REPO / 'demo/org/desktop'
ASSETS = REPO / 'demo/org/app/app/src/main/assets'
FILES = ('desktop.py', 'lightlab_tasks.py', 'requirements.txt', 'run.cmd', 'QUICKSTART.txt')
STAMP = (2026, 11, 14, 0, 0, 0)


def add(archive, name, data):
    entry = zipfile.ZipInfo(name, date_time=STAMP)
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    archive.writestr(entry, data, compresslevel=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()

    version = args.version
    source = (DESKTOP / 'desktop.py').read_text(encoding='utf-8')
    if f"VERSION='{version}'" not in source:
        raise SystemExit(f'desktop.py VERSION does not match {version}')

    tasks = json.loads((ASSETS / 'tasks.json').read_text(encoding='utf-8'))['tasks']
    assert len(tasks) == 3
    for task in tasks:
        digest = hashlib.sha256((ASSETS / task['model_asset']).read_bytes()).hexdigest()
        assert digest == task['model_sha256'], f"model digest mismatch: {task['model_asset']}"
        assert not task['native_control_inputs']

    args.output.parent.mkdir(parents=True, exist_ok=True)
    part = args.output.with_suffix('.zip.part')
    with zipfile.ZipFile(part, 'w') as archive:
        for name in FILES:
            add(archive, name, (DESKTOP / name).read_bytes())
        for asset in sorted(ASSETS.rglob('*')):
            if asset.is_file():
                add(archive, 'assets/' + asset.relative_to(ASSETS).as_posix(), asset.read_bytes())
    part.replace(args.output)

    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert all(name in names for name in FILES)
        assert 'assets/tasks.json' in names
        for task in tasks:
            assert 'assets/' + task['model_asset'] in names

    manifest = {
        'version': version,
        'file': args.output.name,
        'bytes': args.output.stat().st_size,
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'entries': len(names),
        'models': {task['id']: {'asset': task['model_asset'], 'sha256': task['model_sha256']}
                   for task in tasks},
        'android_release_unchanged': True,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
