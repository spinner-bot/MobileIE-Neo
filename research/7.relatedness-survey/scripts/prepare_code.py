"""Inventory a pinned archive and expose private text copies; never execute imported research code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

B = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('resource_id')
    ap.add_argument('receipt', help='Source ZIP receipt relative to workstation')
    a = ap.parse_args()
    events = [json.loads(s) for s in (B / 'catalog/analysis-log.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'begin' and events[-1]['resource_id'] == a.resource_id
    receipt = json.loads((B / a.receipt).read_text())
    original = B / receipt['archive_path']
    assert hashlib.sha256(original.read_bytes()).hexdigest() == receipt['sha256']
    r = B / 'resources' / a.resource_id
    manifest = r / 'derived/docs/upstream-code-manifest.json'
    assert not manifest.exists(), 'Preserve earlier preparation/version'
    cache = (B / '../../.cache/relatedness-survey/derived' / a.resource_id).resolve()
    inventory, texts = [], []
    with zipfile.ZipFile(original) as z:
        prefix = z.namelist()[0].split('/')[0] + '/'
        for info in z.infolist():
            if info.is_dir():
                continue
            source = info.filename.removeprefix(prefix)
            assert not source.startswith('/') and '..' not in Path(source).parts
            raw = z.read(info.filename)
            digest = hashlib.sha256(raw).hexdigest()
            inventory.append({'path':source, 'bytes':len(raw), 'sha256':digest})
            if Path(source).suffix.lower() in ('.py', '.yaml', '.yml', '.json', '.md', '.txt', '.html', '.sh') or Path(source).name in ('LICENSE', '.gitignore'):
                flat = source.replace('/', '--') + '.txt'
                dest = cache / 'docs/upstream-code' / flat
                link = r / 'derived/docs/upstream-code' / flat
                dest.parent.mkdir(parents=True, exist_ok=True)
                link.parent.mkdir(parents=True, exist_ok=True)
                assert not link.exists() and not link.is_symlink(), 'Flat path collision'
                dest.write_bytes(raw)
                link.symlink_to(os.path.relpath(dest, link.parent))
                texts.append({'source_path':source, 'readable_path':str(link.relative_to(r)),
                              'sha256':digest, 'bytes':len(raw)})
    identity = json.loads((r / 'identity.json').read_text())
    manifest.write_text(json.dumps({'purpose':'Resource static inspection preparation; not a read/approval claim',
                                    'revision':identity['revision'], 'archive_receipt':a.receipt,
                                    'files':texts}, ensure_ascii=False, indent=2)+'\n')
    (r / 'derived/docs/archive-inventory.json').write_text(json.dumps(
        {'resource_id':a.resource_id, 'files':inventory, 'executed':False},
        ensure_ascii=False, indent=2)+'\n')
    print('Inventoried', len(inventory), 'files; linked', len(texts), 'texts. No execution/reading claim.')


if __name__ == '__main__':
    main()
