"""Compile the working manuscript with owned tools, caches and attempt records."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from project_assets import ASSETS, REPO, owned_environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=int)
    args = parser.parse_args()
    paper = REPO/'paper'
    stem = f'build-{args.attempt:03d}'
    target = paper/'process'/(stem+'.json')
    assert not target.exists(), 'Keep prior build attempts'
    toolchain = json.loads((paper/'process/toolchain.json').read_text())
    binary = Path(toolchain['binary'])
    assert binary.resolve().is_relative_to(ASSETS.resolve()) and sha(binary) == toolchain['binary_sha256']
    work = ASSETS/'cache/paper-build'/stem
    work.mkdir(parents=True, exist_ok=False)
    env = owned_environment()
    for key in ['TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS', 'TEXMFCNF', 'TEXMFLOCAL', 'TECTONIC_BUNDLE_PATH']:
        env.pop(key, None)
    settings = {'TECTONIC_CACHE_DIR': ASSETS/'runtime-cache/tectonic',
                'XDG_CONFIG_HOME': ASSETS/'runtime-cache/xdg-config',
                'XDG_DATA_HOME': ASSETS/'runtime-cache/xdg-data',
                'TEXMFHOME': ASSETS/'runtime-cache/texmf-home',
                'TEXMFVAR': ASSETS/'runtime-cache/texmf-var',
                'TEXMFCONFIG': ASSETS/'runtime-cache/texmf-config'}
    for name, path in settings.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env['TECTONIC_UNTRUSTED_MODE'] = '1'
    sources = [paper/'main.tex', paper/'references.bib', paper/'evidence-snapshot.json',
               *sorted((paper/'sections').glob('*.tex')), *sorted((paper/'generated').glob('*.tex'))]
    command = [str(binary), '--untrusted', '--keep-logs', '--keep-intermediates', '--outdir', str(work), 'main.tex']
    record = {'started_at': datetime.now().astimezone().isoformat(), 'command': command,
              'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in sources},
              'toolchain_sha256': sha(paper/'process/toolchain.json'),
              'cache_environment': {n: env[n] for n in settings},
              'scope': 'Internal incomplete manuscript; no active-study quality or phone/final-test claim.'}
    started = time.perf_counter()
    log = paper/'process'/(stem+'.log')
    try:
        with log.open('x') as stream:
            process = subprocess.run(command, cwd=paper, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record['returncode'] = process.returncode
        assert process.returncode == 0, 'Typesetting failed; keep this log and use a new attempt after fixing'
        output = work/'main.pdf'
        assert output.read_bytes().startswith(b'%PDF-')
        shutil.copy2(output, paper/'manuscript.pdf')
        record.update(pdf_sha256=sha(output), pdf_bytes=output.stat().st_size,
                      output='paper/manuscript.pdf', completed_normally=True)
    finally:
        record.update(wall_seconds=time.perf_counter()-started, ended_at=datetime.now().astimezone().isoformat())
        target.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'completed_normally': True, 'pdf': record['output'], 'bytes': record['pdf_bytes']}))


if __name__ == '__main__':
    main()
