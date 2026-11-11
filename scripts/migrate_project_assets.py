"""After active experiments exit, create an owned environment and migrate runtime assets."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from project_assets import ASSETS, CACHE, ENVIRONMENT, OWNER, RECORDS, REPO
from project_assets import inventory, mounted_data, owned_environment, project_python_processes, stamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='wait for already-running project Python processes')
    args = parser.parse_args()
    assert REPO == OWNER/'repo/MobileIE-Neo'
    assert Path(sys.prefix).resolve() != (REPO/'.venv').resolve(), 'use system Python, not the borrowed environment'
    lock = (REPO/'.git/project-asset-migration.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = RECORDS/('attempt-'+time.strftime('%Y%m%dT%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    write_json(run/'00-request.json', {'observed_at': stamp(), 'mount': mounted_data(), 'asset_root': str(ASSETS),
                                     'barrier': 'all already-running project venv Python processes must exit'})
    while True:
        processes = project_python_processes()
        if not processes:
            break
        if not args.wait:
            raise RuntimeError('active project processes; migration deferred: '+str(processes))
        print(json.dumps({'stage': 'waiting_for_experiment_exit', 'time': stamp(), 'pids': [r['pid'] for r in processes]}), flush=True)
        time.sleep(10)
    try:
        write_json(run/'01-barrier-passed.json', {'observed_at': stamp(), 'active_project_processes': [], 'mount': mounted_data()})
        ASSETS.mkdir(parents=True, exist_ok=True)
        env = owned_environment()
        python = ENVIRONMENT/'bin/python'
        if not (ENVIRONMENT/'pyvenv.cfg').exists():
            subprocess.run(['/usr/bin/python3.10', '-m', 'venv', '--copies', str(ENVIRONMENT)], check=True, env=env)
        assert 'include-system-site-packages = false' in (ENVIRONMENT/'pyvenv.cfg').read_text()
        commands = [
            [str(python), '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', 'pip==25.3'],
            [str(python), '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', 'https://download.pytorch.org/whl/cu126', 'torch==2.9.1+cu126'],
            [str(python), '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', '-r', str(REPO/'requirements-research.txt')],
            [str(python), '-m', 'pip', 'check'],
        ]
        print(json.dumps({'stage': 'installing_owned_environment', 'time': stamp(), 'prefix': str(ENVIRONMENT)}), flush=True)
        with (run/'environment-install.log').open('w') as stream:
            for command in commands:
                stream.write(json.dumps({'command': command})+'\n')
                stream.flush()
                subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(python), str(REPO/'scripts/verify_project_assets.py'), '--packages-only',
                        '--out', str(run/'02-independent-packages.json')], cwd=REPO, env=env, check=True)
        with (run/'requirements-installed.txt').open('w') as stream:
            subprocess.run([str(python), '-m', 'pip', 'freeze', '--all'], env=env, stdout=stream, check=True)
        assert not project_python_processes(), 'a new project job started during migration preparation'
        source = REPO/'.cache'
        print(json.dumps({'stage': 'inventory_before_cache_move', 'time': stamp()}), flush=True)
        if source.is_symlink():
            assert source.resolve() == CACHE.resolve()
            before = inventory(CACHE)
        else:
            assert source.is_dir() and not CACHE.exists(), 'refuse to overwrite an existing destination'
            before = inventory(source)
            write_json(run/'03-cache-before.json', before)
            assert source.stat().st_dev == ASSETS.stat().st_dev, 'atomic same-volume move required'
            source.rename(CACHE)
            try:
                source.symlink_to(os.path.relpath(CACHE, source.parent), target_is_directory=True)
            except BaseException:
                CACHE.rename(source)
                raise
            for relative, record in before['links'].items():
                old_target = Path(record['resolved'])
                new_target = CACHE/old_target.relative_to(source) if old_target.is_relative_to(source) else old_target
                link = CACHE/relative
                link.unlink()
                link.symlink_to(os.path.relpath(new_target, link.parent))
        after = inventory(CACHE)
        assert before['files'] == after['files'], 'cache file contents changed during move'
        assert before['file_count'] == after['file_count'] and before['bytes'] == after['bytes']
        write_json(run/'04-cache-after.json', after)
        write_json(run/'05-cache-parity.json', {'observed_at': stamp(), 'files': after['file_count'], 'bytes': after['bytes'],
                                             'all_regular_file_SHA256_equal': True, 'source_link': str(source), 'target': str(CACHE)})
        old_env = REPO/'.venv'
        if old_env.is_symlink():
            assert old_env.resolve() == ENVIRONMENT.resolve()
        else:
            assert (old_env/'pyvenv.cfg').is_file()
            archive = ASSETS/('legacy-environment-'+time.strftime('%Y%m%dT%H%M%S'))
            old_env.rename(archive)
            # Quarantine is owned and inactive; remove the foreign import route immediately.
            shared = archive/'lib/python3.10/site-packages/shared-torch.pth'
            if shared.exists():
                shared.unlink()
            cfg = archive/'pyvenv.cfg'
            cfg.write_text(cfg.read_text().replace('include-system-site-packages = true', 'include-system-site-packages = false'))
            old_env.symlink_to(os.path.relpath(ENVIRONMENT, old_env.parent), target_is_directory=True)
            write_json(run/'06-old-env-isolated.json', {'observed_at': stamp(), 'inactive_archive': str(archive),
                       'foreign_pth_removed': not shared.exists(), 'system_site_packages_disabled': True})
        print(json.dumps({'stage': 'verifying_migrated_links_and_CPU_replay', 'time': stamp()}), flush=True)
        subprocess.run([str(python), str(REPO/'scripts/verify_project_assets.py'), '--out', str(run/'07-final-verification.json')],
                       cwd=REPO, env=env, check=True)
        write_json(run/'08-complete.json', {'observed_at': stamp(), 'completed': True,
                   'asset_root': str(ASSETS), 'environment': str(ENVIRONMENT), 'cache': str(CACHE),
                   'repository_links': {name: os.readlink(REPO/name) for name in ['.venv', '.cache']},
                   'scope': 'Owned public-package installation; no other user environment was changed or copied.'})
        write_json(RECORDS/'completed.json', {'completed': True, 'observed_at': stamp(), 'attempt': str(run.relative_to(REPO))})
        print(json.dumps({'stage': 'completed', 'records': str(run), 'time': stamp()}), flush=True)
    except BaseException as error:
        write_json(run/'failure.json', {'observed_at': stamp(), 'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
