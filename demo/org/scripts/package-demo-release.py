#!/usr/bin/env python3
"""Package accepted Android/Python 0.2.0 builds without replacing older releases."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

REPO=Path(__file__).resolve().parents[3]
RELEASE_ROOT=Path('/data/Users/zrh/data/MobileIE-Neo/demo/org/releases')
STATION=Path('/data/Users/zrh/.envs/android/validation/app-final-gates')
ASSETS=REPO/'demo/org/app/app/src/main/assets'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def immutable_copy(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise RuntimeError(f'Refuse to replace immutable release: {target}')
    else:shutil.copyfile(source,target)


def add_bytes(archive,name,data):
    entry=zipfile.ZipInfo(name,date_time=(2026,11,14,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
    entry.create_system=3;entry.external_attr=0o100644 << 16
    archive.writestr(entry,data,compresslevel=9)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='0.2.0')
    args=parser.parse_args();version=args.version
    if not re.fullmatch(r'\d+\.\d+\.\d+',version):parser.error('Expected numeric major.minor.patch')
    evidence=REPO/f'demo/org/results/app-{version}';qa=json.loads((evidence/'qa-summary.json').read_text())
    assert qa['version']==version and qa['accepted_for_simulated_delivery']
    assert qa['instrumentation']['tests']==26 and qa['instrumentation']['failures']==0
    assert qa['numerics']['all_pass'] and qa['task_numerics']['all_pass']
    assert qa['camera_foreground_process_recovery']['lowlight'] and qa['camera_foreground_process_recovery']['underwater']
    assert qa['raw_cold_restore'] and qa['visual_review']['pass']
    for name in ('desktop-gui.json','desktop-numerics.json'):
        assert json.loads((REPO/f'demo/org/results/desktop-{version}'/name).read_text())['pass']
    assert re.search(r'^VERSION\s*=\s*[\'"]'+re.escape(version)+r'[\'"]',(REPO/'demo/org/desktop/desktop.py').read_text(),re.M)
    assert json.loads((evidence/'video-decoder.json').read_text())['pass']
    tasks=json.loads((ASSETS/'tasks.json').read_text())['tasks'];assert len(tasks)==3
    release=RELEASE_ROOT/version;release.mkdir(parents=True,exist_ok=True)
    apk=REPO/'demo/org/app/app/build/outputs/apk/debug/app-debug.apk'
    assert sha(apk)==qa['product_apk_sha256']
    assert f'versionName=\'{version}\'' in (evidence/'apk-badging.txt').read_text()
    with zipfile.ZipFile(apk) as archive:
        assert all(hashlib.sha256(archive.read('assets/'+t['model_asset'])).hexdigest()==t['model_sha256'] and not t['native_control_inputs'] for t in tasks)
        assert all(f'lib/{abi}/libtensorflowlite_jni.so' in archive.namelist() for abi in ('arm64-v8a','x86_64'))
        assert 'assets/credits.txt' in archive.namelist()
    installed=release/f'MobileIE-LightLab-{version}.apk';immutable_copy(apk,installed)
    movie=release/f'MobileIE-LightLab-{version}-emulator-demo.mp4';immutable_copy(STATION/movie.name,movie)
    desktop=release/f'MobileIE-LightLab-Desktop-Python-{version}.zip';temp=desktop.with_suffix('.zip.part')
    with zipfile.ZipFile(temp,'w') as archive:
        for name in ('desktop.py','lightlab_tasks.py','requirements.txt','run.cmd','QUICKSTART.txt'):
            add_bytes(archive,name,(REPO/'demo/org/desktop'/name).read_bytes())
        for asset in sorted(ASSETS.rglob('*')):
            if asset.is_file():add_bytes(archive,'assets/'+asset.relative_to(ASSETS).as_posix(),asset.read_bytes())
    with zipfile.ZipFile(temp) as archive:assert archive.testzip() is None
    immutable_copy(temp,desktop);temp.unlink()
    kit=release/f'MobileIE-LightLab-Android-Demo-{version}.zip';temp=kit.with_suffix('.zip.part')
    with zipfile.ZipFile(temp,'w') as archive:
        add_bytes(archive,installed.name,installed.read_bytes());add_bytes(archive,movie.name,movie.read_bytes())
        add_bytes(archive,'手机操作说明.txt',(REPO/'demo/org/demo-package/微光实验室操作说明.txt').read_bytes())
        for task in tasks:add_bytes(archive,'models/'+task['model_asset'],(ASSETS/task['model_asset']).read_bytes())
        for name in ('tasks.json','credits.txt','MobileIE-LICENSE.txt','task-asset-sources.json'):
            add_bytes(archive,'models/'+name,(ASSETS/name).read_bytes())
        for sample in sorted((ASSETS/'samples/isp').glob('*.rggb')):add_bytes(archive,'RAW-samples/'+sample.name,sample.read_bytes())
        add_bytes(archive,'SHA256SUMS.txt',(sha(installed)+'  '+installed.name+'\n'+sha(movie)+'  '+movie.name+'\n'+''.join(t['model_sha256']+'  models/'+t['model_asset']+'\n' for t in tasks)).encode())
    with zipfile.ZipFile(temp) as archive:assert archive.testzip() is None
    immutable_copy(temp,kit);temp.unlink()
    manifest={'version':version,'date':qa['date'],'application_id':'dev.zrh.mobileie','apk_signing':'development debug signature; not an app-store release',
              'abis':['arm64-v8a','x86_64'],'min_sdk':29,'target_sdk':36,'tasks':tasks,'display_modes':['paired synchronized sliders','independent three-image overview'],
              'android_cpu_threads':4,'desktop_cpu_threads':1,'native_oneplus_tested':False,'native_windows_tested':False,
              'artifacts':[{'name':f.name,'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in (installed,kit,desktop,movie)]}
    result=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n';candidate=release/'release.json.part';candidate.write_text(result)
    immutable_copy(candidate,release/'release.json');candidate.unlink();(evidence/'release.json').write_text(result)
    print(result)


if __name__=='__main__':main()
