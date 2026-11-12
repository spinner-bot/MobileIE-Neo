#!/usr/bin/env python3
"""Create immutable, checksum-verified Android and Python deliverables."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re
import zipfile

REPO=Path(__file__).resolve().parents[3]
RELEASE_ROOT=Path('/data/Users/zrh/data/MobileIE-Neo/demo/org/releases')
MODEL='2ea09b840634469eceab5263bf7b549c33fcd5df2a25701ce5fefc585402ad44'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def immutable_copy(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise RuntimeError(f'Refuse to replace immutable release: {target}')
    else:shutil.copyfile(source,target)


def add_bytes(archive,name,data):
    # Fixed metadata makes repeat packaging byte-identical and safe to audit.
    entry=zipfile.ZipInfo(name,date_time=(2026,11,13,0,0,0))
    entry.compress_type=zipfile.ZIP_DEFLATED
    entry.create_system=3
    entry.external_attr=0o100644 << 16
    archive.writestr(entry,data,compresslevel=9)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',default='0.1.1')
    args=parser.parse_args();version=args.version
    if not re.fullmatch(r'\d+\.\d+\.\d+',version):parser.error('Expected numeric major.minor.patch')
    evidence=REPO/f'demo/org/results/app-{version}'
    qa=json.loads((evidence/'qa-summary.json').read_text())
    assert qa['version']==version
    assert qa['instrumentation']['tests']==14 and qa['instrumentation']['failures']==0
    assert qa['numerics']['all_pass'] and qa['supplementary_export_roundtrip']['failures']==0
    assert json.loads((REPO/f'demo/org/results/desktop-{version}/desktop-gui.json').read_text())['pass']
    assert json.loads((REPO/f'demo/org/results/desktop-{version}/desktop-numerics.json').read_text())['pass']
    assert re.search(r'^VERSION\s*=\s*[\'"]'+re.escape(version)+r'[\'"]',
                     (REPO/'demo/org/desktop/desktop.py').read_text(),re.M)
    release=RELEASE_ROOT/version;release.mkdir(parents=True,exist_ok=True)
    apk=REPO/'demo/org/app/app/build/outputs/apk/debug/app-debug.apk'
    expected_apk=qa['product_apk_sha256'];assert sha(apk)==expected_apk
    assert f'versionName=\'{version}\'' in (evidence/'apk-badging.txt').read_text()
    with zipfile.ZipFile(apk) as archive:
        assert hashlib.sha256(archive.read('assets/model.tflite')).hexdigest()==MODEL
        assert all(f'lib/{abi}/libtensorflowlite_jni.so' in archive.namelist() for abi in ('arm64-v8a','x86_64'))
    installed=release/f'MobileIE-LightLab-{version}.apk';immutable_copy(apk,installed)
    movie=release/f'MobileIE-LightLab-{version}-emulator-demo.mp4'
    immutable_copy(Path('/data/Users/zrh/.envs/android/validation/app-final-gates')/movie.name,movie)
    desktop=release/f'MobileIE-LightLab-Desktop-Python-{version}.zip'
    desktop_temp=desktop.with_suffix('.zip.part')
    with zipfile.ZipFile(desktop_temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in ('desktop.py','requirements.txt','run.cmd','QUICKSTART.txt'):
            add_bytes(archive,name,(REPO/'demo/org/desktop'/name).read_bytes())
        for asset in sorted((REPO/'demo/org/app/app/src/main/assets').rglob('*')):
            if asset.is_file():add_bytes(archive,'assets/'+asset.relative_to(REPO/'demo/org/app/app/src/main/assets').as_posix(),asset.read_bytes())
    with zipfile.ZipFile(desktop_temp) as archive:assert archive.testzip() is None
    immutable_copy(desktop_temp,desktop);desktop_temp.unlink()
    kit=release/f'MobileIE-LightLab-Android-Demo-{version}.zip';kit_temp=kit.with_suffix('.zip.part')
    with zipfile.ZipFile(kit_temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        add_bytes(archive,installed.name,installed.read_bytes());add_bytes(archive,movie.name,movie.read_bytes())
        add_bytes(archive,'手机操作说明.txt',(REPO/'demo/org/demo-package/微光实验室操作说明.txt').read_bytes())
        add_bytes(archive,'SHA256SUMS.txt',(expected_apk+'  '+installed.name+'\n'+sha(movie)+'  '+movie.name+'\n').encode())
    with zipfile.ZipFile(kit_temp) as archive:assert archive.testzip() is None
    immutable_copy(kit_temp,kit);kit_temp.unlink()
    manifest={'version':version,'date':'2026-11-13','application_id':'dev.zrh.mobileie','apk_signing':'development debug signature; not an app-store release',
              'abis':['arm64-v8a','x86_64'],'min_sdk':29,'target_sdk':36,'model_sha256':MODEL,
              'android_cpu_threads':4,'desktop_cpu_threads':1,
              'native_oneplus_tested':False,'native_windows_tested':False,
              'artifacts':[{'name':f.name,'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in (installed,kit,desktop,movie)]}
    text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    (release/'release.json').write_text(text,encoding='utf-8')
    (evidence/'release.json').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
