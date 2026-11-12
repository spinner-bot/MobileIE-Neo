#!/usr/bin/env python3
"""Create immutable, checksum-verified Android and Python deliverables."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

REPO=Path(__file__).resolve().parents[3]
RELEASE=Path('/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.1.0')
EXPECTED_APK='333028916ca87aab4eae129a49877f29806eb2f810f76f776dc9c0c7d21f25a6'
MODEL='2ea09b840634469eceab5263bf7b549c33fcd5df2a25701ce5fefc585402ad44'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def immutable_copy(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise RuntimeError(f'Refuse to replace immutable release: {target}')
    else:shutil.copyfile(source,target)


def main():
    qa=json.loads((REPO/'demo/org/results/app-0.1.0/qa-summary.json').read_text())
    assert qa['instrumentation']=={'tests':14,'failures':0,'seconds':91.375}
    assert qa['numerics']['all_pass'] and qa['supplementary_export_roundtrip']['failures']==0
    assert json.loads((REPO/'demo/org/results/desktop-0.1.0/desktop-gui.json').read_text())['pass']
    assert json.loads((REPO/'demo/org/results/desktop-0.1.0/desktop-numerics.json').read_text())['pass']
    RELEASE.mkdir(parents=True,exist_ok=True)
    apk=REPO/'demo/org/app/app/build/outputs/apk/debug/app-debug.apk'
    assert sha(apk)==EXPECTED_APK
    with zipfile.ZipFile(apk) as archive:
        assert hashlib.sha256(archive.read('assets/model.tflite')).hexdigest()==MODEL
        assert all(f'lib/{abi}/libtensorflowlite_jni.so' in archive.namelist() for abi in ('arm64-v8a','x86_64'))
    installed=RELEASE/'MobileIE-LightLab-0.1.0.apk';immutable_copy(apk,installed)
    movie=RELEASE/'MobileIE-LightLab-0.1.0-emulator-demo.mp4'
    immutable_copy(Path('/data/Users/zrh/.envs/android/validation/app-final-gates')/movie.name,movie)
    desktop=RELEASE/'MobileIE-LightLab-Desktop-Python-0.1.0.zip'
    desktop_temp=desktop.with_suffix('.zip.part')
    with zipfile.ZipFile(desktop_temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in ('desktop.py','requirements.txt','run.cmd','QUICKSTART.txt'):
            archive.write(REPO/'demo/org/desktop'/name,name)
        for asset in sorted((REPO/'demo/org/app/app/src/main/assets').rglob('*')):
            if asset.is_file():archive.write(asset,'assets/'+asset.relative_to(REPO/'demo/org/app/app/src/main/assets').as_posix())
    with zipfile.ZipFile(desktop_temp) as archive:assert archive.testzip() is None
    immutable_copy(desktop_temp,desktop);desktop_temp.unlink()
    kit=RELEASE/'MobileIE-LightLab-Android-Demo-0.1.0.zip';kit_temp=kit.with_suffix('.zip.part')
    with zipfile.ZipFile(kit_temp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        archive.write(installed,installed.name);archive.write(movie,movie.name)
        archive.write(REPO/'demo/org/demo-package/微光实验室操作说明.txt','手机操作说明.txt')
        archive.writestr('SHA256SUMS.txt',EXPECTED_APK+'  '+installed.name+'\n'+sha(movie)+'  '+movie.name+'\n')
    with zipfile.ZipFile(kit_temp) as archive:assert archive.testzip() is None
    immutable_copy(kit_temp,kit);kit_temp.unlink()
    manifest={'version':'0.1.0','date':'2026-11-13','application_id':'dev.zrh.mobileie','apk_signing':'development debug signature; not an app-store release',
              'abis':['arm64-v8a','x86_64'],'min_sdk':29,'target_sdk':36,'model_sha256':MODEL,
              'android_cpu_threads':4,'desktop_cpu_threads':1,
              'native_oneplus_tested':False,'native_windows_tested':False,
              'artifacts':[{'name':f.name,'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in (installed,kit,desktop,movie)]}
    text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    (RELEASE/'release.json').write_text(text,encoding='utf-8')
    (REPO/'demo/org/results/app-0.1.0/release.json').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
