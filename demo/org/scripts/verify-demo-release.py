#!/usr/bin/env python3
"""Verify published bytes, extracted Python task inference/export and older releases."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

REPO=Path(__file__).resolve().parents[3]
ROOT=Path('/data/Users/zrh/data/MobileIE-Neo/demo/org/releases')
STATION=Path('/data/Users/zrh/.envs/android/validation')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',default='0.2.0');a=p.parse_args()
    import numpy as np
    from PIL import Image
    release=ROOT/a.version;manifest=json.loads((release/'release.json').read_text());assert manifest['version']==a.version
    for artifact in manifest['artifacts']:
        path=release/artifact['name'];assert path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256']
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:assert z.testzip() is None
    apk=release/f'MobileIE-LightLab-{a.version}.apk';kit=release/f'MobileIE-LightLab-Android-Demo-{a.version}.zip'
    with zipfile.ZipFile(kit) as z:
        assert hashlib.sha256(z.read(apk.name)).hexdigest()==sha(apk)
        for t in manifest['tasks']:assert hashlib.sha256(z.read('models/'+t['model_asset'])).hexdigest()==t['model_sha256']
    desktop_zip=release/f'MobileIE-LightLab-Desktop-Python-{a.version}.zip';cases=[]
    with tempfile.TemporaryDirectory(prefix='portable-'+a.version+'-',dir=STATION) as directory:
        destination=Path(directory)
        with zipfile.ZipFile(desktop_zip) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(destination)
        for name in ('desktop.py','lightlab_tasks.py','requirements.txt','run.cmd','QUICKSTART.txt'):
            assert sha(destination/name)==sha(REPO/'demo/org/desktop'/name)
        for asset in (REPO/'demo/org/app/app/src/main/assets').rglob('*'):
            if asset.is_file():assert sha(asset)==sha(destination/'assets'/asset.relative_to(REPO/'demo/org/app/app/src/main/assets'))
        sys.path.insert(0,str(destination));core=importlib.import_module('lightlab_tasks');app=importlib.import_module('desktop')
        assert core.ASSETS==destination/'assets' and app.VERSION==a.version
        references=Path('/data/Users/zrh/data/MobileIE-Neo/demo/org/three-task/models-v2')
        for task_id,task in core.TASKS.items():
            engine=core.Engine(task_id)
            for index,asset in enumerate(task['sample_assets'],1):
                value=core.prepare(core.ASSETS/asset,task_id);result,raw,_=engine.run(value);fixed=core.baseline(value)
                delta=0.0
                if task_id!='lowlight':delta=float(np.max(np.abs(raw.astype(float)-np.load(references/f'{task_id}-sample{index}-reference.npy').astype(float))))
                assert np.isfinite(raw).all() and delta<=1e-4 and result.size==fixed.size==value.image.size
                record={'app_version':app.VERSION,'task_id':task_id,'task_name':task['name'],'baseline_name':task['baseline_name'],'model_sha256':task['model_sha256'],'native_control_inputs':[]}
                saved=app.save_results(destination/'exports',{'Input':value.image,'MobileIE':result,'Baseline':fixed},record,value)
                assert json.loads((saved/'record.json').read_text())==record
                with Image.open(saved/'comparison.png') as image:assert image.size==(value.image.width*3,value.image.height+110)
                if task_id=='isp':assert (saved/'input.rggb').read_bytes()==(core.ASSETS/asset).read_bytes()
                cases.append({'task_id':task_id,'sample':index,'pass':True,'official_reference_raw_max_abs':delta,'export_verified':True})
    old_versions=[]
    for old in sorted(ROOT.glob('0.1.*/release.json')):
        metadata=json.loads(old.read_text())
        for artifact in metadata['artifacts']:
            file=old.parent/artifact['name'];assert sha(file)==artifact['sha256'] and file.stat().st_size==artifact['bytes']
        old_versions.append(metadata['version'])
    report={'pass':True,'version':a.version,'all_artifact_sha256_verified':True,'zip_crc_verified':True,'embedded_apk_and_three_models_verified':True,
            'extracted_desktop_source_and_assets_match':True,'portable_inference_and_export_cases':cases,'old_releases_unchanged':old_versions,
            'scope':'Extracted Python ZIP tested on Linux; native Windows and OnePlus 0.2.0 not tested'}
    (REPO/f'demo/org/results/app-{a.version}/package-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
