#!/usr/bin/env python3
"""Actual 0.1.2→0.2.0 data-preserving install on the owned disposable AVD."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

spec=importlib.util.spec_from_file_location('lifecycle',Path(__file__).with_name('verify-android-lifecycle.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
repo=Path(__file__).resolve().parents[3]
assert m.adb('shell','getprop','ro.kernel.qemu').decode().strip()=='1'
assert json.loads((m.BASE/'recording.json').read_text())['version']=='0.2.0'
old=Path('/data/Users/zrh/data/MobileIE-Neo/demo/org/releases/0.1.2/MobileIE-LightLab-0.1.2.apk')
new=repo/'demo/org/app/app/build/outputs/apk/debug/app-debug.apk'
m.adb('shell','am','force-stop',m.PKG);m.adb('shell','pm','clear',m.PKG)
m.adb('install','-r','-d',str(old));m.adb('shell','am','start','-W','-n',m.PKG+'/.MainActivity')
m.wait_text('选一张暗光照片');m.tap_resource(m.PKG+':id/sample')
for node in m.ui().iter('node'):
    if node.get('text','').startswith('01 ·'):
        x1,y1,x2,y2=map(int,re.findall(r'\d+',node.get('bounds')));m.adb('shell','input','tap',str((x1+x2)//2),str((y1+y2)//2));break
else:raise AssertionError('Legacy sample choice missing')
m.wait_text('图片已准备');m.tap_resource(m.PKG+':id/process');m.wait_text('处理完成')
previous=m.session();before=m.record();assert before['app_version']=='0.1.2' and 'task_id' not in before
base=f'files/sessions/{previous}/'
def file_sha(name):return hashlib.sha256(m.adb('exec-out','run-as',m.PKG,'cat',base+name)).hexdigest()
original={name:file_sha(name) for name in ('input.png','gamma.png','mobileie.png')}
m.adb('shell','am','force-stop',m.PKG);m.adb('install','-r',str(new))
m.adb('shell','am','start','-W','-n',m.PKG+'/.MainActivity');m.wait_text('已恢复上次照片');m.wait_text('本次任务：低光增强')
assert m.session()==previous and m.record()==before
assert original=={name:file_sha(name) for name in original}
(m.BASE/'upgrade-legacy-restored.png').write_bytes(m.adb('exec-out','screencap','-p'))
m.tap_resource(m.PKG+':id/process');m.wait_text('处理完成')
after=m.record();assert m.session()==previous and after['app_version']=='0.2.0' and after['task_id']=='lowlight'
assert after['mobileie_complete'] and after['baseline_complete'] and file_sha('input.png')==original['input.png']
report={'pass':True,'from_version':'0.1.2','to_version':'0.2.0','real_apk_replace_install':True,'same_session':True,'input_and_old_outputs_preserved':True,
        'old_disk_record_unchanged_until_reprocess':True,'legacy_gamma_restored_as_baseline':True,'task_label_lowlight':True,'reprocessed_record_has_current_version_and_task':True,
        'scope':'Owned Android 15 emulator, not OnePlus hardware'}
(m.BASE/'upgrade-from-0.1.2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
