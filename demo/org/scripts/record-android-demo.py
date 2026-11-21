#!/usr/bin/env python3
"""Record all three tasks on the owned emulator after exporting gate evidence."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

spec=importlib.util.spec_from_file_location('lifecycle',Path(__file__).with_name('verify-android-lifecycle.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.adb('shell','getprop','ro.kernel.qemu').decode().strip()=='1'
assert json.loads((m.BASE/'numerical-reference/numerical-report.json').read_text())['all_pass']
assert json.loads((m.BASE/'task-reference/task-numerical-report.json').read_text())['all_pass']
version_match=re.search(r'versionName=(\d+\.\d+\.\d+)',m.adb('shell','dumpsys','package',m.PKG).decode());assert version_match
version=version_match.group(1)
m.adb('shell','settings','put','system','font_scale','1.0');m.adb('shell','wm','user-rotation','lock','0')
m.adb('shell','am','force-stop',m.PKG);m.adb('shell','pm','clear',m.PKG)
m.adb('shell','am','start','-W','-n',m.PKG+'/.MainActivity');m.wait_text('当前任务：低光增强')
(m.BASE/'01-fresh-home.png').write_bytes(m.adb('exec-out','screencap','-p'))
log=(m.BASE/'screenrecord.txt').open('wb');m.adb('shell','rm','-f','/sdcard/lightlab-demo.mp4')
recorder=subprocess.Popen(['adb','-s','emulator-5580','shell','screenrecord','--size','720x1560','--bit-rate','1800000','--time-limit','180','/sdcard/lightlab-demo.mp4'],stdout=log,stderr=log)
try:
    for task in ('lowlight','underwater','isp'):
        if task=='lowlight':
            m.tap_resource(m.PKG+':id/sample')
            for node in m.ui().iter('node'):
                if node.get('text','').startswith('01 ·'):
                    x1,y1,x2,y2=map(int,re.findall(r'\d+',node.get('bounds')));m.adb('shell','input','tap',str((x1+x2)//2),str((y1+y2)//2));break
            else:raise AssertionError('Sample choice missing')
        else:m.tap_resource(m.PKG+':id/task_'+task)
        m.wait_text('图片已准备');m.tap_resource(m.PKG+':id/process');m.wait_text('处理完成')
        assert m.record()['task_id']==task
        m.tap_resource(m.PKG+':id/display_dynamic')
        for node in m.ui().iter('node'):
            if node.get('resource-id')==m.PKG+':id/comparison':
                x1,y1,x2,y2=map(int,re.findall(r'\d+',node.get('bounds')));width=x2-x1
                for start,end in ((.1,.4),(.9,.6)):
                    m.adb('shell','input','swipe',str(int(x1+width*start)),str((y1+y2)//2),str(int(x1+width*end)),str((y1+y2)//2),'500')
                break
        else:raise AssertionError('Comparison view missing')
        (m.BASE/f'reviewed-{task}-dynamic.png').write_bytes(m.adb('exec-out','screencap','-p'))
        m.tap_resource(m.PKG+':id/display_static');time.sleep(.4)
        (m.BASE/f'reviewed-{task}-static.png').write_bytes(m.adb('exec-out','screencap','-p'))
    m.adb('shell','input','swipe','540','1900','540','600','300');m.tap_resource(m.PKG+':id/save')
    for _ in range(2):m.adb('shell','input','swipe','540','600','540','1900','300')
    m.wait_text('已保存')
finally:
    for pid in m.adb('shell','pidof','screenrecord',check=False).decode().split():
        if pid.isdigit():m.adb('shell','kill','-2',pid,check=False)
    recorder.wait(timeout=15);log.close()
movie=b''
for _ in range(40):
    movie=m.adb('exec-out','cat','/sdcard/lightlab-demo.mp4')
    if len(movie)>10000 and b'ftyp' in movie[:64] and b'moov' in movie:break
    time.sleep(.25)
else:raise AssertionError('MP4 finalization timed out')
(m.BASE/f'MobileIE-LightLab-{version}-emulator-demo.mp4').write_bytes(movie)
(m.BASE/'recording.json').write_text(json.dumps({'pass':True,'version':version,'bytes':len(movie),'source':'Android 15 emulator, not real phone','workflow':['fresh launch','lowlight sample','underwater sample','real RAW ISP sample','paired synchronized sliders for each task','independent static overview for each task','save RAW and labeled results'],'mp4_ftyp_and_moov_verified':True},indent=2)+'\n')
print('Recorded emulator workflow',len(movie),'bytes')
