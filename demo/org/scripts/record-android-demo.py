#!/usr/bin/env python3
"""Record a fresh-install sample workflow only on the owned emulator.

Clears this test App's data after all numerical artifacts have been collected.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

spec=importlib.util.spec_from_file_location("lifecycle",Path(__file__).with_name("verify-android-lifecycle.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.adb("shell","getprop","ro.kernel.qemu").decode().strip()=="1"
assert json.loads((m.BASE/"numerical-reference/numerical-report.json").read_text())["all_pass"]
version_match=re.search(r"versionName=(\d+\.\d+\.\d+)",m.adb("shell","dumpsys","package",m.PKG).decode())
assert version_match, "Installed version missing"
version=version_match.group(1)
m.adb("shell","settings","put","system","font_scale","1.0")
m.adb("shell","wm","user-rotation","lock","0")
m.adb("shell","am","force-stop",m.PKG)
m.adb("shell","pm","clear",m.PKG)
m.adb("shell","am","start","-W","-n",m.PKG+"/.MainActivity")
m.wait_text("选一张暗光照片")
(m.BASE/"01-fresh-home.png").write_bytes(m.adb("exec-out","screencap","-p"))
log=(m.BASE/"screenrecord.txt").open("wb")
m.adb("shell","rm","-f","/sdcard/lightlab-demo.mp4")
recorder=subprocess.Popen(["adb","-s","emulator-5580","shell","screenrecord","--size","720x1560","--bit-rate","1800000","--time-limit","90","/sdcard/lightlab-demo.mp4"],stdout=log,stderr=log)
try:
    time.sleep(.5)
    m.tap_resource(m.PKG+":id/sample")
    for node in m.ui().iter("node"):
        if node.get("text")=="01 · 书柜":
            x1,y1,x2,y2=map(int,re.findall(r"\d+",node.get("bounds")));m.adb("shell","input","tap",str((x1+x2)//2),str((y1+y2)//2));break
    else:raise AssertionError("Sample choice missing")
    m.wait_text("图片已准备");m.tap_resource(m.PKG+":id/process");m.wait_text("处理完成")
    (m.BASE/"11-reviewed-result.png").write_bytes(m.adb("exec-out","screencap","-p"))
    m.tap_resource(m.PKG+":id/mode_gamma")
    for node in m.ui().iter("node"):
        if node.get("resource-id")==m.PKG+":id/comparison":
            x1,y1,x2,y2=map(int,re.findall(r"\d+",node.get("bounds")));m.adb("shell","input","swipe",str(x1+(x2-x1)//4),str((y1+y2)//2),str(x1+(x2-x1)*4//5),str((y1+y2)//2),"500");break
    m.tap_resource(m.PKG+":id/mode_overview");time.sleep(.4)
    (m.BASE/"12-reviewed-overview.png").write_bytes(m.adb("exec-out","screencap","-p"))
    m.adb("shell","input","swipe","540","1900","540","600","300")
    m.tap_resource(m.PKG+":id/save")
    for _ in range(2):m.adb("shell","input","swipe","540","600","540","1900","300")
    m.wait_text("已保存")
finally:
    for pid in m.adb("shell","pidof","screenrecord",check=False).decode().split():
        if pid.isdigit():m.adb("shell","kill","-2",pid,check=False)
    recorder.wait(timeout=15);log.close()
# The adb shell can exit before MediaMuxer's final moov write reaches FUSE.
# Wait for the actual container, rather than accepting only shell completion.
movie=b""
for _ in range(40):
    movie=m.adb("exec-out","cat","/sdcard/lightlab-demo.mp4")
    if len(movie)>10000 and b"ftyp" in movie[:64] and b"moov" in movie:break
    time.sleep(.25)
else:raise AssertionError("MP4 finalization timed out")
(m.BASE/f"MobileIE-LightLab-{version}-emulator-demo.mp4").write_bytes(movie)
(m.BASE/"recording.json").write_text(json.dumps({"pass":True,"version":version,"bytes":len(movie),"source":"Android 15 emulator, not real phone","workflow":["fresh launch","sample selection","local processing","Gamma slider","three-way overview","save"],"mp4_ftyp_and_moov_verified":True},indent=2)+"\n")
print("Recorded emulator workflow",len(movie),"bytes")
