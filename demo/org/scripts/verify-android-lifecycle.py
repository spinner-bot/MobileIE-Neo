#!/usr/bin/env python3
"""Host-side process death and camera-result recovery gate, on the owned AVD."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

PKG = "dev.zrh.mobileie"
BASE = Path(os.environ["ANDROID_DEV_ROOT"]) / "validation/app-final-gates"


def adb(*args, check=True):
    return subprocess.run(["adb", "-s", "emulator-5580", *args], check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def ui():
    adb("shell", "uiautomator", "dump", "/sdcard/lifecycle-gate.xml")
    return ET.fromstring(adb("exec-out", "cat", "/sdcard/lifecycle-gate.xml"))


def tap_resource(resource):
    for node in ui().iter("node"):
        if node.get("resource-id") == resource:
            x1, y1, x2, y2 = map(int, re.findall(r"\d+", node.get("bounds")))
            adb("shell", "input", "tap", str((x1+x2)//2), str((y1+y2)//2))
            return
    raise AssertionError(f"Resource not found: {resource}")


def wait_text(fragment):
    for _ in range(12):
        root = ui()
        if any(fragment in n.get("text", "") for n in root.iter("node")):
            return
        time.sleep(.25)
    raise AssertionError(f"UI text not found: {fragment}")


def session():
    return json.loads(adb("exec-out", "run-as", PKG, "cat", "files/current.json"))["id"]


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    adb("shell", "settings", "put", "system", "font_scale", "1.0")
    adb("shell", "wm", "user-rotation", "lock", "0")
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "am", "start", "-W", "-n", PKG+"/.MainActivity")
    wait_text("已恢复上次照片")
    previous = session()
    tap_resource(PKG+":id/capture")
    for _ in range(12):
        if any(n.get("resource-id") == "com.android.camera2:id/shutter_button" for n in ui().iter("node")):
            break
        time.sleep(.25)
    old_pid = adb("shell", "pidof", PKG).decode().strip()
    assert old_pid.isdigit(), old_pid
    # Kill only this app while the external camera is foreground; preserve task state.
    adb("shell", "run-as", PKG, "/system/bin/kill", "-9", old_pid)
    tap_resource("com.android.camera2:id/shutter_button")
    time.sleep(1.5)
    root = ui()
    done = [n.get("resource-id") for n in root.iter("node") if n.get("resource-id", "").endswith((":id/done_button", ":id/btn_done", ":id/review_done", ":id/done"))]
    if done:
        tap_resource(done[0])
    wait_text("现场拍摄")
    wait_text("图片已准备")
    current = session()
    new_pid = adb("shell", "pidof", PKG).decode().strip()
    assert current != previous and new_pid != old_pid
    tap_resource(PKG+":id/process")
    wait_text("处理完成")
    record = json.loads(adb("exec-out", "run-as", PKG, "cat", f"files/sessions/{current}/record.json"))
    assert record["source_label"] == "现场拍摄" and record["mobileie_complete"] and record["gamma_complete"]
    (BASE/"10-camera-process-recovered.png").write_bytes(adb("exec-out", "screencap", "-p"))
    report = {"pass": True, "old_pid": old_pid, "new_pid": new_pid, "new_photo_session": current != previous,
              "camera_foreground_process_kill": True, "returned_photo_processed": True,
              "scope": "Android 15 emulator synthetic camera, not OnePlus hardware"}
    (BASE/"camera-process-recovery.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
