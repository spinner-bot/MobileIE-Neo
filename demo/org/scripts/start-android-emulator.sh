#!/usr/bin/env bash
# Start only this user's isolated AVD and adb server.
set -euo pipefail
source /data/Users/zrh/.envs/android/activate.sh
name=zrh-api35
serial=emulator-5580
if [[ ! -f "$ANDROID_AVD_HOME/$name.ini" ]]; then
    printf 'no\n' | avdmanager create avd -n "$name" -k 'system-images;android-35;google_apis;x86_64' -d pixel_5
fi
adb start-server
if ! adb -s "$serial" get-state >/dev/null 2>&1; then
    if ss -ltn | awk '{print $4}' | grep -Eq ':(5580|5581)$'; then
        echo 'Emulator ports are occupied; refusing to reuse an unknown process' >&2; exit 1
    fi
    nohup emulator -avd "$name" -port 5580 -no-window -no-audio -no-boot-anim \
        -no-snapshot -gpu swiftshader -accel on -memory 2048 -cores 2 \
        -camera-back emulated -camera-front emulated \
        > "$ANDROID_DEV_ROOT/logs/emulator-api35.log" 2>&1 < /dev/null &
    echo $! > "$ANDROID_DEV_ROOT/records/emulator-api35.pid"
fi
for attempt in $(seq 1 120); do
    if [[ $(adb -s "$serial" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r') == 1 ]]; then
        adb -s "$serial" shell input keyevent 82
        adb -s "$serial" shell settings put global window_animation_scale 0
        adb -s "$serial" shell settings put global transition_animation_scale 0
        adb -s "$serial" shell settings put global animator_duration_scale 0
        adb -s "$serial" shell getprop > "$ANDROID_DEV_ROOT/records/emulator-properties.txt"
        echo "AVD ready: $serial (private adb localhost:5038)"
        exit 0
    fi
    sleep 2
done
echo 'AVD did not boot within the readiness check; inspect logs before retrying' >&2
exit 1
