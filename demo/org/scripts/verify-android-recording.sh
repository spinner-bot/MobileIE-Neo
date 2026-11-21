#!/usr/bin/env bash
# Decode the actual demo MP4 using Android's media stack on the owned AVD.
set -euo pipefail
source /root/android-dev/activate.sh
serial=emulator-5580
evidence="$ANDROID_DEV_ROOT/validation/app-final-gates"
[[ "$(adb -s "$serial" shell getprop ro.kernel.qemu | tr -d '\r')" == 1 ]]
version=$(adb -s "$serial" shell dumpsys package dev.zrh.mobileie | sed -n 's/.*versionName=\([0-9][0-9.]*\).*/\1/p' | head -n 1)
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
movie="$evidence/MobileIE-LightLab-$version-emulator-demo.mp4"
[[ -s "$movie" ]]
# exec-out does not forward stdin. shell -T preserves binary input without a PTY.
adb -s "$serial" shell -T run-as dev.zrh.mobileie tee files/demo-video.mp4 < "$movie" > /dev/null
adb -s "$serial" shell am instrument -w -r -e class dev.zrh.mobileie.VideoArtifactTest \
  dev.zrh.mobileie.test/androidx.test.runner.AndroidJUnitRunner > "$evidence/video-instrumentation.txt"
grep -q '^OK (1 test)' "$evidence/video-instrumentation.txt"
! grep -Eq 'FAILURES!!!|INSTRUMENTATION_FAILED|Process crashed' "$evidence/video-instrumentation.txt"
adb -s "$serial" exec-out run-as dev.zrh.mobileie tar -C files/video-gate -cf - . | tar -xf - -C "$evidence"
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I - "$evidence/video-decoder.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]));assert report['pass'] and report['decoded_midpoint'] and report['duration_ms']>3000
print(json.dumps(report,indent=2))
PY
