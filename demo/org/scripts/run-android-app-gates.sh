#!/usr/bin/env bash
# Preserve private test outputs for the independent numerical gate (UTP uninstalls).
set -euo pipefail
source /root/android-dev/activate.sh
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
evidence="$ANDROID_DEV_ROOT/validation/app-final-gates"
serial=emulator-5580
mkdir -p "$evidence/android-outputs"
bash "$repo/demo/org/app/build.sh" assembleDebug assembleDebugAndroidTest lintDebug > "$evidence/build-lint.txt" 2>&1
adb -s "$serial" install -r "$repo/demo/org/app/app/build/outputs/apk/debug/app-debug.apk"
adb -s "$serial" install -r "$repo/demo/org/app/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
adb -s "$serial" shell am instrument -w -r \
    -e class dev.zrh.mobileie.CoreInstrumentedTest,dev.zrh.mobileie.FlowInstrumentedTest,dev.zrh.mobileie.LifecycleRegressionTest,dev.zrh.mobileie.ThreeTaskInstrumentedTest \
    dev.zrh.mobileie.test/androidx.test.runner.AndroidJUnitRunner > "$evidence/instrumentation.txt" 2>&1
if ! grep -Eq '^OK \(26 tests\)' "$evidence/instrumentation.txt" || grep -Eq 'FAILURES!!!|INSTRUMENTATION_FAILED|Process crashed' "$evidence/instrumentation.txt"; then
    tail -60 "$evidence/instrumentation.txt"; exit 1
fi
adb -s "$serial" exec-out run-as dev.zrh.mobileie tar -C files/gates -cf - . | tar -xf - -C "$evidence/android-outputs"
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I "$repo/demo/org/scripts/verify-android-numerics.py" \
    --android-dir "$evidence/android-outputs" --output "$evidence/numerical-reference" > "$evidence/numerics.txt" 2>&1
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I "$repo/demo/org/scripts/verify-android-tasks.py" \
    --android-dir "$evidence/android-outputs/tasks" --output "$evidence/task-reference" > "$evidence/task-numerics.txt" 2>&1
apksigner verify --verbose "$repo/demo/org/app/app/build/outputs/apk/debug/app-debug.apk" > "$evidence/signature.txt"
zipalign -c -P 16 4 "$repo/demo/org/app/app/build/outputs/apk/debug/app-debug.apk"
sha256sum "$repo/demo/org/app/app/build/outputs/apk/debug/app-debug.apk" > "$evidence/apk-sha256.txt"
echo "Automated App gates passed. Review screenshots and process restoration separately: $evidence"
