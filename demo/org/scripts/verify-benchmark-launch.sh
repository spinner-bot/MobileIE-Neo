#!/usr/bin/env bash
# Installed-package branch only, using an explicitly labeled stub on the owned AVD.
set -euo pipefail
source /root/android-dev/activate.sh
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
serial=emulator-5580
if adb -s "$serial" shell pm list packages org.benchmark.demo | grep -q '^package:org.benchmark.demo'; then
    echo 'A package already exists; refusing to replace it with a test stub' >&2; exit 1
fi
probe="$ANDROID_DEV_ROOT/validation/benchmark-launch-stub"
mkdir -p "$probe/app/src/main/java/org/benchmark/demo"
cat > "$probe/settings.gradle" <<'EOF'
pluginManagement { repositories { google(); mavenCentral() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name='BenchmarkLaunchStubOnly'
include ':app'
EOF
cat > "$probe/build.gradle" <<'EOF'
plugins { id 'com.android.application' version '8.11.1' apply false }
EOF
cat > "$probe/app/build.gradle" <<'EOF'
plugins { id 'com.android.application' }
android {
    namespace 'org.benchmark.demo'
    compileSdk 36
    defaultConfig { applicationId 'org.benchmark.demo'; minSdk 29; targetSdk 36; versionCode 1; versionName 'test-stub-only' }
}
EOF
cat > "$probe/app/src/main/AndroidManifest.xml" <<'EOF'
<manifest xmlns:android="http://schemas.android.com/apk/res/android"><application android:label="LAUNCH TEST STUB ONLY">
<activity android:name=".StubActivity" android:exported="true"><intent-filter><action android:name="android.intent.action.MAIN"/><category android:name="android.intent.category.LAUNCHER"/></intent-filter></activity>
</application></manifest>
EOF
cat > "$probe/app/src/main/java/org/benchmark/demo/StubActivity.java" <<'EOF'
package org.benchmark.demo;
public class StubActivity extends android.app.Activity {
    @Override public void onCreate(android.os.Bundle state) {
        super.onCreate(state); android.widget.TextView view=new android.widget.TextView(this);
        view.setText("EXTERNAL LAUNCH GATE PASSED — TEST STUB ONLY"); view.setTextSize(22); setContentView(view);
    }
}
EOF
gradle -p "$probe" --no-daemon --max-workers=2 --console=plain assembleDebug
adb -s "$serial" install "$probe/app/build/outputs/apk/debug/app-debug.apk"
trap 'adb -s emulator-5580 uninstall org.benchmark.demo >/dev/null' EXIT
"$ANDROID_DEV_ROOT/validation/reference-env/bin/python" -I - "$repo/demo/org/scripts/verify-android-lifecycle.py" <<'PY'
import importlib.util,json,sys,time
spec=importlib.util.spec_from_file_location('lifecycle_gate',sys.argv[1]); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.adb('shell','am','start','-W','-n',m.PKG+'/.MainActivity')
for _ in range(3): m.adb('shell','input','swipe','540','1850','540','550','250')
m.tap_resource(m.PKG+':id/benchmark')
m.wait_text('EXTERNAL LAUNCH GATE PASSED')
root=m.ui()
assert any(n.get('package')=='org.benchmark.demo' for n in root.iter('node'))
(m.BASE/'benchmark-installed-branch.xml').write_bytes(__import__('xml.etree.ElementTree',fromlist=['tostring']).tostring(root))
(m.BASE/'benchmark-installed-branch.json').write_text(json.dumps({'pass':True,'real_benchmark_app':False,'test_stub_only':True,'external_package_launched':'org.benchmark.demo','scope':'Launch intent only; does not validate actual AI Benchmark screens or inference'},indent=2)+'\n')
print('External launch branch passed with explicitly labeled test stub')
PY
