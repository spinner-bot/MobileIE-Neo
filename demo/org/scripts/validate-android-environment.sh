#!/usr/bin/env bash
# Generates only a disposable toolchain probe, not the MobileIE application.
set -euo pipefail
source /data/Users/zrh/.envs/android/activate.sh
probe="$ANDROID_DEV_ROOT/validation/toolchain-probe"
mkdir -p "$probe/app/src/main/java/dev/zrh/environmentprobe" "$probe/app/src/main/cpp" "$ANDROID_DEV_ROOT/records"
cat > "$probe/settings.gradle" <<'EOF'
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = 'AndroidToolchainProbe'
include ':app'
EOF
cat > "$probe/build.gradle" <<'EOF'
plugins {
    id 'com.android.application' version '8.11.1' apply false
    id 'org.jetbrains.kotlin.android' version '2.2.10' apply false
}
EOF
cat > "$probe/gradle.properties" <<EOF
org.gradle.jvmargs=-Xmx1536m -Djava.io.tmpdir=$TMPDIR -Duser.home=$ANDROID_DEV_ROOT/validation/user-home
org.gradle.workers.max=2
org.gradle.java.installations.auto-detect=false
org.gradle.java.installations.auto-download=false
kotlin.compiler.execution.strategy=in-process
android.useAndroidX=true
EOF
cat > "$probe/app/build.gradle" <<'EOF'
plugins { id 'com.android.application'; id 'org.jetbrains.kotlin.android' }
android {
    namespace 'dev.zrh.environmentprobe'
    compileSdk 36
    buildToolsVersion '35.0.0'
    ndkVersion '27.3.13750724'
    defaultConfig {
        applicationId 'dev.zrh.environmentprobe'
        minSdk 29
        targetSdk 35
        versionCode 1
        versionName 'toolchain-probe'
        ndk { abiFilters 'arm64-v8a', 'x86_64' }
        externalNativeBuild { cmake { cppFlags '-std=c++17' } }
    }
    externalNativeBuild { cmake { path file('src/main/cpp/CMakeLists.txt'); version '3.22.1' } }
    compileOptions { sourceCompatibility JavaVersion.VERSION_17; targetCompatibility JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = '17' }
}
dependencies {
    implementation 'com.google.ai.edge.litert:litert:1.4.2'
    implementation 'com.google.ai.edge.litert:litert-gpu:1.4.2'
    implementation 'androidx.core:core:1.15.0'
}
EOF
cat > "$probe/app/src/main/AndroidManifest.xml" <<'EOF'
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application android:label="Toolchain probe only" android:allowBackup="false" />
</manifest>
EOF
cat > "$probe/app/src/main/java/dev/zrh/environmentprobe/Probe.kt" <<'EOF'
package dev.zrh.environmentprobe
// Compile-only probe. No Activity, camera, images or MobileIE app functionality.
class Probe {
    fun add(a: Int, b: Int): Int = a + b
    external fun nativeVersion(): Int
}
EOF
cat > "$probe/app/src/main/cpp/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.22.1)
project(android_toolchain_probe LANGUAGES CXX)
add_library(toolchain_probe SHARED probe.cpp)
target_compile_features(toolchain_probe PRIVATE cxx_std_17)
target_link_options(toolchain_probe PRIVATE "-Wl,-z,max-page-size=16384")
EOF
cat > "$probe/app/src/main/cpp/probe.cpp" <<'EOF'
#include <jni.h>
extern "C" JNIEXPORT jint JNICALL
Java_dev_zrh_environmentprobe_Probe_nativeVersion(JNIEnv*, jobject) { return 1; }
EOF
java -version
javac -version
gradle --version
adb version
cmake --version
ninja --version
"$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" --version
gradle -p "$probe" --no-daemon --console=plain assembleDebug assembleRelease
key="$probe/probe-signing.keystore"
if [[ ! -f "$key" ]]; then
    keytool -genkeypair -noprompt -keystore "$key" -storepass android -keypass android \
        -alias toolchain-probe -dname 'CN=Disposable toolchain validation' -keyalg RSA -keysize 2048 -validity 365
    chmod 600 "$key"
fi
release="$probe/app/build/outputs/apk/release/app-release-unsigned.apk"
signed="$probe/app/build/outputs/apk/release/app-release-probe-signed.apk"
apksigner sign --ks "$key" --ks-key-alias toolchain-probe --ks-pass pass:android --key-pass pass:android --out "$signed" "$release"
apksigner verify --verbose "$signed"
zipalign -c -P 16 -v 4 "$signed" > "$ANDROID_DEV_ROOT/records/zipalign.txt"
aapt dump badging "$signed" > "$ANDROID_DEV_ROOT/records/probe-apk-badging.txt"
sha256sum "$probe/app/build/outputs/apk/debug/app-debug.apk" "$signed" > "$ANDROID_DEV_ROOT/records/probe-apk-sha256.txt"
sdkmanager --sdk_root="$ANDROID_HOME" --list_installed > "$ANDROID_DEV_ROOT/records/sdk-packages.txt"
echo 'Build, Kotlin, native cross-compilation, signing and alignment checks passed.'
