#!/usr/bin/env bash
# Personal Android build station; does not create the MobileIE application.
set -euo pipefail
ANDROID_DEV_ROOT=/data/Users/zrh/.envs/android
findmnt -rn -M /data >/dev/null || { echo '/data is not mounted' >&2; exit 1; }
mkdir -p "$ANDROID_DEV_ROOT"/{downloads,logs,records,tmp,gradle-cache,android-user-home,validation,sdk/cmdline-tools}
exec 9>"$ANDROID_DEV_ROOT/install.lock"
flock -n 9 || { echo 'Another Android installation is running' >&2; exit 1; }
fetch() {
    local url=$1 destination=$2 algorithm=$3 expected=$4
    if [[ -f "$destination" ]] && printf '%s  %s\n' "$expected" "$destination" | "${algorithm}sum" --check --status; then return; fi
    curl -fL --silent --show-error --retry 3 --connect-timeout 20 --max-time 1800 "$url" -o "$destination.part"
    printf '%s  %s\n' "$expected" "$destination.part" | "${algorithm}sum" --check
    mv "$destination.part" "$destination"
}
fetch 'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.21%2B9/OpenJDK17U-jdk_x64_linux_hotspot_17.0.21_9.tar.gz' "$ANDROID_DEV_ROOT/downloads/temurin-17.0.21+9.tar.gz" sha256 8b9b611d917e3dfd8419d3566b5949226f69b052fed7e109c1368add1dfe8711
if [[ ! -x "$ANDROID_DEV_ROOT/jdk/bin/javac" ]]; then
    mkdir -p "$ANDROID_DEV_ROOT/jdk"
    tar -xzf "$ANDROID_DEV_ROOT/downloads/temurin-17.0.21+9.tar.gz" -C "$ANDROID_DEV_ROOT/jdk" --strip-components=1
fi
fetch https://downloads.gradle.org/distributions/gradle-8.13-bin.zip "$ANDROID_DEV_ROOT/downloads/gradle-8.13-bin.zip" sha256 20f1b1176237254a6fc204d8434196fa11a4cfb387567519c61556e8710aed78
if [[ ! -x "$ANDROID_DEV_ROOT/gradle-8.13/bin/gradle" ]]; then
    unzip -q "$ANDROID_DEV_ROOT/downloads/gradle-8.13-bin.zip" -d "$ANDROID_DEV_ROOT"
fi
fetch https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip "$ANDROID_DEV_ROOT/downloads/commandlinetools-linux-13114758_latest.zip" sha1 5fdcc763663eefb86a5b8879697aa6088b041e70
if [[ ! -x "$ANDROID_DEV_ROOT/sdk/cmdline-tools/19.0/bin/sdkmanager" ]]; then
    unzip -q "$ANDROID_DEV_ROOT/downloads/commandlinetools-linux-13114758_latest.zip" -d "$ANDROID_DEV_ROOT/tmp/command-line-tools"
    mv "$ANDROID_DEV_ROOT/tmp/command-line-tools/cmdline-tools" "$ANDROID_DEV_ROOT/sdk/cmdline-tools/19.0"
fi
ln -sfn 19.0 "$ANDROID_DEV_ROOT/sdk/cmdline-tools/latest"
cat > "$ANDROID_DEV_ROOT/activate.sh" <<'ACTIVATE'
# Source this file to enable zrh's personal Android toolchain in this shell.
export ANDROID_DEV_ROOT=/data/Users/zrh/.envs/android
export JAVA_HOME="$ANDROID_DEV_ROOT/jdk"
export ANDROID_HOME="$ANDROID_DEV_ROOT/sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_USER_HOME="$ANDROID_DEV_ROOT/android-user-home"
export ANDROID_EMULATOR_HOME="$ANDROID_USER_HOME"
export ANDROID_AVD_HOME="$ANDROID_USER_HOME/avd"
export GRADLE_USER_HOME="$ANDROID_DEV_ROOT/gradle-cache"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/27.3.13750724"
export TMPDIR="$ANDROID_DEV_ROOT/tmp"
export ADB_SERVER_SOCKET=tcp:localhost:5038
export PATH="$ANDROID_DEV_ROOT/bin:$JAVA_HOME/bin:$ANDROID_DEV_ROOT/gradle-8.13/bin:$ANDROID_HOME/cmdline-tools/19.0/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/build-tools/35.0.0:$ANDROID_HOME/cmake/3.22.1/bin:$ANDROID_HOME/emulator:$PATH"
ACTIVATE
source "$ANDROID_DEV_ROOT/activate.sh"
# Accept the SDK component licenses for this user-authorized installation.
set +o pipefail
yes | sdkmanager --sdk_root="$ANDROID_HOME" --licenses > "$ANDROID_DEV_ROOT/logs/sdk-licenses.log" 2>&1
license_status=${PIPESTATUS[1]}
set -o pipefail
[[ $license_status -eq 0 ]] || { cat "$ANDROID_DEV_ROOT/logs/sdk-licenses.log"; exit "$license_status"; }
sdkmanager --sdk_root="$ANDROID_HOME" --install \
    'platform-tools' 'platforms;android-34' 'platforms;android-35' 'platforms;android-36' \
    'build-tools;35.0.0' 'build-tools;36.0.0' 'ndk;27.3.13750724' 'cmake;3.22.1' \
    'emulator' 'system-images;android-35;google_apis;x86_64'
if [[ -e /root/android-dev && ! -L /root/android-dev ]]; then
    echo '/root/android-dev exists and is not a link; refusing to replace' >&2; exit 1
fi
if [[ -L /root/android-dev && $(readlink -f /root/android-dev) != "$ANDROID_DEV_ROOT" ]]; then
    echo '/root/android-dev points elsewhere; refusing to replace' >&2; exit 1
fi
ln -sfn "$ANDROID_DEV_ROOT" /root/android-dev
sdkmanager --sdk_root="$ANDROID_HOME" --list_installed > "$ANDROID_DEV_ROOT/records/sdk-packages.txt"
sha256sum "$ANDROID_DEV_ROOT"/downloads/*.zip "$ANDROID_DEV_ROOT"/downloads/*.tar.gz > "$ANDROID_DEV_ROOT/records/download-sha256.txt"
echo 'Installation completed. Source /root/android-dev/activate.sh to enable.'
