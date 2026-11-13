#!/usr/bin/env bash
# Ubuntu host libraries extracted into the personal station; no global install.
set -euo pipefail
station=/data/Users/zrh/.envs/android
mkdir -p "$station/downloads/host-libs" "$station/host-libs" "$station/bin"
cd "$station/downloads/host-libs"
apt-get download libpulse0 libasyncns0 libsndfile1 libflac8 libogg0 libopus0 libvorbis0a libvorbisenc2 libasound2 libasound2-data
for package in ./*.deb; do dpkg-deb -x "$package" "$station/host-libs"; done
sha256sum ./*.deb > "$station/records/host-libs-sha256.txt"
for package in ./*.deb; do dpkg-deb -f "$package" Package Version; done > "$station/records/host-libs-versions.txt"
cat > "$station/bin/emulator" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
station=/data/Users/zrh/.envs/android
export LD_LIBRARY_PATH="$station/host-libs/usr/lib/x86_64-linux-gnu:$station/host-libs/usr/lib/x86_64-linux-gnu/pulseaudio${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$station/sdk/emulator/emulator" "$@"
EOF
chmod +x "$station/bin/emulator"
echo 'Personal emulator host libraries prepared.'
