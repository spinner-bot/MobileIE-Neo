#!/usr/bin/env bash
# Linux-only GUI simulation using packages already extracted into zrh's station.
set -euo pipefail
station=/data/Users/zrh/.envs/android
source "$station/validation/desktop-host/activate.sh"
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
found=false
for number in $(seq 91 99); do
    if [[ ! -e "/tmp/.X$number-lock" && ! -S "/tmp/.X11-unix/X$number" ]]; then
        export DISPLAY=:$number; found=true; break
    fi
done
[[ $found == true ]] || { echo 'No free test display in :91..:99' >&2; exit 1; }
"$DESKTOP_VALIDATION_ROOT/root/usr/bin/Xvfb" "$DISPLAY" -screen 0 1280x960x24 -nolisten tcp -ac -noreset -xkbdir "$XKB_CONFIG_ROOT" > "$DESKTOP_VALIDATION_ROOT/xvfb-test.log" 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
ready=false
for attempt in $(seq 1 20); do
    kill -0 "$xvfb_pid" || exit 1
    if "$station/validation/reference-env/bin/python" -I -c 'import tkinter; r=tkinter.Tk(); r.destroy()' 2>/dev/null; then ready=true; break; fi
    sleep .1
done
[[ $ready == true ]] || { echo 'Virtual display not ready' >&2; exit 1; }
export LIGHTLAB_GUI_TEST=1 LIGHTLAB_TEST_OUTPUT="$station/validation/desktop-evidence"
exec_status=0
"$station/validation/reference-env/bin/python" -I "$repo/demo/org/desktop/test_desktop.py" || exec_status=$?
exit "$exec_status"
