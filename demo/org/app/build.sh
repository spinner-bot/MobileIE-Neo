#!/usr/bin/env bash
set -euo pipefail
source /root/android-dev/activate.sh
cd "$(dirname "$0")"
exec gradle --no-daemon --console=plain "$@"
