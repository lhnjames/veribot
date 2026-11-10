#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/project_root
DISK=/path/to/data_root
mkdir -p "$DISK"/{data,outputs,cache,venv}
ln -sfn "$DISK/data" "$ROOT/data"
ln -sfn "$DISK/outputs" "$ROOT/outputs"
ln -sfn "$DISK/cache" "$ROOT/cache"

usage=$(df --output=pcent "$DISK" | tail -1 | tr -dc '0-9')
if (( usage >= 85 )); then echo "Refusing setup: data disk is ${usage}% full"; exit 2; fi

if ! command -v python3 >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip git rsync
fi
python3 -m venv "$DISK/venv/runtime"
"$DISK/venv/runtime/bin/pip" install --upgrade pip wheel
"$DISK/venv/runtime/bin/pip" install -r "$ROOT/requirements-verifiable.txt"
echo "source $DISK/venv/runtime/bin/activate" > "$ROOT/activate.sh"
