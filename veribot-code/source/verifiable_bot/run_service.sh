#!/usr/bin/env bash
set -euo pipefail
cd /path/to/project_root
source activate.sh
mkdir -p outputs/service
nohup uvicorn verifiable_bot.service:app --host 0.0.0.0 --port "${PORT:-8000}" \
  > outputs/service/server.log 2>&1 &
echo $! > outputs/service/server.pid
