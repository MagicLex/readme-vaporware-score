#!/usr/bin/env bash
# Log one experiment to the leaderboard FG. Registers a model version only on
# keeps (so the registry shows a clean improvement curve, not noise).
# usage: log_exp.sh <keep|discard|crash> "<description>"
set -euo pipefail
cd "$(dirname "$0")/.."
STATUS="$1"; DESC="$2"
LOG=autoresearch/run.log
TAG=jun29

if grep -q '^val_metric:' "$LOG"; then
  V=$(grep '^val_metric:' "$LOG" | awk '{print $2}')
  M=$(grep '^peak_memory_gb:' "$LOG" | awk '{print $2}')
else
  V=0; M=0; STATUS=crash
fi
SHA=$(git rev-parse --short HEAD)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

printf '[{"commit":"%s","val_metric":%s,"peak_memory_gb":%s,"status":"%s","description":"%s","ts":"%s"}]' \
  "$SHA" "$V" "$M" "$STATUS" "$DESC" "$TS" > /tmp/row.json
hops fg insert "autoresearch_experiments_${TAG}" --file /tmp/row.json 2>&1 | grep -iE "inserted|error" || true

if [ "$STATUS" = "keep" ]; then
  hops model register "autoresearch_${TAG}" autoresearch/model \
    --framework sklearn --metrics "val_metric=$V" \
    --description "$DESC; status=keep; metric_direction=max; commit=$SHA" \
    --feature-view vaporware_fv 2>&1 | grep -iE "registered|version|error" || true
fi
echo "logged $STATUS val_metric=$V ($DESC)"
