#!/usr/bin/env bash
SHA=1982953d60cf4878173c7206ea7fd1a0c70f3b00
prev=""
for i in $(seq 1 100); do
  s=$(gh api "repos/StrudelDoodleS/superglm/commits/$SHA/check-runs" \
        --jq '.check_runs[] | "\(.name)|\(.status)|\(.conclusion // "pending")"' 2>/dev/null || true)
  if [ -n "$s" ]; then
    cur=$(printf '%s\n' "$s" | grep -E '\|(failure|timed_out|cancelled)\|' | sort)
    comm -13 <(printf '%s\n' "$prev") <(printf '%s\n' "$cur") | grep -v '^$'
    prev="$cur"
    pend=$(printf '%s\n' "$s" | grep -cE '\|(in_progress|queued)\|' || true)
    if [ "$pend" -eq 0 ]; then
      fails=$(printf '%s\n' "$s" | grep -cE '\|(failure|timed_out|cancelled)\|' || true)
      echo "ALL CHECKS COMPLETE -- failures: $fails"
      exit 0
    fi
  fi
  sleep 30
done
echo "TIMED OUT waiting for checks"
