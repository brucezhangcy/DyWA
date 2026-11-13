#!/usr/bin/env bash
# A launch flag is deliberately not supplied by any queue. Set it only after
# the human owner has approved the particular GPU experiment and its budget.
if [ "${DYWA_GPU_APPROVED:-0}" != 1 ]; then
  echo 'GPU execution is locked. Obtain owner approval for this experiment, then set DYWA_GPU_APPROVED=1 for that invocation.' >&2
  return 64 2>/dev/null || exit 64
fi
