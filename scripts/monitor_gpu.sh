#!/bin/sh

set -eu

OUTPUT_FILE="${1:-gpu_log.csv}"
INTERVAL="${2:-5}"

if ! command -v nvitop >/dev/null 2>&1; then
  echo "Error: nvitop not found in PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Writing GPU metrics to $OUTPUT_FILE every ${INTERVAL}s"
echo "Press Ctrl+C to stop."

nvitop --csv --interval "$INTERVAL" --output "$OUTPUT_FILE"
