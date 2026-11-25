#!/bin/sh
# Same benchmark as the local runner, with a runtime CPU/memory cap.
set -eu
usage='Usage: scripts/benchmark-docker.sh NEW_RESULT_DIRECTORY [benchmark.py options...]

Environment variables:
  BENCH_MEMORY  Container memory and swap limit (default: 4g)'
if [ "${1-}" = "-h" ] || [ "${1-}" = "--help" ]; then
    printf '%s\n' "$usage"
    exit 0
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output=${1:?$usage}
shift
test ! -e "$output" || { echo "Refusing to overwrite $output" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd)
name=$(basename "$output")
docker build -f "$root/scripts/benchmark.Dockerfile" -t xarray-cog-benchmark "$root"
docker run --rm --memory="${BENCH_MEMORY:-4g}" --memory-swap="${BENCH_MEMORY:-4g}" --cpus=1 \
    --user "$(id -u):$(id -g)" \
    -e "BENCH_SOURCE_REVISION=$(git -C "$root" rev-parse HEAD)" \
    -v "$parent:/results" xarray-cog-benchmark \
    "/results/$name" "$@"
