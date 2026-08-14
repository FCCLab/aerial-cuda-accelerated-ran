#!/usr/bin/env bash
# Dev entrypoint: incremental rebuild from bind-mounted source, then run the benchmark.
set -euo pipefail

CUPHY_BUILD="${cuBB_SDK:-/opt/nvidia/cuBB}/cuPHY/build"
BENCH_BIN="${CUPHY_BUILD}/examples/nai_bench/cuphy_ex_nai_bench"
REBUILD_CHANNELS=0
BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild-channels)
            REBUILD_CHANNELS=1
            shift
            ;;
        --)
            shift
            BENCH_ARGS+=("$@")
            break
            ;;
        *)
            BENCH_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "${CUPHY_BUILD}"

if [[ "${REBUILD_CHANNELS}" -eq 1 ]]; then
    echo "=== dev: rebuilding cuphy_channels + cuphy_ex_nai_bench ==="
    cmake --build . --target cuphy_channels cuphy_ex_nai_bench -j"$(nproc)"
else
    echo "=== dev: rebuilding cuphy_ex_nai_bench ==="
    cmake --build . --target cuphy_ex_nai_bench -j"$(nproc)"
fi

echo "=== dev: running ${BENCH_BIN} ==="
exec "${BENCH_BIN}" "${BENCH_ARGS[@]}"
