#!/usr/bin/env bash
# Fast iteration: bind-mount NAI bench (and optional channel) sources, rebuild in-container, run.
#
# Requires a one-time image build:
#   docker compose -f docker/nai_bench/docker-compose.yaml build
#
# Usage (from oai-nvidia/cuBB/):
#   ./docker/nai_bench/dev_run.sh
#   ./docker/nai_bench/dev_run.sh -p 133 -n 500
#   ./docker/nai_bench/dev_run.sh --rebuild-channels -p 133 -n 100
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUBB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${CUBB_ROOT}"

if ! docker image inspect cubb-nai-bench:local >/dev/null 2>&1; then
    echo "Image cubb-nai-bench:local not found. Run first:"
    echo "  docker compose -f docker/nai_bench/docker-compose.yaml build"
    exit 1
fi

docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench-dev "$@"
