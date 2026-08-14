#!/usr/bin/env bash
# Build cuphy_ex_nai_bench inside the Aerial cuBB container and run latency test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUBB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUDA_ARCH="${CUDA_ARCHITECTURES:-80-real;90-real}"

cd "${CUBB_ROOT}"

echo "=== Building cuphy_ex_nai_bench (CUDA_ARCHITECTURES=${CUDA_ARCH}) ==="
docker compose -f docker/nai_bench/docker-compose.yaml build \
  --build-arg "CUDA_ARCHITECTURES=${CUDA_ARCH}"

echo ""
echo "=== Running NAI latency benchmark ==="
docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench "$@"
