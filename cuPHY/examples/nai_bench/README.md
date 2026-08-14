# cuphy_ex_nai_bench

GPU latency microbenchmark for the **NAI graph tail** embedded in the PUSCH full-slot CUDA graph:

`copy_rx_to_ni` → `PdschTx` child graph → `ni_sub_h_tx`

## Docker (recommended)

See [`docker/nai_bench/README.md`](../../../docker/nai_bench/README.md) for build, run, and fast dev workflows.

```bash
# from oai-nvidia/cuBB/
docker compose -f docker/nai_bench/docker-compose.yaml build
docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench -p 133 -n 200
```

## Build without Docker

Requires CMake ≥ 3.25 and a CUDA toolchain matching cuPHY (same as the Aerial cuBB SDK image).

```bash
cd cuPHY
mkdir -p build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/native \
  -DCUPHY_HDF5_VERSION_CHECK=OFF \
  -DBUILD_DOCS=OFF
cmake --build . --target cuphy_ex_nai_bench -j"$(nproc)"
./examples/nai_bench/cuphy_ex_nai_bench -p 133 -n 200
```

## CLI

| Flag | Default | Meaning |
|------|---------|---------|
| `-p` | 133 | UL PRBs |
| `-s` | 14 | OFDM symbols |
| `-a` | 4 | RX antennas |
| `-l` | 1 | UE layers |
| `-w` | 20 | Warmup iterations |
| `-n` | 200 | Timed iterations |
| `--no-breakdown` | off | Only time `full_nai_graph` |

Output includes line stats and an ASCII summary table (mean / p50 / p99 / max in µs).
