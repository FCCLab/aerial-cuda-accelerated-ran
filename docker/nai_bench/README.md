# NAI graph latency benchmark (Docker)

Standalone GPU latency test for the in-cuBB **NAI** (Network-Assisted Interference) CUDA graph tail embedded in the PUSCH full-slot graph:

```
copy_rx_to_ni  →  PdschTx child graph  →  ni_sub_h_tx
```

The benchmark binary is `cuphy_ex_nai_bench` (`cuPHY/examples/nai_bench/`). It builds a minimal graph matching the production PUSCH path, times launches with `cudaEvent`, and prints per-stage stats plus a summary table.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU visible to Docker (`nvidia-smi` works on the host)
- Base image: `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-2-cubb` (pulled automatically on first build)

## Quick start

All commands below are run from **`oai-nvidia/cuBB/`**.

### One-time image build (~2 min, ~33 GB image)

```bash
docker compose -f docker/nai_bench/docker-compose.yaml build
```

For GPUs other than Ampere/Hopper (SM 80/90), set the SASS target at build time:

```bash
CUDA_ARCHITECTURES=80-real docker compose -f docker/nai_bench/docker-compose.yaml build
```

### Run benchmark

```bash
docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench
```

Custom parameters:

```bash
docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench -p 133 -a 4 -n 500
docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench --no-breakdown -p 106 -a 2
```

Helper script (build + run):

```bash
./docker/nai_bench/build_and_run.sh -p 133 -n 500
```

## Fast dev iteration

After the image exists, use bind-mounted source and incremental in-container rebuild (~1–3 s per edit):

```bash
./docker/nai_bench/dev_run.sh -p 133 -n 500
```

After editing NAI channel code (`nai_ni_kernel.cu`, `nai_pdsch_recon.*`, `pusch_rx` NAI paths):

```bash
./docker/nai_bench/dev_run.sh --rebuild-channels -p 133 -n 100
```

Equivalent compose command:

```bash
docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench-dev -p 133 -n 500
```

### What dev mode mounts

| Host path | Container path | When to rebuild |
|-----------|----------------|-----------------|
| `cuPHY/examples/nai_bench/` | `.../examples/nai_bench/` | default (bench target only) |
| `cuPHY/src/cuphy_channels/` | `.../src/cuphy_channels/` | pass `--rebuild-channels` |

Prebuilt `cuPHY/build` and libraries stay in the image; only changed targets are recompiled.

## Benchmark CLI

| Flag | Description | Default |
|------|-------------|---------|
| `-g` | GPU device id | `0` |
| `-p` | UL PRBs / BWP size | `133` |
| `-s` | OFDM symbols per slot | `14` |
| `-a` | RX antenna count | `4` |
| `-l` | UE layers | `1` |
| `-w` | Warmup graph launches | `20` |
| `-n` | Timed iterations | `200` |
| `--no-breakdown` | Skip standalone per-kernel timings | off |
| `-h`, `--help` | Usage | — |

## Example output

```
cuphy_ex_nai_bench
  gpu=NVIDIA GH200 480GB (0)
  n_prb=133 n_sym=14 n_ant=4 n_layers=1 grid=1596x14x4
  warmup=20 iterations=200

full_nai_graph               mean=   35.21 us  p50=   35.17 us  ...

+---------------------- NAI latency summary ----------------------+
| GPU: NVIDIA GH200 480GB                                        |
| Config: 133 PRB, 14 sym, 4 ant, 1 layer(s)  warmup= 20  n= 200 |
+----------------------------+--------+--------+--------+--------+
| Stage                      |   mean |    p50 |    p99 |    max |
|                            |   (us) |   (us) |   (us) |   (us) |
+----------------------------+--------+--------+--------+--------+
| full_nai_graph             |  35.21 |  35.17 |  35.65 |  36.35 |
| copy_rx_to_ni              |   3.55 |   3.49 |   4.16 |   4.16 |
| pdsch_tx_child             |  30.71 |  30.69 |  31.20 |  31.20 |
| ni_sub_h_tx                |   4.68 |   4.64 |   5.18 |   5.18 |
+----------------------------+--------+--------+--------+--------+
| Note: breakdown rows are standalone launches, not concurrent. |
|       full_nai_graph = embedded PUSCH path (single launch).   |
+---------------------------------------------------------------+
```

You may also see `[NAI dbg ...]` lines from cuPHY debug prints in `nai_ni_kernel.cu`; they do not affect timing.

## What is measured (and what is not)

**Included**

- Single CUDA graph launch of the NAI tail: copy RX grid → embedded PdschTx reconstruction → subtract H·tx from NI grid
- Uses `NaiPdschRecon` warmup synthetic TB (same idle/safe path as production when recon is disabled)
- GPU time via `cuGraphLaunch` + `cudaEvent` (mean, p50, p99, min, max in µs)

**Not included**

- Full PUSCH RX pipeline (channel est, LDPC, CRC, etc.)
- MQ export / host-side `copyNaiToMq` latency
- Inter-slot pipelining or dual-stream overlap
- Real UE TB bits from the scheduler (warmup TB only)

For end-to-end NAI in the live stack, enable `CUPHY_NAI_MQ_ENABLE=1` and `PUSCH_PROC_MODE_FULL_SLOT_GRAPHS` in `cuphycontroller`.

## Layout

```
docker/nai_bench/
├── README.md              # this file
├── Dockerfile             # Aerial base + cuphy_ex_nai_bench build
├── docker-compose.yaml    # nai-bench (release) + nai-bench-dev (bind-mount)
├── build_and_run.sh       # full image build + run
├── dev_run.sh             # fast dev: incremental rebuild + run
└── dev_entrypoint.sh      # in-container rebuild helper

cuPHY/examples/nai_bench/
├── cuphy_ex_nai_bench.cpp # benchmark source
├── CMakeLists.txt
└── dummy.cu
```

## Troubleshooting

### `mkdir: cannot create directory 'build': Permission denied`

The Aerial image runs as user `aerial`. The Dockerfile copies sources as root then `chown`s to `aerial` before building. Rebuild with the current Dockerfile.

### `chmod: cannot access '.../dev_entrypoint.sh'`

Ensure `!/docker/nai_bench` is listed in `oai-nvidia/cuBB/.dockerignore`. Dev mode also bind-mounts the script, so `nai-bench-dev` works even without baking it into the image.

### `CUDA error: illegal memory access`

Usually incorrect tensor **element** strides in the benchmark setup. Production uses element indices (not byte offsets) for `rx`/`tx`/`h` layouts; see `pusch_rx.cpp` `updateNaiMqTailGraphNodes()`.

### Build fails without GPU

Set `CUDA_ARCHITECTURES` explicitly for your target SM, e.g. `CUDA_ARCHITECTURES=90-real`.

### Image rebuild after CMake / dependency changes

Dev mode only recompiles selected targets. If you change cuPHY CMake options or add new dependencies, run a full image rebuild:

```bash
docker compose -f docker/nai_bench/docker-compose.yaml build --no-cache
```

## Related production code

| Component | Path |
|-----------|------|
| NAI kernels | `cuPHY/src/cuphy_channels/nai_ni_kernel.cu` |
| PdschTx embed | `cuPHY/src/cuphy_channels/nai_pdsch_recon.cpp` |
| PUSCH graph tail | `cuPHY/src/cuphy_channels/pusch_rx.cpp` (`appendNaiMqTailToFullSlotGraph`, `updateNaiMqTailGraphNodes`) |
