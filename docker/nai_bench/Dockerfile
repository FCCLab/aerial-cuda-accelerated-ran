# Build and run cuphy_ex_nai_bench — isolated NAI (copy_rx -> PdschTx child -> ni_sub) latency test.
#
# Build:
#   docker compose -f docker/nai_bench/docker-compose.yaml build
# Run (requires NVIDIA GPU):
#   docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench
#
# Fast dev (bind-mount source, rebuild target in-container):
#   ./docker/nai_bench/dev_run.sh -p 133 -n 500
#
# From repo root (oai-nvidia/cuBB):
#   docker compose -f docker/nai_bench/docker-compose.yaml run --rm nai-bench -p 133 -n 500

FROM nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-2-cubb

ARG CUDA_ARCHITECTURES=80-real;90-real
ENV DEBIAN_FRONTEND=noninteractive
ENV cuBB_SDK=/opt/nvidia/cuBB

WORKDIR ${cuBB_SDK}

# Base image runs as user "aerial"; COPY creates root-owned files, so build as root then hand back.
USER root
COPY . ${cuBB_SDK}
RUN chown -R aerial:aerial ${cuBB_SDK}

USER aerial
RUN cd cuPHY && mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/native \
      -DCUPHY_HDF5_VERSION_CHECK=OFF \
      -DBUILD_DOCS=OFF \
      -DENABLE_CCACHE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" && \
    cmake --build . --target cuphy_ex_nai_bench -j"$(nproc)"

ENV PATH="${cuBB_SDK}/cuPHY/build/examples/nai_bench:${PATH}"
ENV LD_LIBRARY_PATH="${cuBB_SDK}/cuPHY/build/src/cuphy:${cuBB_SDK}/cuPHY/build/src/cuphy_channels:${LD_LIBRARY_PATH}"

ENTRYPOINT ["/opt/nvidia/cuBB/cuPHY/build/examples/nai_bench/cuphy_ex_nai_bench"]
CMD ["-p", "133", "-w", "20", "-n", "200"]
