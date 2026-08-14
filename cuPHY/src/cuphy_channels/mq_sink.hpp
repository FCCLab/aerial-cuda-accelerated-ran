/*
 * Resgrid plotter POSIX mq export (RX and NAI). RGMC v2 chunks, raw float32 IQ raster.
 * See mq_export.hpp for per-channel enable env vars and queue paths.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "mq_export.hpp"

enum class cuphy_mq_channel : uint8_t { rx = 0, nai = 1 };

inline bool cuphy_mq_export_enabled(cuphy_mq_channel ch)
{
    return (ch == cuphy_mq_channel::nai) ? cuphy_nai_mq_export_enabled() : cuphy_rx_mq_export_enabled();
}

void cuphy_mq_enqueue_from_device(cuphy_mq_channel ch,
    cudaStream_t stream,
    void* dev_src,
    size_t nbytes,
    int dim0,
    int dim1,
    int dim2,
    bool fp16,
    int ul_prb_config,
    int el_stride0,
    int el_stride1,
    int el_stride2);

void cuphy_mq_shutdown();
