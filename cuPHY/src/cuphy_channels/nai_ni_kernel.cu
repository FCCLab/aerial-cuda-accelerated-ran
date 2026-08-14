/*
 * NAI NI grid policy:
 *   1) Copy full RX slot into NI (PRACH/PUCCH/SRS/unallocated PRBs stay as received interference).
 *   2) Inside each PUSCH UE rectangle only: ni = rx - H*tx (PdschTx reconstruction).
 * H layout matches cuPHY tInfoHEst: (N_BS_ANTS, N_LAYERS, NF_alloc, NH).
 */
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "cuphy.h"

// Set NAI_DEBUG_PRINTF=1 to enable throttled device-side printf in the NAI copy/subtract kernels
// (helps verify tx is reconstructed and ni = rx - H*tx is computed correctly).
// Logs appear in the cuPHY container stdout. Set back to 0 once verified.
#ifndef NAI_DEBUG_PRINTF
#define NAI_DEBUG_PRINTF 1
#endif

namespace cuphy_nai_kernels {

#if defined(NAI_DEBUG_PRINTF) && (NAI_DEBUG_PRINTF != 0)
__device__ unsigned g_nai_copy_dbg_launches = 0;
__device__ unsigned g_nai_sub_dbg_launches  = 0;
#endif

__device__ __forceinline__ int layout_off(int i0, int i1, int i2, int s0, int s1, int s2)
{
    return i0 * s0 + i1 * s1 + i2 * s2;
}

__device__ void cmul(float ar, float ai, float br, float bi, float& cr, float& ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

__device__ void load_cx_half(const __half2* base, int off, float& re, float& im)
{
    const __half2 v = base[off];
    const __half* p = reinterpret_cast<const __half*>(&v);
    re = __half2float(p[0]);
    im = __half2float(p[1]);
}

__device__ void store_cx_half(__half2* base, int off, float re, float im)
{
    __half2 v;
    __half* p = reinterpret_cast<__half*>(&v);
    p[0] = __float2half(re);
    p[1] = __float2half(im);
    base[off] = v;
}

__device__ void load_h_f32(const float2* base, int elem_off, float& re, float& im)
{
    const float2 v = base[elem_off];
    re = v.x;
    im = v.y;
}

__device__ int tx_port_for_layer(int layer, uint16_t dmrs_port_bmsk)
{
    int seen = 0;
    for (int p = 0; p < 16; ++p) {
        if ((dmrs_port_bmsk >> p) & 1) {
            if (seen == layer) {
                return p;
            }
            ++seen;
        }
    }
    return layer;
}

__device__ int h_dmrs_slot(float sym_f, const int* dmrs_syms, int n_dmrs)
{
    if (n_dmrs <= 0) {
        return 0;
    }
    if (sym_f <= static_cast<float>(dmrs_syms[0])) {
        return 0;
    }
    if (sym_f >= static_cast<float>(dmrs_syms[n_dmrs - 1])) {
        return n_dmrs - 1;
    }
    for (int i = 0; i < n_dmrs - 1; ++i) {
        const float s0 = static_cast<float>(dmrs_syms[i]);
        const float s1 = static_cast<float>(dmrs_syms[i + 1]);
        if (sym_f >= s0 && sym_f <= s1) {
            return i;
        }
    }
    return n_dmrs - 1;
}

__global__ void copy_rx_to_ni_kernel(const __half2* rx,
    __half2* ni,
    int nf,
    int nt,
    int n_ant,
    int s0,
    int s1,
    int s2)
{
    const int sc = blockIdx.x * blockDim.x + threadIdx.x;
    const int sym = blockIdx.y;
    const int ant = blockIdx.z;
    if (sc >= nf || sym >= nt || ant >= n_ant) {
        return;
    }
    const int off = layout_off(sc, sym, ant, s0, s1, s2);
    ni[off] = rx[off];

#if defined(NAI_DEBUG_PRINTF) && (NAI_DEBUG_PRINTF != 0)
    if (sc == 0 && sym == 0 && ant == 0) {
        const unsigned c = atomicAdd(&g_nai_copy_dbg_launches, 1u);
        if (c < 4u) {
            float re, im;
            load_cx_half(rx, off, re, im);
            printf("[NAI dbg copy] launch=%u sc=0 sym=0 ant=0 RX=(% .4f,% .4f) nf=%d nt=%d nAnt=%d s0=%d s1=%d s2=%d\n",
                   c, re, im, nf, nt, n_ant, s0, s1, s2);
        }
    }
#endif
}

__global__ void ni_sub_h_tx_kernel(const __half2* rx,
    const __half2* tx,
    __half2* ni,
    const float2* h_est,
    int nf_grid,
    int nt,
    int n_ant_rx,
    int rx_s0,
    int rx_s1,
    int rx_s2,
    int n_sc_alloc,
    int n_layers,
    int n_h_pos,
    int h_s0,
    int h_s1,
    int h_s2,
    int h_s3,
    uint16_t dmrs_port_bmsk,
    int prb_start,
    int n_prb,
    int sym_start,
    int n_sym,
    int dmrs_sym_bmsk,
    int tx_nf,
    int tx_nt,
    int tx_n_ports,
    int tx_s0,
    int tx_s1,
    int tx_s2)
{
    const int k_local = blockIdx.x * blockDim.x + threadIdx.x;
    const int s_local = blockIdx.y;
    const int rx_ant = blockIdx.z;
    if (k_local >= n_sc_alloc || s_local >= n_sym || rx_ant >= n_ant_rx) {
        return;
    }

    const int sc_global = prb_start * 12 + k_local;
    const int sym_global = sym_start + s_local;
    if (sc_global >= nf_grid || sym_global >= nt) {
        return;
    }

    int dmrs_syms[4];
    int n_dmrs = 0;
    for (int s = 0; s < 14 && n_dmrs < 4; ++s) {
        if ((dmrs_sym_bmsk >> s) & 1) {
            dmrs_syms[n_dmrs++] = s;
        }
    }
    if (n_dmrs < 1) {
        n_dmrs = 1;
        dmrs_syms[0] = 0;
    }

    const float sym_f = static_cast<float>(sym_global);
    const int h0 = h_dmrs_slot(sym_f, dmrs_syms, n_dmrs);
    int h1 = h0;
    float alpha = 0.f;
    if (n_dmrs > 1 && h0 < n_dmrs - 1) {
        h1 = h0 + 1;
        const float t0 = static_cast<float>(dmrs_syms[h0]);
        const float t1 = static_cast<float>(dmrs_syms[h1]);
        if (t1 > t0) {
            alpha = (sym_f - t0) / (t1 - t0);
        }
    }

    const int rx_off = layout_off(sc_global, sym_global, rx_ant, rx_s0, rx_s1, rx_s2);
    float rx_re, rx_im;
    load_cx_half(rx, rx_off, rx_re, rx_im);

    float pred_re = 0.f;
    float pred_im = 0.f;
#if defined(NAI_DEBUG_PRINTF) && (NAI_DEBUG_PRINTF != 0)
    float dbg_tx_re0 = 0.f;
    float dbg_tx_im0 = 0.f;
    float dbg_hr0 = 0.f;
    float dbg_hi0 = 0.f;
    int   dbg_tx_port0 = -1;
    int   dbg_tx_off0 = -1;
#endif
    for (int layer = 0; layer < n_layers; ++layer) {
        const int tx_port = tx_port_for_layer(layer, dmrs_port_bmsk);
        if (tx_port >= tx_n_ports) {
            break;
        }
        const int tx_off = layout_off(sc_global, sym_global, tx_port, tx_s0, tx_s1, tx_s2);
        float tx_re, tx_im;
        if (sc_global < tx_nf && sym_global < tx_nt) {
            load_cx_half(tx, tx_off, tx_re, tx_im);
        } else {
            tx_re = 0.f;
            tx_im = 0.f;
        }

        const int h_base = rx_ant * h_s0 + layer * h_s1 + k_local * h_s2;
        float h0_re, h0_im, h1_re, h1_im;
        load_h_f32(h_est, h_base + h0 * h_s3, h0_re, h0_im);
        load_h_f32(h_est, h_base + h1 * h_s3, h1_re, h1_im);
        const float hr = (1.f - alpha) * h0_re + alpha * h1_re;
        const float hi = (1.f - alpha) * h0_im + alpha * h1_im;

        float pr, pi;
        cmul(hr, hi, tx_re, tx_im, pr, pi);
        pred_re += pr;
        pred_im += pi;
#if defined(NAI_DEBUG_PRINTF) && (NAI_DEBUG_PRINTF != 0)
        if (layer == 0) {
            dbg_tx_re0 = tx_re;
            dbg_tx_im0 = tx_im;
            dbg_hr0 = hr;
            dbg_hi0 = hi;
            dbg_tx_port0 = tx_port;
            dbg_tx_off0 = tx_off;
        }
#endif
    }

    (void)n_h_pos;
    (void)n_prb;
    store_cx_half(ni, rx_off, rx_re - pred_re, rx_im - pred_im);

#if defined(NAI_DEBUG_PRINTF) && (NAI_DEBUG_PRINTF != 0)
    // Print a small sample for the first PUSCH-active (k_local=0, s_local=0, rx_ant=0) point per launch.
    // Throttled so the log isn't flooded.
    if (k_local == 0 && s_local == 0 && rx_ant == 0) {
        const unsigned c = atomicAdd(&g_nai_sub_dbg_launches, 1u);
        if (c < 12u) {
            const float ni_re = rx_re - pred_re;
            const float ni_im = rx_im - pred_im;
            printf(
              "[NAI dbg sub] launch=%u sc=%d sym=%d ant=%d layer0_port=%d tx_off0=%d "
              "RX=(% .4f,% .4f) TX0=(% .4f,% .4f) H0=(% .4f,% .4f) H*TX=(% .4f,% .4f) NI=(% .4f,% .4f) "
              "| dmrsBmsk=0x%04x prb_start=%d n_prb=%d n_layers=%d nf_grid=%d sym_start=%d n_sym=%d tx_nf=%d tx_nt=%d tx_nP=%d\n",
              c,
              sc_global, sym_global, rx_ant, dbg_tx_port0, dbg_tx_off0,
              rx_re, rx_im,
              dbg_tx_re0, dbg_tx_im0,
              dbg_hr0, dbg_hi0,
              pred_re, pred_im,
              ni_re, ni_im,
              (unsigned)dmrs_port_bmsk, prb_start, n_prb, n_layers, nf_grid, sym_start, n_sym,
              tx_nf, tx_nt, tx_n_ports);
        }
    }
#endif
}

} // namespace cuphy_nai_kernels

cuphyStatus_t cuphy_nai_graph_template_copy_rx_kernel(CUDA_KERNEL_NODE_PARAMS* p, void** kernelParamPtrs)
{
    if (!p || !kernelParamPtrs) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t e = cudaGetFuncBySymbol(&p->func, reinterpret_cast<const void*>(&cuphy_nai_kernels::copy_rx_to_ni_kernel));
    if (e != cudaSuccess) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    p->gridDimX = 1;
    p->gridDimY = 1;
    p->gridDimZ = 1;
    p->blockDimX = 256;
    p->blockDimY = 1;
    p->blockDimZ = 1;
    p->kernelParams = kernelParamPtrs;
    p->sharedMemBytes = 0;
    p->extra = nullptr;
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphy_nai_graph_template_ni_sub_kernel(CUDA_KERNEL_NODE_PARAMS* p, void** kernelParamPtrs)
{
    if (!p || !kernelParamPtrs) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t e = cudaGetFuncBySymbol(&p->func, reinterpret_cast<const void*>(&cuphy_nai_kernels::ni_sub_h_tx_kernel));
    if (e != cudaSuccess) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    p->gridDimX = 1;
    p->gridDimY = 1;
    p->gridDimZ = 1;
    p->blockDimX = 64;
    p->blockDimY = 1;
    p->blockDimZ = 1;
    p->kernelParams = kernelParamPtrs;
    p->sharedMemBytes = 0;
    p->extra = nullptr;
    return CUPHY_STATUS_SUCCESS;
}
