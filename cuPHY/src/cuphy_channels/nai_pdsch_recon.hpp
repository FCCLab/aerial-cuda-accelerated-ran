/*
 * In-cuBB NAI: PdschTx graph warmup + per-slot setup for NI tail embedded in the PUSCH full-slot CUDA graph.
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>

#include "cuphy_api.h"
#include "cuphy.hpp"
#include "ch_est/ch_est_utils.hpp"

struct NaiPuschSlotInput {
    cudaStream_t stream{nullptr};
    const cuphyPuschCellGrpDynPrm_t* cell_grp{nullptr};
    const cuphyPuschDynPrms_t* dyn{nullptr};
    const cuphyCellStatPrm_t* cell_stats{nullptr};
    uint16_t                  n_cell_stats{0};
    uint8_t* tb_payloads_dev{nullptr};
    uint32_t* tb_crcs_dev{nullptr};
    uint32_t* tb_payload_offsets{nullptr};
    void* rx_dev{nullptr};
    int rx_nf{0};
    int rx_nt{0};
    int rx_n_ant{0};
    int rx_stride0{0};
    int rx_stride1{0};
    int rx_stride2{0};
    bool rx_fp16{true};
    void* h_est_dev{nullptr};
    int h_stride0{0};
    int h_stride1{0};
    int h_stride2{0};
    int h_stride3{0};
    int n_rx{0};
    int n_layers{0};
    int n_sc_alloc{0};
    int n_h_dmrs_pos{0};
    uint16_t dmrs_port_bmsk{0};
    const cuphyPuschRxUeGrpPrms_t* drvd_ue_grp{nullptr};
};

/** PdschTx graph embed + per-slot setup for NAI tail inside the PUSCH full-slot CUDA graph. */
class NaiPdschRecon {
public:
    NaiPdschRecon() = default;
    ~NaiPdschRecon();

    NaiPdschRecon(const NaiPdschRecon&) = delete;
    NaiPdschRecon& operator=(const NaiPdschRecon&) = delete;

    /** Lazy-create PdschTx from cell static parameters (once). */
    cuphyStatus_t ensurePdschTx(const cuphyCellStatPrm_t* cell_stats,
        uint16_t n_cells,
        int stream_priority);

    /** One-time: expand PdschTx graph topology so its CUgraph can be embedded under PUSCH full-slot graph. */
    bool warmupGraphTemplateForParentEmbed(cudaStream_t stream,
        const cuphyPuschStatPrms_t& pusch_stat,
        const cuphyCellStatPrm_t* cell_stats,
        uint16_t n_cell_stats);

    /** Per-slot: cuphySetupPdschTx + graph exec param sync for embedded tail (no separate stream launch). */
    cuphyStatus_t prepareEmbeddedGraphSlot(NaiPuschSlotInput& in, void* ni_dev, cudaStream_t graph_stream);

    /**
     * Re-apply warmup PdschTx dyn + graph-exec kernel params (synthetic UE on m_embedWarmTbDev).
     * Used when the embedded child graph cannot be toggled off: cuGraphNodeSetEnabled is not supported for
     * CU_GRAPH_NODE_TYPE_CHILD (only kernel/memcpy/memset).
     */
    cuphyStatus_t refreshEmbeddedGraphToWarmupIdle(cudaStream_t graph_stream);

    cuphyPdschTxHndl_t pdschHandle() const
    {
        return m_pdschHndl;
    }

    void* txGridDev() const
    {
        return m_txDev;
    }

private:
    cuphyStatus_t setupDynForUe(const NaiPuschSlotInput& in,
        uint16_t ue_idx,
        uint16_t ue_grp_idx,
        cudaStream_t stream_override = nullptr,
        const uint64_t* proc_mode_override = nullptr);

    cuphyPdschTxHndl_t m_pdschHndl{nullptr};
    bool               m_pdschReady{false};

    cuphyPdschStatPrms_t           m_statPrms{};
    std::vector<cuphyCellStatPrm_t> m_cellStatStorage;
    std::vector<cuphyPdschDbgPrms_t> m_dbgStorage;
    cuphyTracker_t                 m_tracker{};
    cuphyMemoryFootprint           m_footprint{};

    cuphyPdschDynPrms_t            m_dynPrms{};
    cuphyPdschStatusOut_t          m_statusInfo{};
    cuphyPdschCellGrpDynPrm_t      m_cellGrpDyn{};
    cuphyPdschCellDynPrm_t         m_cellDyn{};
    cuphyPdschUeGrpPrm_t           m_ueGrpDyn{};
    cuphyPdschUePrm_t              m_ueDyn{};
    cuphyPdschCwPrm_t              m_cwDyn{};
    cuphyPdschDmrsPrm_t            m_dmrsDyn{};
    uint16_t                       m_uePrmIdxs[1]{0};
    uint16_t                       m_cwIdxs[1]{0};

    static constexpr int           kPdschOutCells = 8;
    /** Filled by PdschTx when non-null; silences nullptr warning and matches nCells in m_cellGrpDyn. */
    std::array<cuphyPdschCellAerialMetrics_t, kPdschOutCells> m_pdschCellMetrics{};
    std::vector<uint8_t*>          m_tbInputPtrs;
    cuphyPdschDataIn_t             m_dataIn{};
    std::vector<cuphyTensorPrm_t>  m_txTensorStorage;
    cuphyPdschDataOut_t            m_dataOut{};

    uint8_t                        m_rbBitmap[MAX_RBMASK_BYTE_SIZE]{};

    void*                          m_txDev{nullptr};
    size_t                         m_txBytes{0};
    cuphy::tensor_desc             m_txDesc{};

    bool                           m_embedWarmupDone{false};
    void*                          m_embedWarmTbDev{nullptr};
    cuphyPuschCellDynPrm_t         m_embCellDyn{};
    cuphyPuschUeGrpPrm_t           m_embUeGrp{};
    cuphyPuschUePrm_t              m_embUe{};
    cuphyPuschDmrsPrm_t            m_embDmrs{};
    cuphyPuschCellGrpDynPrm_t      m_embCellGrp{};
    uint16_t                       m_embUePrmIdxs[1]{0};
};

/** CUDA graph capture helpers for NAI kernels (device symbols in nai_ni_kernel.cu). */
cuphyStatus_t cuphy_nai_graph_template_copy_rx_kernel(CUDA_KERNEL_NODE_PARAMS* p, void** kernelParamPtrs);
cuphyStatus_t cuphy_nai_graph_template_ni_sub_kernel(CUDA_KERNEL_NODE_PARAMS* p, void** kernelParamPtrs);
