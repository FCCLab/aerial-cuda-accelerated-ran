/*
 * NAI: PdschTx graph embed support for PuschRx full-slot graph (RX->NI copy and subtract run as graph kernel nodes).
 */
#include "nai_pdsch_recon.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdio>
#include "cuphy.h"
#include "cuphy_api.h"
#include "util.hpp"

namespace {

constexpr int kTonesPerPrb = 12;
constexpr int kMaxScGrid = 273 * kTonesPerPrb;
constexpr int kMaxSym = 14;
constexpr int kMaxTxPorts = 16;

void log_nai_once(const char* msg)
{
    static int n = 0;
    if (n++ < 16) {
        std::fprintf(stderr, "cuphy_nai_recon: %s\n", msg);
    }
}

// Throttled trace: per-slot embedded PdschTx graph param refresh (hot path).
void log_nai_embed_slot_prepare_ok()
{
    static std::atomic<int> s_n{0};
    if (s_n.fetch_add(1) >= 32) {
        return;
    }
    std::fprintf(stderr, "cuphy_nai_recon: embed slot OK — cuphyPdschTxUpdateGraphExecKernelParams succeeded\n");
}

} // namespace

NaiPdschRecon::~NaiPdschRecon()
{
    if (m_pdschHndl) {
        (void)cuphyDestroyPdschTx(m_pdschHndl);
        m_pdschHndl = nullptr;
    }
    if (m_txDev) {
        (void)cudaFree(m_txDev);
        m_txDev = nullptr;
    }
    if (m_embedWarmTbDev) {
        (void)cudaFree(m_embedWarmTbDev);
        m_embedWarmTbDev = nullptr;
    }
}

cuphyStatus_t NaiPdschRecon::ensurePdschTx(const cuphyCellStatPrm_t* cell_stats,
    uint16_t n_cells,
    int stream_priority)
{
    if (m_pdschReady) {
        return CUPHY_STATUS_SUCCESS;
    }
    if (!cell_stats || n_cells == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    m_cellStatStorage.assign(cell_stats, cell_stats + n_cells);
    m_dbgStorage.resize(n_cells);
    for (auto& d : m_dbgStorage) {
        d = cuphyPdschDbgPrms_t{"", 0, false, true};
    }

    m_statPrms = {};
    m_statPrms.pOutInfo = &m_tracker;
    m_tracker.pMemoryFootprint = &m_footprint;
    m_statPrms.nCells = n_cells;
    m_statPrms.pCellStatPrms = m_cellStatStorage.data();
    m_statPrms.pDbg = m_dbgStorage.data();
    m_statPrms.read_TB_CRC = false;
    m_statPrms.full_slot_processing = true;
    m_statPrms.stream_priority = stream_priority;
    m_statPrms.nMaxCellsPerSlot = 1;
    m_statPrms.nMaxUesPerCellGroup = 1;
    m_statPrms.enableBatchedMemcpy = 0;

    const cuphyStatus_t st = cuphyCreatePdschTx(&m_pdschHndl, &m_statPrms);
    if (st != CUPHY_STATUS_SUCCESS) {
        log_nai_once("cuphyCreatePdschTx failed");
        return st;
    }

    m_txBytes = static_cast<size_t>(kMaxScGrid) * kMaxSym * kMaxTxPorts * sizeof(__half2);
    if (cudaMalloc(&m_txDev, m_txBytes) != cudaSuccess) {
        log_nai_once("cudaMalloc tx grid failed");
        (void)cuphyDestroyPdschTx(m_pdschHndl);
        m_pdschHndl = nullptr;
        return CUPHY_STATUS_ALLOC_FAILED;
    }

    m_txDesc.set(CUPHY_C_16F, kMaxScGrid, kMaxSym, kMaxTxPorts, cuphy::tensor_flags::align_tight);
    m_txTensorStorage.assign(kPdschOutCells, cuphyTensorPrm_t{});
    m_tbInputPtrs.assign(kPdschOutCells, nullptr);
    m_txTensorStorage[0].pAddr = m_txDev;
    m_txTensorStorage[0].desc  = m_txDesc.handle();
    m_dataOut.pTDataTx         = m_txTensorStorage.data();

    m_pdschReady = true;
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t NaiPdschRecon::setupDynForUe(const NaiPuschSlotInput& in,
    uint16_t ue_idx,
    uint16_t ue_grp_idx,
    cudaStream_t stream_override,
    const uint64_t* proc_mode_override)
{
    const auto* pUePrms = in.cell_grp->pUePrms;
    const auto& ug = in.cell_grp->pUeGrpPrms[ue_grp_idx];
    const auto& ue = pUePrms[ue_idx];

    if (ue.enableTfPrcd) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    std::memset(m_rbBitmap, 0, sizeof(m_rbBitmap));

    m_dmrsDyn.nDmrsCdmGrpsNoData = ug.pDmrsDynPrm->nDmrsCdmGrpsNoData;
    m_dmrsDyn.dmrsScrmId = ug.pDmrsDynPrm->dmrsScrmId;

    const int32_t cellIdxMq = 0;
    const auto& cellDyn = in.cell_grp->pCellPrms[cellIdxMq];

    m_cellDyn = {};
    m_cellDyn.cellPrmStatIdx = cellDyn.cellPrmStatIdx;
    m_cellDyn.cellPrmDynIdx = cellDyn.cellPrmDynIdx;
    m_cellDyn.slotNum = cellDyn.slotNum;
    m_cellDyn.pdschStartSym = ug.puschStartSym;
    m_cellDyn.nPdschSym = ug.nPuschSym;
    m_cellDyn.dmrsSymLocBmsk = ug.dmrsSymLocBmsk;
    m_cellDyn.testModel = 0;

    m_ueGrpDyn = {};
    m_ueGrpDyn.pCellPrm = &m_cellDyn;
    m_ueGrpDyn.pDmrsDynPrm = &m_dmrsDyn;
    m_ueGrpDyn.resourceAlloc = 1;
    m_ueGrpDyn.rbBitmap = m_rbBitmap;
    m_ueGrpDyn.startPrb = ug.startPrb;
    m_ueGrpDyn.nPrb = ug.nPrb;
    m_ueGrpDyn.dmrsSymLocBmsk = ug.dmrsSymLocBmsk;
    m_ueGrpDyn.pdschStartSym = ug.puschStartSym;
    m_ueGrpDyn.nPdschSym = ug.nPuschSym;
    m_ueGrpDyn.nUes = 1;
    m_ueGrpDyn.pUePrmIdxs = m_uePrmIdxs;
    m_uePrmIdxs[0] = 0;

    m_ueDyn = {};
    m_ueDyn.pUeGrpPrm = &m_ueGrpDyn;
    m_ueDyn.scid = ue.scid;
    m_ueDyn.nUeLayers = ue.nUeLayers;
    m_ueDyn.dmrsPortBmsk = ue.dmrsPortBmsk;
    m_ueDyn.BWPStart = 0;
    m_ueDyn.refPoint = 0;
    m_ueDyn.beta_dmrs = 1.f;
    m_ueDyn.beta_qam = 1.f;
    m_ueDyn.rnti = ue.rnti;
    m_ueDyn.dataScramId = ue.dataScramId;
    m_ueDyn.enablePrcdBf = false;
    m_ueDyn.pmwPrmIdx = 0;
    m_ueDyn.nCw = 1;
    m_ueDyn.pCwIdxs = m_cwIdxs;
    m_cwIdxs[0] = 0;

    const uint32_t tb_off = in.tb_payload_offsets ? in.tb_payload_offsets[ue_idx] : 0;

    m_cwDyn = {};
    m_cwDyn.pUePrm = &m_ueDyn;
    m_cwDyn.mcsTableIndex = ue.mcsTableIndex;
    m_cwDyn.mcsIndex = ue.mcsIndex;
    m_cwDyn.targetCodeRate = ue.targetCodeRate;
    m_cwDyn.qamModOrder = ue.qamModOrder;
    m_cwDyn.rv = ue.rv;
    m_cwDyn.tbStartOffset = 0;
    m_cwDyn.tbSize = ue.TBSize;
    m_cwDyn.n_PRB_LBRM = ue.n_PRB_LBRM ? ue.n_PRB_LBRM : 273;
    m_cwDyn.maxLayers = ue.maxLayers ? ue.maxLayers : ue.nUeLayers;
    m_cwDyn.maxQm = ue.maxQm ? ue.maxQm : 8;

    m_cellGrpDyn = {};
    m_cellGrpDyn.nCells = 1;
    m_cellGrpDyn.pCellPrms = &m_cellDyn;
    m_cellGrpDyn.nUeGrps = 1;
    m_cellGrpDyn.pUeGrpPrms = &m_ueGrpDyn;
    m_cellGrpDyn.nUes = 1;
    m_cellGrpDyn.pUePrms = &m_ueDyn;
    m_cellGrpDyn.nCws = 1;
    m_cellGrpDyn.pCwPrms = &m_cwDyn;
    m_cellGrpDyn.nCsiRsPrms = 0;
    m_cellGrpDyn.pCsiRsPrms = nullptr;
    m_cellGrpDyn.nPrecodingMatrices = 0;
    m_cellGrpDyn.pPmwPrms = nullptr;
    m_cellGrpDyn.pCellMetrics = m_pdschCellMetrics.data();

    if (!in.tb_payloads_dev || ue.TBSize == 0 || ue.qamModOrder == 0) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    m_tbInputPtrs.assign(kPdschOutCells, nullptr);
    m_tbInputPtrs[0] = in.tb_payloads_dev + tb_off;
    m_dataIn.pTbInput    = m_tbInputPtrs.data();
    m_dataIn.pBufferType = cuphyPdschDataIn_t::GPU_BUFFER;

    const int outIdx = static_cast<int>(m_cellDyn.cellPrmDynIdx);
    if (outIdx < 0 || outIdx >= kPdschOutCells) {
        log_nai_once("cellPrmDynIdx out of range for NAI PdschTx output");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    m_txTensorStorage[outIdx].pAddr = m_txDev;
    m_txTensorStorage[outIdx].desc  = m_txDesc.handle();

    m_dynPrms = {};
    cudaStream_t strm = stream_override;
    if (!strm) {
        log_nai_once("NAI setupDynForUe: null CUDA stream");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    m_dynPrms.cuStream = strm;
    m_dynPrms.procModeBmsk = proc_mode_override ? *proc_mode_override : static_cast<uint64_t>(PDSCH_PROC_MODE_NO_GRAPHS);
    m_dynPrms.pCellGrpDynPrm = &m_cellGrpDyn;
    m_dynPrms.pDataIn = &m_dataIn;
    m_dynPrms.pTbCRCDataIn = nullptr;
    m_dynPrms.pDataOut     = &m_dataOut;
    m_dynPrms.pStatusInfo  = &m_statusInfo;
    m_statusInfo.status    = cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE;
    m_statusInfo.ueIdx     = ue_idx;
    m_statusInfo.cellPrmStatIdx = m_cellDyn.cellPrmStatIdx;

    (void)cudaMemsetAsync(m_txDev, 0, m_txBytes, strm);

    cuphyStatus_t st = cuphySetupPdschTx(m_pdschHndl, &m_dynPrms, nullptr);
    if (st != CUPHY_STATUS_SUCCESS) {
        log_nai_once("cuphySetupPdschTx failed");
    }
    return st;
}

bool NaiPdschRecon::warmupGraphTemplateForParentEmbed(cudaStream_t stream,
    const cuphyPuschStatPrms_t& pusch_stat,
    const cuphyCellStatPrm_t* cell_stats,
    uint16_t n_cell_stats)
{
    if (m_embedWarmupDone) {
        return true;
    }
    if (!m_pdschHndl || !cell_stats || n_cell_stats == 0 || !stream) {
        return false;
    }
    constexpr size_t kWarmTbBytes = 128u * 1024u;
    if (!m_embedWarmTbDev) {
        if (cudaMalloc(&m_embedWarmTbDev, kWarmTbBytes) != cudaSuccess) {
            log_nai_once("NAI embed warmup: cudaMalloc TB failed");
            return false;
        }
        (void)cudaMemsetAsync(m_embedWarmTbDev, 0, kWarmTbBytes, stream);
    }

    m_embCellDyn = {};
    m_embCellDyn.cellPrmStatIdx = 0;
    m_embCellDyn.cellPrmDynIdx = 0;
    m_embCellDyn.slotNum = 0;

    m_embDmrs = {};
    m_embDmrs.dmrsAddlnPos = 0;
    m_embDmrs.dmrsMaxLen = 1;
    m_embDmrs.nDmrsCdmGrpsNoData = 2;
    m_embDmrs.dmrsScrmId = 0;

    // PdschTx DMRS checks ue_group->nPrb against static cell nPrbDlBwp; warmup must not exceed that BWP.
    const uint32_t maxPrbFromStat = pusch_stat.nMaxPrb ? pusch_stat.nMaxPrb : 273u;
    uint32_t nPrbU32 = std::min(maxPrbFromStat, 273u);
    const uint16_t cellDlBwp = cell_stats[0].nPrbDlBwp;
    if (cellDlBwp > 0) {
        nPrbU32 = std::min(nPrbU32, static_cast<uint32_t>(cellDlBwp));
    }
    const uint16_t nPrb = static_cast<uint16_t>(std::max(1u, nPrbU32));

    m_embUeGrp = {};
    m_embUeGrp.pCellPrm = &m_embCellDyn;
    m_embUeGrp.pDmrsDynPrm = &m_embDmrs;
    m_embUeGrp.startPrb = 0;
    m_embUeGrp.nPrb = nPrb;
    m_embUeGrp.prgSize = 1;
    m_embUeGrp.nUplinkStreams = 1;
    m_embUeGrp.puschStartSym = 0;
    m_embUeGrp.nPuschSym = 14;
    m_embUeGrp.dmrsSymLocBmsk = 0xC03;
    m_embUeGrp.rssiSymLocBmsk = 0;
    m_embUeGrp.nUes = 1;
    m_embUeGrp.pUePrmIdxs = m_embUePrmIdxs;
    m_embUePrmIdxs[0] = 0;

    m_embUe = {};
    m_embUe.pduBitmap = 0x01;
    m_embUe.pUeGrpPrm = &m_embUeGrp;
    m_embUe.ueGrpIdx = 0;
    m_embUe.enableTfPrcd = 0;
    m_embUe.scid = 0;
    m_embUe.dmrsPortBmsk = 1;
    m_embUe.mcsTableIndex = 0;
    m_embUe.mcsIndex = 9;
    m_embUe.targetCodeRate = 4500;
    m_embUe.qamModOrder = 4;
    m_embUe.TBSize = 3997;
    m_embUe.rv = 0;
    m_embUe.rnti = 1;
    m_embUe.dataScramId = 0;
    m_embUe.nUeLayers = 1;
    m_embUe.ndi = 1;
    m_embUe.harqProcessId = 0;
    m_embUe.i_lbrm = 0;
    m_embUe.maxLayers = 1;
    m_embUe.maxQm = 8;
    m_embUe.n_PRB_LBRM = nPrb;
    m_embUe.pUciPrms = nullptr;

    m_embCellGrp = {};
    m_embCellGrp.nCells = 1;
    m_embCellGrp.pCellPrms = &m_embCellDyn;
    m_embCellGrp.nUeGrps = 1;
    m_embCellGrp.pUeGrpPrms = &m_embUeGrp;
    m_embCellGrp.nUes = 1;
    m_embCellGrp.pUePrms = &m_embUe;

    NaiPuschSlotInput win{};
    win.stream = stream;
    win.cell_grp = &m_embCellGrp;
    win.dyn = nullptr;
    win.cell_stats = cell_stats;
    win.n_cell_stats = n_cell_stats;
    win.tb_payloads_dev = static_cast<uint8_t*>(m_embedWarmTbDev);
    win.tb_crcs_dev = nullptr;
    win.tb_payload_offsets = nullptr;

    const uint64_t graphMode = static_cast<uint64_t>(PDSCH_PROC_MODE_GRAPHS);
    if (setupDynForUe(win, 0, 0, stream, &graphMode) != CUPHY_STATUS_SUCCESS) {
        log_nai_once("NAI embed warmup: setupDynForUe failed");
        return false;
    }
    if (cuphySetupPdschTx(m_pdschHndl, &m_dynPrms, nullptr) != CUPHY_STATUS_SUCCESS) {
        log_nai_once("NAI embed warmup: cuphySetupPdschTx failed");
        return false;
    }
    if (cuphyPdschTxUpdateGraphExecKernelParams(m_pdschHndl) != CUPHY_STATUS_SUCCESS) {
        log_nai_once("NAI embed warmup: cuphyPdschTxUpdateGraphExecKernelParams failed");
        return false;
    }

    m_embedWarmupDone = true;
    log_nai_once("NAI embed warmup OK — PdschTx graph template ready for parent-graph child node");
    return true;
}

cuphyStatus_t NaiPdschRecon::refreshEmbeddedGraphToWarmupIdle(cudaStream_t graph_stream)
{
    if (!m_embedWarmupDone || !m_pdschHndl || !graph_stream || m_cellStatStorage.empty() || !m_embedWarmTbDev) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    NaiPuschSlotInput win{};
    win.stream = graph_stream;
    win.cell_grp = &m_embCellGrp;
    win.dyn = nullptr;
    win.cell_stats = m_cellStatStorage.data();
    win.n_cell_stats = static_cast<uint16_t>(m_cellStatStorage.size());
    win.tb_payloads_dev = static_cast<uint8_t*>(m_embedWarmTbDev);
    win.tb_crcs_dev = nullptr;
    win.tb_payload_offsets = nullptr;

    const uint64_t graphMode = static_cast<uint64_t>(PDSCH_PROC_MODE_GRAPHS);
    if (setupDynForUe(win, 0, 0, graph_stream, &graphMode) != CUPHY_STATUS_SUCCESS) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    if (cuphySetupPdschTx(m_pdschHndl, &m_dynPrms, nullptr) != CUPHY_STATUS_SUCCESS) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return cuphyPdschTxUpdateGraphExecKernelParams(m_pdschHndl);
}

cuphyStatus_t NaiPdschRecon::prepareEmbeddedGraphSlot(NaiPuschSlotInput& in, void* ni_dev, cudaStream_t graph_stream)
{
    (void)ni_dev;
    uint16_t ue_grp_idx = 0;
    uint16_t ue_idx = 0;
    bool found = false;
    for (uint32_t ug = 0; ug < in.cell_grp->nUeGrps; ++ug) {
        const auto& ugp = in.cell_grp->pUeGrpPrms[ug];
        if (ugp.nPrb > 0 && ugp.nUes > 0) {
            ue_grp_idx = static_cast<uint16_t>(ug);
            ue_idx = ugp.pUePrmIdxs[0];
            found = true;
            break;
        }
    }
    if (!found) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const auto& ue = in.cell_grp->pUePrms[ue_idx];
    if (!(ue.pduBitmap & 0x01) || ue.enableTfPrcd) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t graphMode = static_cast<uint64_t>(PDSCH_PROC_MODE_GRAPHS);
    if (setupDynForUe(in, ue_idx, ue_grp_idx, graph_stream, &graphMode) != CUPHY_STATUS_SUCCESS) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if (cuphySetupPdschTx(m_pdschHndl, &m_dynPrms, nullptr) != CUPHY_STATUS_SUCCESS) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    const cuphyStatus_t graph_upd_st = cuphyPdschTxUpdateGraphExecKernelParams(m_pdschHndl);
    if (graph_upd_st == CUPHY_STATUS_SUCCESS) {
        log_nai_embed_slot_prepare_ok();
    }
    return graph_upd_st;
}
