/*
 * NAI latency microbenchmark for cuPHY (in-cuBB NAI tail: copy_rx -> PdschTx child graph -> ni_sub).
 *
 * Builds a minimal CUDA graph matching the PUSCH embedded NAI path and reports GPU times via cudaEvent.
 */
#include "nai_pdsch_recon.hpp"
#include "cuphy_api.h"
#include "cuphy.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace {

constexpr int kTonesPerPrb = 12;

#define CUDA_RT_CHECK(expr)                                                                                            \
    do {                                                                                                               \
        const cudaError_t _e = (expr);                                                                                 \
        if (_e != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));               \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CU_DRV_CHECK(expr)                                                                                             \
    do {                                                                                                               \
        const CUresult _r = (expr);                                                                                    \
        if (_r != CUDA_SUCCESS) {                                                                                      \
            const char* _msg = nullptr;                                                                              \
            cuGetErrorString(_r, &_msg);                                                                             \
            std::fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__, __LINE__, _msg ? _msg : "?");            \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CUPHY_CHECK(expr)                                                                                              \
    do {                                                                                                               \
        const cuphyStatus_t _s = (expr);                                                                             \
        if (_s != CUPHY_STATUS_SUCCESS) {                                                                            \
            std::fprintf(stderr, "cuphy error %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(_s));                \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

struct NaiMqGraphHostArgsCopy {
    const void* rx{};
    void*       ni{};
    int         nf{};
    int         nt{};
    int         n_ant{};
    int         s0{};
    int         s1{};
    int         s2{};
};

struct NaiMqGraphHostArgsSub {
    const void* rx{};
    const void* tx{};
    void*       ni{};
    const void* h_est{};
    int         nf_grid{};
    int         nt{};
    int         n_ant_rx{};
    int         rx_s0{};
    int         rx_s1{};
    int         rx_s2{};
    int         n_sc_alloc{};
    int         n_layers{};
    int         n_h_pos{};
    int         h_s0{};
    int         h_s1{};
    int         h_s2{};
    int         h_s3{};
    uint16_t    dmrs_port_bmsk{};
    int         prb_start{};
    int         n_prb{};
    int         sym_start{};
    int         n_sym{};
    int         dmrs_sym_bmsk{};
    int         tx_nf{};
    int         tx_nt{};
    int         tx_n_ports{};
    int         tx_s0{};
    int         tx_s1{};
    int         tx_s2{};
};

struct BenchConfig {
    int   gpu_id{0};
    int   n_prb{133};
    int   n_sym{14};
    int   n_ant{4};
    int   n_layers{1};
    int   warmup{20};
    int   iterations{200};
    bool  breakdown{true};
};

struct LatencyStats {
    double mean_us{0.0};
    double min_us{0.0};
    double max_us{0.0};
    double p50_us{0.0};
    double p99_us{0.0};
};

void usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -g  GPU id (default 0)\n"
        "  -p  UL PRBs / BWP size (default 133)\n"
        "  -s  OFDM symbols (default 14)\n"
        "  -a  RX antenna count (default 4)\n"
        "  -l  UE layers (default 1)\n"
        "  -w  Warmup iterations (default 20)\n"
        "  -n  Timed iterations (default 200)\n"
        "  --no-breakdown  Only report full NAI graph latency\n",
        prog);
}

bool parse_args(int argc, char** argv, BenchConfig& cfg)
{
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return false;
        }
        if (std::strcmp(argv[i], "--no-breakdown") == 0) {
            cfg.breakdown = false;
            continue;
        }
        if (i + 1 >= argc) {
            usage(argv[0]);
            return false;
        }
        if (std::strcmp(argv[i], "-g") == 0) {
            cfg.gpu_id = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-p") == 0) {
            cfg.n_prb = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0) {
            cfg.n_sym = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-a") == 0) {
            cfg.n_ant = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-l") == 0) {
            cfg.n_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-w") == 0) {
            cfg.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-n") == 0) {
            cfg.iterations = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return false;
        }
    }
    return cfg.n_prb > 0 && cfg.n_sym > 0 && cfg.n_ant > 0 && cfg.warmup >= 0 && cfg.iterations > 0;
}

LatencyStats compute_stats(std::vector<float>& samples_ms)
{
    LatencyStats out{};
    if (samples_ms.empty()) {
        return out;
    }
    std::sort(samples_ms.begin(), samples_ms.end());
    double sum = 0.0;
    for (float v : samples_ms) {
        sum += static_cast<double>(v);
    }
    out.mean_us = (sum / static_cast<double>(samples_ms.size())) * 1000.0;
    out.min_us = static_cast<double>(samples_ms.front()) * 1000.0;
    out.max_us = static_cast<double>(samples_ms.back()) * 1000.0;
    out.p50_us = static_cast<double>(samples_ms[samples_ms.size() / 2]) * 1000.0;
    const size_t p99_idx = std::min(samples_ms.size() - 1, static_cast<size_t>(std::lround(0.99 * (samples_ms.size() - 1))));
    out.p99_us = static_cast<double>(samples_ms[p99_idx]) * 1000.0;
    return out;
}

void print_stats(const char* label, const LatencyStats& st)
{
    std::printf("%-28s mean=%8.2f us  p50=%8.2f us  p99=%8.2f us  min=%8.2f us  max=%8.2f us\n",
        label,
        st.mean_us,
        st.p50_us,
        st.p99_us,
        st.min_us,
        st.max_us);
}

struct SummaryRow {
    const char* label;
    LatencyStats stats;
};

void print_summary_table(const BenchConfig& cfg, const cudaDeviceProp& prop, const SummaryRow* rows, int n_rows)
{
    std::printf("\n");
    std::printf("+---------------------- NAI latency summary ----------------------+\n");
    std::printf("| GPU: %-57s |\n", prop.name);
    std::printf("| Config: %3d PRB, %2d sym, %d ant, %d layer(s)  warmup=%3d  n=%4d |\n",
        cfg.n_prb,
        cfg.n_sym,
        cfg.n_ant,
        cfg.n_layers,
        cfg.warmup,
        cfg.iterations);
    std::printf("+----------------------------+--------+--------+--------+--------+\n");
    std::printf("| Stage                      |   mean |    p50 |    p99 |    max |\n");
    std::printf("|                            |   (us) |   (us) |   (us) |   (us) |\n");
    std::printf("+----------------------------+--------+--------+--------+--------+\n");
    for (int i = 0; i < n_rows; ++i) {
        std::printf("| %-26s | %6.2f | %6.2f | %6.2f | %6.2f |\n",
            rows[i].label,
            rows[i].stats.mean_us,
            rows[i].stats.p50_us,
            rows[i].stats.p99_us,
            rows[i].stats.max_us);
    }
    std::printf("+----------------------------+--------+--------+--------+--------+\n");
    if (n_rows > 1) {
        std::printf("| Note: breakdown rows are standalone launches, not concurrent. |\n");
        std::printf("|       full_nai_graph = embedded PUSCH path (single launch).   |\n");
        std::printf("+---------------------------------------------------------------+\n");
    }
    std::printf("\n");
}

LatencyStats time_graph_launches(CUgraphExec graph_exec, cudaStream_t stream, int warmup, int iterations)
{
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_RT_CHECK(cudaEventCreate(&start));
    CUDA_RT_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        CU_DRV_CHECK(cuGraphLaunch(graph_exec, stream));
    }
    CU_DRV_CHECK(cuStreamSynchronize(stream));

    std::vector<float> samples_ms;
    samples_ms.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        CUDA_RT_CHECK(cudaEventRecord(start, stream));
        CU_DRV_CHECK(cuGraphLaunch(graph_exec, stream));
        CUDA_RT_CHECK(cudaEventRecord(stop, stream));
        CUDA_RT_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_RT_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples_ms.push_back(ms);
    }

    CUDA_RT_CHECK(cudaEventDestroy(start));
    CUDA_RT_CHECK(cudaEventDestroy(stop));
    return compute_stats(samples_ms);
}

cuphyCellStatPrm_t make_cell_stat(int n_prb, int n_ant)
{
    cuphyCellStatPrm_t cell{};
    cell.phyCellId = 1;
    cell.nRxAnt = static_cast<uint16_t>(n_ant);
    cell.nRxAntSrs = static_cast<uint16_t>(n_ant);
    cell.nTxAnt = static_cast<uint16_t>(n_ant);
    cell.nPrbUlBwp = static_cast<uint16_t>(n_prb);
    cell.nPrbDlBwp = static_cast<uint16_t>(n_prb);
    cell.mu = 1;
    cell.pPuschCellStatPrms = nullptr;
    cell.pPucchCellStatPrms = nullptr;
    return cell;
}

struct NaiBenchResources {
    void* rx_dev{nullptr};
    void* ni_dev{nullptr};
    void* h_dev{nullptr};
    size_t grid_bytes{0};
    size_t h_bytes{0};
    int nf{0};
    int nt{0};
    int n_ant{0};
    int rx_s0{0};
    int rx_s1{0};
    int rx_s2{0};
    int n_sc_alloc{0};
    int n_h_pos{2};
    int h_s0{0};
    int h_s1{0};
    int h_s2{0};
    int h_s3{0};
    int tx_nf{0};
    int tx_nt{0};
    int tx_n_ports{16};
    int tx_s0{0};
    int tx_s1{0};
    int tx_s2{0};
};

bool alloc_resources(const BenchConfig& cfg, NaiBenchResources& res)
{
    res.nf = cfg.n_prb * kTonesPerPrb;
    res.nt = cfg.n_sym;
    res.n_ant = cfg.n_ant;
    res.n_sc_alloc = res.nf;
    // Element strides for __half2 tensors (matches PUSCH NAI graph setup).
    res.rx_s0 = 1;
    res.rx_s1 = res.nf;
    res.rx_s2 = res.nf * res.nt;
    res.grid_bytes = static_cast<size_t>(res.nf) * res.nt * res.n_ant * sizeof(__half2);

    res.tx_nf = res.nf;
    res.tx_nt = cfg.n_sym;
    res.tx_s0 = 1;
    res.tx_s1 = res.tx_nf;
    res.tx_s2 = res.tx_nf * res.tx_nt;

    const int n_h_dmrs = 2;
    res.n_h_pos = n_h_dmrs;
    // Element strides for float2 channel estimates: (ant, layer, sc, dmrs_pos).
    res.h_s3 = 1;
    res.h_s2 = res.n_h_pos;
    res.h_s1 = res.n_sc_alloc * res.h_s2;
    res.h_s0 = cfg.n_layers * res.h_s1;
    res.h_bytes = static_cast<size_t>(cfg.n_ant) * static_cast<size_t>(res.h_s0) * sizeof(float2);

    CUDA_RT_CHECK(cudaMalloc(&res.rx_dev, res.grid_bytes));
    CUDA_RT_CHECK(cudaMalloc(&res.ni_dev, res.grid_bytes));
    CUDA_RT_CHECK(cudaMalloc(&res.h_dev, res.h_bytes));
    CUDA_RT_CHECK(cudaMemset(res.rx_dev, 0, res.grid_bytes));
    CUDA_RT_CHECK(cudaMemset(res.ni_dev, 0, res.grid_bytes));
    CUDA_RT_CHECK(cudaMemset(res.h_dev, 0, res.h_bytes));
    return true;
}

void fill_copy_args(NaiMqGraphHostArgsCopy& args, const NaiBenchResources& res)
{
    args.rx = res.rx_dev;
    args.ni = res.ni_dev;
    args.nf = res.nf;
    args.nt = res.nt;
    args.n_ant = res.n_ant;
    args.s0 = res.rx_s0;
    args.s1 = res.rx_s1;
    args.s2 = res.rx_s2;
}

void fill_sub_args(NaiMqGraphHostArgsSub& args,
    const NaiBenchResources& res,
    const BenchConfig& cfg,
    void* tx_dev)
{
    args.rx = res.rx_dev;
    args.tx = tx_dev;
    args.ni = res.ni_dev;
    args.h_est = res.h_dev;
    args.nf_grid = res.nf;
    args.nt = res.nt;
    args.n_ant_rx = res.n_ant;
    args.rx_s0 = res.rx_s0;
    args.rx_s1 = res.rx_s1;
    args.rx_s2 = res.rx_s2;
    args.n_sc_alloc = res.n_sc_alloc;
    args.n_layers = cfg.n_layers;
    args.n_h_pos = res.n_h_pos;
    args.h_s0 = res.h_s0;
    args.h_s1 = res.h_s1;
    args.h_s2 = res.h_s2;
    args.h_s3 = res.h_s3;
    args.dmrs_port_bmsk = 1;
    args.prb_start = 0;
    args.n_prb = cfg.n_prb;
    args.sym_start = 0;
    args.n_sym = cfg.n_sym;
    args.dmrs_sym_bmsk = 0xC03;
    args.tx_nf = res.tx_nf;
    args.tx_nt = res.tx_nt;
    args.tx_n_ports = res.tx_n_ports;
    args.tx_s0 = res.tx_s0;
    args.tx_s1 = res.tx_s1;
    args.tx_s2 = res.tx_s2;
}

void wire_kernel_ptrs(void** copy_ptrs, NaiMqGraphHostArgsCopy& copy_args, void** sub_ptrs, NaiMqGraphHostArgsSub& sub_args)
{
    copy_ptrs[0] = &copy_args.rx;
    copy_ptrs[1] = &copy_args.ni;
    copy_ptrs[2] = &copy_args.nf;
    copy_ptrs[3] = &copy_args.nt;
    copy_ptrs[4] = &copy_args.n_ant;
    copy_ptrs[5] = &copy_args.s0;
    copy_ptrs[6] = &copy_args.s1;
    copy_ptrs[7] = &copy_args.s2;

    sub_ptrs[0] = &sub_args.rx;
    sub_ptrs[1] = &sub_args.tx;
    sub_ptrs[2] = &sub_args.ni;
    sub_ptrs[3] = &sub_args.h_est;
    sub_ptrs[4] = &sub_args.nf_grid;
    sub_ptrs[5] = &sub_args.nt;
    sub_ptrs[6] = &sub_args.n_ant_rx;
    sub_ptrs[7] = &sub_args.rx_s0;
    sub_ptrs[8] = &sub_args.rx_s1;
    sub_ptrs[9] = &sub_args.rx_s2;
    sub_ptrs[10] = &sub_args.n_sc_alloc;
    sub_ptrs[11] = &sub_args.n_layers;
    sub_ptrs[12] = &sub_args.n_h_pos;
    sub_ptrs[13] = &sub_args.h_s0;
    sub_ptrs[14] = &sub_args.h_s1;
    sub_ptrs[15] = &sub_args.h_s2;
    sub_ptrs[16] = &sub_args.h_s3;
    sub_ptrs[17] = &sub_args.dmrs_port_bmsk;
    sub_ptrs[18] = &sub_args.prb_start;
    sub_ptrs[19] = &sub_args.n_prb;
    sub_ptrs[20] = &sub_args.sym_start;
    sub_ptrs[21] = &sub_args.n_sym;
    sub_ptrs[22] = &sub_args.dmrs_sym_bmsk;
    sub_ptrs[23] = &sub_args.tx_nf;
    sub_ptrs[24] = &sub_args.tx_nt;
    sub_ptrs[25] = &sub_args.tx_n_ports;
    sub_ptrs[26] = &sub_args.tx_s0;
    sub_ptrs[27] = &sub_args.tx_s1;
    sub_ptrs[28] = &sub_args.tx_s2;
}

CUgraphExec build_single_kernel_graph(CUDA_KERNEL_NODE_PARAMS& driver,
    void** param_ptrs,
    cuphyStatus_t (*template_fn)(CUDA_KERNEL_NODE_PARAMS*, void**))
{
    CUPHY_CHECK(template_fn(&driver, param_ptrs));
    CUgraph graph{};
    CU_DRV_CHECK(cuGraphCreate(&graph, 0));
    CUgraphNode node{};
    CU_DRV_CHECK(cuGraphAddKernelNode(&node, graph, nullptr, 0, &driver));
    CUgraphExec graph_exec{};
    CU_DRV_CHECK(cuGraphInstantiate(&graph_exec, graph, 0));
    CU_DRV_CHECK(cuGraphDestroy(graph));
    return graph_exec;
}

CUgraphExec build_full_nai_graph(NaiPdschRecon& recon,
    CUDA_KERNEL_NODE_PARAMS& copy_driver,
    void** copy_ptrs,
    CUDA_KERNEL_NODE_PARAMS& sub_driver,
    void** sub_ptrs,
    CUgraphNode* out_copy_node,
    CUgraphNode* out_pdsch_child,
    CUgraphNode* out_sub_node)
{
    CUgraph graph{};
    CU_DRV_CHECK(cuGraphCreate(&graph, 0));

    CUgraphNode copy_node{};
    CU_DRV_CHECK(cuGraphAddKernelNode(&copy_node, graph, nullptr, 0, &copy_driver));

    const CUgraph pdsch_template = cuphyPdschTxGetGraphTemplate(recon.pdschHandle());
    if (!pdsch_template) {
        std::fprintf(stderr, "cuphyPdschTxGetGraphTemplate returned null\n");
        std::exit(1);
    }

    CUgraphNode dep = copy_node;
    CUgraphNode pdsch_child{};
    CU_DRV_CHECK(cuGraphAddChildGraphNode(&pdsch_child, graph, &dep, 1, pdsch_template));

    dep = pdsch_child;
    CUgraphNode sub_node{};
    CU_DRV_CHECK(cuGraphAddKernelNode(&sub_node, graph, &dep, 1, &sub_driver));

    CUgraphExec graph_exec{};
    CU_DRV_CHECK(cuGraphInstantiate(&graph_exec, graph, 0));
    CUPHY_CHECK(cuphyPdschTxSyncEmbeddedChildInParentGraphExec(recon.pdschHandle(), graph_exec, pdsch_child));
    CU_DRV_CHECK(cuGraphDestroy(graph));

    if (out_copy_node) {
        *out_copy_node = copy_node;
    }
    if (out_pdsch_child) {
        *out_pdsch_child = pdsch_child;
    }
    if (out_sub_node) {
        *out_sub_node = sub_node;
    }
    return graph_exec;
}

} // namespace

int main(int argc, char** argv)
{
    BenchConfig cfg{};
    if (!parse_args(argc, argv, cfg)) {
        return 1;
    }

    CUDA_RT_CHECK(cudaSetDevice(cfg.gpu_id));
    CU_DRV_CHECK(cuInit(0));

    cudaStream_t stream{};
    CUDA_RT_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cuphyCellStatPrm_t cell_stat = make_cell_stat(cfg.n_prb, cfg.n_ant);
    cuphyPuschStatPrms_t pusch_stat{};
    pusch_stat.nMaxPrb = static_cast<uint32_t>(cfg.n_prb);

    NaiPdschRecon recon;
    if (recon.ensurePdschTx(&cell_stat, 1, 0) != CUPHY_STATUS_SUCCESS) {
        std::fprintf(stderr, "NAI bench: ensurePdschTx failed\n");
        return 1;
    }
    if (!recon.warmupGraphTemplateForParentEmbed(stream, pusch_stat, &cell_stat, 1)) {
        std::fprintf(stderr, "NAI bench: warmupGraphTemplateForParentEmbed failed\n");
        return 1;
    }
    CUPHY_CHECK(recon.refreshEmbeddedGraphToWarmupIdle(stream));
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    NaiBenchResources res{};
    alloc_resources(cfg, res);

    NaiMqGraphHostArgsCopy copy_args{};
    NaiMqGraphHostArgsSub sub_args{};
    fill_copy_args(copy_args, res);
    fill_sub_args(sub_args, res, cfg, recon.txGridDev());

    void* copy_ptrs[9]{};
    void* sub_ptrs[32]{};
    wire_kernel_ptrs(copy_ptrs, copy_args, sub_ptrs, sub_args);

    CUDA_KERNEL_NODE_PARAMS copy_driver{};
    CUDA_KERNEL_NODE_PARAMS sub_driver{};
    CUPHY_CHECK(cuphy_nai_graph_template_copy_rx_kernel(&copy_driver, copy_ptrs));
    CUPHY_CHECK(cuphy_nai_graph_template_ni_sub_kernel(&sub_driver, sub_ptrs));

    copy_driver.gridDimX = static_cast<unsigned>((static_cast<unsigned>(res.nf) + 255u) / 256u);
    copy_driver.gridDimY = static_cast<unsigned>(res.nt);
    copy_driver.gridDimZ = static_cast<unsigned>(res.n_ant);

    sub_driver.gridDimX = static_cast<unsigned>((static_cast<unsigned>(res.n_sc_alloc) + 63u) / 64u);
    sub_driver.gridDimY = static_cast<unsigned>(cfg.n_sym);
    sub_driver.gridDimZ = static_cast<unsigned>(cfg.n_ant);

    CUPHY_CHECK(cuphyPdschTxUpdateGraphExecKernelParams(recon.pdschHandle()));

    CUgraphNode pdsch_child{};
    CUgraphExec full_graph = build_full_nai_graph(
        recon, copy_driver, copy_ptrs, sub_driver, sub_ptrs, nullptr, &pdsch_child, nullptr);

    CUgraphExec copy_graph{};
    CUgraphExec sub_graph{};
    CUgraphExec pdsch_graph{};
    if (cfg.breakdown) {
        copy_graph = build_single_kernel_graph(copy_driver, copy_ptrs, cuphy_nai_graph_template_copy_rx_kernel);
        sub_graph = build_single_kernel_graph(sub_driver, sub_ptrs, cuphy_nai_graph_template_ni_sub_kernel);
        const CUgraph pdsch_template = cuphyPdschTxGetGraphTemplate(recon.pdschHandle());
        CU_DRV_CHECK(cuGraphInstantiate(&pdsch_graph, pdsch_template, 0));
        CUPHY_CHECK(cuphyPdschTxUpdateGraphExecKernelParams(recon.pdschHandle()));
    }

    cudaDeviceProp prop{};
    CUDA_RT_CHECK(cudaGetDeviceProperties(&prop, cfg.gpu_id));

    std::printf("cuphy_ex_nai_bench\n");
    std::printf("  gpu=%s (%d)\n", prop.name, cfg.gpu_id);
    std::printf("  n_prb=%d n_sym=%d n_ant=%d n_layers=%d grid=%dx%dx%d\n",
        cfg.n_prb,
        cfg.n_sym,
        cfg.n_ant,
        cfg.n_layers,
        res.nf,
        res.nt,
        res.n_ant);
    std::printf("  warmup=%d iterations=%d\n\n", cfg.warmup, cfg.iterations);

    const LatencyStats full_stats = time_graph_launches(full_graph, stream, cfg.warmup, cfg.iterations);
    print_stats("full_nai_graph", full_stats);

    LatencyStats copy_stats{};
    LatencyStats pdsch_stats{};
    LatencyStats sub_stats{};
    if (cfg.breakdown) {
        copy_stats = time_graph_launches(copy_graph, stream, cfg.warmup, cfg.iterations);
        pdsch_stats = time_graph_launches(pdsch_graph, stream, cfg.warmup, cfg.iterations);
        sub_stats = time_graph_launches(sub_graph, stream, cfg.warmup, cfg.iterations);
        std::printf("\nPer-kernel breakdown (standalone launches, not concurrent):\n");
        print_stats("copy_rx_to_ni", copy_stats);
        print_stats("pdsch_tx_child", pdsch_stats);
        print_stats("ni_sub_h_tx", sub_stats);
        std::printf("\nNote: full_nai_graph is the embedded path used in PUSCH (single graph launch).\n");
    }

    if (cfg.breakdown) {
        const SummaryRow rows[] = {
            {"full_nai_graph", full_stats},
            {"copy_rx_to_ni", copy_stats},
            {"pdsch_tx_child", pdsch_stats},
            {"ni_sub_h_tx", sub_stats},
        };
        print_summary_table(cfg, prop, rows, static_cast<int>(sizeof(rows) / sizeof(rows[0])));
    } else {
        const SummaryRow rows[] = {
            {"full_nai_graph", full_stats},
        };
        print_summary_table(cfg, prop, rows, 1);
    }

    if (copy_graph) {
        CU_DRV_CHECK(cuGraphExecDestroy(copy_graph));
    }
    if (sub_graph) {
        CU_DRV_CHECK(cuGraphExecDestroy(sub_graph));
    }
    if (pdsch_graph) {
        CU_DRV_CHECK(cuGraphExecDestroy(pdsch_graph));
    }
    CU_DRV_CHECK(cuGraphExecDestroy(full_graph));

    CUDA_RT_CHECK(cudaFree(res.rx_dev));
    CUDA_RT_CHECK(cudaFree(res.ni_dev));
    CUDA_RT_CHECK(cudaFree(res.h_dev));
    CUDA_RT_CHECK(cudaStreamDestroy(stream));
    return 0;
}
