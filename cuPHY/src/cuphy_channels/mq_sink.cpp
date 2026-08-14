/*
 * RX and NAI resgrid plotter export: POSIX mq, RGMC v2 chunks, raw float32 IQ raster.
 * RX: CUPHY_RX_MQ_ENABLE / CUPHY_RX_MQ_PATH.  NAI: CUPHY_NAI_MQ_ENABLE / CUPHY_NAI_MQ_PATH.
 */

#include "mq_sink.hpp"
#include "mq_export.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <signal.h>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>

#if defined(__linux__)
#include <fcntl.h>
#include <mqueue.h>
#endif

namespace {

constexpr int kTonesPerPrb = 12;
constexpr int kMaxResgridExportPrb = 133;
constexpr size_t kMaxStagingBytes = 32u * 1024u * 1024u;

int env_int(const char* name, int def)
{
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return def;
    if (x < -1 || x > 65536) return def;
    return static_cast<int>(x);
}

struct PendingJob {
    cudaEvent_t event{};
    int device{-1};
    size_t nbytes{};
    int dim0{};
    int dim1{};
    int dim2{};
    bool fp16{};
    int ul_prb_config{};
    int el_stride0{};
    int el_stride1{};
    int el_stride2{};
};

inline int layout_offset_elems(int i0, int i1, int i2, int s0, int s1, int s2)
{
    return i0 * s0 + i1 * s1 + i2 * s2;
}

void convert_half2_to_plotter_frame(const __half2* src,
    int nf,
    int nt,
    int n_ant,
    int s0,
    int s1,
    int s2,
    int out_prb,
    int out_sym,
    int ant_pick,
    std::vector<float>& out)
{
    out.assign(static_cast<size_t>(out_prb) * static_cast<size_t>(out_sym) * 2u, 0.f);
    const int n_prb_src = nf / kTonesPerPrb;
    const int prb_use = (out_prb < n_prb_src) ? out_prb : n_prb_src;
    const int sym_use = (out_sym < nt) ? out_sym : nt;
    for (int sym = 0; sym < sym_use; ++sym) {
        for (int prb = 0; prb < prb_use; ++prb) {
            float acc_re = 0.f;
            float acc_im = 0.f;
            float acc_mag_sq = 0.f;
            int cnt_coh = 0;
            int cnt_mag = 0;
            for (int t = 0; t < kTonesPerPrb; ++t) {
                const int sc = prb * kTonesPerPrb + t;
                if (ant_pick < 0) {
                    for (int a = 0; a < n_ant; ++a) {
                        const int off = layout_offset_elems(sc, sym, a, s0, s1, s2);
                        const __half2 v = src[off];
                        const __half* ph = reinterpret_cast<const __half*>(&v);
                        const float re = __half2float(ph[0]);
                        const float im = __half2float(ph[1]);
                        acc_mag_sq += re * re + im * im;
                        ++cnt_mag;
                    }
                } else {
                    const int a = (ant_pick < n_ant) ? ant_pick : 0;
                    const int off = layout_offset_elems(sc, sym, a, s0, s1, s2);
                    const __half2 v = src[off];
                    const __half* ph = reinterpret_cast<const __half*>(&v);
                    acc_re += __half2float(ph[0]);
                    acc_im += __half2float(ph[1]);
                    ++cnt_coh;
                }
            }
            const size_t flat = static_cast<size_t>(sym) * static_cast<size_t>(out_prb) + static_cast<size_t>(prb);
            if (ant_pick < 0) {
                const float mag = (cnt_mag > 0) ? std::sqrt(acc_mag_sq / static_cast<float>(cnt_mag)) : 0.f;
                out[flat * 2u] = mag;
                out[flat * 2u + 1u] = 0.f;
            } else {
                out[flat * 2u] = (cnt_coh > 0) ? (acc_re / static_cast<float>(cnt_coh)) : 0.f;
                out[flat * 2u + 1u] = (cnt_coh > 0) ? (acc_im / static_cast<float>(cnt_coh)) : 0.f;
            }
        }
    }
}

void convert_float2_to_plotter_frame(const float2* src,
    int nf,
    int nt,
    int n_ant,
    int s0,
    int s1,
    int s2,
    int out_prb,
    int out_sym,
    int ant_pick,
    std::vector<float>& out)
{
    out.assign(static_cast<size_t>(out_prb) * static_cast<size_t>(out_sym) * 2u, 0.f);
    const int n_prb_src = nf / kTonesPerPrb;
    const int prb_use = (out_prb < n_prb_src) ? out_prb : n_prb_src;
    const int sym_use = (out_sym < nt) ? out_sym : nt;
    for (int sym = 0; sym < sym_use; ++sym) {
        for (int prb = 0; prb < prb_use; ++prb) {
            float acc_re = 0.f;
            float acc_im = 0.f;
            float acc_mag_sq = 0.f;
            int cnt_coh = 0;
            int cnt_mag = 0;
            for (int t = 0; t < kTonesPerPrb; ++t) {
                const int sc = prb * kTonesPerPrb + t;
                if (ant_pick < 0) {
                    for (int a = 0; a < n_ant; ++a) {
                        const int off = layout_offset_elems(sc, sym, a, s0, s1, s2);
                        const float2 v = src[off];
                        acc_mag_sq += v.x * v.x + v.y * v.y;
                        ++cnt_mag;
                    }
                } else {
                    const int a = (ant_pick < n_ant) ? ant_pick : 0;
                    const int off = layout_offset_elems(sc, sym, a, s0, s1, s2);
                    const float2 v = src[off];
                    acc_re += v.x;
                    acc_im += v.y;
                    ++cnt_coh;
                }
            }
            const size_t flat = static_cast<size_t>(sym) * static_cast<size_t>(out_prb) + static_cast<size_t>(prb);
            if (ant_pick < 0) {
                const float mag = (cnt_mag > 0) ? std::sqrt(acc_mag_sq / static_cast<float>(cnt_mag)) : 0.f;
                out[flat * 2u] = mag;
                out[flat * 2u + 1u] = 0.f;
            } else {
                out[flat * 2u] = (cnt_coh > 0) ? (acc_re / static_cast<float>(cnt_coh)) : 0.f;
                out[flat * 2u + 1u] = (cnt_coh > 0) ? (acc_im / static_cast<float>(cnt_coh)) : 0.f;
            }
        }
    }
}

#if defined(__linux__)
struct ResgridMqChunkHdr {
    uint32_t magic;
    uint32_t version;
    uint32_t frame_seq;
    uint16_t chunk_idx;
    uint16_t n_chunks;
    uint32_t logical_len;
    uint32_t byte_off;
    uint16_t prb;
    uint16_t symbols;
};
static constexpr uint32_t kResgridMqChunkMagic = 0x52474D43u;
static constexpr uint32_t kResgridMqChunkVersion = 2u;

static long read_proc_long_mq(const char* path, long fallback)
{
    FILE* f = std::fopen(path, "re");
    if (!f) return fallback;
    long v = fallback;
    if (std::fscanf(f, "%ld", &v) != 1) v = fallback;
    std::fclose(f);
    return v;
}

static long effective_mq_msgsize_fallback()
{
    const char* ev = std::getenv("RESGRID_MQ_MSGSIZE");
    if (ev && ev[0]) {
        char* end = nullptr;
        long v = std::strtol(ev, &end, 10);
        if (end != ev && v >= 256) return v;
    }
    const long sysmax = read_proc_long_mq("/proc/sys/fs/mqueue/msgsize_max", 8192);
    return sysmax > 0 ? sysmax : 8192;
}
#endif

enum class MqChannelKind { kRx, kNai };

/** One POSIX mq export path (RX or NAI); separate worker thread so channels do not block each other. */
struct MqChannel {
    const char*   tag;
    MqChannelKind kind;

    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<bool> run{false};
    std::atomic<bool> started{false};
    bool ready_for_worker = false;
    bool done_from_worker = true;
    PendingJob job{};
    std::atomic<uint64_t> frames_written{0};
    std::atomic<uint64_t> frames_dropped{0};
    void* staging = nullptr;
    size_t staging_bytes = 0;

#if !defined(__linux__)
    bool path_ok() const { return false; }
    void close_mq_peer() {}
    void open_mq_if_needed() {}
    bool mq_send_logical_frame(const void*, size_t, uint32_t, uint16_t, uint16_t) { return false; }
#else
    mqd_t mq_wr = (mqd_t)-1;
    long mq_msgsize_cap = 8192;
    std::string mq_path;
    std::atomic<uint64_t> mq_reconnects{0};
    std::atomic<uint32_t> mq_frame_seq{0};

    bool path_ok() const
    {
        return (kind == MqChannelKind::kRx) ? cuphy_mq_export::rx_mq_path_configured()
                                            : cuphy_mq_export::nai_mq_path_configured();
    }

    const char* mq_path_name() const
    {
        return (kind == MqChannelKind::kRx) ? cuphy_mq_export::rx_mq_path() : cuphy_mq_export::nai_mq_path();
    }

    void close_mq_peer()
    {
        if (mq_wr != (mqd_t)-1) {
            (void)mq_close(mq_wr);
            mq_wr = (mqd_t)-1;
        }
        mq_msgsize_cap = 8192;
    }

    void open_mq_if_needed()
    {
        if (mq_wr != (mqd_t)-1) return;
        const char* name = mq_path_name();
        if (!name || name[0] != '/') return;
        mq_path = name;
        const mqd_t m = mq_open(name, O_WRONLY);
        if ((int)m < 0) {
            if (errno != ENOENT) {
                static std::atomic<int> log_once{0};
                if (log_once.fetch_add(1) < 4) {
                    std::fprintf(stderr, "%s: mq_open(%s): %s\n", tag, name, std::strerror(errno));
                }
            }
            return;
        }
        mq_attr attr {};
        if (mq_getattr(m, &attr) == 0 && attr.mq_msgsize > 0)
            mq_msgsize_cap = attr.mq_msgsize;
        else
            mq_msgsize_cap = effective_mq_msgsize_fallback();
        mq_wr = m;
        static std::atomic<int> oklog{0};
        if (oklog.fetch_add(1) < 3) {
            std::fprintf(stderr, "%s: writer connected to %s (msgsize_cap=%ld)\n", tag, mq_path.c_str(), mq_msgsize_cap);
        }
    }

    bool mq_send_logical_frame(const void* data, size_t len, uint32_t frame_seq, uint16_t prb, uint16_t symbols)
    {
        if (mq_wr == (mqd_t)-1) return false;
        const size_t hdr_sz = sizeof(ResgridMqChunkHdr);
        if ((size_t)mq_msgsize_cap <= hdr_sz) {
            static std::atomic<int> caplog{0};
            if (caplog.fetch_add(1) < 2) {
                std::fprintf(stderr, "%s: mq_msgsize_cap too small (%ld)\n", tag, mq_msgsize_cap);
            }
            return false;
        }
        const size_t chunk_cap = (size_t)mq_msgsize_cap - hdr_sz;
        const auto* p = reinterpret_cast<const uint8_t*>(data);
        const uint32_t n_chunks_u32 = len == 0 ? 1u : (uint32_t)((len + chunk_cap - 1u) / chunk_cap);
        const uint16_t n_chunks = (uint16_t)std::min<uint32_t>(n_chunks_u32, 65535u);
        if ((uint32_t)n_chunks != n_chunks_u32) return false;
        size_t off = 0;
        for (uint16_t ci = 0; ci < n_chunks; ++ci) {
            const size_t remain = len - off;
            const size_t take = std::min(chunk_cap, remain);
            ResgridMqChunkHdr h {};
            h.magic = kResgridMqChunkMagic;
            h.version = kResgridMqChunkVersion;
            h.frame_seq = frame_seq;
            h.chunk_idx = ci;
            h.n_chunks = n_chunks;
            h.logical_len = (uint32_t)len;
            h.byte_off = (uint32_t)off;
            h.prb = prb;
            h.symbols = symbols;
            std::vector<uint8_t> msg(hdr_sz + take);
            std::memcpy(msg.data(), &h, hdr_sz);
            if (take) std::memcpy(msg.data() + hdr_sz, p + off, take);
            if (mq_send(mq_wr, reinterpret_cast<const char*>(msg.data()), msg.size(), 0) != 0) return false;
            off += take;
        }
        return true;
    }
#endif /* __linux__ */

    void ensure_staging(size_t nbytes)
    {
        if (nbytes <= staging_bytes && staging) return;
        if (staging) {
            (void)cudaFreeHost(staging);
            staging = nullptr;
            staging_bytes = 0;
        }
        if (cudaSuccess != cudaMallocHost(&staging, nbytes)) {
            staging = nullptr;
            staging_bytes = 0;
            return;
        }
        staging_bytes = nbytes;
    }

    void worker_loop()
    {
        std::vector<float> plot_frame;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mutex);
                cv.wait(lk, [this] { return !run.load(std::memory_order_acquire) || ready_for_worker; });
                if (!run.load(std::memory_order_acquire) && !ready_for_worker) break;
                if (!ready_for_worker) continue;
                ready_for_worker = false;
                lk.unlock();

                PendingJob j = job;

                if (j.device >= 0) {
                    (void)cudaSetDevice(j.device);
                }
                const cudaError_t ev_sync = cudaEventSynchronize(j.event);
                if (ev_sync != cudaSuccess) {
                    static std::atomic<int> evlog{0};
                    if (evlog.fetch_add(1) < 8) {
                        std::fprintf(stderr,
                            "%s: cudaEventSynchronize: %s\n",
                            tag,
                            cudaGetErrorString(ev_sync));
                    }
                }
                (void)cudaEventDestroy(j.event);

                const int n_prb_tensor = j.dim0 / kTonesPerPrb;
                const int ul_cfg = (j.ul_prb_config > 0) ? j.ul_prb_config : n_prb_tensor;
                const int ul_prb_capped = std::min(ul_cfg, kMaxResgridExportPrb);
                int out_prb = env_int("CUPHY_RESGRID_PRB", env_int("CUDA_IQ_PRB", ul_prb_capped));
                int out_sym = env_int("CUPHY_RESGRID_SYMBOLS", env_int("CUDA_IQ_SYMBOLS", j.dim1));
                if (out_prb < 1) out_prb = ul_prb_capped;
                if (out_sym < 1) out_sym = j.dim1;
                out_prb = std::min(out_prb, ul_prb_capped);
                out_prb = std::min(out_prb, n_prb_tensor);
                int ant_pick = env_int("CUPHY_RESGRID_ANTENNA", 0);

                int s0 = j.el_stride0;
                int s1 = j.el_stride1;
                int s2 = j.el_stride2;
                if (s0 == 0 && s1 == 0 && s2 == 0) {
                    s0 = 1;
                    s1 = j.dim0;
                    s2 = j.dim0 * j.dim1;
                }

                const size_t elem_size = j.fp16 ? sizeof(__half2) : sizeof(float2);
                const size_t max_el =
                    (size_t)(j.dim0 - 1) * (size_t)s0 + (size_t)(j.dim1 - 1) * (size_t)s1 + (size_t)(j.dim2 - 1) * (size_t)s2;
                const size_t need_bytes = (max_el + 1u) * elem_size;
                if (j.nbytes < need_bytes) {
                    static std::atomic<int> nbytes_warn{0};
                    if (nbytes_warn.fetch_add(1) < 4) {
                        std::fprintf(stderr,
                            "%s: nbytes=%zu < required %zu for strides (%d,%d,%d) dims (%d,%d,%d); drop frame\n",
                            tag,
                            j.nbytes,
                            need_bytes,
                            s0,
                            s1,
                            s2,
                            j.dim0,
                            j.dim1,
                            j.dim2);
                    }
                    frames_dropped.fetch_add(1, std::memory_order_relaxed);
                } else if (staging && j.nbytes <= staging_bytes) {
                    if (j.fp16) {
                        convert_half2_to_plotter_frame(reinterpret_cast<const __half2*>(staging),
                            j.dim0,
                            j.dim1,
                            j.dim2,
                            s0,
                            s1,
                            s2,
                            out_prb,
                            out_sym,
                            ant_pick,
                            plot_frame);
                    } else {
                        convert_float2_to_plotter_frame(reinterpret_cast<const float2*>(staging),
                            j.dim0,
                            j.dim1,
                            j.dim2,
                            s0,
                            s1,
                            s2,
                            out_prb,
                            out_sym,
                            ant_pick,
                            plot_frame);
                    }
#if defined(__linux__)
                    const size_t wire_bytes = plot_frame.size() * sizeof(float);
                    const uint16_t prb_tag = static_cast<uint16_t>(std::min(out_prb, 65535));
                    const uint16_t sym_tag = static_cast<uint16_t>(std::min(out_sym, 65535));

                    open_mq_if_needed();
                    if (mq_wr == (mqd_t)-1) {
                        frames_dropped.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        const uint32_t fs = mq_frame_seq.fetch_add(1u, std::memory_order_relaxed) + 1u;
                        if (!mq_send_logical_frame(plot_frame.data(), wire_bytes, fs, prb_tag, sym_tag)) {
                            const int werr = errno;
                            close_mq_peer();
                            mq_reconnects.fetch_add(1, std::memory_order_relaxed);
                            static std::atomic<int> mqwlog{0};
                            if (mqwlog.fetch_add(1) < 8) {
                                std::fprintf(stderr,
                                    "%s: mq_send failed (%s); closed (reconnects=%llu), will reopen\n",
                                    tag,
                                    std::strerror(werr),
                                    static_cast<unsigned long long>(mq_reconnects.load(std::memory_order_relaxed)));
                            }
                            frames_dropped.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            frames_written.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
#else
                    static std::atomic<int> nlinux{0};
                    if (nlinux.fetch_add(1) < 2) {
                        std::fprintf(stderr, "%s: POSIX mq export is Linux-only; build for Linux to send frames\n", tag);
                    }
                    frames_dropped.fetch_add(1, std::memory_order_relaxed);
#endif
                }

                lk.lock();
                done_from_worker = true;
                lk.unlock();
                cv.notify_all();
            }
        }
    }

    void ensure_worker_started()
    {
        bool f = false;
        if (!started.compare_exchange_strong(f, true)) return;
        static std::once_flag ignore_sigpipe;
        std::call_once(ignore_sigpipe, [] { (void)signal(SIGPIPE, SIG_IGN); });
        run.store(true, std::memory_order_release);
        done_from_worker = true;
        ready_for_worker = false;
        worker = std::thread([this] { worker_loop(); });
    }

    void enqueue(cudaStream_t stream,
        void* dev_src,
        size_t nbytes,
        int dim0,
        int dim1,
        int dim2,
        bool fp16,
        int ul_prb_config,
        int el_stride0,
        int el_stride1,
        int el_stride2)
    {
#if !defined(__linux__)
        static std::atomic<int> nflog{0};
        if (nflog.fetch_add(1) < 2) {
            std::fprintf(stderr, "%s: POSIX mq export is Linux-only\n", tag);
        }
        (void)stream;
        (void)dev_src;
        (void)nbytes;
        (void)dim0;
        (void)dim1;
        (void)dim2;
        (void)fp16;
        (void)ul_prb_config;
        (void)el_stride0;
        (void)el_stride1;
        (void)el_stride2;
        return;
#else
        if (!path_ok()) {
            static std::atomic<int> cfglog{0};
            if (cfglog.fetch_add(1) < 4) {
                if (kind == MqChannelKind::kRx) {
                    std::fprintf(stderr,
                        "%s: set CUPHY_RX_MQ_PATH or CUDA_IQ_MQ_PATH to a name starting with / "
                        "when %s is set; skipping export\n",
                        tag,
                        CUPHY_RX_MQ_ENABLE_ENV);
                } else {
                    std::fprintf(stderr,
                        "%s: set %s to a name starting with / when %s is set; skipping export\n",
                        tag,
                        CUPHY_NAI_MQ_PATH_ENV,
                        CUPHY_NAI_MQ_ENABLE_ENV);
                }
            }
            return;
        }
#endif
        if (!dev_src || nbytes == 0 || dim0 <= 0 || dim1 <= 0 || dim2 <= 0) return;

        int rs0 = el_stride0;
        int rs1 = el_stride1;
        int rs2 = el_stride2;
        if (rs0 == 0 && rs1 == 0 && rs2 == 0) {
            rs0 = 1;
            rs1 = dim0;
            rs2 = dim0 * dim1;
        }

        const size_t elem_size = fp16 ? sizeof(__half2) : sizeof(float2);
        int          job_dim0 = dim0;
        int          job_dim1 = dim1;
        int          job_dim2 = dim2;
        int          jr0 = rs0;
        int          jr1 = rs1;
        int          jr2 = rs2;
        size_t       job_nbytes = nbytes;

        const int ant0_only = env_int("CUPHY_RESGRID_ANT0_ONLY", 1);
        if (ant0_only != 0 && dim2 > 1) {
            const bool     tight_sc_sym_ant = (jr0 == 1 && jr1 == dim0 && jr2 == dim0 * dim1);
            const size_t   plane_elems = static_cast<size_t>(dim0) * static_cast<size_t>(dim1);
            const size_t   plane_bytes = plane_elems * elem_size;
            if (tight_sc_sym_ant && nbytes >= plane_bytes) {
                job_dim2 = 1;
                job_nbytes = plane_bytes;
            }
        }

        if (job_nbytes > kMaxStagingBytes) {
            frames_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        ensure_worker_started();

        {
            std::lock_guard<std::mutex> lk(mutex);
            if (!done_from_worker) {
                frames_dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            done_from_worker = false;
        }

        ensure_staging(job_nbytes);
        if (!staging) {
            std::lock_guard<std::mutex> lk(mutex);
            done_from_worker = true;
            cv.notify_all();
            frames_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        int cuda_dev = 0;
        (void)cudaGetDevice(&cuda_dev);

        const cudaError_t mc_err = cudaMemcpyAsync(staging, dev_src, job_nbytes, cudaMemcpyDeviceToHost, stream);
        if (mc_err != cudaSuccess) {
            static std::atomic<int> mclog{0};
            if (mclog.fetch_add(1) < 8) {
                std::fprintf(stderr,
                    "%s: cudaMemcpyAsync(%zu bytes): %s\n",
                    tag,
                    job_nbytes,
                    cudaGetErrorString(mc_err));
            }
            std::lock_guard<std::mutex> lk(mutex);
            done_from_worker = true;
            cv.notify_all();
            frames_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        cudaEvent_t ev{};
        if (cudaSuccess != cudaEventCreateWithFlags(&ev, cudaEventDisableTiming)) {
            std::lock_guard<std::mutex> lk(mutex);
            done_from_worker = true;
            cv.notify_all();
            frames_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (cudaSuccess != cudaEventRecord(ev, stream)) {
            cudaEventDestroy(ev);
            std::lock_guard<std::mutex> lk(mutex);
            done_from_worker = true;
            cv.notify_all();
            frames_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        {
            std::lock_guard<std::mutex> lk(mutex);
            job.event = ev;
            job.device = cuda_dev;
            job.nbytes = job_nbytes;
            job.dim0 = job_dim0;
            job.dim1 = job_dim1;
            job.dim2 = job_dim2;
            job.fp16 = fp16;
            job.ul_prb_config = ul_prb_config;
            job.el_stride0 = jr0;
            job.el_stride1 = jr1;
            job.el_stride2 = jr2;
            ready_for_worker = true;
        }
        cv.notify_one();
    }

    void shutdown_impl()
    {
        if (!started.load(std::memory_order_acquire)) return;
        {
            std::lock_guard<std::mutex> lk(mutex);
            run.store(false, std::memory_order_release);
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
#if defined(__linux__)
        close_mq_peer();
#endif
        if (staging) {
            (void)cudaFreeHost(staging);
            staging = nullptr;
            staging_bytes = 0;
        }
        started.store(false, std::memory_order_release);
    }
};

static MqChannel g_rx{"cuphy_rx_mq", MqChannelKind::kRx};
static MqChannel g_nai{"cuphy_nai_mq", MqChannelKind::kNai};

} // namespace

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
    int el_stride2)
{
    if (!cuphy_mq_export_enabled(ch)) return;
#if !defined(__linux__)
    static std::atomic<int> nflog{0};
    if (nflog.fetch_add(1) < 2) {
        const char* env = (ch == cuphy_mq_channel::nai) ? CUPHY_NAI_MQ_ENABLE_ENV : CUPHY_RX_MQ_ENABLE_ENV;
        std::fprintf(stderr, "cuphy_mq: %s ignored on non-Linux (POSIX mq export is Linux-only)\n", env);
    }
    return;
#else
    if (ch == cuphy_mq_channel::nai) {
        static std::atomic<int> enlog{0};
        if (enlog.fetch_add(1) < 1) {
            const char* p = cuphy_mq_export::nai_mq_path();
            std::fprintf(stderr,
                "cuphy_mq: %s=1 path=%s (resgrid IQ, same wire as RX)\n",
                CUPHY_NAI_MQ_ENABLE_ENV,
                p ? p : "(unset)");
        }
        g_nai.enqueue(
            stream, dev_src, nbytes, dim0, dim1, dim2, fp16, ul_prb_config, el_stride0, el_stride1, el_stride2);
    } else {
        g_rx.enqueue(
            stream, dev_src, nbytes, dim0, dim1, dim2, fp16, ul_prb_config, el_stride0, el_stride1, el_stride2);
    }
#endif
}

void cuphy_mq_shutdown()
{
#if defined(__linux__)
    g_rx.shutdown_impl();
    g_nai.shutdown_impl();
#endif
}
