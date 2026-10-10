/*
 * RX and NAI resgrid plotter export over ZeroMQ PUB (cppzmq).
 * RX: CUPHY_RX_ZMQ_ENABLE / CUPHY_RX_ZMQ_BIND.  NAI: CUPHY_NAI_ZMQ_ENABLE / CUPHY_NAI_ZMQ_BIND.
 * The bind value is a full address, used as given: ipc:///tmp/resgrid_nai.ipc or tcp://127.0.0.1:5556.
 *
 * Each frame is one multipart message:
 *   part 1: text topic/metadata, e.g. "resgrid_nai seq=1 prb=133 symbols=14 bytes=14896"
 *   part 2: raw float32 IQ
 * A PUB socket drops the message when no subscriber is connected. Sends are non-blocking.
 */

#include "zmq_sink.hpp"
#include "zmq_export.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <signal.h>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#include <zmq.hpp>
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
/** Bind address is used unchanged. Only ipc://, tcp://, and inproc:// are accepted. */
static std::string zmq_endpoint_from_path(const char* path)
{
    if (!cuphy_mq_export::is_zmq_endpoint(path)) return {};
    return path;
}

static std::string ipc_file_from_endpoint(const std::string& endpoint)
{
    if (endpoint.rfind("ipc://", 0) != 0) return {};
    return endpoint.substr(6);
}

static int zmq_sndhwm()
{
    int v = 1;
    const char* ev = std::getenv(RESGRID_ZMQ_SNDHWM_ENV);
    if (!ev || !ev[0]) ev = std::getenv("RESGRID_MQ_MAXMSG");
    if (ev && ev[0]) {
        char* end = nullptr;
        const long parsed = std::strtol(ev, &end, 10);
        if (end != ev && parsed >= 1) v = static_cast<int>(std::min(parsed, 64L));
    }
    return v;
}
#endif

enum class MqChannelKind { kRx, kNai };

/** One ZeroMQ PUB export (RX or NAI); separate worker thread so channels do not block each other. */
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
    int mq_send_logical_frame(const void*, size_t, uint32_t, uint16_t, uint16_t) { return -1; }
#else
    std::unique_ptr<zmq::context_t> zmq_ctx;
    std::unique_ptr<zmq::socket_t> zmq_pub;
    std::string zmq_endpoint;
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

    const char* topic() const
    {
        return (kind == MqChannelKind::kRx) ? "resgrid_rx" : "resgrid_nai";
    }

    void close_mq_peer()
    {
        zmq_pub.reset();
        zmq_ctx.reset();
    }

    void open_mq_if_needed()
    {
        if (zmq_pub) return;
        const char* name = mq_path_name();
        zmq_endpoint = zmq_endpoint_from_path(name);
        if (zmq_endpoint.empty()) return;

        const std::string ipc_file = ipc_file_from_endpoint(zmq_endpoint);
        if (!ipc_file.empty()) (void)unlink(ipc_file.c_str());

        try {
            zmq_ctx = std::make_unique<zmq::context_t>(1);
            zmq_pub = std::make_unique<zmq::socket_t>(*zmq_ctx, zmq::socket_type::pub);
            const int hwm = zmq_sndhwm();
            zmq_pub->set(zmq::sockopt::sndhwm, hwm);
            zmq_pub->set(zmq::sockopt::linger, 0);
            zmq_pub->bind(zmq_endpoint);
            if (!ipc_file.empty()) (void)chmod(ipc_file.c_str(), 0666);
            static std::atomic<int> oklog{0};
            if (oklog.fetch_add(1) < 3) {
                std::fprintf(stderr,
                    "%s: ZeroMQ PUB bound %s topic=%s (sndhwm=%d; multipart header+IQ, drop when unread)\n",
                    tag,
                    zmq_endpoint.c_str(),
                    topic(),
                    hwm);
            }
        } catch (const zmq::error_t& e) {
            static std::atomic<int> log_once{0};
            if (log_once.fetch_add(1) < 4) {
                std::fprintf(stderr, "%s: zmq bind %s failed: %s\n", tag, zmq_endpoint.c_str(), e.what());
            }
            close_mq_peer();
        }
    }

    /** 0 sent, 1 discarded (queue full), -1 transport error.
     *  With no subscriber a PUB send still returns success and ZeroMQ drops the message. */
    int mq_send_logical_frame(const void* data, size_t len, uint32_t frame_seq, uint16_t prb, uint16_t symbols)
    {
        if (!zmq_pub) return -1;
        char meta[192];
        const int meta_n = std::snprintf(meta,
            sizeof(meta),
            "%s seq=%u prb=%u symbols=%u bytes=%zu",
            topic(),
            frame_seq,
            static_cast<unsigned>(prb),
            static_cast<unsigned>(symbols),
            len);
        if (meta_n <= 0) return -1;
        try {
            zmq::message_t header_msg(meta, static_cast<size_t>(meta_n));
            zmq::message_t body_msg(data, len);
            const auto sent_hdr = zmq_pub->send(header_msg, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
            if (!sent_hdr) return (zmq_errno() == EAGAIN) ? 1 : -1;
            const auto sent_body = zmq_pub->send(body_msg, zmq::send_flags::dontwait);
            if (!sent_body) return -1;
            return 0;
        } catch (const zmq::error_t& e) {
            static std::atomic<int> sendlog{0};
            if (sendlog.fetch_add(1) < 4) {
                std::fprintf(stderr, "%s: zmq send failed: %s\n", tag, e.what());
            }
            return -1;
        }
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
                    if (!zmq_pub) {
                        frames_dropped.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        const uint32_t fs = mq_frame_seq.fetch_add(1u, std::memory_order_relaxed) + 1u;
                        const int sent = mq_send_logical_frame(plot_frame.data(), wire_bytes, fs, prb_tag, sym_tag);
                        if (sent == 0) {
                            frames_written.fetch_add(1, std::memory_order_relaxed);
                        } else if (sent == 1) {
                            frames_dropped.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            const int werr = zmq_errno();
                            close_mq_peer();
                            mq_reconnects.fetch_add(1, std::memory_order_relaxed);
                            static std::atomic<int> mqwlog{0};
                            if (mqwlog.fetch_add(1) < 8) {
                                std::fprintf(stderr,
                                    "%s: zmq_send failed (%s); closed (reconnects=%llu), will reopen\n",
                                    tag,
                                    zmq_strerror(werr),
                                    static_cast<unsigned long long>(mq_reconnects.load(std::memory_order_relaxed)));
                            }
                            frames_dropped.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
#else
                    static std::atomic<int> nlinux{0};
                    if (nlinux.fetch_add(1) < 2) {
                        std::fprintf(stderr, "%s: ZeroMQ export is Linux-only; build for Linux to send frames\n", tag);
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
            std::fprintf(stderr, "%s: ZeroMQ export is Linux-only\n", tag);
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
                        "%s: set %s to ipc:// or tcp:// when %s is set; skipping export\n",
                        tag,
                        CUPHY_RX_ZMQ_BIND_ENV,
                        CUPHY_RX_ZMQ_ENABLE_ENV);
                } else {
                    std::fprintf(stderr,
                        "%s: set %s to ipc:// or tcp:// when %s is set; skipping export\n",
                        tag,
                        CUPHY_NAI_ZMQ_BIND_ENV,
                        CUPHY_NAI_ZMQ_ENABLE_ENV);
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

static MqChannel g_rx{"cuphy_rx_zmq", MqChannelKind::kRx};
static MqChannel g_nai{"cuphy_nai_zmq", MqChannelKind::kNai};

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
        const char* env = (ch == cuphy_mq_channel::nai) ? CUPHY_NAI_ZMQ_ENABLE_ENV : CUPHY_RX_ZMQ_ENABLE_ENV;
        std::fprintf(stderr, "cuphy_zmq: %s ignored on non-Linux (ZeroMQ export is Linux-only)\n", env);
    }
    return;
#else
    if (ch == cuphy_mq_channel::nai) {
        static std::atomic<int> enlog{0};
        if (enlog.fetch_add(1) < 1) {
            const char* p = cuphy_mq_export::nai_mq_path();
            std::fprintf(stderr,
                "cuphy_zmq: %s=1 bind=%s\n",
                CUPHY_NAI_ZMQ_ENABLE_ENV,
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
