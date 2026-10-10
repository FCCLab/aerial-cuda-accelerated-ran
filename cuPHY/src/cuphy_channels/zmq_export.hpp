/*
 * Shared ZeroMQ PUB export enable/path helpers for RX and NAI plotters.
 *
 * Primary env is the full ZeroMQ bind address (ipc://, tcp://, or inproc://).
 *   RX:  CUPHY_RX_ZMQ_ENABLE,  CUPHY_RX_ZMQ_BIND=ipc:///tmp/resgrid_rx.ipc
 *   NAI: CUPHY_NAI_ZMQ_ENABLE, CUPHY_NAI_ZMQ_BIND=ipc:///tmp/resgrid_nai.ipc
 *   TCP example: CUPHY_NAI_ZMQ_BIND=tcp://127.0.0.1:5556
 *   Send queue: RESGRID_ZMQ_SNDHWM (default 1, max 64)
 *
 * Older *_MQ_* names are still accepted.
 *
 * NAI plotter checklist (cuPHY PUSCH):
 *   - Export must be enabled in the cuphycontroller process environment before PuschRx is constructed
 *     (CUPHY_NAI_ZMQ_ENABLE=1 and optional CUPHY_NAI_ZMQ_BIND; default ipc:///tmp/resgrid_nai.ipc).
 *     L1 binds that address and publishes a two-part message (topic line, then float32 IQ).
 *   - PUSCH dynamic procMode must include full-slot graphs so the NI buffer is produced on-graph and
 *     copyNaiToMq can run (see PUSCH_PROC_MODE_FULL_SLOT_GRAPHS).
 *   - NAI embed init must succeed (warmup PdschTx / BWP-consistent synthetic UE); otherwise m_niGridDev
 *     is never filled by the NAI tail and no NAI frames are sent.
 */
#pragma once

#include <cstdlib>
#include <cstring>

#define CUPHY_RX_ZMQ_ENABLE_ENV "CUPHY_RX_ZMQ_ENABLE"
#define CUPHY_NAI_ZMQ_ENABLE_ENV "CUPHY_NAI_ZMQ_ENABLE"
#define CUPHY_RX_ZMQ_BIND_ENV "CUPHY_RX_ZMQ_BIND"
#define CUPHY_NAI_ZMQ_BIND_ENV "CUPHY_NAI_ZMQ_BIND"
#define RESGRID_ZMQ_SNDHWM_ENV "RESGRID_ZMQ_SNDHWM"

namespace cuphy_mq_export {

inline bool env_truthy(const char* v)
{
    if (!v || !*v) return false;
    if (v[0] == '0' && v[1] == '\0') return false;
    if (v[0] == 'n' || v[0] == 'N') return false;
    return true;
}

inline bool env_enabled(const char* primary,
    const char* legacy1 = nullptr,
    const char* legacy2 = nullptr,
    const char* legacy3 = nullptr,
    const char* legacy4 = nullptr)
{
    if (env_truthy(std::getenv(primary))) return true;
    if (legacy1 && env_truthy(std::getenv(legacy1))) return true;
    if (legacy2 && env_truthy(std::getenv(legacy2))) return true;
    if (legacy3 && env_truthy(std::getenv(legacy3))) return true;
    if (legacy4 && env_truthy(std::getenv(legacy4))) return true;
    return false;
}

/** Full bind/connect address. A short name such as /resgrid_nai is not an endpoint. */
inline bool is_zmq_endpoint(const char* p)
{
    if (!p || !p[0]) return false;
    if (std::strncmp(p, "ipc://", 6) == 0) return true;
    if (std::strncmp(p, "tcp://", 6) == 0) return true;
    if (std::strncmp(p, "inproc://", 9) == 0) return true;
    return false;
}

inline const char* first_endpoint(const char* a,
    const char* b = nullptr,
    const char* c = nullptr,
    const char* d = nullptr,
    const char* e = nullptr,
    const char* f = nullptr,
    const char* g = nullptr)
{
    const char* names[] = {a, b, c, d, e, f, g};
    for (const char* name : names) {
        if (!name) continue;
        const char* p = std::getenv(name);
        if (is_zmq_endpoint(p)) return p;
    }
    return nullptr;
}

inline const char* rx_mq_path()
{
    return first_endpoint(CUPHY_RX_ZMQ_BIND_ENV, "CUPHY_RX_ZMQ_PATH");
}

/** NAI plotter endpoint (in-cuBB NI grid); mirrors rx_mq_path() naming. */
inline const char* nai_mq_path()
{
    const char* p = first_endpoint(CUPHY_NAI_ZMQ_BIND_ENV, "CUPHY_NAI_ZMQ_PATH");
    return p ? p : "ipc:///tmp/resgrid_nai.ipc";
}

inline bool rx_mq_path_configured()
{
    return rx_mq_path() != nullptr;
}

inline bool nai_mq_path_configured()
{
    return nai_mq_path() != nullptr;
}

} // namespace cuphy_mq_export

inline bool cuphy_rx_mq_export_enabled()
{
    return cuphy_mq_export::env_enabled(CUPHY_RX_ZMQ_ENABLE_ENV,
        "CUPHY_RX_MQ_ENABLE",
        "CUPHY_UL_MQ_ENABLE",
        "CUPHY_RESGRID_MQ_ENABLE",
        "CUPHY_RESGRID_FIFO_ENABLE");
}

inline bool cuphy_nai_mq_export_enabled()
{
    return cuphy_mq_export::env_enabled(
        CUPHY_NAI_ZMQ_ENABLE_ENV, "CUPHY_NAI_MQ_ENABLE", "CUPHY_NAI_BUNDLE_MQ_ENABLE", nullptr);
}
