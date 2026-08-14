/*
 * Shared POSIX mq export enable/path helpers for RX and NAI plotters.
 *
 * Primary env (compose); default mq names:
 *   RX:  CUPHY_RX_MQ_ENABLE,  CUPHY_RX_MQ_PATH=/resgrid_rx
 *   NAI: CUPHY_NAI_MQ_ENABLE, CUPHY_NAI_MQ_PATH=/resgrid_nai (same resgrid IQ wire as RX)
 *
 * Legacy aliases remain accepted for older scripts.
 *
 * NAI plotter checklist (cuPHY PUSCH):
 *   - Export must be enabled in the cuphycontroller process environment before PuschRx is constructed
 *     (CUPHY_NAI_MQ_ENABLE=1 and optional CUPHY_NAI_MQ_PATH; default path /resgrid_nai if unset).
 *   - PUSCH dynamic procMode must include full-slot graphs so the NI buffer is produced on-graph and
 *     copyNaiToMq can run (see PUSCH_PROC_MODE_FULL_SLOT_GRAPHS).
 *   - NAI embed init must succeed (warmup PdschTx / BWP-consistent synthetic UE); otherwise m_niGridDev
 *     is never filled by the NAI tail and no NAI frames are sent.
 */
#pragma once

#include <cstdlib>

#define CUPHY_RX_MQ_ENABLE_ENV "CUPHY_RX_MQ_ENABLE"
#define CUPHY_NAI_MQ_ENABLE_ENV "CUPHY_NAI_MQ_ENABLE"
#define CUPHY_NAI_MQ_PATH_ENV "CUPHY_NAI_MQ_PATH"

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
    const char* legacy3 = nullptr)
{
    if (env_truthy(std::getenv(primary))) return true;
    if (legacy1 && env_truthy(std::getenv(legacy1))) return true;
    if (legacy2 && env_truthy(std::getenv(legacy2))) return true;
    if (legacy3 && env_truthy(std::getenv(legacy3))) return true;
    return false;
}

inline const char* rx_mq_path()
{
    const char* p = std::getenv("CUPHY_RX_MQ_PATH");
    if (p && p[0] == '/') return p;
    p = std::getenv("CUPHY_UL_MQ_PATH");
    if (p && p[0] == '/') return p;
    p = std::getenv("CUDA_IQ_MQ_PATH");
    if (p && p[0] == '/') return p;
    p = std::getenv("CUPHY_RESGRID_MQ_PATH");
    if (p && p[0] == '/') return p;
    return nullptr;
}

/** NAI plotter queue (in-cuBB NI grid); mirrors rx_mq_path() naming. */
inline const char* nai_mq_path()
{
    const char* p = std::getenv(CUPHY_NAI_MQ_PATH_ENV);
    if (p && p[0] == '/') return p;
    p = std::getenv("CUPHY_NAI_MQ_OUT_PATH");
    if (p && p[0] == '/') return p;
    p = std::getenv("CUDA_IQ_MQ_PATH_NAI");
    if (p && p[0] == '/') return p;
    p = std::getenv("CUDA_IQ_MQ_PATH_OUT");
    if (p && p[0] == '/') return p;
    p = std::getenv("CUDA_NAI_INPUT_MQ_PATH");
    if (p && p[0] == '/') return p;
    p = std::getenv("CUPHY_NAI_BUNDLE_MQ_PATH");
    if (p && p[0] == '/') return p;
    return "/resgrid_nai";
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
    return cuphy_mq_export::env_enabled(
        CUPHY_RX_MQ_ENABLE_ENV, "CUPHY_UL_MQ_ENABLE", "CUPHY_RESGRID_MQ_ENABLE", "CUPHY_RESGRID_FIFO_ENABLE");
}

inline bool cuphy_nai_mq_export_enabled()
{
    return cuphy_mq_export::env_enabled(CUPHY_NAI_MQ_ENABLE_ENV, "CUPHY_NAI_BUNDLE_MQ_ENABLE", nullptr);
}
