#include "TonemappingPass.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

#define A_CPU 1
#include "thirdparty/fidelityfx-lpm/ffx_a.h"

// AMD's setup callback has no user context. Each setup task owns its scratch block.
thread_local Moer::Render::Raster::TonemappingPass::LpmControlBlock lpm_controls{};

A_STATIC void LpmSetupOut(AU1 index, inAU4 value) {
    lpm_controls[index] = Moer::uint4(value[0], value[1], value[2], value[3]);
}

#include "thirdparty/fidelityfx-lpm/ffx_lpm.h"

float FiniteClamp(float value, float low, float high, float fallback) {
    return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
}

} // namespace

namespace Moer::Render::Raster {

TonemappingPass::LpmControlBlock TonemappingPass::PrepareLpm(const RasterConfig::TonemappingLpmConfig& config) {
    const RasterConfig::TonemappingLpmConfig defaults{};
    AF1                                      saturation[3];
    std::fill_n(saturation, 3, FiniteClamp(config.saturation, -0.5f, 1.0f, defaults.saturation));
    AF1 crosstalk[3]{
        1.0f,
        FiniteClamp(config.highlight_crosstalk.x, 0.001f, 1.0f, defaults.highlight_crosstalk.x),
        FiniteClamp(config.highlight_crosstalk.y, 0.001f, 1.0f, defaults.highlight_crosstalk.y),
    };
    lpm_controls = {};
    const float hdr_max = FiniteClamp(config.hdr_max, 1.0f, 65504.0f, defaults.hdr_max);
    const float range_stops = FiniteClamp(config.exposure_range_stops, 1.0f, 16.0f, defaults.exposure_range_stops);
    const float contrast = FiniteClamp(config.contrast, 0.0f, 1.0f, defaults.contrast);
    LpmSetup(
        false, LPM_CONFIG_709_709, LPM_COLORS_709_709,
        0.0f, hdr_max, range_stops, contrast, 1.0f, saturation, crosstalk
    );
    return lpm_controls;
}

} // namespace Moer::Render::Raster
