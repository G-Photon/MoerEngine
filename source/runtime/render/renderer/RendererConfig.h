#pragma once

#include "RenderAPI.h"
#include <string_view>

namespace Moer {
struct RasterConfig;
struct RaytracingConfig;
namespace Config {
struct GlobalConfig;
}

// Startup defaults, the legacy quality preset, explicit TOML fields, then the shadow master switch.
RENDER_API void LoadRendererConfigFromTomlFile(
    std::string_view path,
    const Config::GlobalConfig& global,
    RasterConfig& raster,
    RaytracingConfig& raytracing
);
} // namespace Moer
