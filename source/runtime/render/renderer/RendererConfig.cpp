#include "RendererConfig.h"
#include "config/GlobalConfig.h"
#include "raster/RasterConfig.h"
#include "raytracing/RaytracingConfig.h"

#include <cmath>
#include <toml++/toml.hpp>
#include <type_traits>
#include <utility>

namespace Moer {
namespace {

template<class T>
bool TryRead(const toml::node_view<const toml::node>& node, T& value) {
    const auto parsed = node.value<T>();
    if (!parsed) {
        return false;
    }
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(*parsed)) {
            return false;
        }
    }
    value = *parsed;
    return true;
}

template<class T>
void Read(const toml::node_view<const toml::node>& node, T& value) {
    static_cast<void>(TryRead(node, value));
}

void Read(const toml::node_view<const toml::node>& node, float3& value) {
    const auto* array = node.as_array();
    if (!array || array->size() != 3) {
        return;
    }
    float3 parsed;
    for (size_t index = 0; index < 3; ++index) {
        if (!TryRead(node[index], parsed[index])) {
            return;
        }
    }
    value = parsed;
}

template<class T, size_t N>
void Read(const toml::node_view<const toml::node>& node, std::array<T, N>& value) {
    const auto* array = node.as_array();
    if (!array || array->size() != N) {
        return;
    }
    auto parsed = value;
    for (size_t index = 0; index < N; ++index) {
        if (!TryRead(node[index], parsed[index])) {
            return;
        }
    }
    value = parsed;
}

#include "generated/RendererConfigEnums.inl"

} // namespace

void LoadRendererConfigFromTomlFile(
    std::string_view path,
    const Config::GlobalConfig& global,
    RasterConfig& out_raster,
    RaytracingConfig& out_raytracing
) {
    const auto toml_config = toml::parse_file(path);
    RasterConfig raster;
    RaytracingConfig raytracing;
    if (global.engine.render.raster.low_quality_mode) {
        raster.ao_mode = EAoMode::SSAO;
        raster.shadow_map_mode = EShadowMapMode::CSM;
        raster.shadow_csm_num_of_cascades = 1;
        raster.shadow_csm_sm_size = 2048;
        raster.shadow_csm_cover_ratio_of_camera[0] = 0.03f;
        raster.aa_mode = EAaMode::FXAA_SIMPLIFIED;
    }
#include "generated/RasterConfigLoad.inl"
#include "generated/RaytracingConfigLoad.inl"
    if (!global.engine.render.raster.enable_shadow) {
        raster.shadow_map_mode = EShadowMapMode::NONE;
    }
    out_raster = std::move(raster);
    out_raytracing = std::move(raytracing);
}

} // namespace Moer
