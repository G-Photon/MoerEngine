#include "config/GlobalConfig.h"
#include "renderer/RendererConfig.h"
#include "renderer/raster/RasterConfig.h"
#include "renderer/raytracing/RaytracingConfig.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
using namespace Moer;

void Expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

struct Fixture {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("moer-config-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path path = directory / "config.toml";
    Fixture() { std::filesystem::create_directory(directory); }
    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
    std::string Write(std::string_view text) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << text;
        if (!stream) {
            throw std::runtime_error("Cannot write test config");
        }
        return path.generic_string();
    }
};

void DefaultsAndTemplate(Fixture& fixture) {
    const auto empty = fixture.Write("");
    const auto global = Config::GlobalConfig::LoadConfigFromTomlFile(empty);
    Expect(global.editor.width == 1920 && global.editor.font_size == 16.f,
           "Empty configuration retains legacy editor fallbacks");
    Expect(global.engine.rhi.type == "Not Specified" && !global.engine.remote.enable,
           "Missing keys retain fallback values rather than template suggestions");
    RasterConfig raster;
    RaytracingConfig raytracing;
    raster.bloom_enabled = false;
    raytracing.max_bounce = 99;
    LoadRendererConfigFromTomlFile(empty, global, raster, raytracing);
    Expect(raster.bloom_enabled && raster.ao_mode == EAoMode::RTAO && raster.aa_mode == EAaMode::SMAA_1X,
           "Empty renderer config starts from original defaults");
    Expect(raster.probe_gi_volumes[1].extent.x == 32.f && raster.probe_gi_volumes[1].intensity_scale == .85f &&
           !raster.probe_gi_volumes[2].enabled && raster.probe_gi_volumes[2].count_y == 4,
           "Probe array partial initializers retain nested defaults");
    Expect(raytracing.max_bounce == 4 && raytracing.grid_config.GetLightsPerCell() == 512 &&
           raytracing.denoiser_cfg.denoiser_type == Render::s_denoiser_mode_relax,
           "Raytracing defaults and compatibility accessors remain available");
    const auto template_path = std::string(MOER_CONFIG_TEMPLATE);
    const auto from_template = Config::GlobalConfig::LoadConfigFromTomlFile(template_path);
    Expect(from_template.editor.font_size == 17.f && from_template.engine.rhi.type == "Vulkan" &&
           from_template.engine.remote.enable && from_template.engine.scene.material_info_log_lines == 20,
           "Generated template preserves deliberate fallback/template differences");
    LoadRendererConfigFromTomlFile(template_path, from_template, raster, raytracing);
    Expect(raster.probe_gi_volumes[1].blend_distance == 3.f && raster.shadow_csm_sm_size == 4096,
           "Generated template is valid TOML with unchanged renderer defaults");
}

void ExplicitFields(Fixture& fixture) {
    const auto path = fixture.Write(R"(
[editor]
width = 800
font_size = 18.5
[engine.render.raster]
low_quality_mode = true
aa_mode = "SMAA_T2X"
ao_mode = "SSDO"
bloom_enabled = false
shadow_csm_num_of_cascades = 3
shadow_csm_cover_ratio_of_camera = [0.1, 0.2, 0.3, 0.4]
tessellated_surface_center = [1.5, 2.0, -3.0]
selected_frame_buffer_index = 999
should_frame_gaussians = true
ai_trt_visualize_buffer_idx = 0
[engine.render.raster.tonemapping_ae]
enabled = false
histogram_low_percentile = 0.25
[[engine.render.raster.probe_gi_volumes]]
count_x = 12
origin = [-2.0, -3.0, -4.0]
[[engine.render.raster.probe_gi_volumes]]
enabled = false
[engine.render.raytracing]
sun_direction = [0.0, 1.0, 0.0]
max_bounce = 8
final_color = "EFC_NORMAL"
[engine.render.raytracing.grid_config]
light_per_ceil = 128
cell_size = 2.0
[engine.render.raytracing.aa_cfg]
jitter_mode = "Halton"
[engine.render.raytracing.restir_di_cfg.spatial_resample_config]
num_spatial_samples = 4
[engine.render.raytracing.denoiser_cfg.hit_dist_params]
A = 5.0
[engine.render.raytracing.export_cfg]
output_texture = "EOT_HDR"
b_export = true
)");
    const auto global = Config::GlobalConfig::LoadConfigFromTomlFile(path);
    RasterConfig raster;
    RaytracingConfig raytracing;
    // A bounded string_view must also be accepted by the new renderer loader.
    const std::string padded = path + ".not-part-of-path";
    LoadRendererConfigFromTomlFile(std::string_view(padded.data(), path.size()), global, raster, raytracing);
    Expect(global.editor.width == 800 && global.editor.font_size == 18.5f, "Global fields load from the same TOML");
    Expect(raster.aa_mode == EAaMode::SMAA_T2X && raster.ao_mode == EAoMode::SSDO && !raster.bloom_enabled,
           "Named enum and bool fields override the legacy preset");
    Expect(raster.shadow_csm_num_of_cascades == 3 && raster.shadow_csm_sm_size == 2048 &&
           raster.shadow_csm_cover_ratio_of_camera[3] == .4f && raster.tessellated_surface_center.z == -3.f,
           "Explicit scalar/fixed array/vector overrides preserve unmentioned preset settings");
    Expect(!raster.tonemapping_ae.enabled && raster.tonemapping_ae.histogram_low_percentile == .25f,
           "Nested raster settings load");
    Expect(raster.probe_gi_volumes[0].count_x == 12 && raster.probe_gi_volumes[0].origin.y == -3.f &&
           !raster.probe_gi_volumes[1].enabled && raster.probe_gi_volumes[1].blend_distance == 3.f,
           "Array of tables overrides individual probe fields without clearing defaults");
    Expect(raster.selected_frame_buffer_index == 0 && !raster.should_frame_gaussians &&
           raster.ai_trt_visualize_buffer_idx == 18 && !raytracing.export_cfg.b_export,
           "One-shot commands and UI indices are excluded from persisted configuration");
    Expect(raytracing.max_bounce == 8 && raytracing.sun_direction.y == 1.f &&
           raytracing.final_color == Render::EFC_NORMAL && raytracing.grid_config.GetLightsPerCell() == 128 &&
           raytracing.aa_cfg.jitter_mode == EJitter::Halton &&
           raytracing.restir_di_cfg.spatial_resample_config.num_spatial_samples == 4 &&
           raytracing.denoiser_cfg.hit_dist_params.A == 5.f &&
           raytracing.export_cfg.output_texture == EOutputTexture::EOT_HDR,
           "Nested raytracing config, names, arrays and compatibility storage load");
    raytracing.grid_config.SetLightsPerCell(64);
    Expect(raytracing.grid_config.light_per_ceil == 64, "Grid accessor aliases its original storage");
}

void PresetAndShadowSwitch(Fixture& fixture) {
    const auto path = fixture.Write(R"(
[engine.render.raster]
low_quality_mode = true
enable_shadow = false
shadow_map_mode = "CSM_AUTO"
)");
    const auto global = Config::GlobalConfig::LoadConfigFromTomlFile(path);
    RasterConfig raster;
    RaytracingConfig raytracing;
    LoadRendererConfigFromTomlFile(path, global, raster, raytracing);
    Expect(raster.ao_mode == EAoMode::SSAO && raster.aa_mode == EAaMode::FXAA_SIMPLIFIED &&
           raster.shadow_csm_num_of_cascades == 1 && raster.shadow_csm_sm_size == 2048 &&
           raster.shadow_csm_cover_ratio_of_camera[0] == .03f && raster.shadow_map_mode == EShadowMapMode::NONE,
           "Legacy quality preset and shadow master switch work without any UI");
}

void InvalidValuesAndAtomicFailure(Fixture& fixture) {
    const auto path = fixture.Write(R"(
[editor]
width = -1
height = "wrong type"
[engine.threading]
submission_batch_window = -1
[engine.render.raster]
aa_mode = "typo"
ao_mode = 999
bloom_enabled = "false"
ssao_intensity = inf
shadow_csm_cover_ratio_of_camera = [0.1, "bad", 0.3, 0.4]
shadow_cache_camera_move_threshold_in_texels = [1.0, 2.0]
tessellated_surface_center = [1.0, 2.0, nan]
shading_extra_ambient_color = [1.0, 2.0]
[engine.render.raytracing]
max_bounce = -1
[engine.render.raytracing.aa_cfg]
jitter_mode = "msaa"
)");
    const auto global = Config::GlobalConfig::LoadConfigFromTomlFile(path);
    Expect(global.editor.width == std::numeric_limits<uint32>::max() && global.editor.height == 1080 &&
           global.engine.threading.submission_batch_window == 2,
           "Legacy GlobalConfig signed/unsigned conversion policies stay unchanged");
    RasterConfig raster;
    RaytracingConfig raytracing;
    LoadRendererConfigFromTomlFile(path, global, raster, raytracing);
    Expect(raster.aa_mode == EAaMode::SMAA_1X && raster.ao_mode == EAoMode::RTAO && raster.bloom_enabled &&
           raster.ssao_intensity == 1.f && raytracing.max_bounce == 4 && raytracing.aa_cfg.jitter_mode == EJitter::MSAA,
           "Invalid renderer types/names, nonfinite floats and unsigned overflow retain defaults");
    Expect(raster.shadow_csm_cover_ratio_of_camera[0] == .005f &&
           raster.shadow_cache_camera_move_threshold_in_texels[1] == 8.f &&
           raster.tessellated_surface_center.x == 0.f && raster.shading_extra_ambient_color.y == 1.f,
           "Malformed vectors and scalar arrays are rejected atomically");
    raster.bloom_enabled = false;
    raytracing.max_bounce = 27;
    fixture.Write("[invalid\n");
    bool threw = false;
    try {
        LoadRendererConfigFromTomlFile(path, global, raster, raytracing);
    } catch (const std::exception&) {
        threw = true;
    }
    Expect(threw && !raster.bloom_enabled && raytracing.max_bounce == 27,
           "Parse failure does not partially replace live configuration");
}
} // namespace

int main(int argc, const char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--inspect-defaults") {
            const Config::GlobalConfig global;
            const RasterConfig raster;
            const RaytracingConfig raytracing;
            std::cout << "ConfigDefaults editor.width=" << global.editor.width
                      << " raster.ssao_intensity=" << raster.ssao_intensity
                      << " raytracing.white_point=" << raytracing.tone_mapping_cfg.white_point << '\n';
            return 0;
        }
        Fixture fixture;
        DefaultsAndTemplate(fixture);
        ExplicitFields(fixture);
        PresetAndShadowSwitch(fixture);
        InvalidValuesAndAtomicFailure(fixture);
        std::cout << "Unified configuration defaults, template, overrides and failure contracts: success\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
