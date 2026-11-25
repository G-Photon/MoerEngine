#include "Engine.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace Moer;

void Expect(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

void Check(const EditorConfig& config, std::string_view test_case, bool startup = false) {
    const auto& raster = config.raster_config;
    const auto& raytracing = config.raytracing_config;
    if (test_case == "preset") {
        Expect(config.selected_render_method == ERenderMethod::Raster &&
               raster.aa_mode == EAaMode::FXAA_SIMPLIFIED && raster.ao_mode == EAoMode::SSAO &&
               raster.shadow_map_mode == EShadowMapMode::NONE && raster.shadow_csm_num_of_cascades == 1 &&
               raster.shadow_csm_sm_size == 2048, "Engine startup did not apply the quality preset/master switch");
    } else if (test_case == "explicit") {
        Expect(config.selected_render_method == ERenderMethod::Raster &&
               raster.aa_mode == EAaMode::FXAA_QUALITY && raster.ao_mode == EAoMode::SSAO &&
               !raster.bloom_enabled && raster.tonemapping_exposure_ev == -1.25f &&
               raster.tonemapping_ae.histogram_low_percentile == .25f &&
               raster.shadow_csm_num_of_cascades == 2 && raster.shadow_csm_sm_size == 1024,
               "Engine startup/console binding replaced explicit raster fields");
    } else if (test_case == "raytracing") {
        Expect(config.selected_render_method == ERenderMethod::Raytracing &&
               raytracing.max_bounce == 2 && raytracing.grid_config.GetLightsPerCell() == 128 &&
               (startup ? raytracing.grid_config.cell_size == 2.f :
                   std::isfinite(raytracing.grid_config.cell_size) && raytracing.grid_config.cell_size > 0.f) &&
               raytracing.denoiser_cfg.denoiser_type == 0 &&
               raytracing.tone_mapping_cfg.white_point == 2.f && !raytracing.tone_mapping_cfg.enable_color_lut &&
               raytracing.aa_cfg.jitter_mode == EJitter::R2,
               "Engine startup did not load nested raytracing fields");
    } else {
        throw std::runtime_error("Specify --config-case=preset|explicit|raytracing with the matching TOML fixture");
    }
}
} // namespace

int main(int argc, const char** argv) {
    using namespace Moer;
    std::string_view test_case;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument.starts_with("--config-case=")) { test_case = argument.substr(14); }
    }
    Engine engine;
    try {
        engine.Init(argc, argv, false);
        const auto config = engine.GetEditorConfig();
        Check(*config, test_case, true);
        uint32 ready_frames = 0;
        engine.Run(Render::EngineHooks{.on_tick_test = [&](Scene& scene) {
            if (!scene.IsReady()) { return; }
            Check(*config, test_case);
            if (ready_frames == 18) { engine.RequestExit(); return; }
            ++ready_frames;
        }});
        engine.ShutDown();
        Expect(ready_frames == 18, "Startup scene did not reach the expected ready frames");
        if (test_case == "raytracing") {
            std::cout << "ConfigStartup grid cell_size: loaded=2, scene_fitted="
                      << config->raytracing_config.grid_config.cell_size << '\n';
        }
        std::cout << "ConfigStartup " << test_case << ": 18 ready frames, configuration retained, shutdown complete\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        engine.ShutDown();
        return 1;
    }
}
