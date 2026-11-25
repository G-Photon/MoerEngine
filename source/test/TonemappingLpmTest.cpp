#include "Engine.h"
#include "config/ConfigManager.h"
#include "log/LogSystem.h"
#include "renderer/raster/TonemappingPass.h"
#include "rhi/executor/RHIExecutor.h"
#include "taskgraph/TaskSystem.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

using namespace Moer;
using namespace Moer::Render;
using namespace Moer::Render::Raster;

namespace {
constexpr uint32                width = 16, height = 32;
constexpr std::array<double, 3> rec709{0.21263900587151036, 0.7151686787677559, 0.07219231536073371};
using Pixels = std::array<float4, width * height>;
static_assert(sizeof(TonemappingPipelineBindlessParam) <= 128);
static_assert(sizeof(TonemappingPass::LpmControlBlock) == 384);

double Luma(const std::array<double, 3>& color) {
    return color[0] * rec709[0] + color[1] * rec709[1] + color[2] * rec709[2];
}

// Independent FP64 reference for Rec.709 without shoulder contrast. Solve the
// curve from its gray/white anchors instead of reproducing AMD's setup algebra.
std::array<double, 3> ReferenceLpm(std::array<double, 3> color, const RasterConfig::TonemappingLpmConfig& config) {
    for (auto& c : color)
        c = std::max(c, 0.0);
    const double peak = *std::max_element(color.begin(), color.end());
    if (peak == 0.0)
        return {};
    const double power         = 1.0 + config.contrast;
    const double white         = std::pow(config.hdr_max, power);
    const double middle        = std::pow(config.hdr_max * 0.18 * std::exp2(-config.exposure_range_stops), power);
    const double bias          = (1.0 / 0.18 - 1.0) / (1.0 / middle - 1.0 / white);
    const double scale         = 1.0 - bias / white;
    const double original_luma = std::pow(Luma(color), power);
    const double target        = original_luma / (scale * original_luma + bias);
    if (target >= 1.0)
        return {1.0, 1.0, 1.0};
    for (auto& c : color)
        c = std::pow(c / peak, power + config.saturation);
    const double                ratio_scale = std::clamp(target / Luma(color), 0.0, 1.0);
    const std::array<double, 3> crosstalk{1.0, config.highlight_crosstalk.x, config.highlight_crosstalk.y};
    std::array<double, 3>       capacity{};
    for (size_t c = 0; c < 3; ++c) {
        color[c] *= ratio_scale;
        capacity[c] = (1.0 - color[c]) * crosstalk[c];
    }
    const double capacity_luma = Luma(capacity);
    const double fill = capacity_luma > 0.0 ? std::clamp(target - Luma(color), 0.0, 1.0) / capacity_luma : 0.0;
    for (size_t c = 0; c < 3; ++c)
        color[c] = std::clamp(color[c] + fill * capacity[c], 0.0, 1.0);
    const double remaining = std::clamp(target - Luma(color), 0.0, 1.0);
    for (size_t c = 0; c < 3; ++c)
        color[c] = std::clamp(color[c] + remaining / rec709[c], 0.0, 1.0);
    return color;
}

Pixels MakePixels(bool include_negative) {
    const std::array seeds{
        float4(0.f, 0.f, 0.f, 1.f),
        float4(-1.f, -2.f, -3.f, 1.f),
        float4(0.18f, 0.18f, 0.18f, 1.f),
        float4(256.f, 256.f, 256.f, 1.f),
        float4(0.5f, 1.f, 4.f, 1.f),
        float4(32.f, 0.f, 0.f, 1.f),
        float4(0.f, 32.f, 0.f, 1.f),
        float4(0.f, 0.f, 32.f, 1.f),
        float4(1000000.f, 1000000.f, 1000000.f, 1.f),
        float4(-0.2f, 2.f, 8.f, 1.f),
        float4(1.e-6f, 2.e-6f, 4.e-6f, 1.f),
        float4(65504.f, 1.f, 0.f, 1.f),
        float4(1.f, 65504.f, 0.f, 1.f),
        float4(0.f, 1.f, 65504.f, 1.f),
        float4(1.f, 1.f, 1.f, 1.f),
        float4(0.2f, 0.4f, 0.8f, 1.f),
    };
    Pixels pixels{};
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = seeds[i % seeds.size()];
        if (!include_negative)
            for (size_t c = 0; c < 3; ++c)
                pixels[i][c] = std::max(pixels[i][c], 0.f);
    }
    return pixels;
}

template<class T>
Array<byte> Bytes(const T& value) {
    Array<byte> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

struct Result {
    Array<float>  pixels;
    float         exposure;
    Array<uint32> histogram;
};

Result Run(ShaderManager& manager, const RasterConfig& config, bool parallel, bool merged) {
    auto&                             device = RenderDevice::Get();
    TonemappingPass                   pass(device, manager, PF_R32G32B32A32_SFLOAT);
    auto                              input  = device.CreateTexture("LpmInput", Extent3D(width, height, 1), PF_R32G32B32A32_SFLOAT, ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_DST);
    auto                              output = device.CreateTexture("LpmOutput", Extent3D(width, height, 1), PF_R32G32B32A32_SFLOAT, ETextureUsageFlags::COLOR_ATTACHMENT | ETextureUsageFlags::TRANSFER_SRC);
    TonemappingPass::RecordParameters params{};
    params.input                 = {input, 0, {}};
    params.output                = {output, 0, {}};
    params.exposure              = pass.GetExposureBuffer();
    params.histogram_groups      = uint3(1, 2, 1);
    auto& shader                 = params.shader;
    shader.resolution            = uint2(width, height);
    shader.resolution_inv        = float2(1.f / width, 1.f / height);
    shader.exposure_ev           = std::exp2(config.tonemapping_exposure_ev);
    shader.reinhard_enabled      = config.tonemapping_reinhard_enabled;
    shader.lpm_enabled           = config.tonemapping_lpm.enabled;
    params.lpm_control           = pass.PrepareLpm(config.tonemapping_lpm);
    auto& ae                     = shader.ae;
    ae.enabled                   = config.tonemapping_ae.enabled;
    ae.debug_visualize           = config.tonemapping_ae.debug_visualize;
    ae.aces_tonemapping_enabled  = config.tonemapping_ae.aces_tonemapping_enabled;
    ae.log2lum_to01_scale        = 1.f / 26.f;
    ae.log2lum_to01_bias         = 10.f / 26.f;
    ae.log2lum_to01_scale_inv    = 26.f;
    ae.log2lum_to01_bias_inv     = -10.f;
    ae.histogram_low_percentile  = 0.1f;
    ae.histogram_high_percentile = 0.9f;
    ae.min_adapted_luminance     = 0.01f;
    ae.max_adapted_luminance     = 5000.f;
    ae.eye_adaptation_speed_up = ae.eye_adaptation_speed_down = 1.f;
    ae.frame_time                                             = 1.f;
    ae.diff_log2_threshold                                    = 0.01f;
    RenderGraph                     graph("LpmRegression", RenderGraph::QueueTopology::FromRHI());
    TonemappingPass::GraphResources resources{
        .input              = graph.ImportTexture("Input", input, {}),
        .exposure           = graph.ImportBuffer("Exposure", params.exposure, {.byte_size = params.exposure->GetByteSize()}),
        .tonemapping_output = graph.ImportTexture("Output", output, {}),
    };
    for (const auto texture : {resources.input, resources.tonemapping_output}) {
        graph.SetInitialState(texture, RenderGraph::TextureBoundary::Undefined, RenderGraph::QueueRole::None, RenderGraph::AccessMode::None);
    }
    graph.SetInitialState(resources.exposure, RenderGraph::BufferBoundary::Undefined, RenderGraph::QueueRole::None, RenderGraph::AccessMode::None);
    graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
            .Write(resources.input, RenderGraph::TextureAccessType::TransferDestination)
            .Write(resources.exposure, RenderGraph::BufferAccessType::TransferDestination);
    },
                        [=](CommandList& commands) {
                            commands.CopyFrom(Bytes(MakePixels(config.tonemapping_lpm.enabled)), input->GetView());
                            commands.CopyFrom(Bytes(1.f), params.exposure->GetView());
                        },
                        RenderGraph::PassRecordPolicy::ParallelAllowed);
    const auto     prepared    = graph.AddSetupPass("Prepare", params, [](const auto& value) {
        return value;
    });
    const auto     tonemapping = pass.AddRecordPasses(graph, resources, prepared);
    ReadbackFuture image, exposure, histogram;
    graph.AddRecordPass("Readback", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
            .Read(resources.tonemapping_output, RenderGraph::TextureAccessType::TransferSource)
            .Read(tonemapping.histogram, RenderGraph::BufferAccessType::TransferSource)
            .Read(resources.exposure, RenderGraph::BufferAccessType::TransferSource)
            .SideEffect();
    },
                        [&](CommandList& commands) {
                            image     = commands.Readback(output->GetView());
                            exposure  = commands.Readback(params.exposure->GetView());
                            histogram = commands.Readback(graph.GetPhysicalBuffer(tonemapping.histogram)->GetView());
                        },
                        RenderGraph::PassRecordPolicy::ParallelAllowed);
    graph.Export(resources.input, RenderGraph::TextureAccessType::Sampled, RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
    graph.Export(resources.tonemapping_output, RenderGraph::TextureAccessType::TransferSource, RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
    graph.Export(resources.exposure, RenderGraph::BufferAccessType::TransferSource, RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
    if (!graph.Compile())
        throw std::runtime_error(graph.GetCompileError());
    bool recorded;
    if (merged) {
        CommandList commands(EQueueType::Graphics);
        commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
        recorded = graph.RecordAndMergeFrontendCommands(commands, parallel);
        if (recorded)
            RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    } else {
        recorded = graph.ExecuteFrontendRecordingPlan({}, parallel, {}, RenderGraph::ActiveRecordingOptions{.enabled = true});
    }
    if (!recorded)
        throw std::runtime_error(graph.GetCompileError());
    RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
    if (!image.WaitFor(std::chrono::seconds(20)) || !exposure.WaitFor(std::chrono::seconds(20)) ||
        !histogram.WaitFor(std::chrono::seconds(20))) {
        throw std::runtime_error("LPM readback timed out");
    }
    const auto values = image.Get().CopyAs<float>();
    const auto ev     = exposure.Get().CopyAs<float>();
    const auto bins   = histogram.Get().CopyAs<uint32>();
    if (!values || values->size() != width * height * 4 || !ev || ev->size() != 1 ||
        !std::isfinite(ev->front()) || ev->front() <= 0.f || !bins || bins->size() != 256)
        throw std::runtime_error("Invalid LPM readback");
    return {*values, ev->front(), *bins};
}

void Check(const Result& result, const RasterConfig& config, const char* name) {
    const auto input     = MakePixels(config.tonemapping_lpm.enabled);
    double     max_error = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        const auto&           source = input[i];
        const double          ev     = std::exp2(config.tonemapping_exposure_ev) / (config.tonemapping_ae.enabled ? result.exposure : 1.f);
        std::array<double, 3> expected{source.x * ev, source.y * ev, source.z * ev};
        const double          uv_y           = (i / width + 0.5) / height;
        const bool            histogram_band = config.tonemapping_ae.enabled && config.tonemapping_ae.debug_visualize && uv_y >= 0.48 && uv_y < 0.52;
        const bool            exposure_band  = config.tonemapping_ae.enabled && config.tonemapping_ae.debug_visualize && uv_y >= 0.52 && uv_y < 0.56;
        if (histogram_band)
            expected.fill(double(result.histogram[(i % width) * 16 + 8]) / TONEMAPPING_HISTOGRAM_POINT_FRAC_MULTIPLIER / 1000.0);
        else if (exposure_band)
            expected.fill(std::log2(result.exposure) / 32.0 + 0.5);
        else if (config.tonemapping_lpm.enabled)
            expected = ReferenceLpm(expected, config.tonemapping_lpm);
        else if (config.tonemapping_ae.enabled && config.tonemapping_ae.aces_tonemapping_enabled) {
            for (auto& c : expected)
                c = std::clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
        } else if (!config.tonemapping_ae.enabled && config.tonemapping_reinhard_enabled) {
            for (auto& c : expected)
                c /= c + 1.0;
        } else if (config.tonemapping_ae.enabled) {
            const double lum    = source.x * 0.2126 + source.y * 0.7152 + source.z * 0.0722;
            const double scaled = lum * ev;
            const double mapped = scaled * (1.0 + scaled / (5000.0 * 5000.0)) / (1.0 + scaled);
            for (size_t c = 0; c < 3; ++c)
                expected[c] = lum > 0.0 ? source[c] * mapped / lum : 0.0;
        }
        for (size_t c = 0; c < 4; ++c) {
            const float actual = result.pixels[i * 4 + c];
            if (!std::isfinite(actual))
                throw std::runtime_error(std::string(name) + ": nonfinite output");
            const double target = c == 3 ? 1.0 : expected[c];
            if (!std::isfinite(target))
                throw std::runtime_error(std::string(name) + ": nonfinite reference pixel=" + std::to_string(i));
            const double error = std::abs(actual - target);
            max_error          = std::max(max_error, error);
            if (error > 0.00025 * std::max(1.0, std::abs(target)))
                throw std::runtime_error(std::string(name) + ": pixel=" + std::to_string(i) + " channel=" + std::to_string(c) + " expected=" + std::to_string(target) + " actual=" + std::to_string(actual));
        }
    }
    std::cout << "[LPM] " << name << " pixels=" << input.size() << " max_abs_error=" << max_error << " exposure=" << result.exposure << std::endl;
}

void Numeric() {
    auto&        manager = ShaderManager::Get();
    RasterConfig config{};
    config.tonemapping_exposure_ev            = 0.f;
    config.tonemapping_ae.enabled             = false;
    config.tonemapping_lpm.enabled            = true;
    const auto                       controls = TonemappingPass::PrepareLpm(config.tonemapping_lpm);
    std::array<std::future<void>, 8> workers;
    for (auto& worker : workers)
        worker = std::async(std::launch::async, [=] {
            for (int i = 0; i < 20; ++i) {
                auto other     = config.tonemapping_lpm;
                other.contrast = 1.f;
                static_cast<void>(TonemappingPass::PrepareLpm(other));
                const auto value = TonemappingPass::PrepareLpm(config.tonemapping_lpm);
                if (std::memcmp(controls.data(), value.data(), sizeof(value)))
                    throw std::runtime_error("LPM setup race");
            }
        });
    for (auto& worker : workers)
        worker.get();
    auto invalid    = config.tonemapping_lpm;
    invalid.hdr_max = invalid.exposure_range_stops = invalid.contrast = invalid.saturation =
        invalid.highlight_crosstalk.x = invalid.highlight_crosstalk.y = std::numeric_limits<float>::quiet_NaN();
    const auto sanitized                                              = TonemappingPass::PrepareLpm(invalid);
    if (std::memcmp(controls.data(), sanitized.data(), sizeof(controls)))
        throw std::runtime_error("LPM invalid setup fallback");
    std::cout << "[LPM] setup 8 concurrent callers and invalid-input fallback passed" << std::endl;
    constexpr std::array names{"manual_lpm", "manual_lpm_ev_minus2", "manual_lpm_ev_plus2", "auto_lpm", "auto_lpm_diagnostics", "lpm_minimum", "lpm_maximum", "legacy_linear", "legacy_reinhard", "legacy_aces", "legacy_default"};
    for (size_t scenario = 0; scenario < names.size(); ++scenario) {
        auto selected = config;
        if (scenario == 1)
            selected.tonemapping_exposure_ev = -2.f;
        if (scenario == 2)
            selected.tonemapping_exposure_ev = 2.f;
        if (scenario == 3 || scenario == 4)
            selected.tonemapping_ae.enabled = true;
        if (scenario == 4)
            selected.tonemapping_ae.debug_visualize = true;
        if (scenario == 5) {
            selected.tonemapping_lpm.hdr_max              = 1.f;
            selected.tonemapping_lpm.exposure_range_stops = 1.f;
            selected.tonemapping_lpm.contrast             = 0.f;
            selected.tonemapping_lpm.saturation           = -0.5f;
            selected.tonemapping_lpm.highlight_crosstalk  = float2(0.001f, 0.001f);
        }
        if (scenario == 6) {
            selected.tonemapping_lpm.hdr_max              = 65504.f;
            selected.tonemapping_lpm.exposure_range_stops = 16.f;
            selected.tonemapping_lpm.contrast             = 1.f;
            selected.tonemapping_lpm.saturation           = 1.f;
            selected.tonemapping_lpm.highlight_crosstalk  = float2(1.f, 1.f);
        }
        if (scenario >= 7)
            selected.tonemapping_lpm.enabled = false;
        if (scenario == 7)
            selected.tonemapping_reinhard_enabled = false;
        if (scenario >= 9)
            selected.tonemapping_ae.enabled = true;
        if (scenario == 10)
            selected.tonemapping_ae.aces_tonemapping_enabled = false;
        const auto reference = Run(manager, selected, false, false);
        Check(reference, selected, names[scenario]);
        for (const auto mode : {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
            const auto actual = Run(manager, selected, mode.first, mode.second);
            Check(actual, selected, names[scenario]);
            for (size_t i = 0; i < actual.pixels.size(); ++i)
                if (std::abs(actual.pixels[i] - reference.pixels[i]) > 0.00001f)
                    throw std::runtime_error("LPM recording mode mismatch");
        }
    }
    std::cout << "[TESTCASE][PASS] LPM 11 scenarios x 4 recording modes; 22528 GPU pixels verified" << std::endl;
}

int SceneTest(int argc, const char** argv) {
    Engine engine;
    try {
        engine.Init(argc, argv, false);
        auto config                    = engine.GetEditorConfig();
        config->selected_render_method = ERenderMethod::Raster;
        config->active_viewport_mode   = EEditorViewportMode::Game;
        auto& raster                   = config->raster_config;
        raster.probe_gi_enabled        = false;
        constexpr std::array names{"legacy_auto", "lpm_auto", "lpm_manual", "lpm_ev", "lpm_contrast", "lpm_saturation", "lpm_crosstalk", "lpm_diagnostics", "legacy_after_lpm"};
        constexpr uint32     frames_per_phase = 12;
        uint32               frames           = 0;
        engine.Run(EngineHooks{.on_tick_test = [&](Scene& scene) {
            if (!scene.IsReady())
                return;
            const size_t phase = frames / frames_per_phase;
            if (phase == names.size()) {
                engine.RequestExit();
                return;
            }
            if (frames % frames_per_phase == 0) {
                raster.tonemapping_lpm                = {};
                raster.tonemapping_lpm.enabled        = phase > 0 && phase < 8;
                raster.tonemapping_ae.enabled         = phase != 2;
                raster.tonemapping_ae.debug_visualize = phase == 7;
                raster.tonemapping_exposure_ev        = phase == 3 ? -1.f : -2.7f;
                if (phase == 4)
                    raster.tonemapping_lpm.contrast = 0.75f;
                if (phase == 5)
                    raster.tonemapping_lpm.saturation = -0.4f;
                if (phase == 6)
                    raster.tonemapping_lpm.highlight_crosstalk = float2(1.f, 1.f);
                std::cout << "[LPM scene] " << names[phase] << std::endl;
            }
            ++frames;
        }});
        if (frames != names.size() * frames_per_phase)
            throw std::runtime_error("LPM scene phases incomplete");
        engine.ShutDown();
        std::cout << "[TESTCASE][PASS] LPM scene 9 phases 108 frames and normal shutdown" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] " << error.what() << std::endl;
        engine.ShutDown();
        return 1;
    }
}
} // namespace

int main(int argc, const char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--scene")
        return SceneTest(argc, argv);
    if (argc != 1) {
        std::cerr << "Usage: TestTonemappingLpm [--scene]\n";
        return 1;
    }
    bool tasks = false, device = false, shaders = false;
    int  result = 1;
    try {
        LogSystem::Init();
        ConfigManager::GetInstance().Init(std::filesystem::current_path());
        TaskSystem::Init();
        tasks = true;
        RenderDevice::Init(DeviceInitInfo{.rhi_type = ERHIType::Vulkan, .name = "TonemappingLpmTest", .rhi_api_version = "1.3"});
        device  = true;
        shaders = true;
        Numeric();
        result = 0;
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] " << error.what() << std::endl;
    }
    if (shaders)
        ShaderManager::ShutDown();
    if (device)
        RenderDevice::Dispose();
    if (tasks)
        TaskSystem::ShutDown();
    return result;
}
