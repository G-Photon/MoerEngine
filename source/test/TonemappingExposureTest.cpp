#include "config/ConfigManager.h"
#include "log/LogSystem.h"
#include "renderer/raster/TonemappingPass.h"
#include "taskgraph/TaskSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace Moer;
using namespace Moer::Render;
using namespace Moer::Render::Raster;

namespace {
using Histogram = std::array<uint32, TONEMAPPING_HISTOGRAM_BIN_COUNT>;

void RequireNear(float actual, double expected, const char* name) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-5 * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(name) + ": actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
    std::cout << "[TESTCASE][PASS] " << name << " value=" << actual << '\n';
}

float Evaluate(ShaderManager& manager, const Histogram& counts, AutoExposureParam params, float previous) {
    auto&                            device    = RenderDevice::Get();
    const auto                       usage     = EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST |
                                                 EBufferUsageFlags::TRANSFER_SRC;
    auto                             histogram = device.CreateBuffer<uint32>("ExposureTest.Histogram", counts.size(), usage);
    auto                             exposure  = device.CreateBuffer<uint32>("ExposureTest.Exposure", 1, usage | EBufferUsageFlags::TEXTURE_BUFFER);
    auto                             pipeline  = manager.Compute<TonemappingExposurePipeline>("pipelines/postprocess/color/TonemappingExposure.hlsl");
    TonemappingPipelineBindlessParam shader{};
    shader.ae = params;
    RenderGraph graph("ExposureTest", RenderGraph::QueueTopology::FromRHI());
    const auto  h = graph.ImportBuffer("Histogram", histogram, {.byte_size = histogram->GetByteSize()});
    const auto  e = graph.ImportBuffer("Exposure", exposure, {.byte_size = exposure->GetByteSize()});
    for (auto resource : {h, e}) {
        graph.SetInitialState(resource, RenderGraph::BufferBoundary::Undefined, RenderGraph::QueueRole::None, RenderGraph::AccessMode::None);
    }
    graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
            .Write(h, RenderGraph::BufferAccessType::TransferDestination)
            .Write(e, RenderGraph::BufferAccessType::TransferDestination);
    },
                        [&](CommandList& commands) {
                            commands.CopyFrom(std::as_bytes(std::span(counts)), histogram->GetView());
                            commands.CopyFrom(std::as_bytes(std::span(&previous, 1)), exposure->GetView());
                        },
                        RenderGraph::PassRecordPolicy::Serial);
    graph.AddRecordPass("CalculateExposure", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
            .Read(h, RenderGraph::BufferAccessType::ShaderRead)
            .ReadWrite(e, RenderGraph::BufferAccessType::Storage);
    },
                        [&](CommandList& commands) {
                            commands.Compute(pipeline, shader, histogram, exposure).Dispatch(uint3(1, 1, 1));
                        },
                        RenderGraph::PassRecordPolicy::Serial);
    ReadbackFuture readback;
    graph.AddRecordPass("Readback", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
            .Read(e, RenderGraph::BufferAccessType::TransferSource)
            .SideEffect();
    },
                        [&](CommandList& commands) {
                            readback = commands.Readback(exposure->GetView());
                        },
                        RenderGraph::PassRecordPolicy::Serial);
    graph.Export(h, RenderGraph::BufferAccessType::ShaderRead, RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
    graph.Export(e, RenderGraph::BufferAccessType::TransferSource, RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
    CommandList commands;
    commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
    if (!graph.Compile() || !graph.RecordAndMergeFrontendCommands(commands, false)) {
        throw std::runtime_error(graph.GetCompileError());
    }
    RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
    auto result = readback.Get().CopyAs<float>();
    if (!result || result->size() != 1)
        throw std::runtime_error("Exposure readback failed");
    return result->front();
}

AutoExposureParam Defaults() {
    AutoExposureParam params{};
    params.log2lum_to01_scale_inv    = 8.f;
    params.log2lum_to01_bias_inv     = -4.f;
    params.histogram_low_percentile  = 0.f;
    params.histogram_high_percentile = 1.f;
    params.min_adapted_luminance     = .001f;
    params.max_adapted_luminance     = 10000.f;
    params.eye_adaptation_speed_up   = 6.f;
    params.eye_adaptation_speed_down = 1.f;
    params.diff_log2_threshold       = 100.f;
    params.frame_time                = .1f;
    return params;
}

void CheckHistogram(ShaderManager& manager) {
    Histogram    counts{};
    const uint32 multiplier = static_cast<uint32>(TONEMAPPING_HISTOGRAM_POINT_FRAC_MULTIPLIER);
    counts[64]              = 80 * multiplier; // Luminance 0.25.
    counts[192]             = 20 * multiplier; // Luminance 4.
    auto params             = Defaults();
    RequireNear(Evaluate(manager, counts, params, 1.f), 1.0, "histogram-full-range");
    params.histogram_low_percentile  = .75f;
    params.histogram_high_percentile = .95f;
    RequireNear(Evaluate(manager, counts, params, 1.f), (5.0 * .25 + 15.0 * 4.0) / 20.0, "histogram-fractional-both-boundaries");
    params.histogram_low_percentile  = .1f;
    params.histogram_high_percentile = 1.f;
    RequireNear(Evaluate(manager, counts, params, 1.f), (70.0 * .25 + 20.0 * 4.0) / 90.0, "histogram-fractional-low-boundary");
    params.histogram_low_percentile  = 0.f;
    params.histogram_high_percentile = .85f;
    RequireNear(Evaluate(manager, counts, params, 1.f), (80.0 * .25 + 5.0 * 4.0) / 85.0, "histogram-fractional-high-boundary");
    params.histogram_low_percentile  = .85f;
    params.histogram_high_percentile = .95f;
    RequireNear(Evaluate(manager, counts, params, 1.f), 4.0, "histogram-single-partial-bin");
    RequireNear(Evaluate(manager, Histogram{}, params, 1.f), params.min_adapted_luminance, "histogram-empty");
}

void CheckAdaptation(ShaderManager& manager) {
    Histogram counts{};
    counts[128]                = static_cast<uint32>(TONEMAPPING_HISTOGRAM_POINT_FRAC_MULTIPLIER); // Target luminance 1.
    auto params                = Defaults();
    params.diff_log2_threshold = 0.f;
    for (const auto previous : {.75f, 2.f, .01f}) {
        const double speed    = previous < 1.f ? params.eye_adaptation_speed_up : params.eye_adaptation_speed_down;
        const double expected = std::exp2(std::log2(previous) * std::exp(-speed * params.frame_time));
        RequireNear(Evaluate(manager, counts, params, previous), expected, previous < 1.f ? "adaptation-up" : "adaptation-down");
    }
    params.frame_time      = 5.f;
    const float long_frame = Evaluate(manager, counts, params, .01f);
    RequireNear(long_frame, 1.0, "adaptation-long-frame-bounded");
    if (long_frame < .01f || long_frame > 1.f)
        throw std::runtime_error("Adaptation overshot its target");
    params.frame_time = 0.f;
    RequireNear(Evaluate(manager, counts, params, .5f), .5, "adaptation-zero-time");
    params.frame_time = -.2f;
    RequireNear(Evaluate(manager, counts, params, .5f), .5, "adaptation-negative-time");
    params.frame_time              = .1f;
    params.eye_adaptation_speed_up = 0.f;
    RequireNear(Evaluate(manager, counts, params, .5f), .5, "adaptation-zero-speed");
    params.diff_log2_threshold     = .1f;
    params.eye_adaptation_speed_up = 6.f;
    params.frame_time              = 0.f;
    RequireNear(Evaluate(manager, counts, params, .99f), .99, "adaptation-near-target-zero-time");
    params.frame_time              = .1f;
    params.eye_adaptation_speed_up = -1.f;
    RequireNear(Evaluate(manager, counts, params, .99f), .99, "adaptation-near-target-negative-speed");
    params.diff_log2_threshold     = 0.f;
    params.eye_adaptation_speed_up = 2.f;
    auto over_one_second           = [&](uint32 frames) {
        float previous    = .1f;
        params.frame_time = 1.f / frames;
        for (uint32 i = 0; i < frames; ++i)
            previous = Evaluate(manager, counts, params, previous);
        return previous;
    };
    const float  at_30    = over_one_second(30);
    const float  at_60    = over_one_second(60);
    const double expected = std::exp2(std::log2(.1) * std::exp(-2.0));
    RequireNear(at_30, expected, "adaptation-30fps");
    RequireNear(at_60, expected, "adaptation-60fps");
    RequireNear(at_30, at_60, "adaptation-frame-rate-invariance");
}
} // namespace

int main(int argc, const char** argv) {
    bool tasks = false, device = false, shaders = false;
    int  result = 0;
    try {
        LogSystem::Init();
        ConfigManager::GetInstance().Init(std::filesystem::current_path());
        TaskSystem::Init();
        tasks = true;
        RenderDevice::Init(DeviceInitInfo{.rhi_type = ERHIType::Vulkan, .name = "ExposureTest", .rhi_api_version = "1.3"});
        device        = true;
        auto& manager = ShaderManager::Get();
        shaders       = true;
        if (argc == 2 && std::string_view(argv[1]) == "--histogram")
            CheckHistogram(manager);
        else if (argc == 2 && std::string_view(argv[1]) == "--adaptation")
            CheckAdaptation(manager);
        else
            throw std::runtime_error("Expected --histogram or --adaptation");
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] " << error.what() << '\n';
        result = 1;
    }
    if (shaders)
        ShaderManager::ShutDown();
    if (device)
        RenderDevice::Dispose();
    if (tasks)
        TaskSystem::ShutDown();
    return result;
}
