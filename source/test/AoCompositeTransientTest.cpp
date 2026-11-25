#include "Engine.h"
#include "log/LogSystem.h"
#include "renderer/raster/AoPass.h"
#include "renderer/raster/RasterGraphResourceAccess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

using namespace Moer;
using namespace Moer::Render;
using namespace Moer::Render::Raster;

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Statistics {
    uint64 compared_values = 0, cpu_values = 0;
    float maximum_error = 0;
};

float4 Bilinear(const std::vector<float4>& pixels, Extent2D size, float2 uv) {
    const float x = uv.x * size.x - .5f, y = uv.y * size.y - .5f;
    const int left = int(std::floor(x)), top = int(std::floor(y));
    const float dx = x - left, dy = y - top;
    float4 value(0.f);
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
        const auto column = std::clamp(left + i, 0, int(size.x) - 1);
        const auto row = std::clamp(top + j, 0, int(size.y) - 1);
        value += pixels[row * size.x + column] * (i ? dx : 1.f - dx) * (j ? dy : 1.f - dy);
    }
    return value;
}

std::vector<float> RunCase(AoPass& pass, Extent2D size, EAoMode mode, bool half, bool preview,
    bool pooled, bool parallel, Statistics& statistics, bool discard = false) {
    auto& device = RenderDevice::Get();
    RasterConfig config;
    config.ao_mode = mode;
    config.ao_half_resolution = half;
    const Extent2D low(half ? std::max(1u, size.x / 2) : size.x, half ? std::max(1u, size.y / 2) : size.y);
    constexpr auto sampled = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_DST;
    constexpr auto output_usage = sampled | ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::TRANSFER_SRC;
    const auto color = device.CreateTexture("AO.color", size, PF_R32G32B32A32_SFLOAT, sampled);
    const auto ao = device.CreateTexture("AO.scalar", low, PF_R32G32B32A32_SFLOAT, sampled);
    const auto indirect = device.CreateTexture("AO.indirect", low, PF_R32G32B32A32_SFLOAT, sampled);
    const auto normal = device.CreateTexture("AO.normal", size, PF_R32G32B32A32_SFLOAT, sampled);
    const auto depth = device.CreateTexture("AO.depth", size, PF_R32_SFLOAT, sampled);
    const auto persistent_output = pooled ? TextureRef{} : device.CreateTexture("AO.output", size, PF_R32G32B32A32_SFLOAT, output_usage);
    InitializeGraphicsTextures(std::array{color, ao, indirect, normal, depth});
    if (persistent_output) InitializeGraphicsTexture(persistent_output);
    const bool only = mode == EAoMode::SSAO_AO_ONLY || mode == EAoMode::RTAO_AO_ONLY || mode == EAoMode::SSDO_AO_ONLY;
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    uint32 first_allocations = 0;
    std::vector<float> snapshot;
    for (uint32 frame = 0; frame < 2; ++frame) {
        std::vector<float4> occlusion(low.x * low.y), bounce(occlusion.size());
        for (uint32 y = 0; y < low.y; ++y) for (uint32 x = 0; x < low.x; ++x) {
            occlusion[y * low.x + x] = float4(float((x + 2 * y + frame) % 9) / 8.f);
            bounce[y * low.x + x] = float4(.125f + .03f * (x % 3), .25f + .04f * (y % 3), .5f, 0.f);
        }
        ReadbackFuture readback, overwritten, inspected;
        {
            RenderGraph graph("AoCompositeTransient", RenderGraph::QueueTopology::FromRHI());
            const auto input = ImportRasterTexture(graph, "Color", color.Get());
            const auto ambient = ImportRasterTexture(graph, "AO", ao.Get());
            const auto secondary = ImportRasterTexture(graph, "Indirect", indirect.Get());
            const auto normals = ImportRasterTexture(graph, "Normal", normal.Get());
            const auto depths = ImportRasterTexture(graph, "Depth", depth.Get());
            const RGTransientTextureDesc desc{.extent = Extent3D(size.x, size.y, 1), .format = PF_R32G32B32A32_SFLOAT, .usage = output_usage};
            const auto output = pooled ? graph.CreateTransientTexture("Composite", desc) : ImportRasterTexture(graph, "Composite", persistent_output.Get());
            const auto late = graph.CreateTransientTexture("AfterComposite", desc);
            const auto order = graph.ImportToken("SnapshotComplete", &pool);
            graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer);
                for (const auto handle : {input, ambient, secondary, normals, depths})
                    builder.Write(handle, RenderGraph::TextureAccessType::TransferDestination);
            }, [&](CommandList& commands) {
                commands.ClearResource(color->GetView(), float4(.25f, .5f, 2.5f, 1.f));
                commands.CopyFrom(std::as_bytes(std::span(occlusion)), ao->GetView());
                commands.CopyFrom(std::as_bytes(std::span(bounce)), indirect->GetView());
                commands.ClearResource(normal->GetView(), float4(.5f, .5f, 1.f, 1.f));
                commands.ClearResource(depth->GetView(), float4(.5f));
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            pass.AddCompositeToGraph(graph, config, size, low,
                {.ao_only = ambient, .lighting_output = input, .depth = depths, .normal = normals, .ao_output = output, .indirect = secondary});
            graph.AddRecordPass("Snapshot", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(output, RenderGraph::TextureAccessType::TransferSource).Write(order).SideEffect();
            }, [&, output](CommandList& commands) {
                readback = commands.Readback(graph.GetPhysicalTexture(output)->GetView());
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Overwrite", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(order).Write(late, RenderGraph::TextureAccessType::TransferDestination);
            }, [&, late](CommandList& commands) {
                commands.ClearResource(graph.GetPhysicalTexture(late)->GetView(), float4(1.f, 0.f, 1.f, 1.f));
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Inspect", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(late, RenderGraph::TextureAccessType::TransferSource).SideEffect();
                if (preview) builder.Read(output, RenderGraph::TextureAccessType::TransferSource);
            }, [&, output, late](CommandList& commands) {
                overwritten = commands.Readback(graph.GetPhysicalTexture(late)->GetView());
                if (preview) inspected = commands.Readback(graph.GetPhysicalTexture(output)->GetView());
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
            if (pooled) {
                const auto& plan = graph.GetCompiledPlan();
                const auto slot = [&](auto handle) { return plan.resources[handle.handle.resource_id].transient_slot; };
                Require((slot(output) == slot(late)) != preview, "AO composite preview or expired allocation lifetime is wrong");
            }
            CommandList commands;
            commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
            if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator)) throw std::runtime_error(graph.GetCompileError());
            Require(pool.AvailableTextureCount() == 0, "AO composite allocations reusable before completion");
            if (!discard) RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
        }
        if (discard) {
            Require(pool.AvailableTextureCount() == pool.TextureCount(), "Discarded AO composite retained allocations");
            break;
        }
        RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
        const auto values = readback.Get().CopyAs<float>();
        const auto late = overwritten.Get().CopyAs<float>();
        const auto preview_values = inspected.Valid() ? inspected.Get().CopyAs<float>() : decltype(values){};
        Require(values && late && values->size() == size.x * size.y * 4 && values->size() == late->size(), "Invalid AO composite readback");
        if (preview) Require(preview_values && *preview_values == *values, "AO composite preview was corrupted after overwrite");
        for (uint32 y = 0; y < size.y; ++y) for (uint32 x = 0; x < size.x; ++x) {
            const float2 uv((x + .5f) / size.x, (y + .5f) / size.y);
            const float visibility = Bilinear(occlusion, low, uv).x;
            const auto light = Bilinear(bounce, low, uv);
            for (uint32 channel = 0; channel < 4; ++channel) {
                const size_t index = (y * size.x + x) * 4 + channel;
                const std::array color_channels{.25f, .5f, 2.5f};
                const float expected = channel == 3 ? 1.f : only ? visibility :
                    color_channels[channel] * visibility + (mode == EAoMode::SSDO ? light[channel] : 0.f);
                const float error = std::abs((*values)[index] - expected);
                Require(std::isfinite((*values)[index]) && error < 5e-5f, "AO composite differs from independent CPU interpolation/compositing reference");
                Require((*late)[index] == (channel == 1 ? 0.f : 1.f), "AO composite expired allocation was not overwritten");
                statistics.maximum_error = std::max(statistics.maximum_error, error);
                ++statistics.cpu_values;
            }
        }
        snapshot.insert(snapshot.end(), values->begin(), values->end());
        Require(pool.AvailableTextureCount() == pool.TextureCount(), "Completed AO composite allocations were retained");
        if (frame == 0) first_allocations = pool.TextureCount();
        else Require(pool.TextureCount() == first_allocations, "AO composite did not reuse allocations on the second frame");
    }
    return snapshot;
}
} // namespace

void RunAoCompositeTransientContracts() {
    AoPass pass(ShaderManager::Get());
    Statistics statistics;
    uint32 runs = 0;
    for (const auto size : {Extent2D(32, 18), Extent2D(17, 9), Extent2D(1, 17), Extent2D(17, 1), Extent2D(1, 1)})
        for (uint32 mode = 0; mode <= uint32(EAoMode::LINEARIZED_DEPTH_DIV_10); ++mode)
            for (bool half : {false, true}) for (bool preview : {false, true}) {
                const auto baseline = RunCase(pass, size, EAoMode(mode), half, preview, false, false, statistics);
                ++runs;
                for (const auto allocation : {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
                    const auto result = RunCase(pass, size, EAoMode(mode), half, preview, allocation.first, allocation.second, statistics);
                    Require(result == baseline, "AO composite allocation or recording mode changed pixels");
                    statistics.compared_values += result.size();
                    ++runs;
                }
            }
    RunCase(pass, Extent2D(17, 9), EAoMode::SSDO, true, true, true, true, statistics, true);
    LOG_INFO("[TESTCASE][PASS] AoCompositeTransient: runs={} frames={} compared_values={} cpu_values={} maximum_error={}; discarded recording released",
        runs, runs * 2, statistics.compared_values, statistics.cpu_values, statistics.maximum_error);
}
