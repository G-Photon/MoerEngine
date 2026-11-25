#include "log/LogSystem.h"
#include "renderer/raster/HiZBuildPass.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "renderer/raster/RasterTool.h"

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

class DepthPatternPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(DepthPatternPipeline);
    DEFINE_SHADER_BUFFER(depth_values);
    DEFINE_SHADER_ARGS(depth_values);
};

class DepthInspectPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(DepthInspectPipeline);
    DEFINE_SHADER_TEX(depth_texture);
    DEFINE_SHADER_BUFFER(depth_values);
    DEFINE_SHADER_ARGS(depth_texture, depth_values);
};

float DecodeHalf(uint16 value) {
    const int exponent = (value >> 10) & 31;
    const float magnitude = exponent ? std::ldexp(float(1024 + (value & 1023)), exponent - 25) :
                                       std::ldexp(float(value & 1023), -24);
    return value & 0x8000 ? -magnitude : magnitude;
}

struct Statistics {
    uint64 compared_values = 0, cpu_values = 0;
};

std::vector<float> RunCase(DepthPatternPipeline& pattern, DepthInspectPipeline& inspect, HiZBuildPass& hiz,
    Extent2D size, bool preview, bool clear_only, bool pooled, bool parallel,
    Statistics& statistics, bool discard = false) {
    auto& device = RenderDevice::Get();
    const RGTransientTextureDesc descriptor{
        .extent = Extent3D(size.x, size.y, 1), .format = RasterTextures::DepthFormat,
        .usage = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::DEPTH_STENCIL_ATTACHMENT |
                 ETextureUsageFlags::TRANSFER_SRC | ETextureUsageFlags::TRANSFER_DST,
        .aspect_flags = ETextureAspectFlags::DEPTH_SLICE
    };
    const uint32 mip_count = RasterTextures::GetHiZMipCount(uint2(size.x, size.y));
    auto pyramid = device.CreateTexture("DepthTest.HiZ", size, PF_R16_SFLOAT,
        ETextureUsageFlags::SAMPLED | ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::TRANSFER_SRC, mip_count);
    InitializeGraphicsTexture(pyramid);
    const auto values_buffer = device.CreateBuffer<float>("DepthTest.samples", size.x * size.y + 1,
        EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    std::vector<float> snapshot;
    uint32 first_allocations = 0;
    for (uint32 frame = 0; frame < 2; ++frame) {
        std::vector<float> values(size.x * size.y + 1), expected(size.x * size.y);
        values[0] = float(size.x);
        for (uint32 y = 0; y < size.y; ++y) for (uint32 x = 0; x < size.x; ++x) {
            const uint32 index = y * size.x + x;
            values[index + 1] = clear_only ? 0.f : float((x + 3 * y + frame) % 7) / 8.f;
            expected[index] = clear_only ? .375f : values[index + 1];
        }
        ReadbackFuture depth_read, preview_read, late_read;
        std::vector<ReadbackFuture> mip_reads(mip_count);
        std::array<BufferRef, 3> copies;
        for (auto& copy : copies) copy = device.CreateBuffer<float>("DepthTest.copy", size.x * size.y,
            EBufferUsageFlags::TRANSFER_SRC | EBufferUsageFlags::TRANSFER_DST | EBufferUsageFlags::UNORDERED_ACCESS);
        {
            RenderGraph graph("DepthTransientContract", RenderGraph::QueueTopology::FromRHI());
            std::array<RenderGraph::BufferHandle, 3> copy_handles;
            for (uint32 index = 0; index < copies.size(); ++index)
                copy_handles[index] = ImportRasterBuffer(graph, "DepthTest.copy", copies[index].Get());
            TextureRef owner;
            RenderGraph::TextureHandle depth;
            if (pooled) depth = graph.CreateTransientTexture("depth", descriptor);
            else {
                owner = device.CreateDepthBuffer("DepthTest.external", size, RasterTextures::DepthFormat, 1, descriptor.usage)->CastToTextureRef();
                InitializeGraphicsTexture(owner);
                depth = ImportRasterTexture(graph, "depth", owner.Get());
            }
            const auto later = graph.CreateTransientTexture("depth.later", descriptor);
            const auto data = ImportRasterBuffer(graph, "samples", values_buffer.Get());
            const auto output = ImportRasterTexture(graph, "hiz", pyramid.Get());
            const auto captured = graph.ImportToken("snapshot", &pool);
            graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Write(data, RenderGraph::BufferAccessType::TransferDestination);
            }, [&](CommandList& commands) {
                commands.CopyFrom(std::as_bytes(std::span(values)), values_buffer->GetView());
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("DepthAttachment", [=](RenderGraph::PassBuilder& builder) {
                builder.Read(data, RenderGraph::BufferAccessType::ShaderRead)
                    .Write(depth, RenderGraph::TextureAccessType::DepthStencilAttachment);
            }, [&, depth](CommandList& commands) {
                const auto physical = graph.GetPhysicalTexture(depth);
                Require(physical && physical->GetAspectFlags() == ETextureAspectFlags::DEPTH_SLICE,
                    "Depth sample view exposed another aspect");
                DepthAttachment attachment(physical->GetView());
                attachment.clear_depth = clear_only ? .375f : 0.f;
                commands.Gfx(pattern, values_buffer).Draw("Depth pattern", Rect2D(0, 0, size.x, size.y),
                    RasterTool::GetFullScreenDrawDatas(), attachment);
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            hiz.AddToGraph(graph, hiz.Prepare(pyramid), {.depth = depth, .hiz_current = output});
            graph.AddRecordPass("Snapshot", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
                    .Read(depth, RenderGraph::TextureAccessType::Sampled)
                    .Read(output, RenderGraph::TextureAccessType::TransferSource)
                    .Write(copy_handles[0], RenderGraph::BufferAccessType::Storage).Write(captured).SideEffect();
            }, [&, depth](CommandList& commands) {
                commands.Compute(inspect, graph.GetPhysicalTexture(depth)->GetView(), copies[0])
                    .Dispatch(uint3((size.x + 7) / 8, (size.y + 7) / 8, 1), "Inspect depth");
                for (uint32 mip = 0; mip < mip_count; ++mip)
                    mip_reads[mip] = commands.Readback(pyramid->GetView(uint8(mip), 1));
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("ReadDepthCopy", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(copy_handles[0], RenderGraph::BufferAccessType::TransferSource).SideEffect();
            }, [&](CommandList& commands) { depth_read = commands.Readback(copies[0]->GetView()); },
                RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Overwrite", [=](RenderGraph::PassBuilder& builder) {
                builder.Read(captured).Read(data, RenderGraph::BufferAccessType::ShaderRead)
                    .Write(later, RenderGraph::TextureAccessType::DepthStencilAttachment);
            }, [&, later](CommandList& commands) {
                const auto physical = graph.GetPhysicalTexture(later);
                if (pooled) Require((physical == graph.GetPhysicalTexture(depth)) != preview,
                    "Physical depth alias does not match compiled preview lifetime");
                DepthAttachment attachment(physical->GetView());
                attachment.clear_depth = .875f;
                commands.Gfx(pattern, values_buffer).Draw("Depth later overwrite", Rect2D(0, 0, size.x, size.y),
                    RasterTool::GetFullScreenDrawDatas(), attachment);
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Inspect", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
                    .Read(later, RenderGraph::TextureAccessType::Sampled)
                    .Write(copy_handles[1], RenderGraph::BufferAccessType::Storage).SideEffect();
                if (preview) builder.Read(depth, RenderGraph::TextureAccessType::Sampled)
                    .Write(copy_handles[2], RenderGraph::BufferAccessType::Storage);
            }, [&, depth, later](CommandList& commands) {
                commands.Compute(inspect, graph.GetPhysicalTexture(later)->GetView(), copies[1])
                    .Dispatch(uint3((size.x + 7) / 8, (size.y + 7) / 8, 1), "Inspect later depth");
                if (preview) commands.Compute(inspect, graph.GetPhysicalTexture(depth)->GetView(), copies[2])
                    .Dispatch(uint3((size.x + 7) / 8, (size.y + 7) / 8, 1), "Inspect preview depth");
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("ReadInspectedCopies", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(copy_handles[1], RenderGraph::BufferAccessType::TransferSource).SideEffect();
                if (preview) builder.Read(copy_handles[2], RenderGraph::BufferAccessType::TransferSource);
            }, [&](CommandList& commands) {
                late_read = commands.Readback(copies[1]->GetView());
                if (preview) preview_read = commands.Readback(copies[2]->GetView());
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
            if (pooled) {
                const auto& plan = graph.GetCompiledPlan();
                const auto& original = plan.resources[depth.handle.resource_id];
                const auto& late = plan.resources[later.handle.resource_id];
                Require((original.transient_slot == late.transient_slot) != preview, "Depth alias/preview lifetime incorrect");
            }
            CommandList commands;
            commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
            if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator))
                throw std::runtime_error(graph.GetCompileError());
            Require(pool.AvailableTextureCount() == 0, "Depth reusable before GPU completion");
            if (!discard) RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
        }
        if (discard) {
            Require(pool.AvailableTextureCount() == pool.TextureCount(), "Discard retained depth allocations");
            break;
        }
        RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
        const auto result = depth_read.Get();
        const auto actual = result.CopyAs<float>();
        if (!actual || actual->size() != expected.size() || !std::equal(actual->begin(), actual->end(), expected.begin())) {
            LOG_ERROR("Depth readback size={} expected={} status={} reason={}",
                result.ByteSize(), expected.size() * sizeof(float), uint32(result.status), result.error_reason);
            if (actual) for (uint32 index = 0; index < std::min<size_t>(actual->size(), 8); ++index)
                LOG_ERROR("Depth sample index={} actual={} expected={}", index, (*actual)[index], expected[index]);
        }
        Require(actual && actual->size() == expected.size() && std::equal(actual->begin(), actual->end(), expected.begin()),
            "Depth attachment differs from independent pattern/clear reference");
        statistics.cpu_values += actual->size();
        snapshot.insert(snapshot.end(), actual->begin(), actual->end());
        if (preview) {
            const auto inspected = preview_read.Get().CopyAs<float>();
            Require(inspected && inspected->size() == expected.size() && std::equal(inspected->begin(), inspected->end(), expected.begin()),
                "Depth preview corrupted by later alias writes");
        }
        const auto overwritten = late_read.Get().CopyAs<float>();
        Require(overwritten && overwritten->size() == expected.size() &&
            std::all_of(overwritten->begin(), overwritten->end(), [](float value) { return value == .875f; }),
            "Depth later attachment overwrite failed");
        uint32 width = size.x, height = size.y;
        for (uint32 mip = 0; mip < mip_count; ++mip) {
            const auto pixels = mip_reads[mip].Get().CopyAs<uint16>();
            Require(pixels && pixels->size() == expected.size(), "HiZ readback size mismatch");
            for (uint32 index = 0; index < pixels->size(); ++index) {
                const float value = DecodeHalf((*pixels)[index]);
                Require(value == expected[index], "HiZ differs from independent reverse-Z minimum reference");
                snapshot.push_back(value);
                ++statistics.cpu_values;
            }
            if (mip + 1 == mip_count) break;
            const uint32 next_width = std::max(1u, width / 2), next_height = std::max(1u, height / 2);
            std::vector<float> next(next_width * next_height);
            for (uint32 y = 0; y < next_height; ++y) for (uint32 x = 0; x < next_width; ++x) {
                float minimum = 1.f;
                for (uint32 dy = 0; dy < 2; ++dy) for (uint32 dx = 0; dx < 2; ++dx)
                    minimum = std::min(minimum, expected[std::min(2 * y + dy, height - 1) * width + std::min(2 * x + dx, width - 1)]);
                next[y * next_width + x] = minimum;
            }
            width = next_width; height = next_height; expected = std::move(next);
        }
        Require(pool.AvailableTextureCount() == pool.TextureCount(), "Completed depth allocations retained");
        if (!frame) first_allocations = pool.TextureCount();
        else Require(pool.TextureCount() == first_allocations, "Next frame did not reuse depth allocation");
    }
    LOG_INFO("[TESTCASE][PASS] name=DepthTransient size={}x{} preview={} clear={} pooled={} parallel={} discard={}",
        size.x, size.y, preview, clear_only, pooled, parallel, discard);
    return snapshot;
}
} // namespace

void RunDepthTransientContracts() {
    auto& manager = ShaderManager::Get();
    auto pipeline = manager.Raster().Vertex("core/utils/FullScreenQuad.hlsl").Pixel("test/DepthTransient.frag.hlsl")
        .Build<DepthPatternPipeline>(GfxPsoCreateInfo(RHIRasterizeInfo::Preset(), {}, {},
            RHIDepthStencilStateInfo::Preset<DepthStencil::DEPTH_WRITE_GREATER>(), RasterTextures::DepthFormat));
    HiZBuildPass hiz(manager);
    auto inspect = manager.Compute<DepthInspectPipeline>("test/DepthTransient.comp.hlsl");
    Statistics statistics;
    uint32 runs = 0;
    for (auto size : {Extent2D(32, 18), Extent2D(17, 9), Extent2D(1, 17), Extent2D(17, 1), Extent2D(1, 1)})
        for (bool preview : {false, true}) for (bool clear : {false, true}) {
            const auto baseline = RunCase(pipeline, inspect, hiz, size, preview, clear, false, false, statistics);
            ++runs;
            for (auto mode : {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
                const auto actual = RunCase(pipeline, inspect, hiz, size, preview, clear, mode.first, mode.second, statistics);
                Require(actual == baseline, "Depth allocation/recording mode changed pixels");
                statistics.compared_values += actual.size();
                ++runs;
            }
        }
    for (bool clear : {false, true})
        RunCase(pipeline, inspect, hiz, Extent2D(17, 9), true, clear, true, true, statistics, true);
    LOG_INFO("[TESTCASE][PASS] DepthTransient: runs={} frames={} compared_values={} cpu_values={}; exact depth/HiZ, discarded recordings released",
        runs, runs * 2, statistics.compared_values, statistics.cpu_values);
}
