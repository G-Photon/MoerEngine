#include "log/LogSystem.h"
#include "renderer/raster/AoPass.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "renderer/raster/RtaoDenoiserPass.h"

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
    if (!condition)
        throw std::runtime_error(message);
}

float DecodeHalf(uint16 bits) {
    const int   exponent = (bits >> 10) & 31;
    const float value    = exponent ? std::ldexp(float(1024 + (bits & 1023)), exponent - 25) :
                                      std::ldexp(float(bits & 1023), -24);
    return bits & 0x8000 ? -value : value;
}

struct Statistics {
    uint64 cpu_values = 0, compared_values = 0;
    float  maximum_error = 0.f;
};

std::vector<uint16> RunCase(AoPass& ao, RtaoDenoiserPass& denoiser, Extent2D size, bool half, bool preview, uint32 scenario, bool pooled, bool parallel, Statistics& statistics, bool discard = false) {
    auto&            device = RenderDevice::Get();
    const Extent2D   low(half ? std::max(1u, size.x / 2) : size.x, half ? std::max(1u, size.y / 2) : size.y);
    const bool       denoise = scenario == 2;
    const bool       ssdo    = scenario == 1;
    constexpr auto   usage   = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT |
                               ETextureUsageFlags::UNORDERED_ACCESS | ETextureUsageFlags::TRANSFER_SRC | ETextureUsageFlags::TRANSFER_DST;
    const auto       normal  = device.CreateTexture("AoScratch.normal", size, PF_R32G32B32A32_SFLOAT, usage);
    const auto       depth   = device.CreateTexture("AoScratch.depth", size, PF_R32_SFLOAT, usage);
    const auto       color   = device.CreateTexture("AoScratch.color", size, PF_R32G32B32A32_SFLOAT, usage);
    const std::array histories{
        device.CreateTexture("AoScratch.final0", low, RasterTextures::AoFormat, usage),
        device.CreateTexture("AoScratch.final1", low, RasterTextures::AoFormat, usage)
    };
    const auto geometry         = device.CreateTexture("AoScratch.geometry", low, PF_R32G32B32A32_UINT, usage);
    const auto motion_data      = device.CreateBuffer<byte>("AoScratch.motion_data", sizeof(CameraMotionVectorData), EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    const auto scalar_texture   = pooled ? TextureRef{} : device.CreateTexture("AoScratch.noisy", low, RasterTextures::AoFormat, usage);
    const auto motion_texture   = pooled ? TextureRef{} : device.CreateTexture("AoScratch.motion", low, RasterTextures::CameraMotionFormat, usage);
    const auto indirect_texture = pooled ? TextureRef{} : device.CreateTexture("AoScratch.indirect", low, RasterTextures::AoIndirectFormat, usage);
    InitializeGraphicsTextures(std::array{normal, depth, color, histories[0], histories[1], geometry});
    if (!pooled)
        InitializeGraphicsTextures(std::array{scalar_texture, motion_texture, indirect_texture});
    const std::vector<uint4>      positions(low.x * low.y, uint4(0u, 0u, 0u, 0xffff8080u));
    RenderGraphResourcePool       pool(100);
    RenderGraphTransientAllocator allocator(pool);
    uint32                        first_allocations = 0;
    std::vector<uint16>           snapshot;
    for (uint32 frame = 0; frame < 2; ++frame) {
        LOG_INFO("AoScratch case size={}x{} half={} preview={} scenario={} pooled={} parallel={} frame={} discard={}", size.x, size.y, half, preview, scenario, pooled, parallel, frame, discard);
        ReadbackFuture result, noisy_read, motion_read, indirect_read, late_read, preview_read;
        {
            RenderGraph graph("AoScratchTransient", RenderGraph::QueueTopology::FromRHI());
            const auto  normals       = ImportRasterTexture(graph, "Normal", normal.Get());
            const auto  depths        = ImportRasterTexture(graph, "Depth", depth.Get());
            const auto  colors        = ImportRasterTexture(graph, "Color", color.Get());
            const auto  current       = ImportRasterTexture(graph, "Final", histories[frame].Get());
            const auto  previous      = ImportRasterTexture(graph, "Previous", histories[frame ^ 1u].Get());
            const auto  geometries    = ImportRasterTexture(graph, "Geometry", geometry.Get());
            const auto  motion_buffer = ImportRasterBuffer(graph, "MotionData", motion_data.Get());
            const auto  desc          = [&](EPixelFormat format) {
                return RGTransientTextureDesc{.extent = Extent3D(low.x, low.y, 1), .format = format, .usage = usage};
            };
            const auto scratch = [&](std::string_view name, EPixelFormat format, const TextureRef& texture) {
                return pooled ? graph.CreateTransientTexture(name, desc(format)) : ImportRasterTexture(graph, name, texture.Get());
            };
            const auto scalar    = scratch("Noisy", RasterTextures::AoFormat, scalar_texture);
            const auto motion    = scratch("Motion", RasterTextures::CameraMotionFormat, motion_texture);
            const auto indirect  = scratch("Indirect", RasterTextures::AoIndirectFormat, indirect_texture);
            const auto inspected = denoise ? scalar : motion;
            const auto late      = graph.CreateTransientTexture("AfterAO", desc(denoise ? RasterTextures::AoFormat : RasterTextures::CameraMotionFormat));
            const auto order     = graph.ImportToken("SnapshotDone", &pool);
            graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer);
                for (auto handle : {normals, depths, colors, geometries})
                    builder.Write(handle, RenderGraph::TextureAccessType::TransferDestination);
                if (denoise) {
                    builder.Write(scalar, RenderGraph::TextureAccessType::TransferDestination)
                        .Write(motion, RenderGraph::TextureAccessType::TransferDestination);
                }
            },
                                [&, scalar, motion](CommandList& commands) {
                                    commands.ClearResource(normal->GetView(), float4(.5f, .5f, 1.f, 1.f));
                                    commands.ClearResource(depth->GetView(), float4(.5f));
                                    commands.ClearResource(color->GetView(), float4(.25f, .5f, 1.f, 1.f));
                                    commands.CopyFrom(std::as_bytes(std::span(positions)), geometry->GetView());
                                    if (denoise) {
                                        commands.ClearResource(graph.GetPhysicalTexture(scalar)->GetView(), float4(frame ? .6f : .2f));
                                        commands.ClearResource(graph.GetPhysicalTexture(motion)->GetView(), float4(0.f));
                                    }
                                },
                                RenderGraph::PassRecordPolicy::ParallelAllowed);
            if (denoise) {
                RtaoDenoiserPass::RecordParameters parameters;
                parameters.denoise.inv_resolution           = float2(1.f / low.x, 1.f / low.y);
                parameters.denoise.history_ratio            = frame ? .5f : 0.f;
                parameters.denoise.is_reprojection_enable   = 1;
                parameters.denoise.is_validation_enable     = 1;
                parameters.denoise.valid_position_threshold = .01f;
                parameters.denoise.valid_normal_threshold   = .8f;
                denoiser.AddToGraph(graph, parameters, {.camera_motion_vector = motion, .ao_only = scalar, .history_read = previous, .history_write = current, .current_geometry = geometries, .history_geometry = geometries}, half);
            } else {
                AoPass::RecordParameters parameters;
                parameters.mode                                 = ssdo ? EAoMode::SSDO : EAoMode::NONE;
                parameters.frame.denoise                        = true;
                parameters.motion_vector_data.buf               = motion_data;
                parameters.motion_vector_upload.world2clip      = float4x4::Identity();
                parameters.motion_vector_upload.world2clip_prev = float4x4::Identity();
                parameters.ao.clip2world = parameters.ssdo.clip2world = float4x4::Identity();
                parameters.ssdo.world2clip                            = float4x4::Identity();
                parameters.ao.inv_resolution = parameters.ssdo.inv_resolution = float2(1.f / low.x, 1.f / low.y);
                parameters.ao.ao_mode                                         = uint32(parameters.mode);
                parameters.ssdo.ao_mode                                       = uint32(parameters.mode);
                parameters.ssdo.ssdo_sample_count                             = 0;
                ao.AddToGraph(graph, parameters, {.normal = normals, .depth = depths, .noise = colors, .lighting_output = colors, .ao_only = scalar, .camera_motion_vector = motion, .motion_vector_data = motion_buffer, .indirect = indirect});
            }
            graph.AddRecordPass("Snapshot", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(scalar, RenderGraph::TextureAccessType::TransferSource)
                    .Read(motion, RenderGraph::TextureAccessType::TransferSource)
                    .Write(order)
                    .SideEffect();
                if (ssdo)
                    builder.Read(indirect, RenderGraph::TextureAccessType::TransferSource);
                if (denoise)
                    builder.Read(current, RenderGraph::TextureAccessType::TransferSource);
            },
                                [&, scalar, motion, indirect, current](CommandList& commands) {
                                    noisy_read  = commands.Readback(graph.GetPhysicalTexture(scalar)->GetView());
                                    motion_read = commands.Readback(graph.GetPhysicalTexture(motion)->GetView());
                                    if (ssdo)
                                        indirect_read = commands.Readback(graph.GetPhysicalTexture(indirect)->GetView());
                                    if (denoise)
                                        result = commands.Readback(graph.GetPhysicalTexture(current)->GetView());
                                },
                                RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Overwrite", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(order)
                    .Write(late, RenderGraph::TextureAccessType::TransferDestination);
            },
                                [&, late](CommandList& commands) {
                                    commands.ClearResource(graph.GetPhysicalTexture(late)->GetView(), float4(1.f));
                                },
                                RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Inspect", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(late, RenderGraph::TextureAccessType::TransferSource)
                    .SideEffect();
                if (preview)
                    builder.Read(inspected, RenderGraph::TextureAccessType::TransferSource);
            },
                                [&, inspected, late](CommandList& commands) {
                                    late_read = commands.Readback(graph.GetPhysicalTexture(late)->GetView());
                                    if (preview)
                                        preview_read = commands.Readback(graph.GetPhysicalTexture(inspected)->GetView());
                                },
                                RenderGraph::PassRecordPolicy::ParallelAllowed);
            if (!graph.Compile())
                throw std::runtime_error(graph.GetCompileError());
            if (pooled) {
                const auto& plan = graph.GetCompiledPlan();
                const auto  slot = [&](auto handle) {
                    return plan.resources[handle.handle.resource_id].transient_slot;
                };
                Require((slot(inspected) == slot(late)) != preview, "AO scratch alias/preview lifetime is wrong");
                if (!ssdo)
                    Require(slot(indirect) == RenderGraph::InvalidIndex, "Unused AO indirect texture allocated a slot");
            }
            CommandList commands;
            commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
            if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator))
                throw std::runtime_error(graph.GetCompileError());
            Require(pool.AvailableTextureCount() == 0, "AO scratch reused before completion");
            if (!discard)
                RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
        }
        if (discard) {
            Require(pool.AvailableTextureCount() == pool.TextureCount(), "Discarded AO scratch stayed allocated");
            break;
        }
        RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
        const auto scalar = noisy_read.Get().CopyAs<uint8>();
        const auto motion = motion_read.Get().CopyAs<uint16>();
        Require(scalar && motion && scalar->size() == low.x * low.y && motion->size() == scalar->size() * 2, "Invalid AO scratch readback");
        if (preview) {
            const auto preview_result  = preview_read.Get();
            const auto original_result = denoise ? noisy_read.Get() : motion_read.Get();
            const auto bytes           = preview_result.Bytes();
            const auto before          = original_result.Bytes();
            Require(std::equal(bytes.begin(), bytes.end(), before.begin(), before.end()), "AO scratch preview corrupted by overwrite");
        }
        const auto check = [&](float actual, float expected, float tolerance) {
            const auto error = std::abs(actual - expected);
            if (!std::isfinite(actual) || error > tolerance)
                LOG_ERROR("AoScratch mismatch actual={} expected={} tolerance={} checked={}", actual, expected, tolerance, statistics.cpu_values);
            Require(std::isfinite(actual) && error <= tolerance, "AO scratch differs from independent CPU reference");
            statistics.maximum_error = std::max(statistics.maximum_error, error);
            ++statistics.cpu_values;
        };
        for (uint32 y = 0; y < low.y; ++y)
            for (uint32 x = 0; x < low.x; ++x) {
                const auto index = y * low.x + x;
                check((*scalar)[index] / 255.f, denoise ? (frame ? .6f : .2f) : 1.f, 1.f / 255.f);
                const float u = (x + .5f) / low.x, v = (y + .5f) / low.y;
                const auto  check_motion = [&](float actual, float uv, uint32 full, float sign) {
                    const auto reference = [&](uint32 pixel) {
                        return sign * 2.f * (uv - (pixel + .5f) / full);
                    };
                    const float location = uv * full;
                    float       expected = denoise ? 0.f : reference(std::min(uint32(location), full - 1));
                    // At an exact texel boundary, interpolation roundoff can select either neighbor.
                    const auto boundary = uint32(std::lround(location));
                    if (!denoise && boundary > 0 && boundary < full && std::abs(location - boundary) < 1e-5f) {
                        for (const auto pixel : {boundary - 1, boundary})
                            if (std::abs(actual - reference(pixel)) < std::abs(actual - expected))
                                expected = reference(pixel);
                    }
                    check(actual, expected, 5e-4f);
                };
                check_motion(DecodeHalf((*motion)[index * 2]), u, size.x, 1.f);
                check_motion(DecodeHalf((*motion)[index * 2 + 1]), v, size.y, -1.f);
            }
        snapshot.insert(snapshot.end(), scalar->begin(), scalar->end());
        snapshot.insert(snapshot.end(), motion->begin(), motion->end());
        if (denoise) {
            const auto output = result.Get().CopyAs<uint8>();
            const auto late   = late_read.Get().CopyAs<uint8>();
            Require(output && late && output->size() == scalar->size() && late->size() == scalar->size(), "Invalid AO denoise readback");
            for (auto value : *output)
                check(value / 255.f, frame ? .4f : .2f, 1.f / 255.f);
            for (auto value : *late)
                Require(value == 255, "AO scalar scratch was not overwritten");
            snapshot.insert(snapshot.end(), output->begin(), output->end());
        } else {
            const auto late = late_read.Get().CopyAs<uint16>();
            Require(late && late->size() == motion->size(), "Invalid AO motion overwrite readback");
            for (auto value : *late)
                Require(value == 0x3c00, "AO motion scratch was not overwritten");
        }
        if (ssdo) {
            const auto indirect = indirect_read.Get().CopyAs<uint16>();
            Require(indirect && indirect->size() == scalar->size() * 4, "Invalid SSDO indirect readback");
            for (size_t index = 0; index < indirect->size(); ++index)
                if (index % 4 != 3) {
                    check(DecodeHalf((*indirect)[index]), 0.f, 1e-6f);
                    snapshot.push_back((*indirect)[index]);
                }
        }
        Require(pool.AvailableTextureCount() == pool.TextureCount(), "Completed AO scratch stayed allocated");
        if (frame == 0)
            first_allocations = pool.TextureCount();
        else
            Require(pool.TextureCount() == first_allocations, "AO scratch did not reuse completed allocations");
    }
    return snapshot;
}
} // namespace

void RunAoScratchTransientContracts() {
    AoPass           ao(ShaderManager::Get());
    RtaoDenoiserPass denoiser(ShaderManager::Get());
    Statistics       statistics;
    uint32           runs = 0;
    for (auto size : {Extent2D(32, 18), Extent2D(17, 9), Extent2D(1, 17), Extent2D(17, 1), Extent2D(1, 1)})
        for (bool half : {false, true})
            for (bool preview : {false, true})
                for (uint32 scenario = 0; scenario < 3; ++scenario) {
                    const auto baseline = RunCase(ao, denoiser, size, half, preview, scenario, false, false, statistics);
                    ++runs;
                    for (auto mode : {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
                        const auto result = RunCase(ao, denoiser, size, half, preview, scenario, mode.first, mode.second, statistics);
                        Require(result == baseline, "AO scratch allocation/recording policy changed output");
                        statistics.compared_values += result.size();
                        ++runs;
                    }
                }
    for (uint32 scenario = 0; scenario < 3; ++scenario)
        RunCase(ao, denoiser, Extent2D(17, 9), true, true, scenario, true, true, statistics, true);
    LOG_INFO("[TESTCASE][PASS] AoScratchTransient: runs={} frames={} compared_values={} cpu_values={} maximum_error={}; discarded recordings released", runs, runs * 2, statistics.compared_values, statistics.cpu_values, statistics.maximum_error);
}
