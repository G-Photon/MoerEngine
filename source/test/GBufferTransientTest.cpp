#include "log/LogSystem.h"
#include "renderer/raster/DirectionalShadowMaskPass.h"
#include "renderer/raster/LightingPass.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "shaderheaders/shared/scene/SharedSceneStruct.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
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

float DecodeHalf(uint16 bits) {
    const int exponent = (bits >> 10) & 31;
    const float value = exponent ? std::ldexp(float(1024 + (bits & 1023)), exponent - 25) :
                                   std::ldexp(float(bits & 1023), -24);
    return bits & 0x8000 ? -value : value;
}

struct Statistics {
    uint64 compared_bytes = 0, cpu_values = 0;
    float maximum_error = 0;
};

std::vector<byte> RunCase(LightingPass& lighting, DirectionalShadowMaskPass& shadows,
    Extent2D size, bool preview, uint32 scenario, bool pooled, bool parallel,
    Statistics& statistics, bool discard = false) {
    auto& device = RenderDevice::Get();
    constexpr auto usage = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT |
        ETextureUsageFlags::TRANSFER_SRC | ETextureUsageFlags::TRANSFER_DST;
    const auto light_data = device.CreateBuffer<byte>("GBuffer.lighting_data", sizeof(LightingData),
        EBufferUsageFlags::CONSTANT_BUFFER | EBufferUsageFlags::TRANSFER_DST);
    const auto lights = device.CreateBuffer<byte>("GBuffer.lights", sizeof(GLight),
        EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    const auto lut = device.CreateTexture("GBuffer.lut", Extent2D(1, 1), PF_R32G32B32A32_SFLOAT, usage);
    const auto depth = device.CreateTexture("GBuffer.depth", size, PF_R32_SFLOAT, usage);
    const auto shadow_map = device.CreateTexture("GBuffer.shadow_map", size, PF_R32_SFLOAT, usage);
    InitializeGraphicsTextures(std::array{lut, depth, shadow_map});
    const auto table = device.CreateBindlessArray(8);
    LightingData data{};
    data.clip2world = data.world2view = float4x4::Identity();
    data.camera_position = float3(0.f, 0.f, 4.f);
    data.light_count = scenario ? 1 : 0;
    data.brdf_NDF_mode = uint32(EBrdfNdfMode::GGX);
    data.brdf_G_mode = uint32(EBrdfGMode::G_SCHLICK);
    data.lut_ggx_emu_handle = data.lut_ggx_eavg_handle = table->AllocateTexture(lut->GetView(), {SF_NEAREST, SAM_CLAMP_TO_EDGE});
    data.shadow_map_mode = uint32(scenario == 2 ? EShadowMapMode::CSM : EShadowMapMode::NONE);
    data.shadow_csm_num_of_cascades = 1;
    data.cascade_split_ratios = float4(1.f);
    data.far_clip = 1.f;
    data.world2shadow_clip[0] = float4x4::Identity();
    data.cascade_shadow_map.x = table->AllocateTexture(shadow_map->GetView(), {SF_NEAREST, SAM_CLAMP_TO_EDGE});
    GLight light{};
    light.type = uint32(ELightType::Directional);
    light.direction = float3(0.f, 0.f, -1.f);
    light.color = float3(.8f, .5f, .3f);
    light.intensity = 2.f;
    MaterialPassBindlessParam parameters{};
    parameters.light_buf_hdl = table->AllocateBuffer(lights->GetView());
    parameters.enable_extra_ambient = 1;
    parameters.extra_ambient_color = float3(.3f, .5f, .7f);
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    uint32 first_allocations = 0;
    std::vector<byte> snapshot;
    for (uint32 frame = 0; frame < 2; ++frame) {
        std::vector<uint32> base(size.x * size.y), material(base.size()), normal(base.size(), (3u << 30) | (1023u << 20) | (512u << 10) | 512u);
        std::vector<float> depths(base.size()), shadow_depths(base.size());
        for (uint32 y = 0; y < size.y; ++y) for (uint32 x = 0; x < size.x; ++x) {
            const auto index = y * size.x + x;
            base[index] = 0xff000000u | (40u + 23u * ((x + frame) % 8)) |
                ((30u + 21u * (y % 9)) << 8) | ((50u + 19u * ((x + y) % 8)) << 16);
            material[index] = 0xffff0000u | (20u + 25u * ((x + frame) % 8)) | ((90u + 13u * (y % 9)) << 8);
            depths[index] = (x + y + frame) % 5 ? .5f : 0.f;
            shadow_depths[index] = x < size.x / 2 ? .8f : .2f;
        }
        ReadbackFuture color_read, shadow_read;
        std::array<ReadbackFuture, 5> originals, inspected, late_reads;
        {
            RenderGraph graph("GBufferTransientContract", RenderGraph::QueueTopology::FromRHI());
            const std::array formats{RasterTextures::BaseColorFormat, RasterTextures::NormalFormat,
                RasterTextures::MaterialFormat, RasterTextures::ShadowMaskFormat, RasterTextures::LightingOutputFormat};
            std::vector<TextureRef> owners;
            const auto image = [&](std::string_view name, EPixelFormat format, bool transient) {
                if (transient) return graph.CreateTransientTexture(name, RGTransientTextureDesc{
                    .extent = Extent3D(size.x, size.y, 1), .format = format, .usage = usage});
                const auto texture = device.CreateTexture(name, size, format, usage);
                InitializeGraphicsTexture(texture);
                owners.push_back(texture);
                return ImportRasterTexture(graph, name, texture.Get());
            };
            std::array<RenderGraph::TextureHandle, 5> targets, late;
            for (uint32 index = 0; index < targets.size(); ++index) {
                targets[index] = image(std::format("GBuffer.target{}", index), formats[index], pooled);
                late[index] = image(std::format("GBuffer.late{}", index), formats[index], true);
            }
            const auto depth_handle = ImportRasterTexture(graph, "Depth", depth.Get());
            const auto map_handle = ImportRasterTexture(graph, "ShadowMap", shadow_map.Get());
            const auto lut_handle = ImportRasterTexture(graph, "LUT", lut.Get());
            const auto data_handle = ImportRasterBuffer(graph, "LightingData", light_data.Get());
            const auto lights_handle = ImportRasterBuffer(graph, "Lights", lights.Get());
            const auto bindings = graph.ImportToken("Bindings", table.Get());
            const auto captured = graph.ImportToken("Captured", &pool);
            graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer).Write(bindings);
                for (auto handle : {targets[0], targets[1], targets[2], depth_handle, map_handle, lut_handle})
                    builder.Write(handle, RenderGraph::TextureAccessType::TransferDestination);
                builder.Write(data_handle, RenderGraph::BufferAccessType::TransferDestination)
                    .Write(lights_handle, RenderGraph::BufferAccessType::TransferDestination);
            }, [&, targets](CommandList& commands) {
                commands.CopyFrom(std::as_bytes(std::span(base)), graph.GetPhysicalTexture(targets[0])->GetView());
                commands.CopyFrom(std::as_bytes(std::span(normal)), graph.GetPhysicalTexture(targets[1])->GetView());
                commands.CopyFrom(std::as_bytes(std::span(material)), graph.GetPhysicalTexture(targets[2])->GetView());
                commands.CopyFrom(std::as_bytes(std::span(depths)), depth->GetView());
                commands.CopyFrom(std::as_bytes(std::span(shadow_depths)), shadow_map->GetView());
                commands.ClearResource(lut->GetView(), float4(.5f));
                commands.CopyFrom(std::as_bytes(std::span(&data, 1)), light_data->GetView());
                commands.CopyFrom(std::as_bytes(std::span(&light, 1)), lights->GetView());
                commands.UpdateBindlessArray(table);
            }, RenderGraph::PassRecordPolicy::Serial);
            graph.AddRecordPass("ShadowMask", [=](RenderGraph::PassBuilder& builder) {
                builder.Read(bindings).Read(targets[1], RenderGraph::TextureAccessType::Sampled)
                    .Read(depth_handle, RenderGraph::TextureAccessType::Sampled)
                    .Read(map_handle, RenderGraph::TextureAccessType::Sampled)
                    .Read(data_handle, RenderGraph::BufferAccessType::ShaderRead)
                    .Write(targets[3], RenderGraph::TextureAccessType::ColorAttachment);
            }, [&, targets](CommandList& commands) {
                DirectionalShadowMaskPass::RecordParameters resolved;
                resolved.normal_owner = graph.GetPhysicalTexture(targets[1]);
                resolved.depth_owner = depth;
                resolved.lighting_data = light_data;
                resolved.bindless = table;
                resolved.output = graph.GetPhysicalTexture(targets[3]);
                resolved.render_area = Rect2D(0, 0, size.x, size.y);
                shadows.Record(commands, resolved);
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Lighting", [=](RenderGraph::PassBuilder& builder) {
                builder.Read(bindings).Read(depth_handle, RenderGraph::TextureAccessType::Sampled)
                    .Read(lut_handle, RenderGraph::TextureAccessType::Sampled)
                    .Read(data_handle, RenderGraph::BufferAccessType::ShaderRead)
                    .Read(lights_handle, RenderGraph::BufferAccessType::ShaderRead);
                for (uint32 index = 0; index < 4; ++index) builder.Read(targets[index], RenderGraph::TextureAccessType::Sampled);
                builder.Write(targets[4], RenderGraph::TextureAccessType::ColorAttachment);
            }, [&, targets](CommandList& commands) {
                LightingPass::RecordParameters resolved;
                resolved.pass_param = parameters;
                resolved.extra_ambient_intensity = .6f;
                resolved.lighting_data = light_data;
                resolved.light_buffer_owner = lights;
                resolved.bindless = table;
                resolved.base_color_owner = graph.GetPhysicalTexture(targets[0]);
                resolved.normal_owner = graph.GetPhysicalTexture(targets[1]);
                resolved.metal_rough_ao_owner = graph.GetPhysicalTexture(targets[2]);
                resolved.depth_owner = depth;
                resolved.shadow_mask_owner = graph.GetPhysicalTexture(targets[3]);
                resolved.output = graph.GetPhysicalTexture(targets[4]);
                resolved.render_area = Rect2D(0, 0, size.x, size.y);
                lighting.Record(commands, resolved);
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Snapshot", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer).Write(captured).SideEffect();
                for (auto target : targets) builder.Read(target, RenderGraph::TextureAccessType::TransferSource);
            }, [&, targets](CommandList& commands) {
                for (uint32 index = 0; index < targets.size(); ++index)
                    originals[index] = commands.Readback(graph.GetPhysicalTexture(targets[index])->GetView());
                shadow_read = originals[3]; color_read = originals[4];
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Overwrite", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer).Read(captured);
                for (auto target : late) builder.Write(target, RenderGraph::TextureAccessType::TransferDestination);
            }, [&, late](CommandList& commands) {
                for (auto target : late) commands.ClearResource(graph.GetPhysicalTexture(target)->GetView(), float4(1.f));
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("Inspect", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer).SideEffect();
                for (uint32 index = 0; index < targets.size(); ++index) {
                    builder.Read(late[index], RenderGraph::TextureAccessType::TransferSource);
                    if (preview) builder.Read(targets[index], RenderGraph::TextureAccessType::TransferSource);
                }
            }, [&, targets, late](CommandList& commands) {
                for (uint32 index = 0; index < targets.size(); ++index) {
                    late_reads[index] = commands.Readback(graph.GetPhysicalTexture(late[index])->GetView());
                    if (preview) inspected[index] = commands.Readback(graph.GetPhysicalTexture(targets[index])->GetView());
                }
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
            if (pooled) {
                const auto& plan = graph.GetCompiledPlan();
                const auto slot = [&](auto handle) { return plan.resources[handle.handle.resource_id].transient_slot; };
                for (uint32 index = 0; index < targets.size(); ++index) {
                    bool reused = false;
                    for (auto other : late) reused |= slot(targets[index]) == slot(other);
                    Require(reused != preview, "GBuffer alias/preview lifetime is wrong");
                }
            }
            CommandList commands;
            commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
            if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator)) throw std::runtime_error(graph.GetCompileError());
            Require(pool.AvailableTextureCount() == 0, "GBuffer allocation reusable before completion");
            if (!discard) RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
        }
        if (discard) {
            Require(pool.AvailableTextureCount() == pool.TextureCount(), "Discarded GBuffer recording retained allocations");
            break;
        }
        RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
        const auto colors = color_read.Get().CopyAs<uint16>();
        const auto masks = shadow_read.Get().CopyAs<uint8>();
        Require(colors && masks && colors->size() == base.size() * 4 && masks->size() == base.size(), "Invalid GBuffer readback");
        for (uint32 index = 0; index < originals.size(); ++index) {
            const auto original = originals[index].Get();
            if (index < 3) {
                const auto values = original.CopyAs<uint32>();
                const auto& expected = index == 0 ? base : index == 1 ? normal : material;
                Require(values && std::equal(values->begin(), values->end(), expected.begin(), expected.end()), "GBuffer input packing or attachment identity changed");
            }
            snapshot.insert(snapshot.end(), original.Bytes().begin(), original.Bytes().end());
            if (preview) {
                const auto after = inspected[index].Get();
                Require(std::equal(original.Bytes().begin(), original.Bytes().end(), after.Bytes().begin(), after.Bytes().end()), "GBuffer preview corrupted by subsequent alias writes");
            }
            const auto late = late_reads[index].Get();
            const auto bytes = late.CopyAs<uint8>();
            if (index < 4) for (auto value : *bytes) Require(value == 255, "Late GBuffer overwrite failed");
            else {
                const auto half_values = late.CopyAs<uint16>();
                Require(half_values.has_value(), "Invalid late lighting readback");
                for (auto value : *half_values) Require(DecodeHalf(value) == 1.f, "Late lighting overwrite failed");
            }
        }
        for (uint32 y = 0; y < size.y; ++y) for (uint32 x = 0; x < size.x; ++x) {
            const auto index = y * size.x + x;
            const bool background = depths[index] == 0;
            const double shadow = scenario == 2 && !background && x < size.x / 2 ? 0. : 1.;
            Require((*masks)[index] == (shadow ? 255 : 0), "Shadow mask differs from independent reverse-Z/CSM reference");
            ++statistics.cpu_values;
            const auto n = Normalizef(float3(1.f / 1023.f, 1.f / 1023.f, 1.f));
            const auto v = Normalizef(float3(-(2.f * (x + .5f) / size.x - 1.f), -(1.f - 2.f * (y + .5f) / size.y), 4.f - depths[index]));
            const auto h = Normalizef(v + float3(0.f, 0.f, 1.f));
            const double nov = std::max(0., Dot(n, v)), nol = n.z, noh = std::max(0., Dot(n, h)), voh = std::max(0., Dot(v, h));
            const double roughness = ((material[index] >> 8) & 255) / 255.;
            const double metallic = (material[index] & 255) / 255.;
            const double alpha2 = std::pow(roughness, 4), denominator = noh * noh * (alpha2 - 1.) + 1.;
            const double distribution = alpha2 / (3.141592653589793 * denominator * denominator);
            const double k = (roughness + 1.) * (roughness + 1.) / 8.;
            const double visibility = (nov / (nov * (1. - k) + k)) * (nol / (nol * (1. - k) + k)) / (4. * nov * nol);
            for (uint32 channel = 0; channel < 4; ++channel) {
                const double albedo = channel < 3 ? ((base[index] >> (channel * 8)) & 255) / 255. : 1.;
                const double f0 = albedo * (1. - metallic) + .04 * metallic;
                const double fresnel = f0 + (1. - f0) * std::pow(1. - voh, 5);
                double expected = channel == 3 ? 1. : .6 * parameters.extra_ambient_color[channel] * albedo;
                if (scenario && channel < 3) expected += distribution * fresnel * visibility * light.intensity * light.color[channel] * nol * shadow;
                if (background && channel < 3) expected = channel == 0 ? 1. : 0.;
                const float actual = DecodeHalf((*colors)[index * 4 + channel]);
                const float error = std::abs(actual - float(expected));
                Require(std::isfinite(actual) && error <= .0005f + .002f * std::abs(float(expected)), "Lighting differs from independent GGX/ambient/background reference");
                statistics.maximum_error = std::max(statistics.maximum_error, error);
                ++statistics.cpu_values;
            }
        }
        Require(pool.AvailableTextureCount() == pool.TextureCount(), "Completed GBuffer allocations retained");
        if (frame == 0) first_allocations = pool.TextureCount();
        else Require(pool.TextureCount() == first_allocations, "Next frame did not reuse GBuffer allocations");
    }
    LOG_INFO("[TESTCASE][PASS] name=GBufferTransient size={}x{} preview={} scenario={} pooled={} parallel={} discard={}", size.x, size.y, preview, scenario, pooled, parallel, discard);
    return snapshot;
}
} // namespace

void RunGBufferTransientContracts() {
    LightingPass lighting(ShaderManager::Get());
    DirectionalShadowMaskPass shadows(ShaderManager::Get());
    Statistics statistics;
    uint32 runs = 0;
    for (auto size : {Extent2D(32, 18), Extent2D(17, 9), Extent2D(1, 17), Extent2D(17, 1), Extent2D(1, 1)})
        for (bool preview : {false, true}) for (uint32 scenario = 0; scenario < 3; ++scenario) {
            const auto baseline = RunCase(lighting, shadows, size, preview, scenario, false, false, statistics);
            ++runs;
            for (auto mode : {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
                const auto values = RunCase(lighting, shadows, size, preview, scenario, mode.first, mode.second, statistics);
                Require(values == baseline, "GBuffer allocation/recording policy changed encoded pixels");
                statistics.compared_bytes += values.size();
                ++runs;
            }
        }
    for (uint32 scenario = 0; scenario < 3; ++scenario)
        RunCase(lighting, shadows, Extent2D(17, 9), true, scenario, true, true, statistics, true);
    LOG_INFO("[TESTCASE][PASS] GBufferTransient: runs={} frames={} compared_bytes={} cpu_values={} maximum_error={}; discarded recordings released",
        runs, runs * 2, statistics.compared_bytes, statistics.cpu_values, statistics.maximum_error);
}
