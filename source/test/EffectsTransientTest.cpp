#include "Engine.h"
#include "log/LogSystem.h"
#include "renderer/raster/BilateralFilterDenoiserPass.h"
#include "renderer/raster/GaussianCompositePass.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "renderer/raster/SsrPass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
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

constexpr auto usage = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT |
    ETextureUsageFlags::TRANSFER_SRC | ETextureUsageFlags::TRANSFER_DST;

std::vector<float4> FilterReference(const std::vector<float4>& input, Extent2D size, const RasterConfig& config) {
    if (config.denoiser_mode == EDenoiserMode::NONE) return input;
    std::vector<float4> result(input.size());
    const auto luminance = [](const float4& color) { return .299 * color.x + .587 * color.y + .114 * color.z; };
    const int radius = config.denoiser_bfd_kernel_radius;
    for (int y = 0; y < int(size.y); ++y) {
        for (int x = 0; x < int(size.x); ++x) {
            std::array<double, 3> color{};
            double total = 0;
            const auto center = luminance(input[y * size.x + x]);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (x + dx < 0 || x + dx >= int(size.x) || y + dy < 0 || y + dy >= int(size.y)) continue;
                    const auto& sample = input[(y + dy) * size.x + x + dx];
                    const auto delta = luminance(sample) - center;
                    const double weight = std::exp(-double(dx * dx + dy * dy) / config.denoiser_bfd_spatial_sigma_square) *
                        std::exp(-delta * delta / config.denoiser_bfd_range_sigma_square);
                    for (uint32 channel = 0; channel < 3; ++channel) color[channel] += weight * sample[channel];
                    total += weight;
                }
            }
            result[y * size.x + x] = float4(float(color[0] / total), float(color[1] / total), float(color[2] / total), 1.f);
        }
    }
    return result;
}

struct Scenario {
    bool denoise, composite, pattern, preview;
    uint32 ssr; // 0: disabled; 1: material rejection; 2: reflection tracing.
};

struct Statistics {
    uint64 compared_values = 0, cpu_values = 0, reflected_values = 0;
    float maximum_cpu_error = 0;
};

std::vector<float> RunCase(BilateralFilterDenoiserPass& denoiser, SsrPass& ssr, GaussianCompositePass& composite,
    Extent2D size, Scenario scenario, bool pooled, bool parallel, Statistics& statistics, bool discard = false) {
    auto& device = RenderDevice::Get();
    RasterConfig config;
    config.denoiser_mode = scenario.denoise ? EDenoiserMode::BILATERAL_FILTER : EDenoiserMode::NONE;
    config.denoiser_bfd_kernel_radius = 2;
    config.denoiser_bfd_spatial_sigma_square = 2.5f;
    config.denoiser_bfd_range_sigma_square = .3f;
    config.ssr_is_ssr_enabled = scenario.ssr != 0;
    config.ssr_is_force_ground_enable_ssr = false;
    config.ssr_metallic_threshold = scenario.ssr == 1 ? 2.f : 0.f;
    config.ssr_roughness_threshold = 1.f;
    config.ssr_is_enable_jitter = false;
    config.ssr_step_base = .15f;
    config.ssr_sample_count = 16;
    Camera camera;
    camera.Initialize(float3(0.f), -90.f, 0.f, 60.f, float(size.x) / size.y, .01f, 100.f);
    const auto projection = camera.GetProjectionMatrix();
    const float depth_value = (-3.f * projection[2][2] + projection[2][3]) /
        (-3.f * projection[3][2] + projection[3][3]);
    const auto normal_texture = device.CreateTexture("Effects.normal", size, PF_R32G32B32A32_SFLOAT, usage);
    const auto material_texture = device.CreateTexture("Effects.material", size, PF_R32G32B32A32_SFLOAT, usage);
    const auto depth_texture = device.CreateTexture("Effects.depth", size, PF_R32_SFLOAT, usage);
    const auto gaussian_color = device.CreateTexture("Effects.gaussian", size, PF_R32G32B32A32_SFLOAT, usage);
    const auto transmittance = device.CreateTexture("Effects.transmittance", size, PF_R32_SFLOAT, usage);
    InitializeGraphicsTextures(std::array{normal_texture, material_texture, depth_texture, gaussian_color, transmittance});
    const auto status = device.CreateBuffer<uint32>("Effects.status", 4,
        EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    const uint32 tiles = ((size.x + 15) / 16) * ((size.y + 15) / 16);
    const auto ranges = device.CreateBuffer<uint32>("Effects.ranges", tiles * 2,
        EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    std::vector<float> snapshot;
    uint32 first_frame_allocations = 0;
    for (uint32 frame = 0; frame < 2; ++frame) {
        std::vector<float4> input(size.x * size.y);
        for (uint32 y = 0; y < size.y; ++y) {
            for (uint32 x = 0; x < size.x; ++x) {
                input[y * size.x + x] = scenario.pattern ?
                    float4(.25f + .2f * ((x + y + frame) % 5), .1f + .1f * (y % 4), 1.5f + .15f * (x % 3), 1.f) :
                    float4(.25f, .5f, 2.5f, 1.f);
            }
        }
        const auto filtered = FilterReference(input, size, config);
        ReadbackFuture output_readback, preview_readback, late_readback;
        {
            RenderGraph graph("EffectsTransientContract", RenderGraph::QueueTopology::FromRHI());
            std::vector<TextureRef> persistent;
            const auto image = [&](std::string_view name, bool storage = false) {
                const auto flags = storage ? usage | ETextureUsageFlags::UNORDERED_ACCESS : usage;
                if (pooled) return graph.CreateTransientTexture(name, RGTransientTextureDesc{
                    .extent = Extent3D(size.x, size.y, 1), .format = PF_R32G32B32A32_SFLOAT, .usage = flags
                });
                auto texture = device.CreateTexture(name, size, PF_R32G32B32A32_SFLOAT, flags);
                InitializeGraphicsTexture(texture);
                persistent.push_back(texture);
                return ImportRasterTexture(graph, name, texture.Get());
            };
            const auto source = image("Effects.input"), denoised = image("Effects.denoised"), reflected = image("Effects.reflected");
            const auto merged = image("Effects.merged", true);
            const auto normal = ImportRasterTexture(graph, "Normal", normal_texture.Get());
            const auto material = ImportRasterTexture(graph, "Material", material_texture.Get());
            const auto depth = ImportRasterTexture(graph, "Depth", depth_texture.Get());
            const auto color = ImportRasterTexture(graph, "Gaussian", gaussian_color.Get());
            const auto trans = ImportRasterTexture(graph, "Transmittance", transmittance.Get());
            const auto status_handle = ImportRasterBuffer(graph, "Status", status.Get());
            const auto ranges_handle = ImportRasterBuffer(graph, "Ranges", ranges.Get());
            graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer);
                for (const auto resource : {source, normal, material, depth, color, trans})
                    builder.Write(resource, RenderGraph::TextureAccessType::TransferDestination);
                builder.Write(status_handle, RenderGraph::BufferAccessType::TransferDestination)
                    .Write(ranges_handle, RenderGraph::BufferAccessType::TransferDestination);
            }, [&, source](CommandList& commands) {
                commands.CopyFrom(std::as_bytes(std::span(input)), graph.GetPhysicalTexture(source)->GetView());
                commands.ClearResource(normal_texture->GetView(), float4(.5f, 1.f, .5f, 1.f));
                commands.ClearResource(material_texture->GetView(), float4(1.f, .2f, 0.f, 1.f));
                commands.ClearResource(depth_texture->GetView(), float4(depth_value));
                commands.ClearResource(gaussian_color->GetView(), float4(.375f, .1f, .05f, .5f));
                commands.ClearResource(transmittance->GetView(), float4(.5f));
                commands.ClearResource(status->GetView(), 0u);
                commands.ClearResource(ranges->GetView(), 0u);
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            denoiser.AddToGraph(graph, config, size, {.input = source, .denoiser_output = denoised});
            auto output = scenario.denoise ? denoised : source;
            ssr.AddToGraph(graph, config, camera, size, {.input = output, .normal = normal, .depth = depth,
                .metal_rough_ao = material, .ssr_output = reflected});
            if (scenario.ssr) output = reflected;
            if (scenario.composite) {
                GaussianCompositePass::RecordParameters parameters{};
                parameters.settings.exposure_ev = 1.f;
                composite.AddToGraph(graph, parameters, output, color, trans, status_handle, ranges_handle, merged);
                output = merged;
            }
            const auto late = graph.CreateTransientTexture("Effects.after", RGTransientTextureDesc{
                .extent = Extent3D(size.x, size.y, 1), .format = PF_R32G32B32A32_SFLOAT, .usage = usage
            });
            graph.AddRecordPass("SnapshotAndOverwrite", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(output, RenderGraph::TextureAccessType::TransferSource)
                    .Write(late, RenderGraph::TextureAccessType::TransferDestination);
            }, [&, output, late](CommandList& commands) {
                output_readback = commands.Readback(graph.GetPhysicalTexture(output)->GetView());
                commands.ClearResource(graph.GetPhysicalTexture(late)->GetView(), float4(1.f, 0.f, 1.f, 1.f));
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            graph.AddRecordPass("PreviewAndCheckOverwrite", [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Read(late, RenderGraph::TextureAccessType::TransferSource).SideEffect();
                if (scenario.preview && scenario.denoise) builder.Read(denoised, RenderGraph::TextureAccessType::TransferSource);
            }, [&, denoised, late](CommandList& commands) {
                late_readback = commands.Readback(graph.GetPhysicalTexture(late)->GetView());
                if (scenario.preview && scenario.denoise) preview_readback = commands.Readback(graph.GetPhysicalTexture(denoised)->GetView());
            }, RenderGraph::PassRecordPolicy::ParallelAllowed);
            if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
            if (pooled) {
                const auto& plan = graph.GetCompiledPlan();
                const auto slot = [&](auto handle) { return plan.resources[handle.handle.resource_id].transient_slot; };
                for (const auto [handle, enabled] : std::array{
                    std::pair{denoised, scenario.denoise}, std::pair{reflected, scenario.ssr != 0}, std::pair{merged, scenario.composite}}) {
                    Require((slot(handle) != RenderGraph::InvalidIndex) == enabled, "Disabled effect received an allocation slot");
                }
                if (scenario.denoise && scenario.ssr) {
                    Require(slot(source) == slot(reflected), "SSR did not reuse the expired input allocation");
                    if (scenario.preview) Require(slot(denoised) != slot(late) && (!scenario.composite || slot(denoised) != slot(merged)),
                        "Preview did not extend denoiser output lifetime");
                }
            }
            CommandList commands;
            commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
            if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator)) throw std::runtime_error(graph.GetCompileError());
            Require(pool.AvailableTextureCount() == 0, "Effect allocation reusable before completion");
            if (!discard) RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
        }
        if (discard) {
            Require(pool.AvailableTextureCount() == pool.TextureCount(), "Discarded effects recording retained allocations");
            break;
        }
        RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
        const auto values = output_readback.Get().CopyAs<float>();
        const auto late = late_readback.Get().CopyAs<float>();
        const auto preview = preview_readback.Valid() ? preview_readback.Get().CopyAs<float>() : decltype(values){};
        Require(values && late && values->size() == input.size() * 4 && late->size() == values->size(), "Effects readback invalid");
        for (size_t index = 0; index < values->size(); ++index) {
            const uint32 channel = index % 4;
            const size_t pixel = index / 4;
            Require(std::isfinite((*values)[index]), "Effects output is not finite");
            Require((*late)[index] == (channel == 1 ? 0.f : 1.f), "Expired allocation overwrite failed");
            float expected = filtered[pixel][channel];
            if (scenario.composite && channel < 3) {
                const std::array<double, 3> straight{.75, .2, .1};
                const double srgb = straight[channel];
                const double linear = srgb <= .04045 ? srgb / 12.92 : std::pow((srgb + .055) / 1.055, 2.4);
                expected = float(linear + .5 * expected);
            }
            if (scenario.ssr != 2) {
                const auto error = std::abs((*values)[index] - expected);
                Require(error < 5e-5f, "Effects output differs from independent CPU filtering/compositing reference");
                statistics.maximum_cpu_error = std::max(statistics.maximum_cpu_error, error);
                ++statistics.cpu_values;
            } else if (scenario.pattern && std::abs((*values)[index] - expected) > 5e-4f) ++statistics.reflected_values;
            if (preview) Require(preview->size() == values->size() && std::abs((*preview)[index] - filtered[pixel][channel]) < 5e-5f,
                "Denoiser preview was corrupted after subsequent passes");
        }
        snapshot.insert(snapshot.end(), values->begin(), values->end());
        Require(pool.AvailableTextureCount() == pool.TextureCount(), "Completed effects retained allocations");
        if (frame == 0) first_frame_allocations = pool.TextureCount();
        else Require(pool.TextureCount() == first_frame_allocations, "Second frame did not reuse completed allocations");
    }
    Require(!composite.GetOutput().tex, "Graph composite created a persistent standalone output");
    LOG_INFO("[TESTCASE][PASS] name=EffectsTransient size={}x{} denoise={} ssr={} composite={} pattern={} preview={} pooled={} parallel={} discard={}",
        size.x, size.y, scenario.denoise, scenario.ssr, scenario.composite, scenario.pattern, scenario.preview, pooled, parallel, discard);
    return snapshot;
}
} // namespace

void RunEffectsTransientContracts() {
    auto& manager = ShaderManager::Get();
    BilateralFilterDenoiserPass denoiser(manager, PF_R32G32B32A32_SFLOAT);
    SsrPass ssr(manager, PF_R32G32B32A32_SFLOAT);
    GaussianCompositePass composite(manager, RenderDevice::Get().CreateBindlessArray());
    Statistics statistics;
    uint32 runs = 0;
    for (const auto size : {Extent2D(32, 18), Extent2D(1, 17), Extent2D(17, 1)}) {
        for (bool pattern : {false, true}) for (bool denoise : {false, true})
        for (uint32 reflection = 0; reflection < 3; ++reflection) for (bool merge : {false, true})
        for (bool preview : {false, true}) {
            const Scenario scenario{denoise, merge, pattern, preview, reflection};
            const auto baseline = RunCase(denoiser, ssr, composite, size, scenario, false, false, statistics);
            ++runs;
            for (const auto mode : {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
                const auto values = RunCase(denoiser, ssr, composite, size, scenario, mode.first, mode.second, statistics);
                Require(values == baseline, "Allocation or recording mode changed effect pixels");
                statistics.compared_values += values.size();
                ++runs;
            }
        }
    }
    Require(statistics.reflected_values > 0, "SSR tracing fixture never changed any pixel");
    RunCase(denoiser, ssr, composite, Extent2D(17, 9), {true, true, true, false, 2}, true, true, statistics, true);
    LOG_INFO("[TESTCASE][PASS] EffectsTransient numeric: runs={} frames={} compared_values={} cpu_values={} maximum_cpu_error={} reflected_values={}; discarded recording released",
        runs, runs * 2, statistics.compared_values, statistics.cpu_values, statistics.maximum_cpu_error, statistics.reflected_values);
}
