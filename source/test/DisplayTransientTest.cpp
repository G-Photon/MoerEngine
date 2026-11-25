#include "Engine.h"
#include "config/CVarSystem.h"
#include "log/LogSystem.h"
#include "renderer/raster/GaussianCompositePass.h"
#include "renderer/raster/GaussianImageExport.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "scene/gaussian/GaussianAsset.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stb/stb_image.h>
#include <stdexcept>
#include <vector>

using namespace Moer;
using namespace Moer::Render;
using namespace Moer::Render::Raster;

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr auto image_usage = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT |
    ETextureUsageFlags::TRANSFER_SRC | ETextureUsageFlags::TRANSFER_DST;

struct TestDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("moer-display-transient-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    bool keep = false;
    TestDirectory() {
        if (const char* output = std::getenv("MOER_DISPLAY_EVIDENCE_DIR"); output && *output) {
            path = std::filesystem::u8path(output);
            keep = true;
        }
        Require(std::filesystem::create_directory(path), "Display test directory must be new");
    }
    ~TestDirectory() {
        if (!keep) { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    }
};

std::vector<uint8> ReadPng(const std::filesystem::path& path, Extent2D expected) {
    std::ifstream file(path, std::ios::binary);
    std::vector<uint8> bytes((std::istreambuf_iterator<char>(file)), {});
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load_from_memory(bytes.data(), int(bytes.size()), &width, &height, &channels, 3), stbi_image_free);
    Require(pixels && width == int(expected.x) && height == int(expected.y), "PNG dimensions/decode failed");
    return {pixels.get(), pixels.get() + size_t(width) * height * 3};
}

uint8 SrgbByte(float linear) {
    const float srgb = linear <= .0031308f ? linear * 12.92f :
        1.055f * std::pow(linear, 1.f / 2.4f) - .055f;
    return uint8(std::lround(std::clamp(srgb, 0.f, 1.f) * 255.f));
}

void ExportCase(Extent2D size, EPixelFormat format, bool parallel, bool transient, bool overflow,
                const std::filesystem::path& path) {
    auto& device = RenderDevice::Get();
    GaussianImageExport exporter;
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    std::vector<uint8> bytes(size.x * size.y * 4);
    std::vector<float4> floats(size.x * size.y);
    for (uint32 pixel = 0; pixel < floats.size(); ++pixel) {
        for (uint32 channel = 0; channel < 4; ++channel) {
            bytes[pixel * 4 + channel] = channel == 3 ? 255 : uint8((pixel * 29 + channel * 71) % 256);
            floats[pixel][channel] = channel == 3 ? 1.f : float(bytes[pixel * 4 + channel]) / 255.f;
        }
    }
    if (overflow) { std::ofstream existing(path, std::ios::binary); existing << "keep-existing"; }
    {
        RenderGraph graph("DisplayExportContract", RenderGraph::QueueTopology::FromRHI());
        const RGTransientTextureDesc desc{
            .extent = Extent3D(size.x, size.y, 1), .format = format, .usage = image_usage
        };
        TextureRef imported;
        if (!transient) {
            imported = device.CreateTexture("ImportedDisplay", size, format, image_usage);
            InitializeGraphicsTexture(imported);
        }
        const auto source = transient ? graph.CreateTransientTexture("Display", desc) :
            ImportRasterTexture(graph, "Display", imported.Get());
        const auto status_buffer = device.CreateBuffer<uint32>("DisplayExport.status", 4,
            EBufferUsageFlags::TRANSFER_SRC | EBufferUsageFlags::TRANSFER_DST);
        const auto status = graph.ImportBuffer("Status", status_buffer, {.byte_size = 16});
        graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Write(source, RenderGraph::TextureAccessType::TransferDestination)
                .Write(status, RenderGraph::BufferAccessType::TransferDestination);
        }, [&, source](CommandList& commands) {
            const auto data = format == PF_R8G8B8A8_UNORM ? std::as_bytes(std::span(bytes)) :
                std::as_bytes(std::span(floats));
            commands.CopyFrom(data, graph.GetPhysicalTexture(source)->GetView());
            const std::array<uint32, 4> flags{0, uint32(overflow), 0, 0};
            commands.CopyFrom(std::as_bytes(std::span(flags)), status_buffer->GetView());
        }, RenderGraph::PassRecordPolicy::ParallelAllowed);
        // The physical wrapper and the logical-image overload share the snapshot path.
        const auto exported = transient ? exporter.AddToGraph(graph,
            GaussianImageExport::GraphImage{.resource = source, .format = format},
            status_buffer->GetView(), uint2(size.x, size.y), path, 1) :
            exporter.AddToGraph(graph, imported->GetView(), status_buffer->GetView(),
                uint2(size.x, size.y), path, 1);
        Require(exported.IsValid(), "PNG graph request rejected");
        Require(!exporter.AddToGraph(graph, TextureView{}, {}, {}, path, 1).IsValid() &&
                    exporter.GetStatus().is_busy && exporter.GetStatus().message.find("previous export") != std::string::npos,
                "Invalid physical request cleared an active export");
        Require(!exporter.AddToGraph(graph, GaussianImageExport::GraphImage{}, {}, {}, path, 1).IsValid() &&
                    exporter.GetStatus().is_busy && exporter.GetStatus().message.find("previous export") != std::string::npos,
                "Invalid logical request cleared an active export");
        const auto late = graph.CreateTransientTexture("AfterSnapshot", desc);
        graph.AddRecordPass("OverwriteExpiredSource", [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .DependsOn(exported).Write(late, RenderGraph::TextureAccessType::TransferDestination).SideEffect();
        }, [&, late](CommandList& commands) {
            commands.ClearResource(graph.GetPhysicalTexture(late)->GetView(), float4(1.f, 0.f, 1.f, 1.f));
        }, RenderGraph::PassRecordPolicy::ParallelAllowed);
        if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
        if (transient) {
            const auto& plan = graph.GetCompiledPlan();
            Require(plan.resources[source.handle.resource_id].transient_slot ==
                plan.resources[late.handle.resource_id].transient_slot,
                "PNG snapshot retained its expired source allocation");
        }
        CommandList commands;
        commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
        if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator))
            throw std::runtime_error(graph.GetCompileError());
        Require(pool.AvailableTextureCount() == 0, "PNG resource reusable before GPU completion");
        RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    }
    RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
    exporter.Finish();
    Require(!exporter.GetStatus().is_busy, "PNG export did not finish");
    if (overflow) {
        Require(!exporter.GetStatus().is_successful, "Overflow frame was exported");
        std::ifstream existing(path, std::ios::binary);
        const std::string content((std::istreambuf_iterator<char>(existing)), {});
        Require(content == "keep-existing", "Failed export replaced the existing file");
    } else {
        Require(exporter.GetStatus().is_successful, exporter.GetStatus().message.c_str());
        const auto decoded = ReadPng(path, size);
        for (size_t pixel = 0; pixel < floats.size(); ++pixel) {
            for (uint32 channel = 0; channel < 3; ++channel) {
                Require(std::abs(int(decoded[pixel * 3 + channel]) - int(SrgbByte(floats[pixel][channel]))) <= 1,
                        "PNG pixels differ after the source allocation was overwritten");
            }
        }
    }
    Require(pool.AvailableTextureCount() == pool.TextureCount(), "PNG pool allocations retained after completion");
    LOG_INFO("[TESTCASE][PASS] name=DisplayExport size={}x{} format={} parallel={} transient={} overflow={}",
        size.x, size.y, uint32(format), parallel, transient, overflow);
}

void RejectMismatchedSource(bool wrong_format, bool parallel, const std::filesystem::path& path) {
    { std::ofstream existing(path, std::ios::binary); existing << "keep-existing"; }
    GaussianImageExport exporter;
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    {
        RenderGraph graph("RejectMismatchedDisplay", RenderGraph::QueueTopology::FromRHI());
        const auto source = graph.CreateTransientTexture("Display", RGTransientTextureDesc{
            .extent = Extent3D(3, 5, 1), .format = PF_R8G8B8A8_UNORM, .usage = image_usage
        });
        graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Write(source, RenderGraph::TextureAccessType::TransferDestination);
        }, [&, source](CommandList& commands) {
            commands.ClearResource(graph.GetPhysicalTexture(source)->GetView(), float4(1.f));
        }, RenderGraph::PassRecordPolicy::ParallelAllowed);
        const auto status = RenderDevice::Get().CreateBuffer<uint32>("RejectedDisplay.status", 4,
            EBufferUsageFlags::TRANSFER_SRC | EBufferUsageFlags::TRANSFER_DST);
        Require(exporter.AddToGraph(graph, GaussianImageExport::GraphImage{
            .resource = source, .format = wrong_format ? PF_R32G32B32A32_SFLOAT : PF_R8G8B8A8_UNORM
        }, status->GetView(), wrong_format ? uint2(3, 5) : uint2(2, 5), path, 1).IsValid(),
            "Mismatch must be checked when the physical source is resolved");
        if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
        CommandList commands;
        commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
        Require(!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator),
                "Mismatched display source was recorded");
        const auto expected = wrong_format ? "declared format" : "resolution does not match";
        Require(graph.GetCompileError().find(expected) != std::string::npos &&
                    exporter.GetStatus().message.find(expected) != std::string::npos,
                "Recording failed for an unrelated reason");
        // Failed frontend recording is discarded, with no GPU submission.
    }
    exporter.Finish();
    Require(!exporter.GetStatus().is_busy && !exporter.GetStatus().is_successful,
            "Rejected display request remained busy or succeeded");
    std::ifstream existing(path, std::ios::binary);
    const std::string content((std::istreambuf_iterator<char>(existing)), {});
    Require(content == "keep-existing", "Rejected display request replaced the existing file");
    Require(pool.AvailableTextureCount() == pool.TextureCount(), "Rejected display resources retained");
    LOG_INFO("[TESTCASE][PASS] name=DisplayExportMismatch wrong_format={} parallel={}", wrong_format, parallel);
}

void DebugCase(GaussianCompositePass& composite, Extent2D size, uint32 scenario, bool parallel) {
    auto& device = RenderDevice::Get();
    const auto color = device.CreateTexture("DebugColor", size, PF_R32G32B32A32_SFLOAT, image_usage);
    const auto transmittance = device.CreateTexture("DebugTransmittance", size, PF_R32_SFLOAT, image_usage);
    const auto status = device.CreateBuffer<uint32>("DebugStatus", 4,
        EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    const uint32 tiles = ((size.x + 15) / 16) * ((size.y + 15) / 16);
    const auto ranges = device.CreateBuffer<uint32>("DebugRanges", tiles * 2,
        EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    ReadbackFuture readback;
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    {
        RenderGraph graph("GaussianDebugTransient", RenderGraph::QueueTopology::FromRHI());
        const auto color_handle = graph.ImportTexture("Color", color, {});
        const auto transmittance_handle = graph.ImportTexture("Transmittance", transmittance, {});
        const auto status_handle = graph.ImportBuffer("Status", status, {.byte_size = 16});
        const auto ranges_handle = graph.ImportBuffer("Ranges", ranges, {.byte_size = tiles * 8});
        for (const auto texture : {color_handle, transmittance_handle}) {
            graph.SetInitialState(texture, RenderGraph::TextureBoundary::Undefined,
                RenderGraph::QueueRole::None, RenderGraph::AccessMode::None);
            graph.Export(texture, RenderGraph::TextureAccessType::Sampled,
                RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
        }
        for (const auto buffer : {status_handle, ranges_handle}) {
            graph.SetInitialState(buffer, RenderGraph::BufferBoundary::Undefined,
                RenderGraph::QueueRole::None, RenderGraph::AccessMode::None);
            graph.Export(buffer, RenderGraph::BufferAccessType::ShaderRead,
                RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
        }
        const auto output = graph.CreateTransientTexture("Display", RGTransientTextureDesc{
            .extent = Extent3D(size.x, size.y, 1), .format = PF_R8G8B8A8_UNORM, .usage = image_usage
        });
        graph.AddRecordPass("Upload", [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Write(color_handle, RenderGraph::TextureAccessType::TransferDestination)
                .Write(transmittance_handle, RenderGraph::TextureAccessType::TransferDestination)
                .Write(status_handle, RenderGraph::BufferAccessType::TransferDestination)
                .Write(ranges_handle, RenderGraph::BufferAccessType::TransferDestination);
        }, [&](CommandList& commands) {
            commands.ClearResource(color->GetView(), float4(.1f, .2f, .3f, 1.f));
            commands.ClearResource(transmittance->GetView(), float4(.5f));
            const std::array<uint32, 4> flags{0, uint32(scenario == 5), 0, 0};
            commands.CopyFrom(std::as_bytes(std::span(flags)), status->GetView());
            std::vector<uint32> tile_ranges(tiles * 2);
            for (uint32 tile = 0; tile < tiles; ++tile) tile_ranges[tile * 2 + 1] = 63;
            commands.CopyFrom(std::as_bytes(std::span(tile_ranges)), ranges->GetView());
        }, RenderGraph::PassRecordPolicy::ParallelAllowed);
        constexpr std::array modes{EGaussianDebugView::Transmittance, EGaussianDebugView::TileCoverage,
            EGaussianDebugView::ShadowVisibility, EGaussianDebugView::ShadowDepth,
            EGaussianDebugView::ShadowDepth, EGaussianDebugView::Transmittance};
        GaussianCompositePass::RecordParameters parameters{
            .color = color, .transmittance = transmittance, .status = status, .ranges = ranges,
            .debug_view = modes[scenario], .is_shadow_depth_valid = scenario == 3
        };
        graph.AddRecordPass("Debug", [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(color_handle, RenderGraph::TextureAccessType::Sampled)
                .Read(transmittance_handle, RenderGraph::TextureAccessType::Sampled)
                .Read(status_handle, RenderGraph::BufferAccessType::ShaderRead)
                .Read(ranges_handle, RenderGraph::BufferAccessType::ShaderRead)
                .Write(output, RenderGraph::TextureAccessType::ColorAttachment);
        }, [&, output, parameters](CommandList& commands) {
            composite.RecordDebug(commands, parameters, graph.GetPhysicalTexture(output));
        }, RenderGraph::PassRecordPolicy::ParallelAllowed);
        graph.AddRecordPass("Readback", [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Read(output, RenderGraph::TextureAccessType::TransferSource).SideEffect();
        }, [&, output](CommandList& commands) {
            readback = commands.Readback(graph.GetPhysicalTexture(output)->GetView());
        }, RenderGraph::PassRecordPolicy::ParallelAllowed);
        if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
        CommandList commands;
        commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
        if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator))
            throw std::runtime_error(graph.GetCompileError());
        RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    }
    RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
    const auto pixels = readback.Get().CopyAs<uint8>();
    Require(pixels && pixels->size() == size.x * size.y * 4, "Gaussian debug readback failed");
    const std::array<float3, 6> expected{
        float3(.5f), float3(.5f, .25f, .5f), float3(.2f, .4f, .6f),
        float3(.1f, .2f, .3f), float3(0.f), float3(1.f, 0.f, 1.f)
    };
    for (size_t index = 0; index < pixels->size(); ++index) {
        const uint32 channel = index % 4;
        const int reference = channel == 3 ? 255 : int(std::lround(expected[scenario][channel] * 255.f));
        Require(std::abs(int((*pixels)[index]) - reference) <= 1, "Gaussian debug display differs from CPU reference");
    }
    Require(pool.AvailableTextureCount() == pool.TextureCount(), "Debug output retained after completion");
    LOG_INFO("[TESTCASE][PASS] name=DisplayDebug size={}x{} scenario={} parallel={}", size.x, size.y, scenario, parallel);
}
} // namespace

void RunDisplayTransientContracts() {
    TestDirectory directory;
    uint32 exports = 0;
    for (const auto size : {Extent2D(17, 9), Extent2D(1, 17), Extent2D(17, 1)}) {
        for (const auto format : {PF_R8G8B8A8_UNORM, PF_R32G32B32A32_SFLOAT}) {
            for (bool parallel : {false, true}) {
                for (bool transient : {false, true}) {
                    ExportCase(size, format, parallel, transient, false,
                        directory.path / (std::to_string(exports++) + ".png"));
                }
            }
        }
    }
    for (const auto format : {PF_R8G8B8A8_UNORM, PF_R32G32B32A32_SFLOAT}) {
        for (bool parallel : {false, true}) {
            ExportCase(Extent2D(3, 5), format, parallel, true, true,
                directory.path / (std::to_string(exports++) + ".png"));
        }
    }
    for (bool wrong_format : {false, true}) {
        for (bool parallel : {false, true}) {
            RejectMismatchedSource(wrong_format, parallel,
                directory.path / ("rejected-" + std::to_string(exports++) + ".png"));
        }
    }
    GaussianCompositePass composite(ShaderManager::Get(), RenderDevice::Get().CreateBindlessArray());
    for (const auto size : {Extent2D(17, 9), Extent2D(1, 17), Extent2D(17, 1)}) {
        for (uint32 scenario = 0; scenario < 6; ++scenario) {
            for (bool parallel : {false, true}) DebugCase(composite, size, scenario, parallel);
        }
    }
    std::cout << "[TESTCASE][PASS] DisplayTransient: 24 PNG pixel comparisons, 4 rejected overflow frames, 4 rejected mismatches, 56 busy rejections, 36 debug cases" << std::endl;
}

int RunDisplayTransientScene(int argc, const char** argv, bool effects, bool ao_composite, bool ao_scratch, bool gbuffer) {
    TestDirectory directory;
    Engine engine;
    try {
        engine.Init(argc, argv, false);
        const auto config = engine.GetEditorConfig();
        const bool empty_gbuffer = gbuffer && argc > 1 && std::string_view(argv[1]) == "--gbuffer-empty-scene";
        if (empty_gbuffer) {
            const auto inputs = directory.path / "input";
            Require(std::filesystem::create_directory(inputs), "Cannot create Gaussian-only fixture directory");
            const auto source = inputs / "gaussian-only.ply";
            std::ofstream file(source);
            file << "ply\nformat ascii 1.0\nelement vertex 1\n";
            for (const char* property : {"x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"})
                file << "property float " << property << '\n';
            file << "end_header\n0 0 -3 -1 -1 -1 1 0 0 0 2 1 -.5 -.5\n";
            Require(file.good(), "Cannot write Gaussian-only fixture");
            file.close();
            config->scene_path = source.generic_string();
        }
        config->selected_render_method = ERenderMethod::Raster;
        config->raster_config.probe_gi_enabled = false;
        config->raster_config.shadow_map_mode = EShadowMapMode::CSM;
        config->validation_selected_frame_buffer_name = "tonemapping_output";
        constexpr std::array modes{EGaussianDebugView::Color, EGaussianDebugView::Transmittance,
            EGaussianDebugView::TileCoverage, EGaussianDebugView::ShadowVisibility, EGaussianDebugView::ShadowDepth};
        const uint32 phase_count = empty_gbuffer ? 5 : gbuffer ? 20 : ao_scratch ? 24 : ao_composite ? 16 : effects ? 12 : uint32(modes.size());
        const auto recording = CVar::Find("Render.Raster.RDG.ParallelRecording");
        Require(recording.has_value(), "Scene recording policy was not registered");
        uint32 phase = 0, phase_frames = 0, ready_frames = 0, exported_images = 0;
        bool imported = false, request_sent = false;
        Extent2D scene_size(320, 180);
        engine.Run(EngineHooks{
            .on_tick_test = [&](Scene& scene) {
                if (!scene.IsReady()) return;
                Require(++ready_frames < 2400, "Display scene exceeded its bounded frame budget");
                if (!imported && empty_gbuffer) {
                    Require(scene.GetGaussianSceneContent() == EGaussianSceneContent::GaussianOnly,
                        "Skybox boundary test unexpectedly contains meshes");
                    std::cout << "[GBuffer scene] GaussianOnly fixture loaded; mesh path disabled" << std::endl;
                    imported = true;
                }
                if (!imported) {
                    auto asset = std::make_shared<GaussianAsset>();
                    GGaussian gaussian{};
                    const auto& camera = scene.GetMainCamera().camera;
                    const auto position = camera.GetPosition() + camera.GetFront() * 3.f;
                    gaussian.position = float4(position, 1.f);
                    gaussian.scale_opacity = float4(.2f, .2f, .2f, .75f);
                    gaussian.rotation = float4(0.f, 0.f, 0.f, 1.f);
                    gaussian.sh[0] = float4((.8f - .5f) / .2820947918f,
                        (.2f - .5f) / .2820947918f, (.1f - .5f) / .2820947918f, 0.f);
                    asset->bounds = Box3D(position - float3(.6f), position + float3(.6f));
                    asset->gaussians.push_back(gaussian);
                    Require(scene.CreateGaussianInstance(asset) != entt::null, "Cannot create Gaussian scene fixture");
                    imported = true;
                }
                if (phase == phase_count) { engine.RequestExit(); return; }
                if (phase_frames == 0) {
                    scene_size = phase % 2 ? Extent2D(257, 129) : Extent2D(320, 180);
                    config->active_viewport_mode = phase % 2 ? EEditorViewportMode::Game : EEditorViewportMode::Scene;
                    auto settings = scene.GetGaussianViewSettings();
                    settings.display_mode = EGaussianDisplayMode::Engine;
                    settings.debug_view = effects || ao_composite || ao_scratch || gbuffer ? EGaussianDebugView::Color : modes[phase];
                    if (gbuffer) {
                        auto& raster = config->raster_config;
                        constexpr std::array previews{"base_color", "normal", "metal_rough_ao", "shadow_mask", "lighting_output"};
                        const uint32 mode = phase / previews.size();
                        raster.ao_mode = EAoMode::NONE;
                        raster.ssr_is_ssr_enabled = false;
                        raster.shadow_map_mode = mode == 0 ? EShadowMapMode::NONE : EShadowMapMode::CSM;
                        raster.shadow_pcss_enabled = mode == 2;
                        raster.tessellated_surface_enabled = mode == 2;
                        raster.probe_gi_enabled = mode == 3;
                        raster.probe_gi_trace_ray_count = 8;
                        raster.probe_gi_update_brick_budget = 1;
                        config->scene_view_gizmos.show_probe_gi = mode == 3;
                        config->active_viewport_mode = mode == 3 ? EEditorViewportMode::Scene : config->active_viewport_mode;
                        config->validation_selected_frame_buffer_name = previews[phase % previews.size()];
                        std::cout << "[GBuffer scene] phase=" << phase << " mode=" << mode
                                  << " parallel=" << recording->value << " preview=" << config->validation_selected_frame_buffer_name << std::endl;
                    }
                    if (ao_scratch) {
                        auto& raster = config->raster_config;
                        const uint32 mode = phase / 6;
                        constexpr std::array previews{"ao_noisy", "camera_motion_vector", "ao_indirect"};
                        raster.ao_mode = mode == 0 ? EAoMode::NONE : mode == 1 ? EAoMode::SSDO : EAoMode::RTAO;
                        raster.ao_half_resolution = phase % 2 != 0;
                        raster.rtao_denoiser_enable = mode == 3;
                        raster.rtao_sample_mode = mode == 3 ? ERtaoSampleMode::COSINE_WEIGHTED : ERtaoSampleMode::UNIFORM;
                        raster.ssr_is_ssr_enabled = false;
                        config->validation_selected_frame_buffer_name = previews[(phase / 2) % previews.size()];
                        std::cout << "[AO scratch scene] phase=" << phase << " mode=" << uint32(raster.ao_mode)
                                  << " half=" << raster.ao_half_resolution << " denoise=" << raster.rtao_denoiser_enable
                                  << " parallel=" << recording->value << " preview=" << config->validation_selected_frame_buffer_name << std::endl;
                    }
                    if (ao_composite) {
                        auto& raster = config->raster_config;
                        raster.ao_mode = EAoMode(phase / 2);
                        raster.ao_half_resolution = phase % 2 != 0;
                        raster.rtao_denoiser_enable = true;
                        raster.ssr_is_ssr_enabled = false;
                        config->validation_selected_frame_buffer_name = "ao_output";
                        std::cout << "[AO composite scene] phase=" << phase << " mode=" << phase / 2
                                  << " half=" << raster.ao_half_resolution << " parallel=" << recording->value
                                  << " preview=ao_output" << std::endl;
                    }
                    if (effects) {
                        auto& raster = config->raster_config;
                        const uint32 flags = (phase % 12) / 3;
                        raster.denoiser_mode = flags & 1 ? EDenoiserMode::BILATERAL_FILTER : EDenoiserMode::NONE;
                        raster.ssr_is_ssr_enabled = (flags & 2) != 0;
                        constexpr std::array previews{"denoiser_output", "ssr_output", "tonemapping_output"};
                        config->validation_selected_frame_buffer_name = previews[phase % 3];
                        std::cout << "[Effects scene] phase=" << phase << " flags=" << flags
                                  << " parallel=" << recording->value
                                  << " preview=" << config->validation_selected_frame_buffer_name << std::endl;
                    }
                    Require(scene.SetGaussianViewSettings(settings), "Scene rejected Gaussian display settings");
                    std::cout << "[Display scene] phase=" << phase << " size=" << scene_size.x << "x" << scene_size.y << std::endl;
                }
                ++phase_frames;
                const auto path = directory.path / (std::to_string(phase) + ".png");
                if (!request_sent && phase_frames >= 18) {
                    std::string error;
                    Require(scene.RequestGaussianImageExport(path, error), error.c_str());
                    request_sent = true;
                } else if (request_sent && !scene.GetGaussianImageExportStatus().is_busy) {
                    const auto status = scene.GetGaussianImageExportStatus();
                    Require(status.is_successful, status.message.c_str());
                    const auto& counts = config->raster_status.gaussian_status;
                    Require(counts.is_active && counts.has_completed_counts && !counts.has_overflow &&
                        counts.unique_gaussian_count == 1 && counts.visible_count == 1 && counts.tile_pair_count > 0,
                        "Gaussian scene fixture was not visibly rasterized");
                    std::cout << "[Display scene] visible=" << counts.visible_count
                              << " tile_pairs=" << counts.tile_pair_count << std::endl;
                    const auto pixels = ReadPng(path, scene_size);
                    if (!effects && !ao_composite && !ao_scratch && !gbuffer && modes[phase] == EGaussianDebugView::Transmittance) {
                        const auto [minimum, maximum] = std::minmax_element(pixels.begin(), pixels.end());
                        Require(*minimum < 230 && *maximum > 250,
                                "Gaussian transmittance PNG contains no visible splat against its background");
                    }
                    ++exported_images;
                    ++phase;
                    phase_frames = 0;
                    request_sent = false;
                }
            },
            .on_capture_ui_composition = [&] {
                return UiCompositionFrameData{
                    .enabled = true, .scene_extent_resolved = true, .output_resolution = engine.GetResolution(),
                    .scene_color_resolution = float2(scene_size.x, scene_size.y)
                };
            },
            .on_capture_ui_draw_frame = [&] {
                const auto size = engine.GetResolution();
                UiDrawFramePacket packet;
                packet.main_viewport.display_size = float2(size.x, size.y);
                return packet;
            }
        });
        Require(imported && phase == phase_count && exported_images == phase_count,
                "Display scene did not finish every debug/export phase");
        engine.ShutDown();
        if (empty_gbuffer) {
            std::cout << "[TESTCASE][PASS] GBufferTransient empty scene: 5 GaussianOnly skybox/fallback/resize phases" << std::endl;
        } else if (gbuffer) {
            std::cout << "[TESTCASE][PASS] GBufferTransient scene: 20 geometry/shadow/lighting/tessellation/probe/preview phases with visible Gaussian PNG exports" << std::endl;
        } else if (ao_scratch) {
            std::cout << "[TESTCASE][PASS] AoScratchTransient scene: 24 scratch/half-resolution/denoiser/preview phases with visible Gaussian PNG exports" << std::endl;
        } else if (ao_composite) {
            std::cout << "[TESTCASE][PASS] AoCompositeTransient scene: 16 AO/half-resolution/preview phases with visible Gaussian PNG exports" << std::endl;
        } else if (effects) {
            std::cout << "[TESTCASE][PASS] EffectsTransient scene: 12 denoiser/SSR/preview phases with visible Gaussian PNG exports" << std::endl;
        } else {
            std::cout << "[TESTCASE][PASS] DisplayTransient scene: 5 Gaussian views and PNG exports, resize, Scene/Game gizmos" << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] DisplayTransient scene " << error.what() << std::endl;
        engine.ShutDown();
        return 1;
    }
}
