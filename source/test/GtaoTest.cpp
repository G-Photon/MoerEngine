#include "Engine.h"
#include "config/ConfigManager.h"
#include "log/LogSystem.h"
#include "renderer/raster/AoPass.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "rendergraph/RenderGraphUpload.h"
#include "taskgraph/TaskSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace Moer;
using namespace Moer::Render;
using namespace Moer::Render::Raster;

namespace {
void Require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

Array<float> Evaluate(ShaderManager& manager, bool wall, bool half_resolution = false, float radius = .6f, float intensity = 1.f, bool sky = false, bool tilted = false) {
    constexpr uint32                size        = 64;
    const uint32                    output_size = half_resolution ? size / 2 : size;
    std::array<float, size * size>  depth_pixels;
    std::array<float4, size * size> normal_pixels;
    for (uint32 y = 0; y < size; ++y) {
        for (uint32 x = 0; x < size; ++x) {
            const float world_x         = (x + .5f) / size * 2.f - 1.f;
            depth_pixels[y * size + x]  = sky ? 0.f : tilted            ? .5f - world_x * .25f :
                                                  wall && x >= size / 2 ? .85f :
                                                                          .5f;
            const float3 normal         = tilted ? float3(.25f, 0, 1) / std::sqrt(1.0625f) : float3(0, 0, 1);
            normal_pixels[y * size + x] = float4(normal * .5f + float3(.5f), 1.f);
        }
    }
    auto&                        device        = RenderDevice::Get();
    const auto                   sampled       = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_DST;
    const auto                   attachment    = ETextureUsageFlags::COLOR_ATTACHMENT | ETextureUsageFlags::TRANSFER_SRC;
    auto                         depth         = device.CreateTexture("Gtao.Depth", Extent2D(size, size), PF_R32_SFLOAT, sampled);
    auto                         normal        = device.CreateTexture("Gtao.Normal", Extent2D(size, size), PF_R32G32B32A32_SFLOAT, sampled);
    auto                         output        = device.CreateTexture("Gtao.Output", Extent2D(output_size, output_size), PF_R32_SFLOAT, attachment);
    auto                         motion        = device.CreateTexture("Gtao.Motion", Extent2D(output_size, output_size), PF_R32G32_SFLOAT, attachment);
    auto                         camera_buffer = device.CreateBuffer<byte>("Gtao.Camera", sizeof(CameraMotionVectorData), EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TRANSFER_DST);
    const CameraMotionVectorData camera{float4x4::Identity(), float4x4::Identity()};
    auto                         table = device.CreateBindlessArray();
    GtaoPipelineBindlessParam    params{};
    params.clip2world            = float4x4::Identity();
    params.camera_position       = float3(0, 0, 2);
    params.radius                = radius;
    params.inv_resolution        = float2(1.f / output_size);
    params.intensity             = intensity;
    params.slice_count           = 8;
    params.step_count            = 8;
    params.depth_tex             = table->AllocateTexture(depth->GetView(), Sampler{SF_NEAREST, SAM_CLAMP_TO_EDGE});
    params.normal_tex            = table->AllocateTexture(normal->GetView(), Sampler{SF_NEAREST, SAM_CLAMP_TO_EDGE});
    params.camera_mv_data_handle = table->AllocateBuffer(camera_buffer->GetView());
    CommandList upload;
    RecordNewGraphicsTextureUpload(upload, std::as_bytes(std::span(depth_pixels)), depth->GetView());
    RecordNewGraphicsTextureUpload(upload, std::as_bytes(std::span(normal_pixels)), normal->GetView());
    RecordGraphicsBufferUpload(upload, std::as_bytes(std::span(&camera, 1)), camera_buffer->GetView());
    upload.UpdateBindlessArray(table);
    RHIExecutor::Get().Submit(EQueueType::Graphics, upload.Submit(), ERHIExecSubmitFlags::FlushGPU);
    GfxPsoCreateInfo info(RHIRasterizeInfo::Preset(), {}, {RHIColorAttachmentInfo::Preset(PF_R32_SFLOAT), RHIColorAttachmentInfo::Preset(PF_R32G32_SFLOAT)});
    auto             pipeline = manager.Raster().Vertex("core/utils/FullScreenQuad.hlsl").Pixel("pipelines/postprocess/lighting_effects/Gtao.hlsl").Build<GtaoPipeline>(std::move(info));
    RenderGraph      graph("Gtao.Numeric", RenderGraph::QueueTopology::FromRHI());
    auto             d = ImportRasterTexture(graph, "Depth", depth.Get());
    auto             n = ImportRasterTexture(graph, "Normal", normal.Get());
    auto             c = ImportRasterBuffer(graph, "Camera", camera_buffer.Get());
    auto             o = graph.ImportTexture("Output", output, {});
    auto             m = graph.ImportTexture("Motion", motion, {});
    for (auto handle : {o, m}) {
        graph.SetInitialState(handle, RenderGraph::TextureBoundary::Undefined, RenderGraph::QueueRole::None, RenderGraph::AccessMode::None);
        graph.Export(handle, RenderGraph::TextureAccessType::TransferSource, RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
    }
    graph.AddRecordPass("Evaluate", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
            .Read(d, RenderGraph::TextureAccessType::Sampled)
            .Read(n, RenderGraph::TextureAccessType::Sampled)
            .Read(c, RenderGraph::BufferAccessType::ShaderRead)
            .Write(o, RenderGraph::TextureAccessType::ColorAttachment)
            .Write(m, RenderGraph::TextureAccessType::ColorAttachment);
    },
                        [&, params, output_size](CommandList& commands) {
                            commands.Gfx(pipeline, table, params).Draw("Gtao", Rect2D(0, 0, output_size, output_size), RasterTool::GetFullScreenDrawDatas(), ColorAttachment{output}, ColorAttachment{motion});
                        },
                        RenderGraph::PassRecordPolicy::Serial);
    ReadbackFuture result, motion_result;
    graph.AddRecordPass("Readback", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
            .Read(o, RenderGraph::TextureAccessType::TransferSource)
            .Read(m, RenderGraph::TextureAccessType::TransferSource)
            .SideEffect();
    },
                        [&](CommandList& commands) {
                            result        = commands.Readback(output->GetView());
                            motion_result = commands.Readback(motion->GetView());
                        },
                        RenderGraph::PassRecordPolicy::Serial);
    CommandList commands;
    RecordGraphicsUploadGraph(graph, commands);
    RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    auto pixels        = result.Get().CopyAs<float>();
    auto motion_pixels = motion_result.Get().CopyAs<float>();
    Require(pixels && pixels->size() == output_size * output_size, "GTAO readback has invalid size");
    Require(motion_pixels && motion_pixels->size() == output_size * output_size * 2, "GTAO motion readback has invalid size");
    for (float pixel : *pixels)
        Require(std::isfinite(pixel) && pixel >= 0 && pixel <= 1, "GTAO visibility is not finite within [0,1]");
    if (!half_resolution) {
        for (float value : *motion_pixels)
            Require(std::abs(value) < 1e-6f, "Stationary GTAO generated camera motion");
    }
    return std::move(*pixels);
}

void CheckNumeric(ShaderManager& manager) {
    for (bool half : {false, true}) {
        auto plane  = Evaluate(manager, false, half);
        auto tilted = Evaluate(manager, false, half, .6f, 1.f, false, true);
        std::cout << "[GTAO] plane_min=" << *std::min_element(plane.begin(), plane.end())
                  << " tilted_min=" << *std::min_element(tilted.begin(), tilted.end()) << '\n';
        for (float value : plane)
            Require(std::abs(value - 1.f) < 2e-4f, "Unoccluded plane is darkened, including at screen edges");
        for (float value : tilted)
            Require(std::abs(value - 1.f) < 2e-4f, "Unoccluded tilted plane is darkened");
        for (const auto& pixels : {Evaluate(manager, true, half, 0.f), Evaluate(manager, true, half, .6f, 0.f), Evaluate(manager, false, half, .6f, 1.f, true)}) {
            for (float value : pixels)
                Require(std::abs(value - 1.f) < 1e-6f, "GTAO zero radius/intensity or sky fallback failed");
        }
        const auto   wall         = Evaluate(manager, true, half);
        const auto   stronger     = Evaluate(manager, true, half, .6f, 2.f);
        const auto   short_radius = Evaluate(manager, true, half, .1f);
        const uint32 size         = half ? 32 : 64;
        const size_t boundary     = size_t(size / 2) * size + size / 2 - 1;
        std::cout << "[GTAO] half=" << half << " plane=" << plane[boundary] << " wall=" << wall[boundary]
                  << " short_radius=" << short_radius[boundary] << '\n';
        Require(wall[boundary] < .96f, "Nearby occluder did not reduce GTAO visibility");
        Require(short_radius[boundary] > .97f, "Occluder outside world radius still shadows GTAO");
        for (size_t i = 0; i < wall.size(); ++i) {
            Require(std::abs(stronger[i] - std::max(0.f, 2.f * wall[i] - 1.f)) < 1e-5f, "GTAO intensity does not scale occlusion");
        }
    }
    std::cout << "[TESTCASE][PASS] GTAO planes/tilt/edges, occluder/radius, sky, intensity, full/half, motion\n";
}

int RunScene(int argc, const char** argv) {
    Engine engine;
    try {
        engine.Init(argc, argv, false);
        auto config                    = engine.GetEditorConfig();
        config->selected_render_method = ERenderMethod::Raster;
        config->active_viewport_mode   = EEditorViewportMode::Game;
        auto& raster                   = config->raster_config;
        raster.probe_gi_enabled        = false;
        uint32 frames                  = 0;
        engine.Run(EngineHooks{.on_tick_test = [&](Scene& scene) {
            if (!scene.IsReady())
                return;
            if (frames == 144) {
                engine.RequestExit();
                return;
            }
            const uint32 phase        = frames / 24;
            raster.ao_mode            = phase == 4 ? EAoMode::SSAO : phase == 2 ? EAoMode::GTAO_AO_ONLY :
                                                                                  EAoMode::GTAO;
            raster.ao_half_resolution = phase % 2 != 0;
            raster.gtao_intensity     = phase == 3 ? 0.f : 1.f;
            if (frames % 24 == 0)
                std::cout << "[GTAO.Scene] phase=" << phase << '\n';
            ++frames;
        }});
        Require(frames == 144, "GTAO scene did not complete all phases");
        engine.ShutDown();
        std::cout << "[TESTCASE][PASS] GTAO scene full/half, AO-only, intensity, SSAO switch/re-enable\n";
        return 0;
    } catch (const std::exception& error) {
        engine.ShutDown();
        std::cerr << "[TESTCASE][FAIL] " << error.what() << '\n';
        return 1;
    }
}
} // namespace

int main(int argc, const char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--scene")
        return RunScene(argc, argv);
    bool tasks = false, device = false, shaders = false;
    int  result = 0;
    try {
        LogSystem::Init();
        ConfigManager::GetInstance().Init(std::filesystem::current_path());
        TaskSystem::Init();
        tasks = true;
        RenderDevice::Init(DeviceInitInfo{.rhi_type = ERHIType::Vulkan, .name = "GtaoTest", .rhi_api_version = "1.3"});
        device        = true;
        auto& manager = ShaderManager::Get();
        shaders       = true;
        CheckNumeric(manager);
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
