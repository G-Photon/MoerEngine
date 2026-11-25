#include "Core.h"
#include "Engine.h"
#include "config/ConfigManager.h"
#include "log/LogSystem.h"
#include "renderer/raster/AaPass.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "taskgraph/TaskSystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace Moer;
using namespace Moer::Render;
using namespace Moer::Render::Raster;

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr auto usage = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT |
                       ETextureUsageFlags::TRANSFER_DST | ETextureUsageFlags::TRANSFER_SRC;

std::vector<float> RunNumeric(
    AaPass& aa, EAaMode mode, Extent2D size, bool solid, bool parallel, bool preview, bool discard = false
) {
    auto& device = RenderDevice::Get();
    RasterConfig config{};
    config.aa_mode = mode;
    Camera camera;
    camera.Initialize(float3(0.f), 0.f, 0.f, 60.f, float(size.x) / size.y, 0.1f, 100.f);
    aa.ResetHistory();
    AaPass::FrameTextures textures{
        .input = device.CreateTexture("AA.TestInput", size, PF_R32G32B32A32_SFLOAT, usage),
        .output = device.CreateTexture("AA.TestOutput", size, PF_R32G32B32A32_SFLOAT, usage),
        .depth = device.CreateTexture("AA.TestDepth", size, PF_R32_SFLOAT, usage),
        .history = {
            device.CreateTexture("AA.History0", size, PF_R32G32B32A32_SFLOAT, usage),
            device.CreateTexture("AA.History1", size, PF_R32G32B32A32_SFLOAT, usage)
        }
    };
    InitializeGraphicsTextures(std::array{textures.input, textures.output, textures.depth,
                                         textures.history[0], textures.history[1]});
    const std::vector<float4> pixels = [&] {
        std::vector<float4> result(size.x * size.y);
        for (uint32 y = 0; y < size.y; ++y) {
            for (uint32 x = 0; x < size.x; ++x) {
                const bool high = ((x + y / 2) % 7 < 3);
                result[y * size.x + x] = solid ? float4(0.25f, 0.5f, 0.75f, 1.f) :
                    (high ? float4(0.8f, 0.25f, 0.2f, 1.f) : float4(0.1f, 0.6f, 0.7f, 1.f));
            }
        }
        return result;
    }();
    const std::vector<float> depths(pixels.size(), 0.5f);
    RenderGraphResourcePool pool(100);
    RenderGraphTransientAllocator allocator(pool);
    std::vector<float> snapshot;
    const bool smaa = mode >= EAaMode::SMAA_1X;
    const uint32 expected_slots = preview ? 3 : (smaa ? 2 : 1);
    for (uint32 frame = 0; frame < 3; ++frame) {
        const uint8 phase = aa.NextSmaaT2xPhase();
        ReadbackFuture output_readback, edges_readback, weights_readback, late_readback;
        {
            RenderGraph graph("AaTransientContract", RenderGraph::QueueTopology::FromRHI());
            const auto input = ImportRasterTexture(graph, "AA.TestInput", textures.input.Get());
            const auto depth = ImportRasterTexture(graph, "AA.TestDepth", textures.depth.Get());
            const auto output = ImportRasterTexture(graph, "AA.TestOutput", textures.output.Get());
            graph.AddRecordPass(
                "Upload",
                [=](RenderGraph::PassBuilder& builder) {
                    builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                        .Write(input, RenderGraph::TextureAccessType::TransferDestination)
                        .Write(depth, RenderGraph::TextureAccessType::TransferDestination);
                },
                [&](CommandList& commands) {
                    commands.CopyFrom(std::as_bytes(std::span(pixels)), textures.input->GetView());
                    commands.CopyFrom(std::as_bytes(std::span(depths)), textures.depth->GetView());
                },
                RenderGraph::PassRecordPolicy::ParallelAllowed
            );
            const auto result = aa.AddRecordPasses(graph, config, camera, textures, phase,
                {.input = input, .depth = depth, .aa_output = output}, preview);
            Require(result.edges.IsValid() && result.weights.IsValid() == (smaa || preview),
                    "AA allocated unused weights or omitted a required image");
            const RGTransientTextureDesc desc{
                .extent = Extent3D(size.x, size.y, 1), .format = PF_R32G32B32A32_SFLOAT, .usage = usage
            };
            const auto late = graph.CreateTransientTexture("AfterAA.Scratch", desc);
            graph.AddRecordPass(
                "AfterAA",
                [=](RenderGraph::PassBuilder& builder) {
                    builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                        .Read(output, RenderGraph::TextureAccessType::TransferSource)
                        .Write(late, RenderGraph::TextureAccessType::TransferDestination);
                },
                [&graph, late](CommandList& commands) {
                    commands.ClearResource(graph.GetPhysicalTexture(late)->GetView(), float4(0.125f, 0.25f, 0.5f, 1.f));
                },
                RenderGraph::PassRecordPolicy::ParallelAllowed
            );
            graph.AddRecordPass(
                "ReadResults",
                [=](RenderGraph::PassBuilder& builder) {
                    builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                        .Read(output, RenderGraph::TextureAccessType::TransferSource)
                        .Read(late, RenderGraph::TextureAccessType::TransferSource).SideEffect();
                    if (preview) {
                        builder.Read(result.edges, RenderGraph::TextureAccessType::TransferSource)
                            .Read(result.weights, RenderGraph::TextureAccessType::TransferSource);
                    }
                },
                [&, result, late](CommandList& commands) {
                    output_readback = commands.Readback(textures.output->GetView());
                    late_readback = commands.Readback(graph.GetPhysicalTexture(late)->GetView());
                    if (preview) {
                        edges_readback = commands.Readback(graph.GetPhysicalTexture(result.edges)->GetView());
                        weights_readback = commands.Readback(graph.GetPhysicalTexture(result.weights)->GetView());
                    }
                },
                RenderGraph::PassRecordPolicy::ParallelAllowed
            );
            if (!graph.Compile()) throw std::runtime_error(graph.GetCompileError());
            const auto& plan = graph.GetCompiledPlan();
            const auto& edges = plan.resources[result.edges.handle.resource_id];
            const auto late_slot = plan.resources[late.handle.resource_id].transient_slot;
            Require(!edges.imported && edges.transient_slot != RenderGraph::InvalidIndex,
                    "AA edges are not allocation-backed transients");
            if (preview) {
                Require(late_slot != edges.transient_slot &&
                    late_slot != plan.resources[result.weights.handle.resource_id].transient_slot,
                    "Framebuffer inspection did not extend scratch lifetimes");
            } else {
                Require(late_slot == edges.transient_slot || (smaa &&
                    late_slot == plan.resources[result.weights.handle.resource_id].transient_slot),
                    "AA scratch was not aliased after its last consumer");
            }
            CommandList commands;
            commands.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
            if (!graph.RecordAndMergeFrontendCommands(commands, parallel, {}, &allocator)) {
                throw std::runtime_error(graph.GetCompileError());
            }
            Require(pool.TextureCount() == expected_slots && pool.AvailableTextureCount() == 0,
                    "AA scratch was reusable before recorded GPU work completed");
            if (!discard) {
                RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
            }
        }
        if (discard) {
            Require(pool.AvailableTextureCount() == pool.TextureCount(), "Discarded recording retained AA scratch");
            break;
        }
        RHIExecutor::Get().Sync(ERHISyncDepth::RHI);
        const auto data = output_readback.Get().CopyAs<float>();
        const auto late = late_readback.Get().CopyAs<float>();
        Require(data && data->size() == pixels.size() * 4 && late && late->size() == data->size(),
                "AA result readback failed");
        bool filtered_edge = false;
        for (size_t pixel = 0; pixel < pixels.size(); ++pixel) {
            for (uint32 channel = 0; channel < 4; ++channel) {
                const float value = (*data)[pixel * 4 + channel];
                Require(std::isfinite(value), "AA produced non-finite pixels");
                if (channel < 3 && (solid || mode == EAaMode::NONE)) {
                    Require(std::abs(value - pixels[pixel][channel]) < 2e-5f,
                            "AA changed a constant color or the disabled path");
                }
                if (channel < 3 && !solid) {
                    const float3 minimum(0.1f, 0.25f, 0.2f), maximum(0.8f, 0.6f, 0.7f);
                    Require(value >= minimum[channel] - 2e-5f && value <= maximum[channel] + 2e-5f,
                            "AA output left the input color range");
                    filtered_edge |= std::abs(value - pixels[pixel][channel]) > 1e-4f;
                }
                const float4 expected_late(0.125f, 0.25f, 0.5f, 1.f);
                Require((*late)[pixel * 4 + channel] == expected_late[channel],
                        "Aliased successor scratch has incorrect contents");
                snapshot.push_back(value);
            }
        }
        if (!solid && mode != EAaMode::NONE && size.x == 32 && size.y == 24) {
            Require(filtered_edge, "Enabled AA did not filter the diagonal test pattern");
        }
        if (preview) {
            for (const auto* future : {&edges_readback, &weights_readback}) {
                const auto values = future->Get().CopyAs<float>();
                Require(values && values->size() == data->size(), "AA preview readback failed");
                for (float value : *values) Require(std::isfinite(value), "AA preview was corrupted by aliasing");
                if (!smaa && future == &weights_readback) {
                    for (float value : *values) Require(value == 0.f, "Inactive SMAA weights preview was not cleared");
                }
                snapshot.insert(snapshot.end(), values->begin(), values->end());
            }
        }
        Require(pool.AvailableTextureCount() == pool.TextureCount(), "GPU completion did not release AA scratch");
        aa.CommitFrame(mode, phase, camera.GetViewProjectionMatrix());
    }
    LOG_INFO("[TESTCASE][PASS] name=AaTransient mode={} size={}x{} solid={} parallel={} preview={} discard={} pool_slots={} frames={}",
        uint32(mode), size.x, size.y, solid, parallel, preview, discard, pool.TextureCount(), discard ? 0 : 3);
    return snapshot;
}

void Numeric() {
    auto& device = RenderDevice::Get();
    CommandList uploads;
    uploads.SetResourceStateOwnership(ERHIResourceStateOwnership::Compiled);
    AaPass aa(device, ShaderManager::Get(), uploads, PF_R32G32B32A32_SFLOAT);
    RHIExecutor::Get().Submit(EQueueType::Graphics, uploads.Submit(), ERHIExecSubmitFlags::FlushGPU);
    size_t compared_values = 0;
    float max_recording_difference = 0.f;
    for (const auto size : {Extent2D(32, 24), Extent2D(1, 17), Extent2D(17, 1)}) {
        for (const auto mode : {EAaMode::NONE, EAaMode::FXAA_SIMPLIFIED, EAaMode::FXAA_QUALITY,
                                EAaMode::SMAA_1X, EAaMode::SMAA_T2X}) {
            for (bool solid : {true, false}) {
                for (bool preview : {false, true}) {
                    const auto serial = RunNumeric(aa, mode, size, solid, false, preview);
                    const auto parallel = RunNumeric(aa, mode, size, solid, true, preview);
                    Require(serial.size() == parallel.size(), "AA recording modes returned different shapes");
                    for (size_t index = 0; index < serial.size(); ++index) {
                        const float difference = std::abs(serial[index] - parallel[index]);
                        max_recording_difference = std::max(max_recording_difference, difference);
                        Require(difference <= 1e-6f,
                                "AA serial and parallel pixels differ");
                    }
                    compared_values += serial.size();
                }
            }
        }
    }
    RunNumeric(aa, EAaMode::SMAA_T2X, Extent2D(3, 5), true, true, true, true);
    LOG_INFO("[AA numeric] compared_values={} maximum_recording_difference={}", compared_values, max_recording_difference);
    LOG_INFO("[TESTCASE][PASS] AaTransient numeric: 120 runs x 3 frames; aliasing, preview lifetimes, completion reuse and discard verified");
}

int SceneSmoke(int argc, const char** argv) {
    Engine engine;
    try {
        engine.Init(argc, argv, false);
        auto config = engine.GetEditorConfig();
        config->selected_render_method = ERenderMethod::Raster;
        config->active_viewport_mode = EEditorViewportMode::Game;
        auto& raster = config->raster_config;
        raster.probe_gi_enabled = false;
        raster.ao_mode = EAoMode::SSAO;
        raster.ssr_is_ssr_enabled = false;
        constexpr uint32 frames_per_phase = 18;
        uint32 ready_frames = 0;
        bool names_verified = false;
        Array<std::string> stable_names;
        engine.Run(EngineHooks{
            .on_tick_test = [&](Scene& scene) {
                if (!scene.IsReady()) return;
                const uint32 phase = ready_frames / frames_per_phase;
                if (phase == 10) { engine.RequestExit(); return; }
                if (ready_frames % frames_per_phase == 0) {
                    raster.aa_mode = static_cast<EAaMode>(phase / 2);
                    config->validation_selected_frame_buffer_name = phase % 2 ? "aa_texture_2" : "aa_texture_1";
                    std::cout << "[AA scene] mode=" << phase / 2 << " preview="
                              << config->validation_selected_frame_buffer_name << std::endl;
                }
                ++ready_frames;
            },
            .on_capture_ui_composition = [&] {
                const auto size = engine.GetResolution();
                return UiCompositionFrameData{
                    .enabled = true, .scene_extent_resolved = true, .output_resolution = size,
                    .scene_color_resolution = float2(size.x, size.y)
                };
            },
            .on_capture_ui_draw_frame = [&] {
                const auto size = engine.GetResolution();
                UiDrawFramePacket packet;
                packet.main_viewport.display_size = float2(size.x, size.y);
                return packet;
            },
            .on_raster_register_frame_buffer_names = [&](const Array<std::string>& names) {
                const auto edges = std::find(names.begin(), names.end(), "aa_texture_1");
                const auto weights = std::find(names.begin(), names.end(), "aa_texture_2");
                Require(edges != names.end() && weights == edges + 1 &&
                    std::find(names.begin(), names.end(), "output") == names.end(),
                    "Logical framebuffer names or indices changed");
                if (names_verified) Require(names == stable_names, "Framebuffer name order changed across frames");
                stable_names = names;
                names_verified = true;
            }
        });
        Require(names_verified && ready_frames == 180, "AA scene did not complete its preview phases");
        engine.ShutDown();
        std::cout << "[TESTCASE][PASS] AaTransient scene: 10 preview phases, 180 ready frames, stable logical names" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] AaTransient scene " << error.what() << std::endl;
        engine.ShutDown();
        return 1;
    }
}

} // namespace

int main(int argc, const char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--scene") return SceneSmoke(argc, argv);
    bool device_ready = false, tasks_ready = false;
    try {
        LogSystem::Init();
        ConfigManager::GetInstance().Init(std::filesystem::current_path());
        TaskSystem::Init();
        tasks_ready = true;
        RenderDevice::Init(DeviceInitInfo{.rhi_type = ERHIType::Vulkan, .name = "AaTransientTest", .rhi_api_version = "1.3"});
        device_ready = true;
        Numeric();
        ShaderManager::ShutDown();
        RenderDevice::Dispose();
        device_ready = false;
        TaskSystem::ShutDown();
        tasks_ready = false;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] AaTransient " << error.what() << std::endl;
        if (device_ready) { ShaderManager::ShutDown(); RenderDevice::Dispose(); }
        if (tasks_ready) TaskSystem::ShutDown();
        return 1;
    }
}
