#include "Engine.h"
#include "config/ConfigManager.h"
#include "log/LogSystem.h"
#include "renderer/raster/AssetTool.h"
#include "renderer/raster/IblEnvironment.h"
#include "renderer/raster/RasterGraphResourceAccess.h"
#include "renderer/raster/RasterTool.h"
#include "rendergraph/RenderGraphUpload.h"
#include "taskgraph/TaskSystem.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace Moer;
using namespace Moer::Render;
using namespace Moer::Render::Raster;

namespace {

class IblEvaluationPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(IblEvaluationPipeline);
    DEFINE_SHADER_TEX(diffuse_map);
    DEFINE_SHADER_TEX(specular_map);
    DEFINE_SHADER_TEX(brdf_lut);
    DEFINE_SHADER_SAMPLER(linear_sampler);
    DEFINE_SHADER_ARGS(diffuse_map, specular_map, brdf_lut, linear_sampler);
};

void Require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

float FromHalf(uint16 value) {
    const uint32 exponent    = (value >> 10) & 31u;
    const float  significand = float(value & 1023u);
    if (exponent == 31)
        throw std::runtime_error("IBL produced a nonfinite value");
    const float result = exponent == 0 ? std::ldexp(significand, -24) :
                                         std::ldexp(significand + 1024.f, int(exponent) - 25);
    return value & 32768u ? -result : result;
}

TextureRef ConstantEnvironment(RenderDevice& device, float4 value) {
    auto                   cube = device.CreateCubeMap("Test.IBL.Input", Extent2D(8, 8), PF_R32G32B32A32_SFLOAT, ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_DST);
    std::array<float4, 64> pixels;
    pixels.fill(value);
    CommandList commands;
    for (uint32 face = 0; face < 6; ++face) {
        RecordNewGraphicsTextureUpload(commands, std::span<const byte>(reinterpret_cast<const byte*>(pixels.data()), sizeof(pixels)), cube->GetView().Slice(face, 1));
    }
    RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    return cube;
}

float3 CubeDirection(uint32 face, float u, float v) {
    const float length = std::sqrt(1.f + u * u + v * v);
    switch (face) {
        case 0:
            return float3(1, -v, -u) / length;
        case 1:
            return float3(-1, -v, u) / length;
        case 2:
            return float3(u, 1, v) / length;
        case 3:
            return float3(u, -1, -v) / length;
        case 4:
            return float3(u, -v, 1) / length;
        default:
            return float3(-u, -v, -1) / length;
    }
}

TextureRef DirectionEnvironment(RenderDevice& device) {
    constexpr uint32 size = 32;
    auto             cube = device.CreateCubeMap("Test.IBL.Directions", Extent2D(size, size), PF_R32G32B32A32_SFLOAT, ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_DST);
    CommandList      commands;
    for (uint32 face = 0; face < 6; ++face) {
        std::array<float4, size * size> pixels;
        for (uint32 y = 0; y < size; ++y) {
            for (uint32 x = 0; x < size; ++x) {
                const auto direction = CubeDirection(face, 2.f * (x + .5f) / size - 1.f, 2.f * (y + .5f) / size - 1.f);
                pixels[y * size + x] = float4(direction * .5f + float3(.5f), 1.f);
            }
        }
        RecordNewGraphicsTextureUpload(commands, std::span<const byte>(reinterpret_cast<const byte*>(pixels.data()), sizeof(pixels)), cube->GetView().Slice(face, 1));
    }
    RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    return cube;
}

std::array<double, 2> ReferenceBrdf(double no_v, double roughness) {
    constexpr uint32 samples       = 16384;
    const double     vx            = std::sqrt(1.0 - no_v * no_v);
    const double     alpha_squared = std::pow(roughness, 4);
    const double     k             = roughness * roughness / 2.0;
    const auto       geometry      = [k](double cosine) {
        return cosine / (cosine * (1.0 - k) + k);
    };
    std::array<double, 2> result{};
    for (uint32 i = 0; i < samples; ++i) {
        uint32 bits = i, reversed = 0;
        for (uint32 bit = 0; bit < 32; ++bit) {
            reversed = (reversed << 1) | (bits & 1u);
            bits >>= 1;
        }
        const double xi   = double(reversed) / 4294967296.0;
        const double hz   = std::sqrt((1.0 - xi) / (1.0 + (alpha_squared - 1.0) * xi));
        const double hx   = std::sqrt(1.0 - hz * hz) * std::cos(2.0 * std::numbers::pi * i / samples);
        const double vo_h = std::max(0.0, vx * hx + no_v * hz);
        const double no_l = std::max(0.0, 2.0 * vo_h * hz - no_v);
        if (no_l > 0.0) {
            const double visibility = geometry(no_v) * geometry(no_l) * vo_h / (hz * no_v);
            const double fresnel    = std::pow(1.0 - vo_h, 5);
            result[0] += (1.0 - fresnel) * visibility / samples;
            result[1] += fresnel * visibility / samples;
        }
    }
    return result;
}

std::vector<Array<uint16>> ReadTexture(TextureRef texture) {
    RenderGraph                 graph("Test.IBL.Readback", RenderGraph::QueueTopology::FromRHI());
    auto                        handle = ImportRasterTexture(graph, "Texture", texture.Get());
    std::vector<ReadbackFuture> futures(texture->GetNumMips() * texture->GetNumArray());
    graph.AddRecordPass("Readback", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
            .Read(handle, RenderGraph::TextureAccessType::TransferSource)
            .SideEffect();
    },
                        [=, &futures](CommandList& commands) {
                            for (uint32 mip = 0; mip < texture->GetNumMips(); ++mip) {
                                for (uint32 face = 0; face < texture->GetNumArray(); ++face) {
                                    futures[mip * texture->GetNumArray() + face] = commands.Readback(
                                        texture->GetView(mip, 1).Slice(face, 1)
                                    );
                                }
                            }
                        },
                        RenderGraph::PassRecordPolicy::Serial);
    CommandList commands;
    RecordGraphicsUploadGraph(graph, commands);
    RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    std::vector<Array<uint16>> result;
    for (auto& future : futures) {
        auto pixels = future.Get().CopyAs<uint16>();
        Require(pixels && !pixels->empty(), "IBL texture readback is empty");
        result.push_back(std::move(*pixels));
    }
    return result;
}

void CheckConstant(TextureRef texture, const float4& expected) {
    for (const auto& image : ReadTexture(texture)) {
        for (size_t i = 0; i < image.size(); ++i) {
            Require(std::abs(FromHalf(image[i]) - expected[i % 4]) < 0.005f, "IBL convolution changed a uniform environment");
        }
    }
}

void CheckMaterial(ShaderManager& manager, const IblEnvironment& ibl, const float4& radiance) {
    auto&            device = RenderDevice::Get();
    GfxPsoCreateInfo info(RHIRasterizeInfo::Preset(), {}, {RHIColorAttachmentInfo::Preset(PF_R16G16B16A16_SFLOAT)});
    auto             pipeline = manager.Raster().Vertex("core/utils/FullScreenQuad.hlsl").Pixel("test/IblEvaluation.frag.hlsl").Build<IblEvaluationPipeline>(std::move(info));
    auto             output   = device.CreateTexture("Test.IBL.Material", Extent2D(5, 1), PF_R16G16B16A16_SFLOAT, ETextureUsageFlags::COLOR_ATTACHMENT | ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_SRC);
    RenderGraph      graph("Test.IBL.Material", RenderGraph::QueueTopology::FromRHI());
    const auto       diffuse  = ImportRasterTexture(graph, "Diffuse", ibl.GetDiffuse().Get());
    const auto       specular = ImportRasterTexture(graph, "Specular", ibl.GetSpecular().Get());
    const auto       brdf     = ImportRasterTexture(graph, "Brdf", ibl.GetBrdf().Get());
    const auto       target   = graph.ImportTexture("Output", output, {});
    graph.SetInitialState(target, RenderGraph::TextureBoundary::Undefined, RenderGraph::QueueRole::None, RenderGraph::AccessMode::None);
    graph.Export(target, RenderGraph::TextureAccessType::Sampled, RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read);
    graph.AddRecordPass("Evaluate", [=](RenderGraph::PassBuilder& builder) {
        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
            .Read(diffuse, RenderGraph::TextureAccessType::Sampled)
            .Read(specular, RenderGraph::TextureAccessType::Sampled)
            .Read(brdf, RenderGraph::TextureAccessType::Sampled)
            .Write(target, RenderGraph::TextureAccessType::ColorAttachment);
    },
                        [&pipeline, &ibl, output](CommandList& commands) {
                            commands.Gfx(pipeline, ibl.GetDiffuse(), ibl.GetSpecular(), ibl.GetBrdf(), Sampler{SF_LINEAR, SAM_CLAMP_TO_EDGE})
                                .Draw("Evaluate", Rect2D(0, 0, 5, 1), RasterTool::GetFullScreenDrawDatas(), ColorAttachment{output, AC_CLEAR_STORE, float4(0)});
                        },
                        RenderGraph::PassRecordPolicy::Serial);
    CommandList commands;
    RecordGraphicsUploadGraph(graph, commands);
    RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    const auto   pixels = ReadTexture(output).front();
    const float3 albedo(.8f, .4f, .2f);
    for (uint32 test = 0; test < 5; ++test) {
        const float metallic      = test == 0 ? 0.f : 1.f;
        const float roughness     = test == 3 ? .5f / 128.f : test == 4 ? 127.5f / 128.f :
                                                                          .5f;
        const auto  expected_brdf = ReferenceBrdf(127.5 / 128.0, roughness);
        for (uint32 channel = 0; channel < 3; ++channel) {
            const float  f0       = metallic == 0 ? .04f : albedo[channel];
            const double expected = test == 2 ? 0.0 : radiance[channel] * (albedo[channel] * (1.f - metallic) * (1.f - f0) + f0 * expected_brdf[0] + expected_brdf[1]);
            Require(std::abs(FromHalf(pixels[test * 4 + channel]) - expected) < .035, "Material IBL does not match split-sum evaluation");
        }
    }
    std::cout << "[TESTCASE][PASS] IBL dielectric/metallic, roughness endpoints, zero AO\n";
}

void CheckSkyboxLoading(ShaderManager& manager) {
    const auto root   = ConfigManager::GetInstance().GetEditorResourcePath() / "textures";
    const auto folder = root / ("Test.IBL." +
                                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Require(std::filesystem::create_directory(folder), "Cannot create unique skybox fixture directory");
    struct Cleanup {
        std::filesystem::path folder;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(folder, ignored);
        }
    } cleanup{folder};
    const std::array names{"posx.jpg", "negx.jpg", "posy.jpg", "negy.jpg", "posz.jpg", "negz.jpg"};
    const auto       write_face = [&](const char* name, uint8 width, uint8 height) {
        // Tiny uncompressed TGA fixture; stb detects content independently of the file extension.
        std::vector<uint8> image(18 + size_t(width) * height * 4, 128);
        std::fill_n(image.begin(), 18, 0);
        image[2]  = 2;
        image[12] = width;
        image[14] = height;
        image[16] = 32;
        image[17] = 0x28;
        std::ofstream stream(folder / name, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(image.data()), std::streamsize(image.size()));
        Require(bool(stream), "Cannot write skybox fixture");
    };
    auto&      device           = RenderDevice::Get();
    auto       cfg              = TexConfig::Asset(folder.string()).Format(PF_R8G8B8A8_SRGB).Usage(ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_DST);
    const auto require_rejected = [&] {
        CommandList commands;
        TextureRef  texture;
        bool        rejected = false;
        try {
            AssetTool::LoadTexture<TexCubeTag>(device, commands, texture, cfg);
        } catch (const std::runtime_error& error) {
            rejected = std::string_view(error.what()).find("skybox") != std::string_view::npos ||
                       std::string_view(error.what()).find("Skybox") != std::string_view::npos;
        }
        Require(rejected && !texture, "Invalid skybox was not rejected before texture creation");
    };
    require_rejected();
    for (const char* name : names)
        write_face(name, 2, 2);
    std::filesystem::remove(folder / names.back());
    require_rejected();
    {
        std::ofstream corrupt(folder / names.back(), std::ios::binary);
        corrupt << "invalid image";
    }
    require_rejected();
    write_face(names.back(), 2, 1);
    require_rejected();
    write_face(names.back(), 4, 4);
    require_rejected();
    write_face(names.back(), 2, 2);
    for (const auto& path : {folder.string(), folder.filename().string()}) {
        cfg.asset_path_relative = path;
        TextureRef  texture;
        CommandList commands;
        AssetTool::LoadTexture<TexCubeTag>(device, commands, texture, cfg);
        RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
        IblEnvironment ibl(device, manager);
        ibl.Update(texture);
        CheckConstant(ibl.GetDiffuse(), float4(.21586f, .21586f, .21586f, 1.f));
    }
    std::cout << "[TESTCASE][PASS] IBL skybox absolute/relative path, sRGB decode, missing/nonsquare/mismatched faces\n";
}

void CheckPrecompute(ShaderManager& manager) {
    auto&          device = RenderDevice::Get();
    IblEnvironment ibl(device, manager);
    const float4   radiance(2.f, .5f, 4.f, 1.f);
    auto           source = ConstantEnvironment(device, radiance);
    ibl.Update(source);
    CheckConstant(ibl.GetDiffuse(), radiance);
    CheckConstant(ibl.GetSpecular(), radiance);
    CheckMaterial(manager, ibl, radiance);
    const auto first_diffuse  = ibl.GetDiffuse();
    const auto first_specular = ibl.GetSpecular();
    const auto first_brdf     = ibl.GetBrdf();
    ibl.Update(source);
    Require(first_diffuse.Get() == ibl.GetDiffuse().Get() && first_specular.Get() == ibl.GetSpecular().Get(), "Unchanged IBL source was recomputed");
    for (const auto& image : ReadTexture(first_brdf)) {
        for (size_t i = 0; i < image.size(); i += 4) {
            const float a = FromHalf(image[i]);
            const float b = FromHalf(image[i + 1]);
            Require(a >= 0 && b >= 0 && a + b <= 1.02f, "IBL BRDF integration exceeds unit energy");
        }
        for (uint32 y : {25u, 76u, 115u}) {
            for (uint32 x : {31u, 83u, 121u}) {
                const auto expected = ReferenceBrdf((x + .5) / 128, (y + .5) / 128);
                for (uint32 channel = 0; channel < 2; ++channel) {
                    const float actual = FromHalf(image[(y * 128 + x) * 4 + channel]);
                    Require(std::abs(actual - expected[channel]) < .025, "BRDF LUT disagrees with CPU integration");
                }
            }
        }
    }
    auto black = ConstantEnvironment(device, float4(0, 0, 0, 1));
    ibl.Update(black);
    CheckConstant(ibl.GetDiffuse(), float4(0, 0, 0, 1));
    CheckConstant(ibl.GetSpecular(), float4(0, 0, 0, 1));
    Require(first_brdf.Get() == ibl.GetBrdf().Get(), "Environment-independent BRDF LUT was rebuilt");
    ibl.Update(DirectionEnvironment(device));
    for (auto texture : {ibl.GetDiffuse(), ibl.GetSpecular()}) {
        const uint32 size   = texture->GetExtent().x;
        const auto   images = ReadTexture(texture);
        for (uint32 face = 0; face < 6; ++face) {
            for (uint32 y = size / 4; y < size * 3 / 4; y += size / 8) {
                for (uint32 x = size / 4; x < size * 3 / 4; x += size / 8) {
                    const auto  direction = CubeDirection(face, 2.f * (x + .5f) / size - 1.f, 2.f * (y + .5f) / size - 1.f);
                    const float scale     = texture.Get() == ibl.GetDiffuse().Get() ? 1.f / 3.f : .5f;
                    const auto  expected  = float3(.5f) + direction * scale;
                    for (uint32 channel = 0; channel < 3; ++channel) {
                        Require(std::abs(FromHalf(images[face][(y * size + x) * 4 + channel]) - expected[channel]) < .02f, "IBL cube face orientation or cosine convolution is incorrect");
                    }
                }
            }
        }
    }
    std::cout << "[TESTCASE][PASS] IBL HDR/black, cube orientation, analytic diffuse, CPU BRDF, energy, cache/replacement\n";
}

int RunScene(int argc, const char** argv) {
    Engine engine;
    try {
        engine.Init(argc, argv, false);
        auto config                    = engine.GetEditorConfig();
        config->selected_render_method = ERenderMethod::Raster;
        auto& raster                   = config->raster_config;
        raster.ao_mode                 = EAoMode::SSAO;
        raster.probe_gi_enabled        = false;
        uint32 ready_frames            = 0;
        engine.Run(EngineHooks{.on_tick_test = [&](Scene& scene) {
            if (!scene.IsReady())
                return;
            if (ready_frames == 144) {
                engine.RequestExit();
                return;
            }
            const uint32 phase   = ready_frames / 24;
            raster.ibl_enabled   = phase != 0 && phase != 4;
            raster.ibl_intensity = phase == 2 ? 0.f : phase == 3 ? 2.f :
                                                                   1.f;
            if (ready_frames % 24 == 0)
                std::cout << "[IBL.Scene] phase=" << phase << '\n';
            ++ready_frames;
        }});
        Require(ready_frames == 144, "IBL scene did not complete its phases");
        engine.ShutDown();
        std::cout << "[TESTCASE][PASS] IBL scene enable/strength/disable/re-enable\n";
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
    bool task_system = false, render_device = false, shader_manager = false;
    int  result = 0;
    try {
        LogSystem::Init();
        ConfigManager::GetInstance().Init(std::filesystem::current_path());
        TaskSystem::Init();
        task_system = true;
        RenderDevice::Init(DeviceInitInfo{.rhi_type = ERHIType::Vulkan, .name = "IblLightingTest", .rhi_api_version = "1.3"});
        render_device  = true;
        auto& manager  = ShaderManager::Get();
        shader_manager = true;
        CheckPrecompute(manager);
        CheckSkyboxLoading(manager);
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] " << error.what() << '\n';
        result = 1;
    }
    if (shader_manager)
        ShaderManager::ShutDown();
    if (render_device)
        RenderDevice::Dispose();
    if (task_system)
        TaskSystem::ShutDown();
    return result;
}
