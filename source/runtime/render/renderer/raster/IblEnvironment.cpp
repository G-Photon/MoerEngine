#include "IblEnvironment.h"
#include "RasterGraphResourceAccess.h"
#include "RasterTool.h"
#include "rendergraph/RenderGraphUpload.h"
#include "rhi/executor/RHIExecutor.h"

#include <algorithm>
#include <format>

namespace Moer::Render::Raster {

IblEnvironment::IblEnvironment(RenderDevice& device, ShaderManager& manager) : device(device) {
    GfxPsoCreateInfo info(
        RHIRasterizeInfo::Preset(), {}, {RHIColorAttachmentInfo::Preset(PF_R16G16B16A16_SFLOAT)}
    );
    pipeline = manager.Raster()
                   .Vertex("core/utils/FullScreenQuad.hlsl")
                   .Pixel("pipelines/raster/deferred/lighting/IblPrecompute.frag.hlsl")
                   .Build<IblPrecomputePipeline>(std::move(info));
}

void IblEnvironment::Update(TextureRef environment_map) {
    if (!environment_map || source.Get() == environment_map.Get()) {
        return;
    }
    constexpr auto usage         = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT |
                                   ETextureUsageFlags::TRANSFER_SRC;
    auto           next_diffuse  = device.CreateCubeMap("IBL.Diffuse", Extent2D(32, 32), PF_R16G16B16A16_SFLOAT, usage);
    auto           next_specular = device.CreateCubeMap(
        "IBL.Specular", Extent2D(128, 128), PF_R16G16B16A16_SFLOAT, usage, 8
    );
    auto next_brdf = brdf ? brdf : device.CreateTexture("IBL.Brdf", Extent2D(128, 128), PF_R16G16B16A16_SFLOAT, usage);

    RenderGraph graph("IBL.Precompute", RenderGraph::QueueTopology::FromRHI());
    const auto  input      = ImportRasterTexture(graph, "Environment", environment_map.Get());
    const auto  add_target = [&](TextureRef target, uint32 mode) {
        const auto handle = graph.ImportTexture(
            mode == 0 ? "Diffuse" : mode == 1 ? "Specular" :
                                                "Brdf",
            target,
            {.mip_count = target->GetNumMips(), .layer_count = target->GetNumArray()}
        );
        graph.SetInitialState(
            handle, RenderGraph::TextureBoundary::Undefined,
            RenderGraph::QueueRole::None, RenderGraph::AccessMode::None
        );
        graph.Export(
            handle, RenderGraph::TextureAccessType::Sampled,
            RenderGraph::QueueRole::Graphics, RenderGraph::AccessMode::Read
        );
        for (uint32 mip = 0; mip < target->GetNumMips(); ++mip) {
            for (uint32 face = 0; face < target->GetNumArray(); ++face) {
                const RenderGraph::TextureRange range{
                    .mip_first = mip, .mip_count = 1, .layer_first = face, .layer_count = 1
                };
                const IblPrecomputeParam param{
                    face, mode, mode == 1 ? float(mip) / float(target->GetNumMips() - 1) : 0.f, 256
                };
                graph.AddRecordPass(
                    std::format("Filter.{}.Mip{}.Face{}", mode, mip, face),
                    [=](RenderGraph::PassBuilder& builder) {
                        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                            .Read(input, RenderGraph::TextureAccessType::Sampled)
                            .Write(handle, RenderGraph::TextureAccessType::ColorAttachment, range);
                    },
                    [this, environment_map, target, param, mip, face](CommandList& commands) {
                        commands.Gfx(pipeline, environment_map, Sampler{SF_LINEAR, SAM_CLAMP_TO_EDGE}, param)
                            .Draw(
                                "IBL.Filter",
                                Rect2D(0, 0, std::max(1u, target->GetExtent().x >> mip),
                                       std::max(1u, target->GetExtent().y >> mip)),
                                RasterTool::GetFullScreenDrawDatas(),
                                ColorAttachment{target.Get(), AC_CLEAR_STORE, float4(0), mip, face}
                            );
                    },
                    RenderGraph::PassRecordPolicy::ParallelAllowed
                );
            }
        }
    };
    add_target(next_diffuse, 0);
    add_target(next_specular, 1);
    if (!brdf) {
        add_target(next_brdf, 2);
    }
    CommandList commands;
    RecordGraphicsUploadGraph(graph, commands);
    // The exported Sampled boundary and ordered graphics submissions put the
    // convolution before frame reads; command references retain old resources.
    RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
    source   = std::move(environment_map);
    diffuse  = std::move(next_diffuse);
    specular = std::move(next_specular);
    brdf     = std::move(next_brdf);
}

} // namespace Moer::Render::Raster
