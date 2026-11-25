// 构建 GPU 遮挡剔除使用的层次化反向 Z 深度金字塔。
#pragma once

#include "RasterResource.h"
#include "misc/Types.h"
#include "rendergraph/RenderGraph.h"
#include "shader/ShaderPipeline.h"
#include "shader/ShaderResourceManager.h"
#include "shaderheaders/shared/raster/culling/ShaderParameters.h"

namespace Moer::Render::Raster {

class HiZBuildPipeline : public ComputePipeline {
public:
    DEFINE_COMPUTE_PIPELINE_CLASS(HiZBuildPipeline);

    DEFINE_SHADER_CONSTANT(HiZBuildParam, param);
    DEFINE_SHADER_TEX(src_texture);
    DEFINE_SHADER_TEX(dst_texture);

    DEFINE_SHADER_ARGS(param, src_texture, dst_texture);
};

class RENDER_API HiZBuildPass {
public:
    struct GraphResources {
        RenderGraph::TextureHandle depth{};
        RenderGraph::TextureHandle hiz_current{};
    };

    struct MipDispatch {
        TextureView source{};
        TextureView destination{};
        uint2       source_size{};
        uint2       destination_size{};
        bool        is_mip0{false};
    };

    /**
     * Immutable, strongly-owned recording input.  RenderGraph workers must not
     * read RasterContext::textures while CommitHiZHistory may swap the current
     * and previous images on the render thread.
     */
    struct RecordParameters {
        TextureRef         source_owner{};
        TextureRef         destination_owner{};
        Array<MipDispatch> dispatches{};

        [[nodiscard]] bool IsEmpty() const noexcept {
            return dispatches.empty();
        }
    };

    explicit HiZBuildPass(RasterContext& context) : HiZBuildPass(context.manager) {}

    explicit HiZBuildPass(ShaderManager& manager) {
        pipeline = manager.Compute<HiZBuildPipeline>(
            "pipelines/raster/culling/HiZBuild.comp.hlsl"
        );
    }

    [[nodiscard]] RecordParameters Prepare(const TextureRef& destination) const {
        RecordParameters parameters{};
        if (!destination) {
            return parameters;
        }

        const uint32 mip_count = destination->GetNumMips();
        if (mip_count == 0) {
            return parameters;
        }

        parameters.destination_owner = destination;
        parameters.dispatches.reserve(mip_count);

        const uint2 mip0_size(destination->GetWidth(), destination->GetHeight());
        parameters.dispatches.emplace_back(MipDispatch{
            .destination      = parameters.destination_owner->GetView(0, 1),
            .source_size      = mip0_size,
            .destination_size = mip0_size,
            .is_mip0          = true,
        });

        for (uint32 mip = 1; mip < mip_count; ++mip) {
            parameters.dispatches.emplace_back(MipDispatch{
                .source = parameters.destination_owner->GetView(static_cast<uint8>(mip - 1), 1),
                .destination = parameters.destination_owner->GetView(static_cast<uint8>(mip), 1),
                .source_size = uint2(std::max(1u, mip0_size.x >> (mip - 1)), std::max(1u, mip0_size.y >> (mip - 1))),
                .destination_size = uint2(std::max(1u, mip0_size.x >> mip), std::max(1u, mip0_size.y >> mip)),
                .is_mip0 = false,
            });
        }
        return parameters;
    }

    void RecordMip(CommandList& cmd_list, const RecordParameters& parameters, uint32 mip) {
        const MipDispatch& dispatch = parameters.dispatches.at(mip);
        HiZBuildParam param{};
        param.src_size = dispatch.source_size;
        param.dst_size = dispatch.destination_size;
        param.is_mip0 = dispatch.is_mip0 ? 1u : 0u;
        cmd_list.Compute(pipeline, param, dispatch.source, dispatch.destination)
            .Dispatch(uint3((param.dst_size.x + 7) / 8, (param.dst_size.y + 7) / 8, 1), "Build Hi-Z");
    }

    void AddToGraph(
        RenderGraph& graph,
        const RasterContext& context,
        GraphResources resources
    );
    void AddToGraph(RenderGraph& graph, RecordParameters parameters, GraphResources resources);

private:
    HiZBuildPipeline pipeline;
};

} // namespace Moer::Render::Raster
