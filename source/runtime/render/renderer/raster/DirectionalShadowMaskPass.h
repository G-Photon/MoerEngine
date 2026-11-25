// 构建延迟光照使用的屏幕空间方向光阴影因子。
#pragma once

#include "misc/Types.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/lighting_pass/ShaderParameters.h"

#include "RasterResource.h"
#include "rendergraph/RenderGraph.h"

namespace Moer::Render::Raster {

class DirectionalShadowMaskPassPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(DirectionalShadowMaskPassPipeline);
    DEFINE_SHADER_BUFFER(lighting_data);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_TEX(gbuffer_normal);
    DEFINE_SHADER_TEX(gbuffer_depth);
    DEFINE_SHADER_SAMPLER(normal_sampler);
    DEFINE_SHADER_SAMPLER(depth_sampler);
    DEFINE_SHADER_ARGS(lighting_data, bdls, gbuffer_normal, gbuffer_depth, normal_sampler, depth_sampler);
};

class RENDER_API DirectionalShadowMaskPass {
public:
    struct GraphResources {
        RenderGraph::TextureHandle normal{};
        RenderGraph::TextureHandle depth{};
        RenderGraph::BufferHandle  lighting_data{};
        RenderGraph::TokenHandle   shadow_maps{};
        RenderGraph::TextureHandle shadow_mask{};
    };

    struct RecordParameters {
        TextureRef       normal_owner{};
        TextureRef       depth_owner{};
        StaticArray<DepthBufferRef, CSM_MAX_CASCADES> cascade_shadow_owners{};
        StaticArray<TextureRef, RasterContext::PointShadowData::MAX_POINT_SHADOWS>
            point_shadow_owners{};
        BufferRef        lighting_data{};
        BindlessArrayRef bindless{};
        TextureRef       output{};
        Rect2D           render_area{};
    };

    DirectionalShadowMaskPass(RasterContext& context);
    explicit DirectionalShadowMaskPass(ShaderManager& manager);

    [[nodiscard]] RecordParameters Prepare(const RasterContext& context) const;
    void Record(CommandList& cmd_list, const RecordParameters& parameters);
    void AddToGraph(
        RenderGraph& graph,
        const RasterContext& context,
        GraphResources resources,
        std::span<const RenderGraph::SetupPassHandle> setup_dependencies = {}
    );

private:
    DirectionalShadowMaskPassPipeline pipeline;
};
} // namespace Moer::Render::Raster
