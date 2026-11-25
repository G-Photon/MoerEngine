#pragma once

#include "math/Function.h"
#include "scene/camera/Camera.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/post_process/ShaderParameters.h"

#include "RasterConfig.h"
#include "RasterResource.h"
#include "RasterTool.h"
#include "rendergraph/RenderGraph.h"

namespace Moer::Render::Raster {

class SsrPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(SsrPipeline);
    DEFINE_SHADER_CONSTANT(SsrPipelineBindlessParam, param);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_ARGS(bdls, param);
};

/**
 * MARK: SSR Pass
 * 
 * GUI Help:
 *   You can set `Roughness Threshold` and `Metallic Threshold` to control which material will be reflected.
 *   If you want all materials to be reflected, set `Roughness Threshold` to 0.0 and `Metallic Threshold` to 1.0.
 *   `Force Ground Enable SSR` will force the ground to be reflected, which is useful for testing.
 *   TODO: move the above text to gui
 * 
 * TODO: use HiZ buffer to accelerate SSR (now, a simple heuristic and binary search is used for SSR)
 * TODO: glossy ssr
 * TODO: performance optimization
 * TODO: fix some artifacts (jitter)
 */
class SsrPass {
public:
    struct GraphResources {
        RenderGraph::TextureHandle input{};
        RenderGraph::TextureHandle normal{};
        RenderGraph::TextureHandle depth{};
        RenderGraph::TextureHandle metal_rough_ao{};
        RenderGraph::TextureHandle ssr_output{};
    };

    struct RecordParameters {
        bool                         enabled{false};
        SsrPipelineBindlessParam     shader{};
        BindlessArrayRef             bindless{};
        TextureWithHandle            input{};
        TextureWithHandle            output{};
    };

    SsrPass(RasterContext& context) : SsrPass(context.manager, RasterTextures::PostprocessOutputFormat) {}

    SsrPass(ShaderManager& manager, EPixelFormat output_format) {
        GfxPsoCreateInfo pso_full_screen_info(
            RHIRasterizeInfo::Preset(),
            {},
            {RHIColorAttachmentInfo::Preset(output_format)}
        );

        ssr_pipeline = manager.Raster()
                           .Vertex("core/utils/FullScreenQuad.hlsl")
                           .Pixel("pipelines/postprocess/lighting_effects/Ssr.hlsl")
                           .Build<SsrPipeline>(std::move(pso_full_screen_info));
    }

    [[nodiscard]] RecordParameters Prepare(
        const RasterConfig& ui_config,
        const Camera&       camera,
        Extent2D            extent
    ) const {
        RecordParameters parameters{};
        parameters.enabled  = ui_config.ssr_is_ssr_enabled != 0;
        if (!parameters.enabled) {
            return parameters;
        }

        auto& param = parameters.shader;
        param.clip2world                     = Transpose(camera.GetViewProjectionMatrixInv());
        param.world2clip                     = Transpose(camera.GetViewProjectionMatrix());
        param.camera_position                = camera.GetPosition();
        param.near_clip                      = camera.GetNearClip();
        param.resolution                     = float2(extent.x, extent.y);
        param.far_clip                       = camera.GetFarClip();
        param.ssr_roughness_threshold        = ui_config.ssr_roughness_threshold;
        param.ssr_metallic_threshold         = ui_config.ssr_metallic_threshold;
        param.ssr_step_base                  = ui_config.ssr_step_base;
        param.ssr_sample_count               = ui_config.ssr_sample_count;
        param.ssr_is_enable_jitter           = ui_config.ssr_is_enable_jitter;
        param.ssr_is_force_ground_enable_ssr = ui_config.ssr_is_force_ground_enable_ssr;
        return parameters;
    }

    void Record(CommandList& cmd_list, const RecordParameters& parameters) {
        if (!parameters.enabled) {
            return;
        }
        cmd_list.Gfx(ssr_pipeline, parameters.bindless, parameters.shader)
            .Draw(
                "SSR Pass",
                parameters.output.GetRect2D(),
                std::move(RasterTool::GetFullScreenDrawDatas()),
                ColorAttachment(parameters.output.tex)
            );
    }

    RENDER_API void AddToGraph(
        RenderGraph& graph,
        const RasterConfig& ui_config,
        const Camera& camera,
        Extent2D extent,
        GraphResources resources
    );

private:
    SsrPipeline ssr_pipeline;
};

} // namespace Moer::Render::Raster
