#pragma once

// 过滤 AI 光照结果中的噪点，并写入光栅降噪输出目标。
#include "math/Function.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/post_process/ShaderParameters.h"

#include "RasterConfig.h"
#include "RasterResource.h"
#include "RasterTool.h"
#include "rendergraph/RenderGraph.h"

namespace Moer::Render::Raster {

class BilateralFilterDenoiserPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(BilateralFilterDenoiserPipeline);
    DEFINE_SHADER_CONSTANT(BilateralFilterDenoiserPipelineBindlessParam, param);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_ARGS(bdls, param);
};

/** 启用双边滤波降噪时，过滤 TensorRT 输出并写入专用降噪目标。 */
class BilateralFilterDenoiserPass {
public:
    struct GraphResources {
        RenderGraph::TextureHandle input{};
        RenderGraph::TextureHandle denoiser_output{};
    };

    struct RecordParameters {
        bool                                           enabled{false};
        BilateralFilterDenoiserPipelineBindlessParam   shader{};
        BindlessArrayRef                               bindless{};
        TextureWithHandle                              input{};
        TextureWithHandle                              output{};
    };

    BilateralFilterDenoiserPass(RasterContext& context) :
        BilateralFilterDenoiserPass(context.manager, RasterTextures::PostprocessOutputFormat) {}

    BilateralFilterDenoiserPass(ShaderManager& manager, EPixelFormat output_format) {
        GfxPsoCreateInfo pipeline_info(
            RHIRasterizeInfo::Preset(),
            {},
            {RHIColorAttachmentInfo::Preset(output_format)}
        );

        pipeline = manager.Raster()
                       .Vertex("core/utils/FullScreenQuad.hlsl")
                       .Pixel("pipelines/postprocess/denoise/BilateralFilterDenoiser.hlsl")
                       .Build<BilateralFilterDenoiserPipeline>(std::move(pipeline_info));
    }

    [[nodiscard]] RecordParameters Prepare(
        const RasterConfig& raster_config,
        Extent2D            extent
    ) const {
        RecordParameters parameters{};
        parameters.enabled  = raster_config.denoiser_mode != EDenoiserMode::NONE;
        if (!parameters.enabled) {
            return parameters;
        }
        parameters.shader.inv_resolution =
            1.0f / float2(extent.x, extent.y);
        parameters.shader.kernel_radius = raster_config.denoiser_bfd_kernel_radius;
        parameters.shader.spatial_sigma_square =
            raster_config.denoiser_bfd_spatial_sigma_square;
        parameters.shader.range_sigma_square =
            raster_config.denoiser_bfd_range_sigma_square;
        return parameters;
    }

    void Record(CommandList& cmd_list, const RecordParameters& parameters) {
        if (!parameters.enabled) {
            return;
        }
        cmd_list.Gfx(pipeline, parameters.bindless, parameters.shader)
            .Draw(
                "BilateralFilterDenoiser Pass",
                parameters.output.GetRect2D(),
                std::move(RasterTool::GetFullScreenDrawDatas()),
                ColorAttachment(parameters.output.tex)
            );
    }

    RENDER_API void AddToGraph(
        RenderGraph& graph,
        const RasterConfig& ui_config,
        Extent2D extent,
        GraphResources resources
    );

private:
    BilateralFilterDenoiserPipeline pipeline;
};

} // namespace Moer::Render::Raster
