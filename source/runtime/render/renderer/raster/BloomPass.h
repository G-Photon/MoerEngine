// 提取高亮区域、构建 Bloom 金字塔，并合成回场景颜色。
#pragma once

#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/post_process/ShaderParameters.h"

#include "RasterConfig.h"
#include "RasterResource.h"
#include "rendergraph/RenderGraph.h"

namespace Moer::Render::Raster {

class BloomPassPrefilterPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(BloomPassPrefilterPipeline);
    DEFINE_SHADER_TEX(input_tex);
    DEFINE_SHADER_SAMPLER(linear_sampler);
    DEFINE_SHADER_CONSTANT(BloomPrefilterParam, param);
    DEFINE_SHADER_ARGS(input_tex, linear_sampler, param);
};

class BloomPassUpSamplePipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(BloomPassUpSamplePipeline);
    DEFINE_SHADER_TEX(upsample_tex);
    DEFINE_SHADER_TEX(downsample_tex);
    DEFINE_SHADER_SAMPLER(linear_sampler);
    DEFINE_SHADER_CONSTANT(BloomUpsampleParam, param);
    DEFINE_SHADER_ARGS(upsample_tex, downsample_tex, linear_sampler, param);
};

class BloomPassDownSamplePipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(BloomPassDownSamplePipeline);
    DEFINE_SHADER_TEX(src_tex);
    DEFINE_SHADER_SAMPLER(linear_sampler);
    DEFINE_SHADER_CONSTANT(BloomDownsampleParam, param);
    DEFINE_SHADER_ARGS(src_tex, linear_sampler, param);
};

class BloomApplyPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(BloomApplyPipeline);
    DEFINE_SHADER_TEX(bloom_tex);
    DEFINE_SHADER_SAMPLER(linear_sampler);
    DEFINE_SHADER_CONSTANT(BloomApplyParam, param);
    DEFINE_SHADER_ARGS(bloom_tex, linear_sampler, param);
};

class RENDER_API BloomPass {
public:
    struct GraphResources {
        RenderGraph::TextureHandle input{};
        RenderGraph::TextureHandle downsample_chain{};
        RenderGraph::TextureHandle upsample_chain{};
    };

    BloomPass(RasterContext& context);
    BloomPass(ShaderManager& manager, EPixelFormat scene_format);

    RenderGraph::PassHandle AddRecordPasses(
        RenderGraph&   graph,
        GraphResources resources,
        uint32         mip_count
    );
    GraphResources AddToGraph(
        RenderGraph&               graph,
        bool                       enabled,
        RenderGraph::TextureHandle input,
        Extent2D                   extent
    );

private:
    void RecordPrefilter(CommandList& commands, const TextureRef& input, const TextureRef& downsample);
    void RecordDownsample(CommandList& commands, const TextureRef& downsample, uint32 mip);
    void RecordUpsample(
        CommandList&      commands,
        const TextureRef& source,
        const TextureRef& downsample,
        const TextureRef& upsample,
        uint32            mip
    );
    void RecordApply(CommandList& commands, const TextureRef& input, const TextureRef& bloom);

    BloomPassPrefilterPipeline  prefilter_pipeline;
    BloomPassUpSamplePipeline   upsample_pipeline;
    BloomPassDownSamplePipeline downsample_pipeline;
    BloomApplyPipeline          apply_pipeline;
};
} // namespace Moer::Render::Raster
