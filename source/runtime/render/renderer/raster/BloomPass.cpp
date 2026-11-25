#include "BloomPass.h"
#include "misc/Types.h"

// 实现 Bloom 的预过滤、下采样、上采样和加法合成阶段。
#include "RasterResource.h"
#include "RasterTextures.h"
#include "RasterTool.h"

#include <algorithm>

namespace Moer::Render::Raster {

namespace {
Rect2D MipRect(const TextureRef& texture, uint32 mip = 0) {
    return Rect2D(0, 0, std::max(1u, texture->GetWidth() >> mip), std::max(1u, texture->GetHeight() >> mip));
}
} // namespace

BloomPass::BloomPass(RasterContext& context) :
    BloomPass(context.manager, RasterTextures::LightingOutputFormat) {}

BloomPass::BloomPass(ShaderManager& manager, EPixelFormat scene_format) {
    const RHIRasterizeInfo rasterize_info = RHIRasterizeInfo::Preset();
    const EPixelFormat     bloom_format   = PF_B10G11R11_UFLOAT_PACK32;

    GfxPsoCreateInfo prefilter_info(
        rasterize_info, {}, {RHIColorAttachmentInfo::Preset<Moer::Render::Blend::NONE>(bloom_format)}
    );
    prefilter_pipeline = manager.Raster()
                             .Vertex("core/utils/FullScreenQuad.hlsl")
                             .Pixel("pipelines/postprocess/color/BloomPrefilter.frag.hlsl")
                             .Build<BloomPassPrefilterPipeline>(std::move(prefilter_info));

    GfxPsoCreateInfo downsample_info(
        rasterize_info, {}, {RHIColorAttachmentInfo::Preset<Moer::Render::Blend::NONE>(bloom_format)}
    );
    downsample_pipeline = manager.Raster()
                              .Vertex("core/utils/FullScreenQuad.hlsl")
                              .Pixel("pipelines/postprocess/color/BloomDownSample.frag.hlsl")
                              .Build<BloomPassDownSamplePipeline>(std::move(downsample_info));

    GfxPsoCreateInfo upsample_info(
        rasterize_info, {}, {RHIColorAttachmentInfo::Preset<Moer::Render::Blend::NONE>(bloom_format)}
    );
    upsample_pipeline = manager.Raster()
                            .Vertex("core/utils/FullScreenQuad.hlsl")
                            .Pixel("pipelines/postprocess/color/BloomUpSample.frag.hlsl")
                            .Build<BloomPassUpSamplePipeline>(std::move(upsample_info));

    GfxPsoCreateInfo apply_info(
        rasterize_info,
        {},
        {RHIColorAttachmentInfo::Preset<Moer::Render::Blend::ADDITIVE_BLEND>(scene_format)}
    );
    apply_pipeline = manager.Raster()
                         .Vertex("core/utils/FullScreenQuad.hlsl")
                         .Pixel("pipelines/postprocess/color/BloomApply.frag.hlsl")
                         .Build<BloomApplyPipeline>(std::move(apply_info));
}

void BloomPass::RecordPrefilter(CommandList& cmd_list, const TextureRef& input, const TextureRef& downsample) {
    const BloomPrefilterParam param{.threshold = 20.0f, .knee = 0.5f};
    cmd_list
        .Gfx(
            prefilter_pipeline,
            input->GetView(0, 1),
            Sampler{SF_LINEAR, SAM_CLAMP_TO_EDGE},
            param
        )
        .Draw(
            "Bloom Prefilter Pass",
            MipRect(downsample),
            std::move(RasterTool::GetFullScreenDrawDatas()),
            ColorAttachment{
                downsample, EAttachmentAction::AC_CLEAR_STORE, float4(0, 0, 0, 0), 0
            }
        );
}

void BloomPass::RecordDownsample(CommandList& cmd_list, const TextureRef& downsample, uint32 mip) {
    const auto                 size = MipRect(downsample, mip - 1).extent;
    const BloomDownsampleParam param{.inv_size = float2(1.0f / size.x, 1.0f / size.y)};
    cmd_list.Gfx(downsample_pipeline, downsample->GetView(mip - 1, 1), Sampler{SF_LINEAR, SAM_CLAMP_TO_EDGE}, param)
        .Draw(
            std::format("Downsample Pass #{}", mip),
            MipRect(downsample, mip),
            std::move(RasterTool::GetFullScreenDrawDatas()),
            ColorAttachment{downsample, AC_CLEAR_STORE, float4(0), mip}
        );
}

void BloomPass::RecordUpsample(
    CommandList&      cmd_list,
    const TextureRef& source,
    const TextureRef& downsample,
    const TextureRef& upsample,
    uint32            mip
) {
    const auto               size = MipRect(source, mip + 1).extent;
    const BloomUpsampleParam param{
        .filter_radius = 1.2f, .inv_size = float2(1.0f / size.x, 1.0f / size.y)
    };
    cmd_list
        .Gfx(
            upsample_pipeline,
            source->GetView(mip + 1, 1),
            downsample->GetView(mip, 1),
            Sampler{SF_LINEAR, SAM_CLAMP_TO_EDGE},
            param
        )
        .Draw(
            std::format("Bloom Upsample Pass #{}", mip),
            MipRect(upsample, mip),
            std::move(RasterTool::GetFullScreenDrawDatas()),
            ColorAttachment{upsample, AC_CLEAR_STORE, float4(0), mip}
        );
}

void BloomPass::RecordApply(CommandList& cmd_list, const TextureRef& input, const TextureRef& bloom) {
    const BloomApplyParam param{.bloom_intensity = 0.1f};
    cmd_list
        .Gfx(
            apply_pipeline,
            bloom->GetView(0, 1),
            Sampler{SF_LINEAR, SAM_CLAMP_TO_EDGE},
            param
        )
        .Draw(
            "Apply Bloom to Scene",
            MipRect(input),
            std::move(RasterTool::GetFullScreenDrawDatas()),
            ColorAttachment{input, EAttachmentAction::AC_LOAD_STORE, float4(0, 0, 0, 0), 0}
        );
}

} // namespace Moer::Render::Raster
