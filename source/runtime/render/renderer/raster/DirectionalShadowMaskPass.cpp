// 实现全屏方向光阴影遮罩 Pass。
#include "DirectionalShadowMaskPass.h"

#include "RasterTool.h"

namespace Moer::Render::Raster {
DirectionalShadowMaskPass::DirectionalShadowMaskPass(RasterContext& context) :
    DirectionalShadowMaskPass(context.manager) {}

DirectionalShadowMaskPass::DirectionalShadowMaskPass(ShaderManager& manager) {
    GfxPsoCreateInfo pso_full_screen_info(
        RHIRasterizeInfo::Preset(),
        {},
        {RHIColorAttachmentInfo::Preset(RasterTextures::ShadowMaskFormat)}
    );

    pipeline = manager.Raster()
                   .Vertex("core/utils/FullScreenQuad.hlsl")
                   .Pixel("pipelines/raster/deferred/lighting/shadows/ShadowMask.frag.hlsl")
                   .Build<DirectionalShadowMaskPassPipeline>(std::move(pso_full_screen_info));
}

DirectionalShadowMaskPass::RecordParameters
DirectionalShadowMaskPass::Prepare(const RasterContext& context) const {
    RecordParameters parameters{
        .lighting_data = context.lighting_data_buffer.buf,
        .bindless      = context.bdls,
    };
    const auto size = context.GetResolution();
    parameters.render_area = Rect2D(0, 0, size.x, size.y);
    for (size_t index = 0; index < parameters.cascade_shadow_owners.size(); ++index) {
        parameters.cascade_shadow_owners[index] = context.csm_data.shadow_map_textures[index].tex;
    }
    for (size_t index = 0; index < parameters.point_shadow_owners.size(); ++index) {
        parameters.point_shadow_owners[index] = context.point_shadow_data.shadow_cubes[index].tex;
    }
    return parameters;
}

void DirectionalShadowMaskPass::Record(
    CommandList&            cmd_list,
    const RecordParameters& parameters
) {
    cmd_list
        .Gfx(pipeline, parameters.lighting_data, parameters.bindless,
            parameters.normal_owner->GetView(), parameters.depth_owner->GetView(),
            Sampler{SF_LINEAR, SAM_REPEAT}, Sampler{SF_NEAREST, SAM_CLAMP_TO_EDGE})
        .Draw(
            "Directional Shadow Mask Pass",
            parameters.render_area,
            std::move(RasterTool::GetFullScreenDrawDatas()),
            ColorAttachment(parameters.output)
        );
}

} // namespace Moer::Render::Raster
