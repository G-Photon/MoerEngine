#pragma once

// 对几何缓冲和方向光阴影遮罩应用延迟 PBR 光照。

#include "misc/Types.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/lighting_pass/Constants.h"

#include "RasterConfig.h"
#include "RasterResource.h"
#include "RasterTool.h"
#include "rendergraph/RenderGraph.h"

namespace Moer::Render::Raster {

class PbrMaterialShadingPipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(PbrMaterialShadingPipeline);
    DEFINE_SHADER_BUFFER(lighting_data);
    DEFINE_SHADER_BINDLESS_ARRAY(bdls);
    DEFINE_SHADER_TEX(gbuffer_base_color);
    DEFINE_SHADER_TEX(gbuffer_normal);
    DEFINE_SHADER_TEX(gbuffer_material);
    DEFINE_SHADER_TEX(gbuffer_depth);
    DEFINE_SHADER_TEX(shadow_mask);
    DEFINE_SHADER_SAMPLER(gbuffer_sampler);
    DEFINE_SHADER_SAMPLER(depth_sampler);
    DEFINE_SHADER_CONSTANTS(LIGHTING_PASS_CONSTANTS);
    DEFINE_SHADER_ARGS(lighting_data, bdls, param, extra_ambient_intensity,
        gbuffer_base_color, gbuffer_normal, gbuffer_material, gbuffer_depth, shadow_mask,
        gbuffer_sampler, depth_sampler);
};

class LightingPass {
public:
    struct GraphResources {
        RenderGraph::TextureHandle base_color{};
        RenderGraph::TextureHandle normal{};
        RenderGraph::TextureHandle metal_rough_ao{};
        RenderGraph::TextureHandle depth{};
        RenderGraph::TextureHandle shadow_mask{};
        RenderGraph::BufferHandle  lighting_data{};
        RenderGraph::TextureHandle cubemap{};
        RenderGraph::TokenHandle   probe_volume{};
        RenderGraph::TextureHandle lighting_output{};
        RenderGraph::BufferHandle  scene_lights{};
        Array<TextureRef>          gaussian_shadow_textures{};
        BufferRef                  gaussian_shadow_data{};
    };

    /**
     * Prepare captures scene inputs. Record resolves GBuffer attachments from
     * the graph after transient allocation; commands retain their physical owners.
     */
    struct RecordParameters {
        MaterialPassBindlessParam pass_param{};
        float                     extra_ambient_intensity = 0.0f;
        BufferRef                 lighting_data{};
        BufferRef                 light_buffer_owner{};
        BindlessArrayRef          bindless{};
        TextureRef                base_color_owner{};
        TextureRef                normal_owner{};
        TextureRef                metal_rough_ao_owner{};
        TextureRef                depth_owner{};
        TextureRef                cubemap_owner{};
        TextureRef                shadow_mask_owner{};
        TextureRef                output{};
        Rect2D                    render_area{};
    };

    explicit LightingPass(RasterContext& context) : LightingPass(context.manager) {}

    explicit LightingPass(ShaderManager& manager) {
        GfxPsoCreateInfo pipeline_info(
            RHIRasterizeInfo::Preset(),
            {},
            {RHIColorAttachmentInfo::Preset(RasterTextures::LightingOutputFormat)}
        );

        pipeline = manager.Raster()
                       .Vertex("core/utils/FullScreenQuad.hlsl")
                       .Pixel("pipelines/raster/deferred/lighting/RasterLightingPass.frag.hlsl")
                       .Build<PbrMaterialShadingPipeline>(std::move(pipeline_info));
    }

    [[nodiscard]] RecordParameters
    Prepare(const RasterContext& context, const RasterConfig& ui_config) const {
        RecordParameters parameters{};
        auto&            pass_param = parameters.pass_param;
        pass_param.extra_ambient_color     = ui_config.shading_extra_ambient_color;
        parameters.extra_ambient_intensity = ui_config.shading_extra_ambient_intensity;
        pass_param.enable_extra_ambient    = ui_config.shading_enable_extra_ambient;
        pass_param.shading_mode            = static_cast<uint32>(ui_config.shading_mode);

        const auto& gpu_scene_res     = context.GetGpuSceneRes();
        pass_param.light_buf_hdl      = gpu_scene_res.light_buf.hdl;
        pass_param.cubemap_handle     = context.textures.cubemap_tex.hdl;

        parameters.lighting_data        = context.lighting_data_buffer.buf;
        parameters.light_buffer_owner   = gpu_scene_res.light_buf.buf;
        parameters.bindless             = context.bdls;
        parameters.cubemap_owner        = context.textures.cubemap_tex.tex;
        const auto size = context.GetResolution();
        parameters.render_area          = Rect2D(0, 0, size.x, size.y);
        return parameters;
    }

    void Record(CommandList& cmd_list, const RecordParameters& parameters) {
        cmd_list
            .Gfx(
                pipeline,
                parameters.lighting_data,
                parameters.bindless,
                parameters.pass_param,
                parameters.extra_ambient_intensity,
                parameters.base_color_owner->GetView(),
                parameters.normal_owner->GetView(),
                parameters.metal_rough_ao_owner->GetView(),
                parameters.depth_owner->GetView(),
                parameters.shadow_mask_owner->GetView(),
                Sampler{SF_LINEAR, SAM_REPEAT},
                Sampler{SF_NEAREST, SAM_CLAMP_TO_EDGE}
            )
            .Draw(
                "Lighting Pass",
                parameters.render_area,
                std::move(RasterTool::GetFullScreenDrawDatas()),
                ColorAttachment(parameters.output)
            );
    }

    void AddToGraph(
        RenderGraph& graph,
        const RasterContext& context,
        const RasterConfig& ui_config,
        GraphResources resources
    );

private:
    PbrMaterialShadingPipeline pipeline;
};

} // namespace Moer::Render::Raster
