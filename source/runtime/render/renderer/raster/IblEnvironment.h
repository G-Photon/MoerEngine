#pragma once

#include "rendergraph/RenderGraph.h"
#include "shader/ShaderPipeline.h"
#include "shader/ShaderResourceManager.h"
#include "shaderheaders/shared/raster/lighting_pass/ShaderParameters.h"

namespace Moer::Render::Raster {

class IblPrecomputePipeline : public RasterPipeline {
public:
    DEFINE_RASTER_PIPELINE_CLASS(IblPrecomputePipeline);
    DEFINE_SHADER_TEX(environment_map);
    DEFINE_SHADER_SAMPLER(environment_sampler);
    DEFINE_SHADER_CONSTANT(IblPrecomputeParam, param);
    DEFINE_SHADER_ARGS(environment_map, environment_sampler, param);
};

// Environment-dependent filtering is immutable between source changes. In-flight
// commands retain the old textures when a new environment replaces this cache.
class RENDER_API IblEnvironment {
public:
    IblEnvironment(RenderDevice& device, ShaderManager& manager);
    // The immutable cubemap must reach Sampled on Graphics before this work.
    // Returned textures are consumed by later Graphics submissions, without a host wait.
    void Update(TextureRef environment_map);

    [[nodiscard]] const TextureRef& GetDiffuse() const {
        return diffuse;
    }
    [[nodiscard]] const TextureRef& GetSpecular() const {
        return specular;
    }
    [[nodiscard]] const TextureRef& GetBrdf() const {
        return brdf;
    }

private:
    RenderDevice&         device;
    IblPrecomputePipeline pipeline;
    TextureRef            source;
    TextureRef            diffuse;
    TextureRef            specular;
    TextureRef            brdf;
};

} // namespace Moer::Render::Raster
