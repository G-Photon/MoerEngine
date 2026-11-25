#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)
#include "pipelines/RasterCommon.hlsli"
#include "pipelines/raster/deferred/lighting/shadows/Shadows.hlsli"
#include "shared/raster/ShaderParameters.h"

[[vk::binding(0, 0)]] ConstantBuffer<Moer::LightingData> lighting_data;
[[vk::binding(1, 0)]] Texture2D<float4> gbuffer_normal;
[[vk::binding(2, 0)]] Texture2D<float> gbuffer_depth;
[[vk::binding(3, 0)]] SamplerState normal_sampler;
[[vk::binding(4, 0)]] SamplerState depth_sampler;
DEFINE_SHADER_BINDLESS_CONSTANTS()

float main(float2 in_uv : TEXCOORD0) : SV_TARGET {
    // MARK: GBuffer
    float  depth  = gbuffer_depth.Sample(depth_sampler, in_uv);

    // 天空/背景像素在 Reverse-Z 下 depth == 0，无需 PCSS，直接返回全亮
    if (depth < 1e-6)
        return 1.0;

    float3 normal = normalize(Raster::UnpackNormal(gbuffer_normal.Sample(normal_sampler, in_uv).rgb));
    float3 position = WorldPosFromDepth(depth, in_uv, lighting_data.clip2world);

    float shadow = calculate_shadow(lighting_data, position, in_uv, normal);

    return shadow;
}
