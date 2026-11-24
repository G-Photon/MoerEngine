#include "pipelines/raster/deferred/lighting/Ibl.hlsli"

[[vk::binding(0, 0)]] TextureCube<float4> diffuse_map;
[[vk::binding(1, 0)]] TextureCube<float4> specular_map;
[[vk::binding(2, 0)]] Texture2D<float4> brdf_lut;
[[vk::binding(3, 0)]] SamplerState linear_sampler;

float4 main(float4 position : SV_Position) : SV_Target {
    uint test = uint(position.x);
    float metallic = test == 0 ? 0.0f : 1.0f;
    float roughness = test == 3 ? 0.0f : test == 4 ? 1.0f : 0.5f;
    float ao = test == 2 ? 0.0f : 1.0f;
    return float4(EvaluateIbl(diffuse_map, specular_map, brdf_lut, linear_sampler,
        float3(0, 0, 1), float3(0, 0, 1), float3(0.8f, 0.4f, 0.2f), metallic,
        roughness, ao, 7.0f), 1.0f);
}
