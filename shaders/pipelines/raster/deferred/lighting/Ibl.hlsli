#ifndef MOER_RASTER_IBL_HLSLI
#define MOER_RASTER_IBL_HLSLI

float3 EvaluateIbl(
    TextureCube<float4> diffuse_map, TextureCube<float4> specular_map,
    Texture2D<float4> brdf_lut, SamplerState linear_sampler,
    float3 N, float3 V, float3 albedo, float metallic, float roughness, float ao,
    float max_specular_lod
) {
    float NoV = saturate(dot(N, V));
    float3 f0 = lerp(0.04f.xxx, albedo, metallic);
    float3 fresnel = f0 + (max((1.0f - roughness).xxx, f0) - f0) * pow(1.0f - NoV, 5.0f);
    float3 diffuse = diffuse_map.SampleLevel(linear_sampler, N, 0).rgb * albedo *
                     (1.0f - metallic) * (1.0f - fresnel);
    float3 prefiltered = specular_map.SampleLevel(
        linear_sampler, reflect(-V, N), roughness * max_specular_lod).rgb;
    float2 brdf = brdf_lut.SampleLevel(linear_sampler, float2(NoV, roughness), 0).rg;
    return (diffuse + prefiltered * (f0 * brdf.x + brdf.y)) * ao;
}

#endif
