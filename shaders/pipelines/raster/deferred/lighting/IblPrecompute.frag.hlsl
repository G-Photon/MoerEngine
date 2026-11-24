#include "core/math/STL.hlsli"
#include "shared/raster/ShaderParameters.h"

[[vk::binding(0, 0)]] TextureCube<float4> environment_map;
[[vk::binding(1, 0)]] SamplerState environment_sampler;
[[vk::push_constant]] ConstantBuffer<Moer::IblPrecomputeParam> param;

float3 CubeDirection(float2 uv, uint face) {
    float2 p = uv * 2.0f - 1.0f;
    if (face == 0) return normalize(float3(1, -p.y, -p.x));
    if (face == 1) return normalize(float3(-1, -p.y, p.x));
    if (face == 2) return normalize(float3(p.x, 1, p.y));
    if (face == 3) return normalize(float3(p.x, -1, -p.y));
    if (face == 4) return normalize(float3(p.x, -p.y, 1));
    return normalize(float3(-p.x, -p.y, -1));
}

float GeometrySchlick(float cosine, float k) {
    return cosine / max(cosine * (1.0f - k) + k, 1e-6f);
}

float2 IntegrateBrdf(float NoV, float roughness) {
    float3 V = float3(sqrt(saturate(1.0f - NoV * NoV)), 0, NoV);
    float2 result = 0;
    float k = roughness * roughness * 0.5f;
    [loop] for (uint i = 0; i < param.sample_count; ++i) {
        float3 H = STL::ImportanceSampling::NDF::GetRay(
            STL::Sequence::Hammersley2D(i, param.sample_count), roughness);
        float VoH = saturate(dot(V, H));
        float3 L = 2.0f * VoH * H - V;
        float NoL = saturate(L.z);
        if (NoL > 0) {
            float visibility = GeometrySchlick(NoV, k) * GeometrySchlick(NoL, k) * VoH /
                               max(H.z * NoV, 1e-6f);
            float Fc = pow(1.0f - VoH, 5.0f);
            result += float2(1.0f - Fc, Fc) * visibility;
        }
    }
    return result / float(param.sample_count);
}

float4 main(float2 uv : TEXCOORD0) : SV_Target {
    if (param.mode == 2) return float4(IntegrateBrdf(uv.x, uv.y), 0, 1);

    float3 N = CubeDirection(uv, param.face);
    if (param.mode == 1 && param.roughness == 0) {
        return float4(environment_map.SampleLevel(environment_sampler, N, 0).rgb, 1);
    }
    float3x3 basis = STL::Geometry::GetBasis(N);
    float3 radiance = 0;
    float weight = 0;
    [loop] for (uint i = 0; i < param.sample_count; ++i) {
        float2 xi = STL::Sequence::Hammersley2D(i, param.sample_count);
        float3 local = param.mode == 0 ? STL::ImportanceSampling::Cosine::GetRay(xi) :
                                        STL::ImportanceSampling::NDF::GetRay(xi, param.roughness);
        float3 direction = mul(local, basis);
        if (param.mode == 1) direction = reflect(-N, direction);
        float NoL = param.mode == 0 ? 1.0f : saturate(dot(N, direction));
        radiance += environment_map.SampleLevel(environment_sampler, direction, 0).rgb * NoL;
        weight += NoL;
    }
    // Cosine sampling estimates irradiance / pi directly; the specular filter
    // uses the normalized N=V GGX convolution from the split-sum approximation.
    return float4(radiance / max(weight, 1e-6f), 1);
}
