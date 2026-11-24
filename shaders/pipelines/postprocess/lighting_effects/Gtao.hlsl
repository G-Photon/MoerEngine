#include "core/common/Bindless.hlsl"
#include "core/common/Common.hlsl"
BINDLESS_BINDINGS(3, 2, 4, 5)
#include "shared/raster/ShaderParameters.h"
DEFINE_SHADER_CONSTANT(Moer::GtaoPipelineBindlessParam, param)
#include "pipelines/postprocess/lighting_effects/AoCommon.hlsl"
#include "thirdparty/XeGTAO/HorizonIntegral.hlsli"

float CalculateGtao(float2 uv) {
    AoSurface center = LoadAoSurface(uv, moer_constants.param.depth_tex, moer_constants.param.normal_tex);
    float radius = moer_constants.param.radius;
    if (!IsAoSurfaceValid(center) || radius <= 1e-6f || moer_constants.param.intensity <= 0.0f) return 1.0f;
    float3 position = WorldPosFromDepth(center.depth, center.uv, moer_constants.param.clip2world);
    float3 view_delta = moer_constants.param.camera_position - position;
    if (dot(view_delta, view_delta) < 1e-12f) return 1.0f;
    float3 V = normalize(view_delta);
    if (dot(center.normal, V) <= 0.0f) return 1.0f;

    float3 screen_dx = WorldPosFromDepth(center.depth,
        center.uv + float2(moer_constants.param.inv_resolution.x, 0), moer_constants.param.clip2world) - position;
    float3 screen_dy = WorldPosFromDepth(center.depth,
        center.uv + float2(0, moer_constants.param.inv_resolution.y), moer_constants.param.clip2world) - position;
    float3 projected_dx = screen_dx - V * dot(screen_dx, V);
    float3 projected_dy = screen_dy - V * dot(screen_dy, V);
    float xx = dot(projected_dx, projected_dx), yy = dot(projected_dy, projected_dy);
    float xy = dot(projected_dx, projected_dy);
    float determinant = xx * yy - xy * xy;
    if (xx <= 0.0f || determinant <= 0.0f) return 1.0f;
    float3 basis_x = projected_dx / sqrt(xx);
    float3 basis_y = cross(V, basis_x);

    uint slices = clamp(moer_constants.param.slice_count, 1u, 16u);
    uint steps = clamp(moer_constants.param.step_count, 1u, 16u);
    float2 pixel = floor(uv / moer_constants.param.inv_resolution);
    float noise = frac(52.9829189f * frac(dot(pixel, float2(0.06711056f, 0.00583715f))));
    float visibility = 0;
    [loop] for (uint slice = 0; slice < slices; ++slice) {
        float phi = PI * (float(slice) + noise) / float(slices);
        float3 tangent = basis_x * cos(phi) + basis_y * sin(phi);
        // Integrate uniform angles around the view vector, then invert the local
        // projection Jacobian to find the corresponding screen sampling direction.
        float tx = dot(tangent, projected_dx), ty = dot(tangent, projected_dy);
        float2 screen_direction = normalize(float2(tx * yy - ty * xy, ty * xx - tx * xy));
        float3 screen_delta = screen_dx * screen_direction.x + screen_dy * screen_direction.y;
        float3 axis = cross(tangent, V);
        float3 projected_normal = center.normal - axis * dot(center.normal, axis);
        float projected_length = length(projected_normal);
        float cos_normal = clamp(dot(projected_normal, V) / max(projected_length, 1e-8f), -1.0f, 1.0f);
        float normal_angle = atan2(dot(projected_normal, tangent), dot(projected_normal, V));
        float2 horizon = cos(normal_angle + float2(PI * 0.5f, -PI * 0.5f));
        float2 low_horizon = horizon;
        float2 radius_uv = screen_direction * moer_constants.param.inv_resolution *
            (radius / max(length(screen_delta), 1e-8f));
        [loop] for (uint step = 0; step < steps; ++step) {
            float fraction = (float(step) + 0.5f + noise * 0.5f) / (float(steps) + 0.5f);
            float2 offset = radius_uv * fraction * fraction;
            [unroll] for (uint side = 0; side < 2; ++side) {
                float2 sample_uv = center.uv + (side == 0 ? offset : -offset);
                if (any(sample_uv < 0.0f) || any(sample_uv >= 1.0f)) continue;
                AoSurface sample_surface = LoadAoSurface(sample_uv, moer_constants.param.depth_tex,
                    moer_constants.param.normal_tex);
                if (!IsAoSurfaceValid(sample_surface)) continue;
                float3 delta = WorldPosFromDepth(sample_surface.depth, sample_surface.uv,
                    moer_constants.param.clip2world) - position;
                // Texel snapping can move a tap out of its intended slice. Points
                // on/below the tangent plane cannot occlude the normal hemisphere.
                if (dot(delta, center.normal) <= radius * 1e-5f) continue;
                float distance_squared = dot(delta, delta);
                if (distance_squared < 1e-12f) continue;
                float distance = sqrt(distance_squared);
                float weight = 1.0f - smoothstep(radius * 0.2f, radius, distance);
                float sample_cos = dot(delta / distance, V);
                horizon[side] = max(horizon[side], lerp(low_horizon[side], sample_cos, weight));
            }
        }
        visibility += GtaoHorizonIntegral(projected_length, cos_normal, normal_angle, horizon);
    }
    return saturate(1.0f - (1.0f - saturate(visibility / float(slices))) * moer_constants.param.intensity);
}

AoOutput main(float2 uv : TEXCOORD0) {
    AoOutput output;
    output.ambient_only = CalculateGtao(uv);
    output.camera_motion_vector = GetCameraMotionVector(uv);
    return output;
}
