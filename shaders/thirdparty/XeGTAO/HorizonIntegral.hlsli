// Horizon visibility integral adapted from Intel XeGTAO's MainPass.
// Source: GameTechDev/XeGTAO, a5b1686c7ea37788eeb3576b5be47f7c03db532c,
// Source/Rendering/Shaders/XeGTAO.hlsli. See LICENSE in this directory.
#ifndef MOER_GTAO_HORIZON_INTEGRAL
#define MOER_GTAO_HORIZON_INTEGRAL

float GtaoHorizonIntegral(float projected_length, float cos_normal, float normal_angle, float2 horizon_cos) {
    const float half_pi = 1.57079632679f;
    float h0 = -acos(clamp(horizon_cos.y, -1.0f, 1.0f));
    float h1 = acos(clamp(horizon_cos.x, -1.0f, 1.0f));
    h0 = normal_angle + clamp(h0 - normal_angle, -half_pi, half_pi);
    h1 = normal_angle + clamp(h1 - normal_angle, -half_pi, half_pi);
    float arc0 = (cos_normal + 2.0f * h0 * sin(normal_angle) - cos(2.0f * h0 - normal_angle)) * 0.25f;
    float arc1 = (cos_normal + 2.0f * h1 * sin(normal_angle) - cos(2.0f * h1 - normal_angle)) * 0.25f;
    return projected_length * (arc0 + arc1);
}

#endif
