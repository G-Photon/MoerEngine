[[vk::binding(0, 0)]] StructuredBuffer<float> depth_values;

float main(float4 position : SV_Position) : SV_Depth {
    const uint2 pixel = uint2(position.xy);
    return depth_values[1 + pixel.y * uint(depth_values[0]) + pixel.x];
}
