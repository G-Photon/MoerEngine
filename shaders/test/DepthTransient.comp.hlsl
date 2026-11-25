[[vk::binding(0, 0)]] Texture2D<float> depth_texture;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> depth_values;

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    depth_texture.GetDimensions(width, height);
    if (id.x < width && id.y < height)
        depth_values[id.y * width + id.x] = depth_texture.Load(int3(id.xy, 0));
}
