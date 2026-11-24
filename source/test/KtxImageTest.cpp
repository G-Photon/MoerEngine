#include "scene/loader/io/ImageIO.h"
#include "log/LogSystem.h"
#include <ktx.h>
#include <gl_format.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Texture = std::unique_ptr<ktxTexture, decltype(&ktxTexture_Destroy)>;

void Require(ktxResult result) {
    if (result != KTX_SUCCESS)
        throw std::runtime_error(ktxErrorString(result));
}

struct Fixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("moer_ktx_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ktx");
    ~Fixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};
} // namespace

int main() {
    try {
        Moer::LogSystem::Init();
        for (Moer::uint32 faces : {1u, 6u}) {
            Fixture              fixture;
            ktxTextureCreateInfo info{};
            info.glInternalformat = GL_RGBA8;
            info.baseWidth = info.baseHeight = 2;
            info.baseDepth                   = 1;
            info.numDimensions               = 2;
            info.numLevels                   = 1;
            info.numLayers                   = faces == 1 ? 2 : 1;
            info.numFaces                    = faces;
            info.isArray                     = info.numLayers > 1;
            ktxTexture* raw                  = nullptr;
            Require(ktxTexture_Create(&info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &raw));
            Texture                  texture(raw, ktxTexture_Destroy);
            const Moer::uint32       image_count = info.numLayers * info.numFaces;
            std::vector<Moer::uint8> source(16 * image_count);
            for (Moer::uint32 layer = 0; layer < image_count; ++layer) {
                for (Moer::uint32 pixel = 0; pixel < 4; ++pixel) {
                    for (Moer::uint32 channel = 0; channel < 4; ++channel)
                        source[layer * 16 + pixel * 4 + channel] = 17 + 20 * layer + 5 * pixel;
                }
                Require(ktxTexture_SetImageFromMemory(texture.get(), 0, layer / faces, layer % faces, source.data() + layer * 16, 16));
            }
            Require(ktxTexture_WriteToNamedFile(texture.get(), fixture.path.string().c_str()));
            for (bool generate : {false, true}) {
                auto                                                 image = Moer::ImageIO::ReadFromFile(fixture.path, 4, PF_R8G8B8A8_UNORM, generate);
                std::unique_ptr<void, decltype(image.data_callback)> owner(image.data, image.data_callback);
                const Moer::uint32                                   stride = generate ? 20 : 16;
                if (!image.IsValid() || image.width != 2 || image.height != 2 || image.layers != info.numLayers ||
                    image.faces != faces || image.mips != (generate ? 2 : 1) || image.data_size != image_count * stride)
                    throw std::runtime_error("KTX array metadata mismatch");
                const auto* bytes = static_cast<const Moer::uint8*>(image.data);
                for (Moer::uint32 layer = 0; layer < image_count; ++layer) {
                    if (std::memcmp(bytes + layer * stride, source.data() + layer * 16, 16) != 0)
                        throw std::runtime_error("KTX base pixels changed in layer " + std::to_string(layer) + " generate=" + std::to_string(generate));
                    if (generate) {
                        const float expected = 24.5f + 20.f * layer;
                        for (Moer::uint32 channel = 0; channel < 4; ++channel)
                            if (std::abs(bytes[layer * stride + 16 + channel] - expected) > 1.f)
                                throw std::runtime_error("KTX generated mip does not average its own layer");
                    }
                }
            }
        }
        std::cout << "[TESTCASE][PASS] KTX array/cube base images and generated mips preserved\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] " << error.what() << '\n';
        return 1;
    }
}
