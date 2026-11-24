#include "log/LogSystem.h"
#include "scene/loader/io/ImageIO.h"
#include <dds.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
struct Fixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("moer_dds_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".dds");
    ~Fixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    void Write(DXGI_FORMAT format, Moer::uint32 size, Moer::uint32 mips, const std::vector<Moer::uint8>& payload, bool legacy = false, bool cube = false) {
        dds::FileHeader header{};
        header.size  = sizeof(header);
        header.flags = static_cast<dds::HeaderFlags>(dds::HeaderFlags::Texture | dds::HeaderFlags::Mipmap);
        header.width = header.height = size;
        header.mipmapCount           = mips;
        header.caps1                 = 0x1000 | (mips > 1 ? 0x400008 : 0);
        header.caps2                 = cube ? dds::Caps2Flags::Cubemap : 0;
        header.pixelFormat.size      = sizeof(header.pixelFormat);
        header.pixelFormat.flags     = dds::PixelFormatFlags::FourCC;
        header.pixelFormat.fourCC    = legacy ? dds::DdsMagicNumber::DXT1 : dds::DdsMagicNumber::DX10;
        const Moer::uint32    magic  = dds::DdsMagicNumber::DDS;
        const dds::Dx10Header dx10{format, dds::ResourceDimension::Texture2D, cube ? 0x4u : 0, 1, 0};
        std::ofstream         file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        if (!legacy)
            file.write(reinterpret_cast<const char*>(&dx10), sizeof(dx10));
        file.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        if (!file)
            throw std::runtime_error("Cannot create DDS fixture");
    }
};

void Check(Fixture& fixture, EPixelFormat format, Moer::uint32 size, Moer::uint32 mips, const std::vector<Moer::uint8>& payload, bool generate, bool check_payload = true) {
    auto                                                 image = Moer::ImageIO::ReadFromFile(fixture.path, 4, PF_R8G8B8A8_UNORM, generate);
    std::unique_ptr<void, decltype(image.data_callback)> owner(image.data, image.data_callback);
    if (!image.IsValid() || image.format != format) {
        throw std::runtime_error("DDS format mismatch: actual=" + std::to_string(image.format) + " expected=" + std::to_string(format));
    }
    if (image.width != size || image.height != size || image.mips != mips ||
        (check_payload && (image.data_size != payload.size() || std::memcmp(image.data, payload.data(), payload.size()) != 0))) {
        throw std::runtime_error("DDS payload or mip mismatch: bytes=" + std::to_string(image.data_size) + " expected=" + std::to_string(payload.size()) + " mips=" + std::to_string(image.mips));
    }
}
} // namespace

int main(int argc, const char** argv) {
    try {
        Moer::LogSystem::Init();
        Fixture                        fixture;
        const std::vector<Moer::uint8> rg{0, 255, 127, 128, 32, 200, 10, 250, 64, 192};
        if (argc == 2 && std::string_view(argv[1]) == "--formats") {
            for (const auto [dxgi, expected] : {std::pair{DXGI_FORMAT_R8G8_UINT, PF_R8G8_UINT}, std::pair{DXGI_FORMAT_R8G8_SINT, PF_R8G8_SINT}, std::pair{DXGI_FORMAT_R8G8_UNORM, PF_R8G8_UNORM}}) {
                fixture.Write(dxgi, 2, 2, rg);
                Check(fixture, expected, 2, 2, rg, false, false);
            }
            std::cout << "[TESTCASE][PASS] DDS signed/unsigned/normalized format mapping\n";
        } else if (argc == 2 && std::string_view(argv[1]) == "--payload") {
            fixture.Write(DXGI_FORMAT_R8G8_UNORM, 2, 2, rg);
            Check(fixture, PF_R8G8_UNORM, 2, 2, rg, false);
            const std::vector<Moer::uint8> bc{0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0};
            for (bool legacy : {false, true}) {
                fixture.Write(DXGI_FORMAT_BC1_UNORM, 4, 1, bc, legacy);
                for (bool generate : {false, true})
                    Check(fixture, PF_BC1_RGB_UNORM_BLOCK, 4, 1, bc, generate);
            }
            std::vector<Moer::uint8> rgba(64, 128);
            fixture.Write(DXGI_FORMAT_R8G8B8A8_UNORM, 4, 1, rgba);
            Check(fixture, PF_R8G8B8A8_UNORM, 4, 1, rgba, false);
            std::vector<Moer::uint8> pyramid(84, 128);
            Check(fixture, PF_R8G8B8A8_UNORM, 4, 3, pyramid, true);
            fixture.Write(DXGI_FORMAT_R8G8_UNORM, 2, 2, {1, 2});
            auto                                                     truncated = Moer::ImageIO::ReadFromFile(fixture.path, 4, PF_R8G8B8A8_UNORM, false);
            std::unique_ptr<void, decltype(truncated.data_callback)> owner(truncated.data, truncated.data_callback);
            if (truncated.IsValid())
                throw std::runtime_error("Truncated DDS payload was accepted");
            for (bool legacy : {false, true}) {
                fixture.Write(DXGI_FORMAT_BC1_UNORM, 4, 1, bc, legacy, true);
                auto                                                cube = Moer::ImageIO::ReadFromFile(fixture.path, 4, PF_R8G8B8A8_UNORM, false);
                std::unique_ptr<void, decltype(cube.data_callback)> cube_owner(cube.data, cube.data_callback);
                if (cube.IsValid())
                    throw std::runtime_error("Unsupported DDS cube was accepted as a 2D image");
            }
            std::filesystem::resize_file(fixture.path, 0);
            auto empty = Moer::ImageIO::ReadFromFile(fixture.path, 4, PF_R8G8B8A8_UNORM, false);
            if (empty.IsValid())
                throw std::runtime_error("Empty DDS was accepted");
            std::cout << "[TESTCASE][PASS] DDS payload=verified existing-mips=verified legacy=verified compressed=preserved rgba-mips=verified truncated=rejected\n";
        } else
            throw std::runtime_error("Expected --formats or --payload");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] " << error.what() << '\n';
        return 1;
    }
}
