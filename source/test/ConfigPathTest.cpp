#include "config/GlobalConfig.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

int main() {
    const auto path   = std::filesystem::temp_directory_path() /
                        ("moer_config_path_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".toml");
    int        result = 0;
    try {
        {
            std::ofstream file(path, std::ios::binary);
            file << "[editor]\nwidth = 321\nheight = 123\n"
                    "[engine.scene]\nscene_path = 'config-path-regression.gltf'\n";
            if (!file)
                throw std::runtime_error("Cannot create config fixture");
        }
        const std::string      terminated = path.string();
        const auto             ordinary   = Moer::Config::GlobalConfig::LoadConfigFromTomlFile(terminated);
        const std::string      extended   = terminated + ".not-part-of-the-view";
        const std::string_view bounded(extended.data(), terminated.size());
        const auto             sliced = Moer::Config::GlobalConfig::LoadConfigFromTomlFile(bounded);
        if (ordinary.editor.width != 321 || ordinary.editor.height != 123 ||
            sliced.editor.width != ordinary.editor.width || sliced.editor.height != ordinary.editor.height ||
            sliced.engine.scene.scene_path != "config-path-regression.gltf" || sliced.engine.rhi.api_version != "1.3") {
            throw std::runtime_error("Config path or default values changed");
        }
        std::cout << "[TESTCASE][PASS] ConfigPath terminated=verified bounded-view=verified defaults=verified\n";
    } catch (const std::exception& error) {
        std::cerr << "[TESTCASE][FAIL] " << error.what() << '\n';
        result = 1;
    }
    std::error_code cleanup_error;
    std::filesystem::remove(path, cleanup_error);
    return result;
}
