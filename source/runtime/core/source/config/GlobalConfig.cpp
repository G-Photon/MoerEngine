#include "config/GlobalConfig.h"
#include <toml++/toml.hpp>

namespace Moer::Config {

#include "generated/GlobalConfigDefaults.inl"

GlobalConfig GlobalConfig::LoadConfigFromTomlFile(const std::string_view& toml_path) {
    GlobalConfig loaded_config{};
    const auto toml_config = toml::parse_file(toml_path.data());
#include "generated/GlobalConfigLoad.inl"
    return loaded_config;
}

} // namespace Moer::Config
