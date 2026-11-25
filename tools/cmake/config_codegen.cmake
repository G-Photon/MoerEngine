# Checked-in outputs also allow source inspection without running the generator.
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
set(moer_config_generator "${moer_root_dir}/tools/config/generate.py")
set(moer_config_schema "${moer_root_dir}/tools/config/schema.json")
set(moer_config_outputs
    source/runtime/core/include/config/generated/GlobalConfig.generated.h
    source/runtime/core/source/config/generated/GlobalConfigDefaults.inl
    source/runtime/core/source/config/generated/GlobalConfigLoad.inl
    source/runtime/render/renderer/generated/RendererConfigEnums.inl
    source/runtime/render/renderer/raster/generated/RasterConfig.generated.h
    source/runtime/render/renderer/raster/generated/RasterConfigDefaults.inl
    source/runtime/render/renderer/generated/RasterConfigLoad.inl
    source/runtime/render/renderer/raytracing/generated/RaytracingConfig.generated.h
    source/runtime/render/renderer/raytracing/generated/RaytracingConfigDefaults.inl
    source/runtime/render/renderer/generated/RaytracingConfigLoad.inl
    template.MoerEngine.toml
)
list(TRANSFORM moer_config_outputs PREPEND "${moer_root_dir}/")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${moer_config_generator}"
    RESULT_VARIABLE moer_config_result
)
if(NOT moer_config_result EQUAL 0)
    message(FATAL_ERROR "Configuration generation failed")
endif()
set(moer_config_stamp "${CMAKE_BINARY_DIR}/config-codegen.stamp")
add_custom_command(
    OUTPUT "${moer_config_stamp}"
    BYPRODUCTS ${moer_config_outputs}
    COMMAND "${Python3_EXECUTABLE}" "${moer_config_generator}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${moer_config_stamp}"
    DEPENDS "${moer_config_generator}" "${moer_config_schema}"
    COMMENT "Generating configuration declarations, defaults, loaders and TOML template"
    VERBATIM
)
add_custom_target(moer_config_codegen DEPENDS "${moer_config_stamp}")
