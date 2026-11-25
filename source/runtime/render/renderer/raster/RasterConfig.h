#pragma once
#include "RenderAPI.h"
#include "misc/STL.h"
#include "misc/Traits.h"
#include "misc/Types.h"

#include "RasterCompileTimeConstants.h"
#include "shaderheaders/shared/raster/ProbeLimits.h"
#include "shaderheaders/shared/raster/SharedEnum.h"

#include <string>

namespace Moer {

// EnumParam(EShadingMode, DEFAULT, DEBUG);
static const UnorderedMap<EShadingMode, std::string> s_shading_mode_name_map = {
    {EShadingMode::DEFAULT_PBR, "Default PBR (GAMES202)"},
    {EShadingMode::DEBUG, "Debug"},
};

// EnumParam(EBrdfNdfMode, BECKMANN, GGX, EXTENDING_GGX);
static const UnorderedMap<EBrdfNdfMode, std::string> s_brdf_ndf_mode_name_map = {
    {EBrdfNdfMode::BECKMANN, "Beckmann"},
    {EBrdfNdfMode::GGX, "GGX"},
    {EBrdfNdfMode::GTR2, "GTR Gamma=2"},
    {EBrdfNdfMode::GTR1, "GTR Gamma=1"},
};

// EnumParam(EBrdfGMode, G_SCHLICK, VIS_UE4, VIS_UNITY, VIS_FILAMENT, VIS_RESPAWN);
static const UnorderedMap<EBrdfGMode, std::string> s_brdf_geometry_mode_name_map = {
    {EBrdfGMode::G_SCHLICK, "GGX-Smith Separable (UE4)"},
    {EBrdfGMode::VIS_UE4, "GGX-Smith Joint (UE4)"},
    {EBrdfGMode::VIS_UNITY, "GGX-Smith Joint (Unity)"},
    {EBrdfGMode::VIS_FILAMENT, "GGX-Smith Joint (Filament)"},
    {EBrdfGMode::VIS_RESPAWN, "GGX-Smith Joint (Respawn E.)"},
};

// Enum 在 ..../ShaderParameters.h中定义，让shader和cpp可以共用枚举值
// EnumParam(EAaMode, NONE, FXAA_SIMPLIFIED, FXAA_QUALITY, SMAA_1X, SMAA_T2X);
static const UnorderedMap<EAaMode, std::string> s_aa_mode_name_map = {
    {EAaMode::NONE, "None"},
    {EAaMode::FXAA_SIMPLIFIED, "FXAA Simplified"},
    {EAaMode::FXAA_QUALITY, "FXAA Quality"},
    {EAaMode::SMAA_1X, "SMAA 1x"},
    {EAaMode::SMAA_T2X, "SMAA T2x"},
};

// EnumParam(EAoMode, NONE, SSAO, SSAO_AO_ONLY, RTAO, RTAO_AO_ONLY, LINEARIZED_DEPTH_DIV_10);
static const UnorderedMap<EAoMode, std::string> s_ao_mode_name_map = {
    {EAoMode::NONE, "None"},
    {EAoMode::SSAO, "SSAO"},
    {EAoMode::SSAO_AO_ONLY, "SSAO AO Only"},
    {EAoMode::RTAO, "RTAO"},
    {EAoMode::RTAO_AO_ONLY, "RTAO AO Only"},
    {EAoMode::SSDO, "SSDO"},
    {EAoMode::SSDO_AO_ONLY, "SSDO AO Only"},
    {EAoMode::LINEARIZED_DEPTH_DIV_10, "Linear. Depth / 10.0"},
};

// EnumParam(EDenoiserMode, NONE, BILATERAL_FILTER);
static const UnorderedMap<EDenoiserMode, std::string> s_denoiser_mode_name_map = {
    {EDenoiserMode::NONE, "None"},
    {EDenoiserMode::BILATERAL_FILTER, "双边滤波 Bilateral Filter"},
};

// EnumParam(ERtaoSampleMode, UNIFORM, COSINE_WEIGHTED);
static const UnorderedMap<ERtaoSampleMode, std::string> s_rtao_sample_mode_name_map = {
    {ERtaoSampleMode::UNIFORM, "Uniform in Semisphere"},
    {ERtaoSampleMode::COSINE_WEIGHTED, "Cosine-Weighted in Semisphere"},
};

// EnumParam(EShadowMapMode, NONE, CSM, CSM_AUTO);
static const UnorderedMap<EShadowMapMode, std::string> s_shadow_map_mode_name_map = {
    {EShadowMapMode::NONE, "None"},
    {EShadowMapMode::POINT_CUBE, "Point Cube"},
    {EShadowMapMode::CSM, "CSM"},
    {EShadowMapMode::CSM_AUTO, "CSM Auto"},
};

enum class EUpsampleMode {
    None = 0,
    BILINEAR,
    DEPTH
};
static const Array<std::string> s_upsample_mode_name_array = {
    "None",
    "BILINEAR",
    "DEPTH",
};

static const Array<std::string> s_rtao_sample_mode = {
    "Uniform in Semisphere",
    "Cosine-Weighted in Semisphere"
};

static const Array<std::string> s_shadow_sampling_mode_name_array = {
    "No Filtering",
    "PCF 1x1",
    "PCF 3x3",
    "PCF 5x5",
};
static const Array<std::string> s_ai_trt_visualize_buffer_array = {
    "Engine1 in_depth",           "Engine1 in_color",        "Engine1 in_motion",
    "Engine1 in_prev_ao",         "Engine1 in_prev_embed",   "Engine1 out_XTX_batch",
    "Engine1 out_XTY_batch",      "Engine1 out_X_model",     "Engine1 out_ao",
    "Engine1 out_upscale_kernel", "Engine1 out_color",       "Engine1 out_prev_ao",
    "Engine1 out_embed",          "Engine2 in_X_model",      "Engine2 in_coeffs_batch",
    "Engine2 in_upscale_kernel",  "Engine2 in_color",        "Engine2 in_prev_ao",
    "Engine2 out_final_output",   "Engine2 out_denoised_ao",
}; // namespace Moer

#include "generated/RasterConfig.generated.h"

} // namespace Moer
