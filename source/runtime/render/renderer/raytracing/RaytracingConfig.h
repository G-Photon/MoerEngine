#pragma once
#include "RenderAPI.h"

#include "misc/Types.h"

// 定义 Raytracing 管线可由编辑器调整的运行时配置。

#include "misc/Traits.h"
#include "shaderheaders/shared/ShaderParameters.h"

namespace Moer {

enum EAnitiAliasMode {
    EAA_TAA = 0,
    EAA_Num
};
enum EOutputTexture {
    EOT_LDR = 0,
    EOT_HDR,
    EOT_Num
};
enum class EJitter {
    MSAA,
    Halton,
    R2,
    WhiteNoise,
    Num
};

#include "generated/RaytracingConfig.generated.h"

} // namespace Moer
