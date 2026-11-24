#pragma once

#ifdef __cplusplus
#include "misc/Traits.h"
namespace Moer::Render {
#else
namespace Moer {
#endif

static const uint RASTER_PROBE_VOLUME_MAX_COUNT     = 4;
static const uint RASTER_PROBE_MAX_COUNT_PER_VOLUME = 512;
static const uint RASTER_PROBE_MAX_COUNT =
    RASTER_PROBE_VOLUME_MAX_COUNT * RASTER_PROBE_MAX_COUNT_PER_VOLUME;

} // namespace Moer
