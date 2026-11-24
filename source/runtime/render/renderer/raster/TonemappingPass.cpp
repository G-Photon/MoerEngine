#include "TonemappingPass.h"
#include "RasterGraphResourceAccess.h"
#include "RasterTool.h"
#include "profile/ProfileScope.h"
#include "rendergraph/RenderGraphUpload.h"
#include "rhi/executor/RHIExecutor.h"
#include "scene/loader/io/ImageIO.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace Moer::Render::Raster {

static_assert(sizeof(TonemappingPipelineBindlessParam) <= 128);

TonemappingPass::TonemappingPass(RasterContext& context) :
    TonemappingPass(context.device, context.manager, context.textures.tonemapping_output.tex->GetFormat()) {}

TonemappingPass::TonemappingPass(RenderDevice& device, ShaderManager& manager, EPixelFormat output_format) {
    GfxPsoCreateInfo pso_full_screen_info(
        RHIRasterizeInfo::Preset(), {}, {RHIColorAttachmentInfo::Preset(output_format)}
    );

    // Pipeline
    tonemapping_pipeline = manager.Raster()
                               .Vertex("core/utils/FullScreenQuad.hlsl")
                               .Pixel("pipelines/postprocess/color/Tonemapping.hlsl")
                               .Build<TonemappingPipeline>(std::move(pso_full_screen_info));

    tonemapping_histogram_pipeline = manager.Compute<TonemappingHistogramPipeline>(
        "pipelines/postprocess/color/TonemappingHistogram.hlsl"
    );

    tonemapping_exposure_pipeline = manager.Compute<TonemappingExposurePipeline>(
        "pipelines/postprocess/color/TonemappingExposure.hlsl"
    );

    // Buffer
    exposure_buffer = device.CreateBuffer<uint32>(
        "raster exposure buffer",
        1,
        EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TEXTURE_BUFFER |
            EBufferUsageFlags::CPU_VISIBLE
    );
}

void TonemappingPass::UpdateColorLut(RenderDevice& device, const std::string& path) {
    if (path == color_lut_path) {
        return;
    }
    // Remember failed requests too; an invalid file must not be decoded every frame.
    color_lut_path = path;
    color_lut      = {};
    color_grading_error.clear();
    if (path.empty()) {
        return;
    }

    try {
        std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
        const auto    length = file ? file.tellg() : std::streampos(-1);
        if (length <= 0 || length > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Cannot read LUT image");
        }
        std::vector<uint8> encoded(static_cast<size_t>(length));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(encoded.data()), encoded.size())) {
            throw std::runtime_error("Cannot read LUT image");
        }
        auto image = ImageIO::ReadFromMemory(encoded.data(), encoded.size(), 4, false);

        std::unique_ptr<void, decltype(image.data_callback)> data(image.data, image.data_callback);
        if (!image.IsValid()) {
            throw std::runtime_error("Cannot decode LUT image");
        }
        if (image.height < 2 || uint64(image.height) * image.height != image.width) {
            throw std::runtime_error("LUT image must have dimensions N*N by N (N >= 2)");
        }
        auto texture = device.CreateTexture(
            "Tonemapping.ColorLUT", Extent2D(image.width, image.height), PF_R8G8B8A8_UNORM, ETextureUsageFlags::SAMPLED | ETextureUsageFlags::TRANSFER_DST
        );
        CommandList commands;
        RecordNewGraphicsTextureUpload(
            commands, {static_cast<const byte*>(image.data), image.data_size}, texture->GetView(), "Tonemapping.UploadColorLUT"
        );
        // CopyFrom owns the bytes. FIFO submission and the exported Sampled
        // boundary order this upload before the frame's graphics reads.
        RHIExecutor::Get().Submit(EQueueType::Graphics, commands.Submit(), ERHIExecSubmitFlags::FlushGPU);
        color_lut = std::move(texture);
    } catch (const std::exception& error) {
        color_grading_error = error.what();
        LOG_WARNING("Color grading LUT '{}': {}", path, color_grading_error);
    }
}

TonemappingPass::RecordParameters TonemappingPass::Prepare(
    const RasterContext& context, const RasterConfig& ui_config,
    TextureWithHandle input_image, float frame_time
) const {
    RecordParameters parameters{};
    parameters.input     = std::move(input_image);
    parameters.output    = context.textures.tonemapping_output;
    parameters.exposure  = exposure_buffer;

    const uint2 input_res = parameters.input.GetSize();
    auto&       param     = parameters.shader;
    {
        auto& ae  = param.ae;
        const auto& uae = ui_config.tonemapping_ae;

        ae.enabled = uae.enabled ? 1 : 0;

        // from [x, y] to [0, 1]: v * scale + bias
        ae.log2lum_to01_scale = 1.0f / (uae.log2lum_max - uae.log2lum_min);
        ae.log2lum_to01_bias  = -uae.log2lum_min * ae.log2lum_to01_scale;
        // from [0, 1] to [x, y]: v * scale_inv + bias_inv
        ae.log2lum_to01_scale_inv = uae.log2lum_max - uae.log2lum_min;
        ae.log2lum_to01_bias_inv  = uae.log2lum_min;

        ae.histogram_low_percentile  = uae.histogram_low_percentile;
        ae.histogram_high_percentile = uae.histogram_high_percentile;

        ae.min_adapted_luminance = uae.min_adapted_luminance;
        ae.max_adapted_luminance = uae.max_adapted_luminance;

        ae.eye_adaptation_speed_up   = uae.eye_adaptation_speed_up;
        ae.eye_adaptation_speed_down = uae.eye_adaptation_speed_down;

        ae.frame_time          = frame_time;
        ae.diff_log2_threshold = uae.diff_log2_threshold;

        ae.debug_visualize          = uae.debug_visualize ? 1 : 0;
        ae.aces_tonemapping_enabled = uae.aces_tonemapping_enabled ? 1 : 0;
    }
    {
        param.resolution     = input_res;
        param.resolution_inv = float2(1.f / input_res.x, 1.f / input_res.y);

        // 这个参数是共用的
        param.exposure_ev = std::exp2f(ui_config.tonemapping_exposure_ev);

        param.reinhard_enabled = ui_config.tonemapping_reinhard_enabled ? 1 : 0;

        param.debug_param = ui_config.debug_param;
        if (ui_config.color_grading_enabled && color_lut) {
            parameters.color_lut     = color_lut;
            param.color_lut_size     = color_lut->GetExtent().y;
            param.color_lut_strength = std::clamp(ui_config.color_grading_strength, 0.0f, 1.0f);
        }
    }

    parameters.histogram_groups = uint3(
        (input_res.x - 1) / TONEMAPPING_HISTOGRAM_GROUP_X + 1,
        (input_res.y - 1) / TONEMAPPING_HISTOGRAM_GROUP_Y + 1,
        1
    );
    return parameters;
}

void TonemappingPass::AddToGraph(
    RenderGraph& graph, const RasterContext& context, const RasterConfig& config,
    TextureWithHandle input_image, float frame_time, GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.TonemappingPass");
    UpdateColorLut(context.device, config.color_grading_lut_path);
    if (config.color_grading_enabled && color_lut) {
        resources.color_lut = ImportRasterTexture(graph, "Tonemapping.ColorLUT", color_lut.Get());
    }
    auto prepared = graph.AddSetupPass(
        "Tonemapping.Prepare", std::move(input_image),
        [this, &context, &config, frame_time](const TextureWithHandle& input) {
            return Prepare(context, config, input, frame_time);
        }
    );
    if (prepared.IsValid()) static_cast<void>(AddRecordPasses(graph, resources, prepared));
}

TonemappingPass::GraphResult TonemappingPass::AddRecordPasses(
    RenderGraph& graph, GraphResources resources, RGPreparedValue<RecordParameters> prepared
) {
    const auto histogram = graph.CreateTransientBuffer("Tonemapping.Histogram", RGTransientBufferDesc{
        .element_count = TONEMAPPING_HISTOGRAM_BIN_COUNT,
        .stride = sizeof(uint32),
        .usage = EBufferUsageFlags::UNORDERED_ACCESS | EBufferUsageFlags::TEXTURE_BUFFER |
                 EBufferUsageFlags::TRANSFER_DST | EBufferUsageFlags::TRANSFER_SRC,
    });
    graph.AddRecordPass(
        "Tonemapping.ClearHistogram",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Write(histogram, RenderGraph::BufferAccessType::TransferDestination);
        },
        [&graph, histogram](CommandList& commands) {
            ScopedGpuMarker marker(commands, "Pass: Tonemapping.ClearHistogram", GpuMarkerPalette::Pass());
            RecordClear(commands, graph.GetPhysicalBuffer(histogram)->GetView());
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    graph.AddRecordPass(
        "Tonemapping.BuildHistogram",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
                .Read(resources.input, RenderGraph::TextureAccessType::Sampled)
                .ReadWrite(histogram, RenderGraph::BufferAccessType::Storage);
        },
        [this, &graph, histogram, prepared](CommandList& commands) {
            ScopedGpuMarker marker(commands, "Pass: Tonemapping.BuildHistogram", GpuMarkerPalette::Pass());
            RecordHistogram(commands, prepared.Get(), graph.GetPhysicalBuffer(histogram)->GetView());
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    graph.AddRecordPass(
        "Tonemapping.CalculateExposure",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
                .Read(histogram, RenderGraph::BufferAccessType::ShaderRead)
                .ReadWrite(resources.exposure, RenderGraph::BufferAccessType::Storage);
        },
        [this, &graph, histogram, prepared](CommandList& commands) {
            ScopedGpuMarker marker(commands, "Pass: Tonemapping.CalculateExposure", GpuMarkerPalette::Pass());
            RecordExposure(commands, prepared.Get(), graph.GetPhysicalBuffer(histogram)->GetView());
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    const auto apply = graph.AddRecordPass(
        "Tonemapping.Apply",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(resources.input, RenderGraph::TextureAccessType::Sampled)
                .Read(histogram, RenderGraph::BufferAccessType::ShaderRead)
                .Read(resources.exposure, RenderGraph::BufferAccessType::ShaderRead)
                .Write(resources.tonemapping_output, RenderGraph::TextureAccessType::ColorAttachment);
            if (resources.color_lut.IsValid()) {
                builder.Read(resources.color_lut, RenderGraph::TextureAccessType::Sampled);
            }
        },
        [this, &graph, histogram, prepared](CommandList& commands) {
            ScopedGpuMarker marker(commands, "Pass: Tonemapping.Apply", GpuMarkerPalette::Pass());
            RecordTonemapping(commands, prepared.Get(), graph.GetPhysicalBuffer(histogram)->GetView());
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    return {.apply = apply, .histogram = histogram};
}

void TonemappingPass::ResetExposure(CommandList& commands) {
    commands.ClearResource(exposure_buffer->GetView(), std::bit_cast<uint32>(1.0f));
}

void TonemappingPass::RecordClear(CommandList& commands, BufferView histogram) {
    commands.ClearResource(histogram, 0);
}

void TonemappingPass::RecordHistogram(
    CommandList& commands, const RecordParameters& parameters, BufferView histogram
) {
    commands.Compute(tonemapping_histogram_pipeline, parameters.shader, parameters.input.tex, histogram)
        .Dispatch(parameters.histogram_groups, "Tonemapping Histogram Pass");
}

void TonemappingPass::RecordExposure(
    CommandList& commands, const RecordParameters& parameters, BufferView histogram
) {
    commands.Compute(tonemapping_exposure_pipeline, parameters.shader, histogram, parameters.exposure)
        .Dispatch(uint3(1, 1, 1), "Tonemapping Exposure Pass");
}

void TonemappingPass::RecordTonemapping(
    CommandList& commands, const RecordParameters& parameters, BufferView histogram
) {
    commands.Gfx(
                tonemapping_pipeline, parameters.shader, parameters.input.tex, parameters.exposure, histogram, parameters.color_lut ? parameters.color_lut : parameters.input.tex, Sampler{SF_LINEAR, SAM_CLAMP_TO_EDGE}
    )
        .Draw("Tonemapping Pass", parameters.output.GetRect2D(), RasterTool::GetFullScreenDrawDatas(), ColorAttachment(parameters.output.tex));
}

} // namespace Moer::Render::Raster
