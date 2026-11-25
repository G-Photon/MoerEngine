// Explicit RDG registration owned by each raster pass.

#include "AaPass.h"
#include "AoPass.h"
#include "BilateralFilterDenoiserPass.h"
#include "BloomPass.h"
#include "CameraGizmoPass.h"
#include "CsmGizmoPass.h"
#include "DirectionalShadowMaskPass.h"
#include "GeometryPass.h"
#include "HiZBuildPass.h"
#include "LightingPass.h"
#include "ProbeGizmoPass.h"
#include "ProbeUpdatePass.h"
#include "RtaoDenoiserPass.h"
#include "ShadowDepthPass.h"
#include "SkyboxPass.h"
#include "SsrPass.h"
#include "TessellatedSurfacePass.h"
#include "misc/Types.h"

#include "profile/ProfileScope.h"
#include "renderer/common/UiCombinePass.h"

#include <algorithm>
#include <bit>
#include <format>
#include <functional>
#include <utility>

namespace Moer::Render::Raster {

namespace {

template<typename Record>
void RecordWithPassMarker(CommandList& cmd_list, std::string_view name, Record&& record) {
    const std::string marker_name = std::format("Pass: {}", name);
    ScopedGpuMarker marker(cmd_list, marker_name, GpuMarkerPalette::Pass());
    std::invoke(std::forward<Record>(record));
}

std::vector<RenderGraph::TextureHandle> ImportShadowMaps(RenderGraph& graph, const RasterContext& context) {
    std::vector<RenderGraph::TextureHandle> result;
    for (const auto& map : context.csm_data.shadow_map_textures) {
        if (map.tex) {
            result.push_back(ImportRasterTexture(graph, "Shadow.cascade", map.tex->GetView().GetTexture()));
        }
    }
    for (const auto& map : context.point_shadow_data.shadow_cubes) {
        if (map.tex) {
            result.push_back(ImportRasterTexture(graph, "Shadow.point", map.tex.Get()));
        }
    }
    return result;
}

} // namespace

ShadowDepthPass::GraphPassHandle ShadowDepthPass::AddToGraph(
    RenderGraph&        graph,
    RasterContext&      context,
    const SceneUpdateBatch& updates,
    const RasterConfig& ui_config,
    const Camera&       camera,
    GraphResources      resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.ShadowDepthPass");
    // Allocation and cache decisions determine the graph's physical resources
    // and node count, so they must finish before Compile. GPU recording remains parallel.
    auto       parameters = std::make_shared<const RecordParameters>(Prepare(context, updates, ui_config, camera));
    const auto table      = graph.ImportToken("Raster.bindless", parameters->bindless.Get());
    RenderGraph::PassHandle publication{};
    if (parameters->update_bindless_array) {
        publication = graph.AddRecordPass(
            "ShadowDepth.PublishBindings",
            [=](RenderGraph::PassBuilder& builder) {
                builder.Write(table).SideEffect();
            },
            [parameters](CommandList& commands) {
                commands.UpdateBindlessArray(parameters->bindless);
            },
            RenderGraph::PassRecordPolicy::Serial
        );
    }
    std::vector<RenderGraph::PassHandle> draws;
    if (!parameters->draws.empty()) {
        const RasterSceneGraphReads scene_reads(graph, context.GetGpuSceneRes());
        const auto indices = ImportRasterBuffer(graph, "Scene.indices", parameters->index_buffer.GetBuffer());
        for (size_t index = 0; index < parameters->draws.size(); ++index) {
            const auto& batch = parameters->draws[index];
            m_culling_pass.AddToGraph(graph, batch.pass_name, batch.culling);
            RenderGraph::BufferHandle matrices{};
            if (batch.multiview) {
                matrices = ImportRasterBuffer(
                    graph, "ShadowDepth.matrices", parameters->point_shadow_view_matrices.Get()
                );
                graph.AddRecordPass(
                    batch.pass_name + ".UploadMatrices",
                    [=](RenderGraph::PassBuilder& builder) {
                        builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                            .Write(matrices, RenderGraph::BufferAccessType::TransferDestination);
                    },
                    [parameters, index](CommandList& commands) {
                        const auto& data = parameters->draws[index].multiview_matrices;
                        commands.CopyFrom(
                            std::span<const byte>(reinterpret_cast<const byte*>(&data), sizeof(data)),
                            parameters->point_shadow_view_matrices->GetView()
                        );
                    },
                    RenderGraph::PassRecordPolicy::ParallelAllowed
                );
            }
            const auto depth = ImportRasterTexture(graph, batch.pass_name, batch.depth_view.GetTexture());
            const auto visible =
                ImportRasterBuffer(graph, "ShadowDepth.visible", batch.culling.visible_instance_ids.GetBuffer());
            const auto indirect =
                ImportRasterBuffer(graph, "ShadowDepth.indirect", batch.culling.draw_commands.GetBuffer());
            const auto count =
                ImportRasterBuffer(graph, "ShadowDepth.count", batch.culling.draw_count.GetBuffer());
            RenderGraph::TextureRange range{};
            range.mip_first   = batch.depth_view.mip_level;
            range.mip_count   = batch.depth_view.num_mips;
            range.layer_first = batch.depth_view.array_layer;
            range.layer_count = batch.depth_view.num_array;
            draws.push_back(graph.AddRecordPass(
                batch.pass_name,
                [=](RenderGraph::PassBuilder& builder) {
                    builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                        .Read(table)
                        .Read(indices, RenderGraph::BufferAccessType::IndexBuffer)
                        .Read(visible, RenderGraph::BufferAccessType::ShaderRead)
                        .Read(indirect, RenderGraph::BufferAccessType::IndirectArgument)
                        .Read(count, RenderGraph::BufferAccessType::IndirectArgument)
                        .Write(depth, RenderGraph::TextureAccessType::DepthStencilAttachment, range);
                    scene_reads.Declare(builder);
                    if (matrices.IsValid()) {
                        builder.Read(matrices, RenderGraph::BufferAccessType::ShaderRead);
                    }
                },
                [this, parameters, index](CommandList& commands) {
                    const auto& draw = parameters->draws[index];
                    if (draw.multiview) {
                        RecordPointShadowMultiviewDraw(commands, *parameters, draw);
                    } else {
                        RecordShadowDraw(commands, *parameters, draw);
                    }
                },
                RenderGraph::PassRecordPolicy::ParallelAllowed
            ));
        }
    }
    if (parameters->gaussian && m_gaussian_pass) {
        const auto gaussian = std::shared_ptr<const GaussianShadowPass::RecordParameters>(
            parameters, &*parameters->gaussian
        );
        draws.push_back(m_gaussian_pass->AddToGraph(graph, gaussian));
    }
    const auto ready = graph.AddRecordPass(
        "ShadowDepth.Ready",
        [=](RenderGraph::PassBuilder& builder) {
            builder.Write(resources.shadow_maps).SideEffect();
            for (auto draw : draws) {
                builder.DependsOn(draw);
            }
            if (publication.IsValid()) {
                builder.DependsOn(publication);
            }
        },
        [](CommandList&) {},
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    return {.pass = ready, .parameters = std::move(parameters)};
}

RenderGraph::PassHandle ProbeUpdatePass::AddToGraph(
    RenderGraph&        graph,
    RasterContext&      context,
    const SceneUpdateBatch& updates,
    const RasterConfig& config,
    const Camera&       camera,
    uint64              frame_index,
    GraphResources      resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.ProbeUpdatePass");
    auto parameters = std::make_shared<const RecordParameters>(Prepare(context, updates, config, camera, frame_index));
    if (!parameters->enabled) {
        return {};
    }
    const auto import = [&](BufferView view) {
        return ImportRasterBuffer(graph, "ProbeUpdate.data", view.GetBuffer());
    };
    std::vector<RenderGraph::BufferHandle> uploads;
    const auto&                            upload = parameters->scene_upload;
    const auto add_upload                         = [&](const BufferRef& buffer, const Array<byte>& data) {
        if (buffer && !data.empty()) {
            uploads.push_back(ImportRasterBuffer(graph, "ProbeUpdate.upload", buffer.Get()));
        }
    };
    add_upload(upload.cell_buffer, upload.cell_data);
    add_upload(upload.volume_buffer, upload.volume_data);
    add_upload(upload.brick_buffer, upload.brick_data);
    add_upload(upload.page_table_buffer, upload.page_table_data);
    add_upload(upload.scene_data_buffer, upload.scene_data);
    auto last = graph.AddRecordPass(
        "ProbeUpdate.Upload",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .RecordWorkload(std::max(1u, static_cast<uint32>(uploads.size())))
                .Write(resources.probe_volume)
                .SideEffect();
            for (auto buffer : uploads) {
                builder.Write(buffer, RenderGraph::BufferAccessType::TransferDestination);
            }
        },
        [parameters](CommandList& commands) {
            ProbeVolumeResource::RecordSceneDataUpload(commands, parameters->scene_upload);
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    const auto probes           = import(parameters->probe_data);
    const auto visibility       = import(parameters->visibility_atlas);
    const auto irradiance       = import(parameters->irradiance_atlas);
    const auto scene            = import(parameters->scene_data);
    const auto volumes          = import(parameters->volume_data);
    const auto bricks           = import(parameters->brick_data);
    const auto visibility_image = ImportRasterTexture(
        graph, "ProbeUpdate.visibility", parameters->visibility_atlas_texture.GetTexture()
    );
    const auto irradiance_image = ImportRasterTexture(
        graph, "ProbeUpdate.irradiance", parameters->irradiance_atlas_texture.GetTexture()
    );
    const auto environment =
        ImportRasterTexture(graph, "ProbeUpdate.environment", context.textures.cubemap_tex.tex.Get());
    const RasterSceneGraphReads            scene_reads(graph, context.GetGpuSceneRes());
    std::vector<RenderGraph::BufferHandle> ray_data;
    const auto&                            scene_resources = context.GetGpuSceneRes();
    for (const auto* resource :
         {&scene_resources.rt_instance_buf,
          &scene_resources.rt_primitive_table_buf,
          &scene_resources.index_buf,
          &scene_resources.light_buf}) {
        if (resource->buf) {
            ray_data.push_back(ImportRasterBuffer(graph, "ProbeUpdate.ray_data", resource->buf.Get()));
        }
    }
    for (uint32 index = 0; index < parameters->update.job_count && parameters->update.scheduled_probe_count;
         ++index) {
        if (!parameters->update.jobs[index].probe_count) {
            continue;
        }
        last = graph.AddRecordPass(
            std::format("ProbeUpdate.Job{}", index),
            [=](RenderGraph::PassBuilder& builder) {
                using BS = RenderGraph::BufferAccessType;
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
                    .ReadWrite(resources.probe_volume)
                    .ReadWrite(probes, BS::Storage)
                    .ReadWrite(visibility, BS::Storage)
                    .ReadWrite(irradiance, BS::Storage)
                    .Read(scene, BS::ShaderRead)
                    .Read(volumes, BS::ShaderRead)
                    .Read(bricks, BS::ShaderRead)
                    .Write(visibility_image, RenderGraph::TextureAccessType::Storage)
                    .Write(irradiance_image, RenderGraph::TextureAccessType::Storage)
                    .Read(environment, RenderGraph::TextureAccessType::Sampled)
                    .SideEffect();
                scene_reads.Declare(builder);
                for (auto buffer : ray_data) {
                    builder.Read(buffer, BS::ShaderRead);
                }
            },
            [this, parameters, index](CommandList& commands) {
                RecordJob(commands, *parameters, index);
            },
            RenderGraph::PassRecordPolicy::ParallelAllowed
        );
    }
    return last;
}

RenderGraph::PassHandle GeometryPass::AddToGraph(
    RenderGraph&   graph,
    RasterContext& context,
    const RasterConfig& raster_config,
    const Camera&  camera,
    GraphResources resources,
    CullingStats& out_stats
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.GeometryPass");
    auto parameters = std::make_shared<const RecordParameters>(Prepare(context, raster_config, camera, out_stats));
    culling_pass.AddToGraph(graph, "Geometry", parameters->culling, resources.hiz_previous);
    const RasterSceneGraphReads scene_reads(graph, context.GetGpuSceneRes());
    const auto indices = ImportRasterBuffer(graph, "Scene.indices", parameters->index_buffer.GetBuffer());
    const auto visible =
        ImportRasterBuffer(graph, "Geometry.visible", parameters->culling.visible_instance_ids.GetBuffer());
    const auto indirect =
        ImportRasterBuffer(graph, "Geometry.indirect", parameters->draw_commands.GetBuffer());
    const auto count = ImportRasterBuffer(graph, "Geometry.count", parameters->draw_count.GetBuffer());
    const auto table = graph.ImportToken("Raster.bindless", parameters->bindless.Get());
    return graph.AddRecordPass(
        "Geometry.Draw",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(table)
                .Read(indices, RenderGraph::BufferAccessType::IndexBuffer)
                .Read(visible, RenderGraph::BufferAccessType::ShaderRead)
                .Read(indirect, RenderGraph::BufferAccessType::IndirectArgument)
                .Read(count, RenderGraph::BufferAccessType::IndirectArgument)
                .Write(resources.base_color, RenderGraph::TextureAccessType::ColorAttachment)
                .Write(resources.normal, RenderGraph::TextureAccessType::ColorAttachment)
                .Write(resources.metal_rough_ao, RenderGraph::TextureAccessType::ColorAttachment)
                .Write(resources.depth, RenderGraph::TextureAccessType::DepthStencilAttachment);
            scene_reads.Declare(builder);
        },
        [this, parameters, resources, &graph](CommandList& commands) {
            auto resolved = *parameters;
            resolved.base_color = graph.GetPhysicalTexture(resources.base_color);
            resolved.normal = graph.GetPhysicalTexture(resources.normal);
            resolved.metal_rough_ao = graph.GetPhysicalTexture(resources.metal_rough_ao);
            RecordDraw(commands, resolved);
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void TessellatedSurfacePass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    const RasterConfig& config,
    const Camera& camera,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.TessellatedSurfacePass");
    if (!supported || !config.tessellated_surface_enabled) {
        return;
    }
    auto prepared = graph.AddSetupPass(
        "TessellatedSurface.Prepare",
        uint8{0},
        [this, &context, &config, &camera](const uint8&) {
            return Prepare(context, config, camera);
        }
    );
    if (!prepared.IsValid()) {
        return;
    }

    const auto data = ImportRasterBuffer(graph, "Tessellation.data", surface_data_buffer.Get());
    graph.AddRecordPass(
        "Tessellation.Upload",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Write(data, RenderGraph::BufferAccessType::TransferDestination);
        },
        [this, prepared](CommandList& commands) {
            RecordUpload(commands, prepared.Get());
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    graph.AddRecordPass(
        "TessellatedSurface",
        [resources, data](RenderGraph::PassBuilder& builder) {
            builder.Read(data, RenderGraph::BufferAccessType::ShaderRead);
            builder.ReadWrite(resources.base_color, RenderGraph::TextureAccessType::ColorAttachment)
                .ReadWrite(resources.normal, RenderGraph::TextureAccessType::ColorAttachment)
                .ReadWrite(resources.metal_rough_ao, RenderGraph::TextureAccessType::ColorAttachment)
                .ReadWrite(resources.depth, RenderGraph::TextureAccessType::DepthStencilAttachment);
        },
        [this, prepared, resources, &graph](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "TessellatedSurface", [&] {
                auto resolved = prepared.Get();
                resolved.base_color.tex = graph.GetPhysicalTexture(resources.base_color);
                resolved.normal.tex = graph.GetPhysicalTexture(resources.normal);
                resolved.metal_rough_ao.tex = graph.GetPhysicalTexture(resources.metal_rough_ao);
                RecordDraw(cmd_list, resolved);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void HiZBuildPass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.HiZBuildPass");
    auto prepared = graph.AddSetupPass(
        "HiZBuild.Prepare",
        uint8{0},
        [this, &context](const uint8&) {
            return Prepare(context);
        }
    );
    if (!prepared.IsValid()) {
        return;
    }

    for (uint32 mip = 0; mip < context.textures.hiz_current.tex->GetNumMips(); ++mip) {
        graph.AddRecordPass(
            std::format("HiZBuild.Mip{}", mip),
            [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
                    .Read(mip == 0 ? resources.depth : resources.hiz_current, RenderGraph::TextureAccessType::Sampled, RenderGraph::TextureRange::Mips(mip == 0 ? 0 : mip - 1, 1))
                    .Write(resources.hiz_current, RenderGraph::TextureAccessType::Storage, RenderGraph::TextureRange::Mips(mip, 1));
            },
            [this, prepared, mip](CommandList& cmd_list) {
                RecordMip(cmd_list, prepared.Get(), mip);
            },
            RenderGraph::PassRecordPolicy::ParallelAllowed
        );
    }
}

void DirectionalShadowMaskPass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    GraphResources resources,
    std::span<const RenderGraph::SetupPassHandle> setup_dependencies
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.DirectionalShadowMaskPass");
    auto prepared = graph.AddSetupPass(
        "DirectionalShadowMask.Prepare",
        uint8{0},
        [this, &context](const uint8&) {
            return Prepare(context);
        },
        setup_dependencies
    );
    if (!prepared.IsValid()) {
        return;
    }

    const auto shadows = ImportShadowMaps(graph, context);
    graph.AddRecordPass(
        "DirectionalShadowMask",
        [resources, shadows](RenderGraph::PassBuilder& builder) {
            for (auto texture : shadows) {
                builder.Read(texture, RenderGraph::TextureAccessType::Sampled);
            }

            builder.Read(resources.normal, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.depth, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.lighting_data, RenderGraph::BufferAccessType::ShaderRead)
                .Read(resources.shadow_maps)
                .Write(resources.shadow_mask, RenderGraph::TextureAccessType::ColorAttachment);
        },
        [this, prepared, resources, &graph](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "DirectionalShadowMask", [&] {
                auto resolved = prepared.Get();
                resolved.normal_owner = graph.GetPhysicalTexture(resources.normal);
                resolved.depth_owner = graph.GetPhysicalTexture(resources.depth);
                resolved.output = graph.GetPhysicalTexture(resources.shadow_mask);
                Record(cmd_list, resolved);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void LightingPass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    const RasterConfig& ui_config,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.LightingPass");
    auto prepared = graph.AddSetupPass(
        "Lighting.Prepare",
        uint8{0},
        [this, &context, &ui_config](const uint8&) {
            return Prepare(context, ui_config);
        }
    );
    if (!prepared.IsValid()) {
        return;
    }

    const auto                  shadows = ImportShadowMaps(graph, context);
    const RasterProbeGraphReads probes(graph, context.probe_volume);
    const auto emu  = ImportRasterTexture(graph, "Lighting.emu", context.textures.lut_ggx_emu.tex.Get());
    const auto eavg = ImportRasterTexture(graph, "Lighting.eavg", context.textures.lut_ggx_eavg.tex.Get());
    Array<RenderGraph::TextureHandle> gaussian_textures;
    for (const auto& texture : resources.gaussian_shadow_textures) {
        gaussian_textures.push_back(ImportRasterTexture(graph, "Lighting.gaussian_shadow", texture.Get()));
    }
    RenderGraph::BufferHandle gaussian_data{};
    if (resources.gaussian_shadow_data) {
        gaussian_data = ImportRasterBuffer(graph, "Lighting.gaussian_views", resources.gaussian_shadow_data.Get());
    }
    graph.AddRecordPass(
        "Lighting",
        [resources, shadows, probes, emu, eavg, gaussian_textures, gaussian_data](RenderGraph::PassBuilder& builder) {
            for (auto texture : gaussian_textures) {
                builder.Read(texture, RenderGraph::TextureAccessType::Sampled);
            }
            if (gaussian_data.IsValid()) {
                builder.Read(gaussian_data, RenderGraph::BufferAccessType::ShaderRead);
            }
            for (auto texture : shadows) {
                builder.Read(texture, RenderGraph::TextureAccessType::Sampled);
            }
            probes.Declare(builder);
            builder.Read(emu, RenderGraph::TextureAccessType::Sampled)
                .Read(eavg, RenderGraph::TextureAccessType::Sampled);

            builder.Read(resources.base_color, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.normal, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.metal_rough_ao, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.depth, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.shadow_mask, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.lighting_data, RenderGraph::BufferAccessType::ShaderRead)
                .Read(resources.cubemap, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.probe_volume)
                .Write(resources.lighting_output, RenderGraph::TextureAccessType::ColorAttachment);
            if (resources.scene_lights.IsValid()) {
                builder.Read(resources.scene_lights, RenderGraph::BufferAccessType::ShaderRead);
            }
        },
        [this, prepared, resources, &graph](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "Lighting", [&] {
                auto resolved = prepared.Get();
                resolved.base_color_owner = graph.GetPhysicalTexture(resources.base_color);
                resolved.normal_owner = graph.GetPhysicalTexture(resources.normal);
                resolved.metal_rough_ao_owner = graph.GetPhysicalTexture(resources.metal_rough_ao);
                resolved.depth_owner = graph.GetPhysicalTexture(resources.depth);
                resolved.shadow_mask_owner = graph.GetPhysicalTexture(resources.shadow_mask);
                resolved.output = graph.GetPhysicalTexture(resources.lighting_output);
                Record(cmd_list, resolved);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void SkyboxPass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    const SceneUpdateBatch& updates,
    const RasterConfig& config,
    const Camera& camera,
    GraphResources resources,
    bool clear_depth
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.SkyboxPass");
    auto prepared = graph.AddSetupPass(
        "Skybox.Prepare",
        uint8{0},
        [this, &context, &updates, &config, &camera, clear_depth](const uint8&) {
            auto parameters = Prepare(context, updates, config, camera);
            parameters.clear_depth = clear_depth;
            return parameters;
        }
    );
    if (!prepared.IsValid()) {
        return;
    }

    graph.AddRecordPass(
        "Skybox",
        [resources, clear_depth](RenderGraph::PassBuilder& builder) {
            // Skybox disables depth writes, but the attachment STORE operation
            // still writes memory at the end of rendering.
            builder.ReadWrite(resources.depth, RenderGraph::TextureAccessType::DepthStencilAttachment)
                .Read(resources.cubemap, RenderGraph::TextureAccessType::Sampled);
            if (clear_depth)
                builder.Write(resources.lighting_output, RenderGraph::TextureAccessType::ColorAttachment);
            else
                builder.ReadWrite(resources.lighting_output, RenderGraph::TextureAccessType::ColorAttachment);
        },
        [this, prepared, resources, &graph](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "Skybox", [&] {
                auto resolved = prepared.Get();
                resolved.output = graph.GetPhysicalTexture(resources.lighting_output);
                Record(cmd_list, resolved);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void ProbeGizmoPass::AddToGraph(
    RenderGraph& graph,
    RasterContext& context,
    RasterConfig config,
    const Camera& camera,
    GraphResources resources,
    std::span<const RenderGraph::SetupPassHandle> setup_dependencies
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.ProbeGizmoPass");
    auto prepared = graph.AddSetupPass(
        "ProbeGizmo.Prepare",
        uint8{0},
        [this, &context, config = std::move(config), &camera](const uint8&) {
            return Prepare(context, config, camera);
        },
        setup_dependencies
    );
    if (!prepared.IsValid()) {
        return;
    }

    const RasterProbeGraphReads probes(graph, context.probe_volume);
    graph.AddRecordPass(
        "ProbeGizmo",
        [resources, probes](RenderGraph::PassBuilder& builder) {
            probes.Declare(builder);

            builder.Read(resources.probe_volume)
                .ReadWrite(resources.lighting_output, RenderGraph::TextureAccessType::ColorAttachment)
                .SideEffect();
        },
        [this, prepared, resources, &graph](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "ProbeGizmo", [&] {
                auto resolved = prepared.Get();
                resolved.output.tex = graph.GetPhysicalTexture(resources.lighting_output);
                Record(cmd_list, resolved);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void AoPass::AddToGraph(
    RenderGraph& graph,
    RasterContext& context,
    const RasterConfig& ui_config,
    const Camera& camera,
    uint64 frame_index,
    AoFrameResources frame,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.AoPass");
    AddToGraph(graph, Prepare(context, ui_config, camera, frame_index, frame), resources);
}

void AoPass::AddToGraph(RenderGraph& graph, RecordParameters parameters, GraphResources resources) {
    auto prepared = std::make_shared<RecordParameters>(std::move(parameters));
    const bool ray_query = prepared->mode == EAoMode::RTAO || prepared->mode == EAoMode::RTAO_AO_ONLY;
    const bool ssdo = prepared->mode == EAoMode::SSDO || prepared->mode == EAoMode::SSDO_AO_ONLY;
    const auto bindings = graph.ImportToken("AmbientOcclusion.bindings", prepared.get());
    graph.AddRecordPass("AmbientOcclusion.PublishBindings",
        [=](RenderGraph::PassBuilder& builder) { builder.Write(bindings); },
        [&graph, prepared, resources, ray_query, ssdo](CommandList& commands) {
            prepared->bindless = RenderDevice::Get().CreateBindlessArray(8);
            const auto bind = [&](RenderGraph::TextureHandle resource, Sampler sampler = {SF_LINEAR, SAM_REPEAT}) {
                return prepared->bindless->AllocateTexture(graph.GetPhysicalTexture(resource)->GetView(), sampler);
            };
            const auto normal = bind(resources.normal);
            const auto depth_linear = bind(resources.depth, {SF_LINEAR, SAM_CLAMP_TO_EDGE});
            const auto depth_nearest = bind(resources.depth, {SF_NEAREST, SAM_CLAMP_TO_EDGE});
            const auto noise = bind(resources.noise);
            const auto motion = prepared->bindless->AllocateBuffer(graph.GetPhysicalBuffer(resources.motion_vector_data)->GetView());
            prepared->ao.normal_tex = prepared->rtao.normal_tex = prepared->ssdo.normal_tex = normal;
            prepared->ao.depth_tex = depth_linear;
            prepared->rtao.depth_tex = prepared->ssdo.depth_tex = depth_nearest;
            prepared->ao.noise_tex = prepared->rtao.noise_tex = prepared->ssdo.noise_tex = noise;
            prepared->ao.camera_mv_data_handle = prepared->rtao.camera_mv_data_handle = prepared->ssdo.camera_mv_data_handle = motion;
            auto& frame = prepared->frame;
            if (frame.denoise) frame.noisy_ao.tex = graph.GetPhysicalTexture(resources.ao_only);
            else frame.final_ao[frame.write_index].tex = graph.GetPhysicalTexture(resources.ao_only);
            frame.camera_motion.tex = graph.GetPhysicalTexture(resources.camera_motion_vector);
            if (ssdo) {
                prepared->ssdo.input_image = bind(resources.lighting_output);
                frame.indirect.tex = graph.GetPhysicalTexture(resources.indirect);
            }
            if (ray_query) frame.geometry[frame.write_index].tex = graph.GetPhysicalTexture(resources.geometry);
            commands.UpdateBindlessArray(prepared->bindless);
        }, RenderGraph::PassRecordPolicy::Serial);

    graph.AddRecordPass(
        "AmbientOcclusion.UploadMotion",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Write(resources.motion_vector_data, RenderGraph::BufferAccessType::TransferDestination);
        },
        [this, prepared](CommandList& commands) {
            RecordUpload(commands, *prepared);
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    graph.AddRecordPass(
        "AmbientOcclusion",
        [resources, ray_query, ssdo, bindings](RenderGraph::PassBuilder& builder) {
            builder
                .ExecuteOn(
                    RenderGraph::QueueRole::Graphics,
                    ray_query ? RenderGraph::GpuWorkType::Compute : RenderGraph::GpuWorkType::Graphics
                )
                .Read(bindings)
                .Read(resources.normal, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.depth, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.noise, RenderGraph::TextureAccessType::Sampled)
                .Write(
                    resources.ao_only,
                    ray_query ? RenderGraph::TextureAccessType::Storage :
                                RenderGraph::TextureAccessType::ColorAttachment
                )
                .Write(
                    resources.camera_motion_vector,
                    ray_query ? RenderGraph::TextureAccessType::Storage :
                                RenderGraph::TextureAccessType::ColorAttachment
                )
                .Read(resources.motion_vector_data, RenderGraph::BufferAccessType::ShaderRead);
            if (ssdo) {
                builder.Read(resources.lighting_output, RenderGraph::TextureAccessType::Sampled)
                    .Write(resources.indirect, RenderGraph::TextureAccessType::ColorAttachment);
            }
            if (ray_query) {
                builder.Write(resources.geometry, RenderGraph::TextureAccessType::Storage);
            }
        },
        [this, prepared](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "AmbientOcclusion", [&] {
                RecordEvaluate(cmd_list, *prepared);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

RenderGraph::TextureHandle AoPass::AddCompositeToGraph(
    RenderGraph&            graph,
    const RasterConfig&     ui_config,
    Extent2D                full_resolution,
    Extent2D                ao_resolution,
    CompositeGraphResources resources
) {
#if !WITH_CUDA
    if (ui_config.ao_mode == EAoMode::NONE) return resources.lighting_output;
#endif
    const bool ssdo = ui_config.ao_mode == EAoMode::SSDO;
    auto prepared = std::make_shared<CompositeRecordParameters>(PrepareComposite(ui_config, full_resolution, ao_resolution));
    const auto bindings = graph.ImportToken("AoComposite.bindings", prepared.get());
    graph.AddRecordPass("AoComposite.PublishBindings",
        [=](RenderGraph::PassBuilder& builder) { builder.Write(bindings); },
        [&, prepared, resources, ssdo](CommandList& commands) {
            prepared->bindless = RenderDevice::Get().CreateBindlessArray(8);
            const auto bind = [&](RenderGraph::TextureHandle resource, Sampler sampler = {SF_LINEAR, SAM_CLAMP_TO_EDGE}) {
                return prepared->bindless->AllocateTexture(graph.GetPhysicalTexture(resource)->GetView(), sampler);
            };
            prepared->output.tex = graph.GetPhysicalTexture(resources.ao_output);
            const Sampler ao_sampler(SF_LINEAR, prepared->shader.is_half_resolution ? SAM_CLAMP_TO_EDGE : SAM_REPEAT);
            prepared->shader.ao_tex = bind(resources.ao_only, ao_sampler);
            prepared->shader.color_tex = bind(resources.lighting_output, {SF_LINEAR, SAM_REPEAT});
            if (prepared->shader.is_half_resolution) {
                prepared->shader.depth_tex = bind(resources.depth, {SF_NEAREST, SAM_CLAMP_TO_EDGE});
                prepared->shader.normal_tex = bind(resources.normal, {SF_LINEAR, SAM_REPEAT});
            }
            if (ssdo) prepared->shader.indirect_tex = bind(resources.indirect, ao_sampler);
            commands.UpdateBindlessArray(prepared->bindless);
        }, RenderGraph::PassRecordPolicy::Serial);

    graph.AddRecordPass(
        "AoComposite",
        [resources, ssdo, bindings, half = ui_config.ao_half_resolution](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Compute)
                .Read(bindings).Read(resources.ao_only, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.lighting_output, RenderGraph::TextureAccessType::Sampled)
                .Write(resources.ao_output, RenderGraph::TextureAccessType::Storage);
            if (half) {
                builder.Read(resources.depth, RenderGraph::TextureAccessType::Sampled)
                    .Read(resources.normal, RenderGraph::TextureAccessType::Sampled);
            }
            if (ssdo) {
                builder.Read(resources.indirect, RenderGraph::TextureAccessType::Sampled);
            }
        },
        [this, prepared](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "AoComposite", [&] {
                RecordComposite(cmd_list, *prepared);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    return resources.ao_output;
}

void RtaoDenoiserPass::AddToGraph(
    RenderGraph&        graph,
    const RasterConfig& ui_config,
    const Camera&       camera,
    AoFrameResources    frame,
    GraphResources      resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.RtaoDenoiserPass");
    if (!frame.denoise) {
        return;
    }
    AddToGraph(graph, Prepare(ui_config, camera, frame), resources, frame.half_resolution);
}

void RtaoDenoiserPass::AddToGraph(
    RenderGraph& graph, RecordParameters parameters, GraphResources resources, bool half_resolution
) {
    auto prepared = std::make_shared<RecordParameters>(std::move(parameters));
    const auto bindings = graph.ImportToken("RtaoDenoise.bindings", prepared.get());
    graph.AddRecordPass("RtaoDenoise.PublishBindings",
        [=](RenderGraph::PassBuilder& builder) { builder.Write(bindings); },
        [&graph, prepared, resources, half_resolution](CommandList& commands) {
            prepared->bindless = RenderDevice::Get().CreateBindlessArray(8);
            const auto bind = [&](RenderGraph::TextureHandle resource) {
                return prepared->bindless->AllocateTexture(graph.GetPhysicalTexture(resource)->GetView(),
                    {SF_LINEAR, half_resolution ? SAM_CLAMP_TO_EDGE : SAM_REPEAT});
            };
            prepared->output.tex = graph.GetPhysicalTexture(resources.history_write);
            prepared->denoise.curr_ao_tex = bind(resources.ao_only);
            prepared->denoise.motion_vector_tex = bind(resources.camera_motion_vector);
            prepared->denoise.history_ao_tex = bind(resources.history_read);
            prepared->denoise.current_geometry_tex = bind(resources.current_geometry);
            prepared->denoise.history_geometry_tex = bind(resources.history_geometry);
            commands.UpdateBindlessArray(prepared->bindless);
        }, RenderGraph::PassRecordPolicy::Serial);

    graph.AddRecordPass(
        "RtaoDenoise",
        [resources, bindings](RenderGraph::PassBuilder& builder) {
            builder.Read(bindings).Read(resources.current_geometry, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.history_geometry, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.camera_motion_vector, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.history_read, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.ao_only, RenderGraph::TextureAccessType::Sampled)
                .Write(resources.history_write, RenderGraph::TextureAccessType::ColorAttachment);
        },
        [this, prepared](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "RtaoDenoise", [&] {
                RecordDenoise(cmd_list, *prepared);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void BilateralFilterDenoiserPass::AddToGraph(
    RenderGraph& graph,
    const RasterConfig& ui_config,
    Extent2D extent,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.BilateralFilterDenoiserPass");
    if (ui_config.denoiser_mode == EDenoiserMode::NONE) return;
    if (!extent.x || !extent.y) throw std::invalid_argument("Bilateral denoising requires a nonzero extent");
    auto prepared = std::make_shared<RecordParameters>(Prepare(ui_config, extent));
    const auto bindings = graph.ImportToken("BilateralDenoise.bindings", prepared.get());
    graph.AddRecordPass("BilateralDenoise.PublishBindings",
        [=](RenderGraph::PassBuilder& builder) { builder.Write(bindings); },
        [&, prepared, resources](CommandList& commands) {
            prepared->bindless = RenderDevice::Get().CreateBindlessArray(4);
            prepared->input.tex = graph.GetPhysicalTexture(resources.input);
            prepared->input.hdl = prepared->bindless->AllocateTexture(
                prepared->input.tex->GetView(), {SF_LINEAR, SAM_REPEAT});
            prepared->output.tex = graph.GetPhysicalTexture(resources.denoiser_output);
            prepared->shader.input_image = prepared->input.hdl;
            commands.UpdateBindlessArray(prepared->bindless);
        }, RenderGraph::PassRecordPolicy::Serial);
    graph.AddRecordPass(
        "BilateralDenoise",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(bindings).Read(resources.input, RenderGraph::TextureAccessType::Sampled)
                .Write(resources.denoiser_output, RenderGraph::TextureAccessType::ColorAttachment);
        },
        [this, prepared](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "BilateralDenoise", [&] {
                Record(cmd_list, *prepared);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void SsrPass::AddToGraph(
    RenderGraph& graph,
    const RasterConfig& ui_config,
    const Camera& camera,
    Extent2D extent,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.SsrPass");
    if (!ui_config.ssr_is_ssr_enabled) return;
    if (!extent.x || !extent.y) throw std::invalid_argument("SSR requires a nonzero extent");
    auto prepared = std::make_shared<RecordParameters>(Prepare(ui_config, camera, extent));
    const auto bindings = graph.ImportToken("ScreenSpaceReflection.bindings", prepared.get());
    graph.AddRecordPass("ScreenSpaceReflection.PublishBindings",
        [=](RenderGraph::PassBuilder& builder) { builder.Write(bindings); },
        [&, prepared, resources](CommandList& commands) {
            prepared->bindless = RenderDevice::Get().CreateBindlessArray(8);
            const auto bind = [&](RenderGraph::TextureHandle resource, Sampler sampler = {SF_LINEAR, SAM_REPEAT}) {
                return prepared->bindless->AllocateTexture(graph.GetPhysicalTexture(resource)->GetView(), sampler);
            };
            prepared->input.tex = graph.GetPhysicalTexture(resources.input);
            prepared->output.tex = graph.GetPhysicalTexture(resources.ssr_output);
            prepared->shader.color_tex = bind(resources.input);
            prepared->shader.normal_tex = bind(resources.normal);
            prepared->shader.depth_tex = bind(resources.depth, {SF_LINEAR, SAM_CLAMP_TO_EDGE});
            prepared->shader.gbuffer_metal_rough_ao = bind(resources.metal_rough_ao);
            commands.UpdateBindlessArray(prepared->bindless);
        }, RenderGraph::PassRecordPolicy::Serial);
    graph.AddRecordPass(
        "ScreenSpaceReflection",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(bindings).Read(resources.input, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.normal, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.depth, RenderGraph::TextureAccessType::Sampled)
                .Read(resources.metal_rough_ao, RenderGraph::TextureAccessType::Sampled)
                .Write(resources.ssr_output, RenderGraph::TextureAccessType::ColorAttachment);
        },
        [this, prepared](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "ScreenSpaceReflection", [&] {
                Record(cmd_list, *prepared);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

AaPass::GraphResult AaPass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    const RasterConfig& ui_config,
    const Camera& camera,
    uint8 smaa_phase,
    GraphResources resources,
    bool inspect_weights
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.AaPass");
    return AddRecordPasses(graph, ui_config, camera, {
        .extent = Extent2D(context.GetResolution()),
        .depth = context.textures.depth_linear_sampler.tex->CastToTextureRef(),
        .history = {context.textures.aa_texture_3.tex, context.textures.aa_texture_4.tex}
    }, smaa_phase, resources, inspect_weights);
}

AaPass::GraphResult AaPass::AddRecordPasses(
    RenderGraph& graph,
    const RasterConfig& ui_config,
    const Camera& camera,
    FrameTextures textures,
    uint8 smaa_phase,
    GraphResources resources,
    bool inspect_weights
) {
    if (!textures.extent.x || !textures.extent.y || !resources.input.IsValid() ||
        !resources.aa_output.IsValid() || smaa_phase > 1) {
        throw std::invalid_argument("AA requires input/output handles, extent and a valid temporal phase");
    }
    if (ui_config.aa_mode == EAaMode::SMAA_T2X &&
        (!textures.depth || !textures.history[0] || !textures.history[1])) {
        throw std::invalid_argument("SMAA T2x requires depth and both persistent history images");
    }
    using State       = RenderGraph::TextureAccessType;
    const auto import = [&](std::string_view name, const TextureRef& texture) {
        return ImportRasterTexture(graph, name, texture.Get());
    };
    const bool smaa = ui_config.aa_mode >= EAaMode::SMAA_1X;
    const bool temporal = ui_config.aa_mode == EAaMode::SMAA_T2X;
    const auto extent = textures.extent;
    const RGTransientTextureDesc scratch_desc{
        .extent = Extent3D(extent.x, extent.y, 1),
        .format = output_format,
        .usage = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT |
                 ETextureUsageFlags::TRANSFER_DST | ETextureUsageFlags::TRANSFER_SRC
    };
    const auto texture_1 = graph.CreateTransientTexture("AA.edges_or_luma", scratch_desc);
    const auto texture_2 = smaa || inspect_weights ?
        graph.CreateTransientTexture("AA.weights", scratch_desc) : RenderGraph::TextureHandle{};
    auto prepared = std::make_shared<RecordParameters>();
    const auto bindings_token = graph.ImportToken("AA.bindings", prepared.get());
    // Resolve pooled images after compilation. A private table avoids mutating
    // the scene's shared bindings from parallel recording callbacks.
    graph.AddRecordPass(
        "AA.PublishBindings",
        [=](RenderGraph::PassBuilder& builder) { builder.Write(bindings_token); },
        [this, &graph, &ui_config, &camera, textures, resources, smaa_phase,
         texture_1, texture_2, temporal, smaa, prepared](CommandList& commands) {
            FrameBindings bindings{};
            bindings.bindless = RenderDevice::Get().CreateBindlessArray(16);
            const auto bind = [&](TextureRef texture, Sampler sampler = {SF_LINEAR, SAM_REPEAT}) {
                TextureWithHandle result{};
                result.tex = std::move(texture);
                result.hdl = bindings.bindless->AllocateTexture(result.tex->GetView(), sampler);
                return result;
            };
            bindings.input = bind(graph.GetPhysicalTexture(resources.input));
            bindings.output.tex = graph.GetPhysicalTexture(resources.aa_output);
            bindings.edges = bind(graph.GetPhysicalTexture(texture_1));
            if (smaa) {
                bindings.weights = bind(graph.GetPhysicalTexture(texture_2));
                bindings.area = bind(smaa_area_tex.tex).hdl;
                bindings.search = bind(smaa_search_tex.tex).hdl;
            }
            if (temporal) {
                bindings.current = bind(textures.history[smaa_phase]);
                bindings.previous = history_valid ? bind(textures.history[smaa_phase ^ 1u]) : bindings.current;
                bindings.depth = bind(
                    textures.depth,
                    {SF_LINEAR, SAM_CLAMP_TO_EDGE}
                ).hdl;
            } else {
                bindings.current = bindings.output;
                bindings.previous = bindings.output;
            }
            *prepared = Prepare(ui_config, camera, smaa_phase, bindings);
            commands.UpdateBindlessArray(bindings.bindless);
        },
        RenderGraph::PassRecordPolicy::Serial
    );
    if (!smaa && inspect_weights) {
        graph.AddRecordPass(
            "AA.ClearUnusedWeights",
            [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                    .Write(texture_2, State::TransferDestination);
            },
            [&graph, texture_2](CommandList& commands) {
                commands.ClearResource(graph.GetPhysicalTexture(texture_2)->GetView(), float4(0.f));
            },
            RenderGraph::PassRecordPolicy::ParallelAllowed
        );
    }
    graph.AddRecordPass(
        smaa ? "AA.Edges" : "AA.Luma",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(bindings_token)
                .Read(resources.input, State::Sampled)
                .Write(texture_1, State::ColorAttachment);
            if (temporal) {
                builder.Read(resources.depth, State::Sampled);
            }
        },
        [this, prepared](CommandList& commands) {
            RecordEdgesOrLuma(commands, *prepared);
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    if (smaa) {
        const auto area = import("AA.area", smaa_area_tex.tex);
        const auto search = import("AA.search", smaa_search_tex.tex);
        graph.AddRecordPass(
            "AA.Weights",
            [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                    .Read(bindings_token)
                    .Read(texture_1, State::Sampled)
                    .Read(area, State::Sampled)
                    .Read(search, State::Sampled)
                    .Write(texture_2, State::ColorAttachment);
            },
            [this, prepared](CommandList& commands) {
                RecordWeights(commands, *prepared);
            },
            RenderGraph::PassRecordPolicy::ParallelAllowed
        );
    }
    const auto blend_output = temporal ? import("AA.current", textures.history[smaa_phase]) : resources.aa_output;
    graph.AddRecordPass(
        "AA.Blend",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(bindings_token)
                .Read(texture_1, State::Sampled)
                .Write(blend_output, State::ColorAttachment);
            if (smaa) {
                builder.Read(resources.input, State::Sampled).Read(texture_2, State::Sampled);
            }
        },
        [this, prepared](CommandList& commands) {
            RecordBlend(commands, *prepared);
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    if (temporal) {
        // An invalid history samples the current output, exactly as Prepare's shader handle does.
        const auto previous = history_valid ? import("AA.previous", textures.history[smaa_phase ^ 1u]) : blend_output;
        graph.AddRecordPass(
            "AA.Resolve",
            [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                    .Read(bindings_token)
                    .Read(blend_output, State::Sampled)
                    .Read(previous, State::Sampled)
                    .Read(texture_1, State::Sampled)
                    .Write(resources.aa_output, State::ColorAttachment);
            },
            [this, prepared](CommandList& commands) {
                RecordResolve(commands, *prepared);
            },
            RenderGraph::PassRecordPolicy::ParallelAllowed
        );
    }
    return {.edges = texture_1, .weights = texture_2};
}

BloomPass::GraphResources BloomPass::AddToGraph(
    RenderGraph&               graph,
    bool                       enabled,
    RenderGraph::TextureHandle input,
    Extent2D                   extent
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.BloomPass");
    GraphResources resources{.input = input};
    if (!enabled) {
        return resources;
    }
    if (extent.width == 0 || extent.height == 0) {
        throw std::invalid_argument("Bloom requires a nonzero input extent");
    }
    const uint32                 mip_count = std::min<uint32>(6, std::bit_width(std::max(extent.width, extent.height)));
    const RGTransientTextureDesc desc{
        .extent    = Extent3D(extent),
        .format    = PF_B10G11R11_UFLOAT_PACK32,
        .usage     = ETextureUsageFlags::SAMPLED | ETextureUsageFlags::COLOR_ATTACHMENT,
        .mip_count = mip_count,
    };
    // The graph owns logical lifetimes; recorded commands retain the physical
    // textures until Completion, after which the existing pool may reuse them.
    resources.downsample_chain = graph.CreateTransientTexture("Bloom.DownsampleChain", desc);
    if (mip_count > 1) {
        resources.upsample_chain = graph.CreateTransientTexture("Bloom.UpsampleChain", desc);
    }
    AddRecordPasses(graph, resources, mip_count);
    return resources;
}

RenderGraph::PassHandle BloomPass::AddRecordPasses(
    RenderGraph&   graph,
    GraphResources resources,
    uint32         mip_count
) {
    using Range = RenderGraph::TextureRange;
    using State = RenderGraph::TextureAccessType;
    if (mip_count == 0) {
        throw std::invalid_argument("Bloom graph requires at least one mip");
    }
    // Every draw is one synchronization boundary. Different mips of the same
    // image can be sampled and written concurrently; never declare whole-chain writes.
    graph.AddRecordPass(
        "Bloom.Prefilter",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(resources.input, State::Sampled, Range::Mips(0, 1))
                .Write(resources.downsample_chain, State::ColorAttachment, Range::Mips(0, 1));
        },
        [this, &graph, resources](CommandList& commands) {
            RecordPrefilter(
                commands,
                graph.GetPhysicalTexture(resources.input),
                graph.GetPhysicalTexture(resources.downsample_chain)
            );
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    for (uint32 mip = 1; mip < mip_count; ++mip) {
        graph.AddRecordPass(
            std::format("Bloom.Downsample{}", mip),
            [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                    .Read(resources.downsample_chain, State::Sampled, Range::Mips(mip - 1, 1))
                    .Write(resources.downsample_chain, State::ColorAttachment, Range::Mips(mip, 1));
            },
            [this, &graph, resources, mip](CommandList& commands) {
                RecordDownsample(commands, graph.GetPhysicalTexture(resources.downsample_chain), mip);
            },
            RenderGraph::PassRecordPolicy::ParallelAllowed
        );
    }
    for (uint32 source_mip = mip_count - 1; source_mip > 0; --source_mip) {
        const uint32 mip = source_mip - 1;
        const auto source = source_mip == mip_count - 1 ? resources.downsample_chain : resources.upsample_chain;
        graph.AddRecordPass(
            std::format("Bloom.Upsample{}", mip),
            [=](RenderGraph::PassBuilder& builder) {
                builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                    .Read(source, State::Sampled, Range::Mips(source_mip, 1))
                    .Read(resources.downsample_chain, State::Sampled, Range::Mips(mip, 1))
                    .Write(resources.upsample_chain, State::ColorAttachment, Range::Mips(mip, 1));
            },
            [this, &graph, resources, source, mip](CommandList& commands) {
                RecordUpsample(
                    commands,
                    graph.GetPhysicalTexture(source),
                    graph.GetPhysicalTexture(resources.downsample_chain),
                    graph.GetPhysicalTexture(resources.upsample_chain),
                    mip
                );
            },
            RenderGraph::PassRecordPolicy::ParallelAllowed
        );
    }
    return graph.AddRecordPass(
        "Bloom.Apply",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Graphics)
                .Read(mip_count == 1 ? resources.downsample_chain : resources.upsample_chain, State::Sampled, Range::Mips(0, 1))
                // Additive blending loads the existing scene color.
                .ReadWrite(resources.input, State::ColorAttachment, Range::Mips(0, 1));
        },
        [this, &graph, resources, mip_count](CommandList& commands) {
            RecordApply(
                commands,
                graph.GetPhysicalTexture(resources.input),
                graph.GetPhysicalTexture(mip_count == 1 ? resources.downsample_chain : resources.upsample_chain)
            );
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void CameraGizmoPass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    const Camera& scene_camera,
    const Camera& main_camera,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.CameraGizmoPass");
    auto prepared = graph.AddSetupPass(
        "CameraGizmo.Prepare",
        uint8{0},
        [this, &scene_camera, &main_camera](const uint8&) {
            return Prepare(scene_camera, main_camera);
        }
    );
    if (!prepared.IsValid()) {
        return;
    }

    graph.AddRecordPass(
        "CameraGizmo",
        [resources](RenderGraph::PassBuilder& builder) {
            builder.ReadWrite(resources.tonemapping_output, RenderGraph::TextureAccessType::ColorAttachment)
                .SideEffect();
        },
        [this, &graph, resources, prepared](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "CameraGizmo", [&] {
                auto parameters = prepared.Get();
                parameters.output.tex = graph.GetPhysicalTexture(resources.tonemapping_output);
                Record(cmd_list, parameters);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

void CsmGizmoPass::AddToGraph(
    RenderGraph& graph,
    const RasterContext& context,
    const RasterConfig& config,
    const SceneViewGizmoConfig& gizmos,
    const Camera& camera,
    const Camera& main_camera,
    GraphResources resources,
    std::span<const RenderGraph::SetupPassHandle> setup_dependencies
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.CsmGizmoPass");
    auto prepared = graph.AddSetupPass(
        "CsmGizmo.Prepare",
        uint8{0},
        [this, &context, &config, &gizmos, &camera, &main_camera](const uint8&) {
            return Prepare(context, config, gizmos, camera, main_camera);
        },
        setup_dependencies
    );
    if (!prepared.IsValid()) {
        return;
    }

    const auto data = ImportRasterBuffer(graph, "CsmGizmo.data", m_cascade_data_buffer.buf.Get());
    graph.AddRecordPass(
        "CsmGizmo.Upload",
        [=](RenderGraph::PassBuilder& builder) {
            builder.ExecuteOn(RenderGraph::QueueRole::Graphics, RenderGraph::GpuWorkType::Transfer)
                .Write(data, RenderGraph::BufferAccessType::TransferDestination);
        },
        [this, prepared](CommandList& commands) {
            RecordUpload(commands, prepared.Get());
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
    graph.AddRecordPass(
        "CsmGizmo",
        [resources, data](RenderGraph::PassBuilder& builder) {
            builder.Read(data, RenderGraph::BufferAccessType::ShaderRead);
            builder.Read(resources.shadow_maps)
                .ReadWrite(resources.tonemapping_output, RenderGraph::TextureAccessType::ColorAttachment)
                .SideEffect();
        },
        [this, &graph, resources, prepared](CommandList& cmd_list) {
            RecordWithPassMarker(cmd_list, "CsmGizmo", [&] {
                auto parameters = prepared.Get();
                parameters.output.tex = graph.GetPhysicalTexture(resources.tonemapping_output);
                RecordDraw(cmd_list, parameters);
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

} // namespace Moer::Render::Raster

namespace Moer::Render {

void UiCombinePass::AddToGraph(
    RenderGraph& graph,
    bool ui_enabled,
    bool writes_external_window,
    bool is_separate_window,
    uint2 resolution,
    float2 scene_color_position,
    float2 scene_color_resolution,
    TextureView window_framebuffer,
    GraphResources resources
) {
    MOER_PROFILE_SCOPE("Raster.AddToGraph.UiCombinePass");
    graph.AddRecordPass(
        "UiCombine",
        [resources, ui_enabled, writes_external_window](RenderGraph::PassBuilder& builder) {
            if (!ui_enabled) {
                builder.Read(resources.processing_input, RenderGraph::TextureAccessType::Sampled)
                    .Write(resources.output, RenderGraph::TextureAccessType::ColorAttachment);
            } else if (writes_external_window) {
                builder.Read(resources.selected_framebuffer, RenderGraph::TextureAccessType::Sampled)
                    .Write(resources.window_framebuffer, RenderGraph::TextureAccessType::ColorAttachment);
            } else {
                builder.Read(resources.selected_framebuffer, RenderGraph::TextureAccessType::Sampled)
                    .Write(resources.output, RenderGraph::TextureAccessType::ColorAttachment);
            }
            builder.SideEffect();
        },
        [=, this, &graph](CommandList& cmd_list) {
            Raster::RecordWithPassMarker(cmd_list, "UiCombine", [&] {
                const TextureView output(graph.GetPhysicalTexture(resources.output));
                const TextureView input(graph.GetPhysicalTexture(
                    ui_enabled ? resources.selected_framebuffer : resources.processing_input
                ));
                Record(cmd_list, Prepare(
                    ui_enabled ? is_separate_window : true,
                    resolution,
                    ui_enabled ? scene_color_position : float2(0.f),
                    ui_enabled ? scene_color_resolution : float2(resolution.x, resolution.y),
                    ui_enabled ? window_framebuffer : output,
                    input,
                    output
                ));
            });
        },
        RenderGraph::PassRecordPolicy::ParallelAllowed
    );
}

} // namespace Moer::Render
