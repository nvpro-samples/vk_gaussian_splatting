/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameters.h"

namespace vk_gaussian_splatting {

// no reset function on purpose
SceneParameters prmScene{};

VramDataParameters    prmData{};
RtxVramDataParameters prmRtxData{};

// no reset function on purpose
uint32_t             prmSelectedPipeline = PIPELINE_MESH;
shaderio::FrameInfo  prmFrame{};
RenderParameters     prmRender{};
RasterParameters     prmRaster{};
RtxParameters        prmRtx{};
ComparisonParameters prmComparison{};

// Storage for respective default values

static VramDataParameters    prmDataDefault{};
static RtxVramDataParameters prmRtxDataDefault{};

static shaderio::FrameInfo  prmFrameDefault{};
static RenderParameters     prmRenderDefault{};
static RasterParameters     prmRasterDefault{};
static RtxParameters        prmRtxDefault{};
static ComparisonParameters prmComparisonDefault{};

void storeDefaultParameters()
{
  prmDataDefault    = prmData;
  prmRtxDataDefault = prmRtxData;

  prmFrameDefault      = prmFrame;
  prmRenderDefault     = prmRender;
  prmRasterDefault     = prmRaster;
  prmRtxDefault        = prmRtx;
  prmComparisonDefault = prmComparison;
}

void resetDataParameters()
{
  prmData = prmDataDefault;
}
void resetRtxDataParameters()
{
  prmRtxData = prmRtxDataDefault;
}
void resetFrameParameters()
{
  prmFrame = prmFrameDefault;
}
void resetRenderParameters()
{
  prmRender = prmRenderDefault;
}
void resetRasterParameters()
{
  prmRaster = prmRasterDefault;
}
void resetRtxParameters()
{
  prmRtx = prmRtxDefault;
}
void resetComparisonParameters()
{
  prmComparison = prmComparisonDefault;
}

void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry)
{
  // Scene - Command-line files added to queue via callback
  // First file resets scene, subsequent files are additive
  parameterRegistry->addCustom({"inputFile", "load a ply, spz or splat file"},
                               1,  // argCount: expects 1 argument (the filename)
                               [](const nvutils::ParameterBase* const, std::span<char const* const> args,
                                  const std::filesystem::path& basePath) -> bool {
                                 if(args.empty())
                                   return false;

                                 std::filesystem::path filePath = args[0];

                                 // Make absolute path if relative
                                 if(filePath.is_relative())
                                   filePath = basePath / filePath;

                                 // First file resets scene, subsequent are additive
                                 bool isFirstFile = prmScene.sceneLoadQueue.empty();
                                 prmScene.pushLoadRequest(filePath, isFirstFile);

                                 return true;  // Successfully handled
                               },
                               {".ply", ".spz", ".splat"}  // Extensions for suffix-based triggering
  );
#ifdef WITH_DEFAULT_SCENE_FEATURE
  parameterRegistry->add({"loadDefaultScene", "0=disable the load of a default scene when no ply file is provided"},
                         &prmScene.enableDefaultScene);
#endif
  // Projects
  parameterRegistry->add({.name            = "inputProject",
                          .help            = "load a vkgs project file",
                          .callbackSuccess = [](const nvutils::ParameterBase* const) { prmScene.projectLoadPorcelain = true; }},
                         {".vkgs"}, &prmScene.projectToLoadFilename);

  // Data
  parameterRegistry->add({"shformat", "0=fp32 1=fp16 2=uint8"}, &prmData.shFormat);
  parameterRegistry->add({"rgbaformat", "0=fp32 1=fp16 2=uint8"}, &prmData.rgbaFormat);
  parameterRegistry->add({"useAABBs", "0=use icosahedron 3D mesh and built-in triangle/ray intersection (Default), 1=use AABBs and parametric intersection shader. DO NOT COMBINE with useTlasInstances=0."},
                         &prmRtxData.useAABBs);
  parameterRegistry->add({"useSpheres", "1=use sphere primitives (VK_NV_ray_tracing_linear_swept_spheres). Mutually exclusive with useAABBs."},
                         &prmRtxData.useSpheres);
  parameterRegistry->add({"useTlasInstances", "1=use one TLAS instance per particle and a small unit particle BLAS (default). 0=use one TLAS entry and a large BLAS."},
                         &prmRtxData.useTlasInstances);
  parameterRegistry->add({"compressBlas", "1=compress BLAS (default). 0=diabled."}, &prmRtxData.compressBlas);

  // Pipelines
  parameterRegistry->add({"pipeline", "0=3dgs-vert 1=3dgs-mesh(default) 2=3dgrt 3=hybrid-3dgs 4=3dgut 5=hybrid-3dgut"},
                         &prmSelectedPipeline);
  parameterRegistry->add({"maxShDegree", "max sh degree used for rendering in [0,1,2,3]"}, &prmFrame.shDegree,
                         int32_t(0), int32_t(3));

  // Rendering
  parameterRegistry->add({"lightingEnabled", "0=off(default) 1=enable lighting"}, &prmRender.lightingEnabled,
                         (int32_t)LIGHTING_DISABLED, (int32_t)LIGHTING_ENABLED);
  parameterRegistry->add({"shadowMode", "0=off(default) 1=hard shadows 2=soft shadows"},
                         (uint32_t*)&prmRender.shadowsMode, (uint32_t)SHADOWS_DISABLED, (uint32_t)SHADOWS_SOFT);
  parameterRegistry->add({"gsShadowMask", "1=shadow-only (gs-shadow) lights darken splat emissive output (RTX pipelines)"},
                         &prmRender.gsShadowMask);
  parameterRegistry->add({"gsShadowMaskMin", "shadow mask floor in [0,1], 0=black shadows (default 0.2)"},
                         &prmRender.gsShadowMaskMin);
  parameterRegistry->add({"gsShadowMaskFromParticles", "1=particles also occlude gs-shadow mask rays (self-shadow risk)"},
                         &prmRender.gsShadowMaskFromParticles);

  // Rasterization
  parameterRegistry->add({"sortStrategy", "particles rasterization strategy 0=GPU radix sort(default) 1=CPU async mono 2=CPU async multi 3=stochastic splat"},
                         &prmRaster.sortingMethod, SORTING_GPU_SYNC_RADIX, SORTING_STOCHASTIC_SPLAT);
  parameterRegistry->add({"extentProjection", "particle extent projection method [0=Eigen (default),1=Conic]"},
                         &prmRaster.extentProjection);

  // Ray Tracing
  parameterRegistry->add({"rtxSortStrategy", "particles raytracing strategy 0=full anyhit(default) 1=pass stochastic 2=stochastic anyhit"},
                         &prmRtx.rtxTraceStrategy, RTX_TRACE_STRATEGY_FULL_ANYHIT, RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT);
  parameterRegistry->add({"rtxMaxBounces", "max light bounces for path tracing [0-16, default=3]"},
                         &prmFrame.rtxMaxBounces, int32_t(0), int32_t(16));
  parameterRegistry->add({"kernelDegree", "kernel degree used by 3DGRT, 3DGUT and Hybrid 3DGUT pipelines in [0,1,2(default),3,4,5]"},
                         &prmRtx.kernelDegree, int32_t(0), int32_t(5));
  parameterRegistry->add({"rtxParticleDepth", "particle depth mode 0=billboard 1=ellipsoid(default)"},
                         &prmRtx.particleDepth, PARTICLE_DEPTH_BILLBOARD, PARTICLE_DEPTH_ELLIPSOID);
  parameterRegistry->add({"rtxShortenRay", "shorten ray in stochastic any-hit billboard mode 0=off 1=on(default)"},
                         &prmRtx.shortenRay);
  parameterRegistry->add({"rtxBillboardBounding", "billboard bounding mode 0=fitted 1=uniform 2=uniform3/4 3=uniform2/3 4=uniform1/2 5=uniform1/3 6=uniform1/4 7=optimal"},
                         (int32_t*)&prmRtxData.billboardBoundingMode, int32_t(0), int32_t(7));
}

}  // namespace vk_gaussian_splatting
