/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nvutils/parameter_registry.hpp>

#include "shaderio.h"

namespace vk_gaussian_splatting {

// Parameters that controls the scene
struct SceneParameters
{
#ifdef WITH_DEFAULT_SCENE_FEATURE
  // do we load a default scene at startup if none is provided through CLI
  bool enableDefaultScene = true;
#endif

  // triggers a scene load at next frame when set to non empty string
  std::filesystem::path sceneToLoadFilename;
  // triggers a project load at next frame when set to non empty string
  std::filesystem::path projectToLoadFilename;
  // triggers an obj file import at next frame when set to non empty string
  std::filesystem::path meshToImportFilename;
};

// Parameters that controls the scene
extern SceneParameters prmScene;

// Parameters that controls data format and storage in VRAM, shared by all pipeline
struct VramDataParameters
{
  int shFormat    = FORMAT_FLOAT32;
  int dataStorage = STORAGE_BUFFERS;
};

// Parameters that controls data storage
extern VramDataParameters prmData;

// Parameters that controls data format and storage in VRAM, specific to RTX pipelines
// Mainly about acceleration structures and particle primitive geometry
struct RtxVramDataParameters
{
  // if true will compact BLAS
  bool compressBlas = true;
  // set to true to use AABBs instead of mesh ICOSA primitives
  // This will also make the Rtx pipeline use parametric intersections
  bool useAABBs = false;
  // if true, use one instance per splat in TLAS and single splat model in BLAS
  // otherwise, only one instance in TLAS and all splats transformed in BLAS
  bool useTlasInstances = true;
};

// Parameters that controls data storage
extern RtxVramDataParameters prmRtxData;

// Parameters common to all rendering pipelines and provided to shaders as a UniformBufffer
// FrameInfo is defined in shaderio.h since declaration is shared with shaders
extern shaderio::FrameInfo prmFrame;

// pipeline selector
extern uint32_t prmSelectedPipeline;

// Parameters common to all rendering pipelines
struct RenderParameters
{
  int  visualize               = VISUALIZE_FINAL;
  bool wireframe               = false;  // display bounding volume
  int  maxShDegree             = 3;      // in [0,3]
  bool showShOnly              = false;
  bool opacityGaussianDisabled = false;
};

// Parameters common to all rendering pipelines
extern RenderParameters prmRender;

// Parameters that control rasterization
struct RasterParameters
{
  int32_t sortingMethod           = SORTING_GPU_SYNC_RADIX;
  bool    cpuLazySort             = true;  // if true, sorting starts only if viewpoint changed
  int     frustumCulling          = FRUSTUM_CULLING_AT_DIST;
  int     distShaderWorkgroupSize = 256;  // best default value set by experimentation on ADA6000
  int     meshShaderWorkgroupSize = 32;   // best default value set by experimentation on ADA6000
  bool    fragmentBarycentric     = false;
  bool    pointCloudModeEnabled   = false;
  int     extentProjection        = EXTENT_CONIC;
  // Whether gaussians should be rendered with mip-splat
  // antialiasing https://niujinshuchong.github.io/mip-splatting/
  bool msAntialiasing = false;
};

// Parameters that control rasterization
extern RasterParameters prmRaster;

// Parameters that control Raytracing (RTX)
struct RtxParameters
{
  // temporalSampling is controlled by temporalSamplingMode, it is not directly exposed
  bool  temporalSampling       = false;  // do we accumulate frame results over time (for DOF and other)
  int   temporalSamplingMode   = TEMPORAL_SAMPLING_AUTO;  // how do we control temporal sampling activation
  int   kernelDegree           = KERNEL_DEGREE_QUADRATIC;
  float kernelMinResponse      = 0.0113f;  // constant value from Paper
  bool  kernelAdaptiveClamping = true;
  int   payloadArraySize       = 18;     // best default value set by experimentation on ADA6000
  bool  usePayloadBuffer       = false;  // Experimental, change the value here for testing, no UI
};

// Parameters that control Raytracing (RTX)
extern RtxParameters prmRtx;

// Invoked by main() to save defaults after command line options are applied at startup
void storeDefaultParameters();

// Reset prmData to defaults
void resetDataParameters();
// Reset prmRtxData to defaults
void resetRtxDataParameters();

// Reset prmFrame to defaults
void resetFrameParameters();
// Reset prmRender to defaults
void resetRenderParameters();
// Reset prmRaster to defaults
void resetRasterParameters();
// Reset prmRtx to defaults
void resetRtxParameters();

// register the set of global parameters
void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry);

}  // namespace vk_gaussian_splatting
