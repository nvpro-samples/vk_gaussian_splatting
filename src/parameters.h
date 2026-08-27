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

#pragma once

#include <deque>
#include <memory>
#include <nvutils/parameter_registry.hpp>

#include "shaderio.h"
#include "image_compare.h"  // For ImageCompare::Mode and ImageCompare::Parameters

namespace vk_gaussian_splatting {

// Forward declarations for pre-configured objects
class SplatSetVk;
struct SplatSetInstanceVk;  // Note: SplatSetInstanceVk is a struct, not class

// Scene load request for queue-based loading
struct SceneLoadRequest
{
  std::filesystem::path path;               // File to load
  bool                  porcelain = false;  // Was the request enqueued by command line or project loader

  // Optional: Pre-configured objects (for project loading)
  // If provided, loader will use these instead of creating new ones
  std::shared_ptr<SplatSetVk>         splatSet;  // Optional: use this splat set asset
  std::shared_ptr<SplatSetInstanceVk> instance;  // Optional: use this instance (with transform/material)

  // Additional instances to create after loading (for Version 1+ project files)
  // These instances share the same splat set as the primary instance
  std::vector<std::shared_ptr<SplatSetInstanceVk>> additionalInstances;
};

// Parameters that controls the scene
struct SceneParameters
{
#ifdef WITH_DEFAULT_SCENE_FEATURE
  // do we load a default scene at startup if none is provided through CLI
  bool enableDefaultScene = true;
#endif

  // Queue-based scene loading (multi-file support)
  std::deque<SceneLoadRequest> sceneLoadQueue;

  // Helper methods for queue management
  void pushLoadRequest(const std::filesystem::path& path, bool porcelain = false)
  {
    SceneLoadRequest request;
    request.path      = path;
    request.porcelain = porcelain;
    sceneLoadQueue.push_back(request);
  }

  void pushLoadRequest(const SceneLoadRequest& request) { sceneLoadQueue.push_back(request); }

  bool hasLoadRequests() const { return !sceneLoadQueue.empty(); }

  // triggers a project load at next frame when set to non empty string
  std::filesystem::path projectToLoadFilename;
  bool                  projectLoadPorcelain = false;
  // triggers an obj file import at next frame when set to non empty string
  std::filesystem::path meshToImportFilename;
};

// Parameters that controls the scene
extern SceneParameters prmScene;

// Parameters that controls data format in VRAM, shared by all pipeline
struct VramDataParameters
{
  int shFormat   = FORMAT_UINT8;
  int rgbaFormat = FORMAT_UINT8;
};

// Parameters that controls data storage
extern VramDataParameters prmData;

// Billboard bounding mode for TLAS instance scale computation (host-side only, no shader macro)
enum class BillboardBoundingMode : int
{
  eBillboardBoundingFitted     = 0,
  eBillboardBoundingUniform    = 1,  // max
  eBillboardBoundingUniform3_4 = 2,  // max * 3/4
  eBillboardBoundingUniform2_3 = 3,  // max * 2/3
  eBillboardBoundingUniform1_2 = 4,  // max * 1/2
  eBillboardBoundingUniform1_3 = 5,  // max * 1/3
  eBillboardBoundingUniform1_4 = 6,  // max * 1/4
  eBillboardBoundingOptimal    = 7,  // per-axis: (max + s_i) / 2
};

// Parameters that controls data format and storage in VRAM, specific to RTX pipelines
// Mainly about acceleration structures and particle primitive geometry
struct RtxVramDataParameters
{
  // if true will compact BLAS
  bool compressBlas = true;
  // set to true to use AABBs instead of mesh ICOSA primitives
  // This will also make the Rtx pipeline use parametric intersections
  bool useAABBs = false;
  // set to true to use sphere primitives (VK_NV_ray_tracing_linear_swept_spheres)
  // Mutually exclusive with useAABBs; forces useTlasInstances = true
  bool useSpheres = false;
  // if true, use one instance per splat in TLAS and single splat model in BLAS
  // (shared TLAS mode: one TLAS per unique asset with local-only particle transforms,
  // per-instance global transform applied to ray before tracing, TLAS reused across instances)
  // otherwise, only one instance in TLAS and all splats transformed in BLAS
  bool useTlasInstances = true;
  // Billboard bounding mode for TLAS instance scale computation
  BillboardBoundingMode billboardBoundingMode = BillboardBoundingMode::eBillboardBoundingFitted;
};

// Parameters that controls data storage
extern RtxVramDataParameters prmRtxData;

// Parameters common to all rendering pipelines and provided to shaders as a UniformBufffer
// FrameInfo is defined in shaderio.h since declaration is shared with shaders
extern shaderio::FrameInfo prmFrame;

// pipeline selector
extern uint32_t prmSelectedPipeline;

// C++ enum classes for type-safe UI combobox registration (host-side only)
// Values match the #defines in shaderio.h for shader compatibility
enum class NormalMethod : uint32_t
{
  eMaxDensityPlane = NORMAL_METHOD_MAX_DENSITY_PLANE,
  eIsoSurface      = NORMAL_METHOD_ISO_SURFACE,
};

enum class DofMode : int
{
  eDofDisabled   = DOF_DISABLED,
  eDofFixedFocus = DOF_FIXED_FOCUS,
  eDofAutoFocus  = DOF_AUTO_FOCUS,
};

enum class ShadowsMode : uint32_t
{
  eShadowsDisabled = SHADOWS_DISABLED,
  eShadowsHard     = SHADOWS_HARD,
  eShadowsSoft     = SHADOWS_SOFT,
};

// Parameters common to all rendering pipelines
struct RenderParameters
{
  // Alternative visualization modes

  int       visualize       = VISUALIZE_FINAL;
  float     hitsVisuShift   = 0.0;
  glm::vec2 hitsVisuMinMax  = glm::vec2(0, 100);
  float     clockVisuShift  = 0.0;
  glm::vec2 clockVisuMinMax = glm::vec2(0.0, 0.5);
  float     depthVisuShift  = 0.0;
  glm::vec2 depthVisuMinMax = glm::vec2(0.0, 20.0);

  // Normal computation method
  NormalMethod normalMethod          = NormalMethod::eMaxDensityPlane;  // Normal vector computation method
  float        thinParticleThreshold = 1e-6f;  // Scale below which a particle axis is considered degenerate

  // Global lighting and shadows controls
  int32_t     lightingEnabled = LIGHTING_DISABLED;              // Enable lighting for all models
  ShadowsMode shadowsMode     = ShadowsMode::eShadowsDisabled;  // Shadows mode for all models (RTX only)

  // GS shadow mask: shadow-only (gs-shadow) lights darken the splat emissive output
  // at the reconstructed surfel (RTX pipelines only). Implies LIGHTING_MODE compilation
  // and surface reconstruction; pure-emissive splat sets keep their baked appearance.
  bool  gsShadowMask              = false;  // master enable
  float gsShadowMaskMin           = 0.2f;   // shadow floor: 0 = shadows go black, 1 = no visible shadow
  bool  gsShadowMaskFromParticles = false;  // also let particles occlude mask rays (self-shadow risk)

  // Color buffer format
  VkFormat colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

  // Clay visualization color (used when visualize == VISUALIZE_CLAY)
  glm::vec3 clayColor = glm::vec3(0.423, 0.337, 0.251);

  // Gaussians specific

  bool wireframe               = false;  // display bounding volume
  bool showShOnly              = false;
  bool opacityGaussianDisabled = false;
};

// Parameters common to all rendering pipelines
extern RenderParameters prmRender;

// Effective lighting compile mode: the GS shadow mask needs the particle shading
// path (and its linear-space/tonemapper contract) compiled in, so enabling it
// implies LIGHTING_ENABLED even when the user keeps lighting off. Pure-emissive
// splat sets still early-exit and keep their baked appearance.
inline int32_t effectiveLightingMode()
{
  return (prmRender.lightingEnabled == LIGHTING_ENABLED || prmRender.gsShadowMask) ? LIGHTING_ENABLED : LIGHTING_DISABLED;
}

// Parameters that control rasterization
struct RasterParameters
{
  int32_t sortingMethod           = SORTING_GPU_SYNC_RADIX;
  bool    cpuLazySort             = true;  // if true, sorting starts only if viewpoint changed
  int     frustumCulling          = FRUSTUM_CULLING_AT_DIST;
  int     sizeCulling             = SIZE_CULLING_DISABLED;  // size-based culling of small splats
  int     distShaderWorkgroupSize = 256;                    // best default value set by experimentation on ADA6000
  int     meshShaderWorkgroupSize = 32;                     // best default value set by experimentation on ADA6000
  bool    fragmentBarycentric     = false;
  bool    pointCloudModeEnabled   = false;
  int     extentProjection        = EXTENT_CONIC;
  // Whether gaussians should be rendered with mip-splat
  // Effective only if covarianceDilation > 0.0
  // antialiasing https://niujinshuchong.github.io/mip-splatting/
  bool msAntialiasing = false;
  // 2D covariance low-pass kernel size.
  // MipSplatting default: 0.1, 3DGS/3DGUT/StochasticSplat/StopThePop default: 0.3
  float covarianceDilation = 0.3f;
  // Use octahedral encoding for normals (reduces mesh->fragment bandwidth)
  bool quantizeNormals = true;
  // FTB sync mode: controls synchronization for depth buffer storage image access
  // FTB_SYNC_INTERLOCK = correct but slower, FTB_SYNC_DISABLED = fast but may have artifacts
  int ftbSyncMode = FTB_SYNC_DISABLED;
  // Depth iso threshold: transmittance threshold for depth picking in rasterization
  float depthIsoThreshold = 0.7f;
};

// Parameters that control rasterization
extern RasterParameters prmRaster;

// Mesh traces pack hit data into payload dist[] and id[0..1].
// Full precision: 15 slots (fp32 normal, UVs, tangent)
// Quantized:       9 slots (octahedral normal + fp16 UVs/tangent)
constexpr int MESH_PAYLOAD_MIN_SIZE_FULL      = 15;
constexpr int MESH_PAYLOAD_MIN_SIZE_QUANTIZED = 11;

// Parameters that control Raytracing (RTX)
struct RtxParameters
{
  // temporalSampling is controlled by temporalSamplingMode, it is not directly exposed
  bool  temporalSampling       = false;  // do we accumulate frame results over time (for DOF and other)
  int   temporalSamplingMode   = TEMPORAL_SAMPLING_AUTO;  // how do we control temporal sampling activation
  int   kernelDegree           = KERNEL_DEGREE_QUADRATIC;
  float kernelMinResponse      = 0.0113f;  // constant value from Paper
  bool  kernelAdaptiveClamping = true;
  int   particleSamplesPerPass  = 18;  // best default value set by experimentation on ADA6000
  int   rtxTraceStrategy        = RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT;  // trace strategy for gaussian intersection
  bool  traceProfile           = false;                           // collect per-hit trace profile for shader feedback
  // Particle shadow parameters
  float particleShadowOffset                 = 0.2f;  // Shadow ray offset for particles (volumetric nature)
  float particleShadowTransmittanceThreshold = 0.8f;  // Transmittance threshold for particle shadow termination
  float particleShadowColorStrength = 0.0f;  // Per-channel absorption from particle color [0=mono, 1=fully colored]
  // Ambient occlusion for emissive splat sets
  bool  particleEmissiveAoEnabled  = false;  // Enable particle emissive AO (compile-time macro gating)
  float particleEmissiveAoRadius   = 0.05f;  // AO hemisphere sampling radius
  float particleEmissiveAoStrength = 1.0f;   // AO darkening intensity (0 = no effect, 1 = full, >1 = exaggerated)
  // Firefly clamp: luminance threshold to suppress stochastic outliers (0 = disabled)
  float fireflyClampThreshold = 13.0f;
  // DLSS minimum radiance threshold: clamp radiance floor for DLSS input (0 = allow negative)
  float dlssMinRadianceThreshold = 0.0f;
  // Depth iso threshold: transmittance threshold for depth picking in ray tracing
  float depthIsoThresholdRTX = 0.7f;
  // Particle depth mode: how depth is computed for particles
  int particleDepth = PARTICLE_DEPTH_ELLIPSOID;
  // Billboard frustum culling: cull particles outside the camera frustum in the any-hit shader
  // Only relevant when particleDepth == PARTICLE_DEPTH_BILLBOARD
  bool billboardFrustumCulling = true;
  // Shorten ray: use payload max distance to early-terminate ray traversal in stochastic any-hit
  // Only relevant when particleDepth == PARTICLE_DEPTH_BILLBOARD and stochastic any-hit
  bool shortenRay = true;
  // Quantize mesh hit payload: pack normal (octahedral) + UVs/tangent (fp16) to reduce
  // payload from 15 to 9 float slots, lowering register pressure and local memory spilling
  bool quantizeMeshPayload = true;
};

// Parameters that control Raytracing (RTX)
extern RtxParameters prmRtx;

// Parameters that control Comparison Mode (alias to ImageCompare::Parameters)
using ComparisonParameters = ImageCompare::Parameters;

extern ComparisonParameters prmComparison;

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
// Reset prmComparison to defaults
void resetComparisonParameters();

// register the set of global parameters
void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry);

}  // namespace vk_gaussian_splatting
