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

#ifndef _DEVICE_HOST_H_
#define _DEVICE_HOST_H_

// type of method used for sorting
#define SORTING_GPU_SYNC_RADIX 0
#define SORTING_CPU_ASYNC_MONO 1
#define SORTING_CPU_ASYNC_MULTI 2
#define SORTING_STOCHASTIC_SPLAT 3  // "Stochastic Splat" (Kheradmand et al., 2025)

// type of model storage
#define STORAGE_BUFFERS 0
#define STORAGE_TEXTURES 1

enum class SplatStorage
{
  eBuffers,
  eTextures
};

// format for SH storage
#define FORMAT_FLOAT32 0
#define FORMAT_FLOAT16 1
#define FORMAT_UINT8 2

enum class SplatFormat
{
  eFloat32,
  eFloat16,
  eUint8
};

// Authorized Sh degrees
enum class SplatDegree
{
  eSh0 = 0,
  eSh1 = 1,
  eSh2 = 2,
  eSh3 = 3
};

// type of pipeline used
#define PIPELINE_VERT 0
#define PIPELINE_MESH 1
#define PIPELINE_RTX 2
#define PIPELINE_HYBRID 3        // Hybrid rendering: raster primary rays (3DGS), raytrace secondary rays (3DGRT)
#define PIPELINE_MESH_3DGUT 4    // 3DGUT (Unscented Transform) rasterization using mesh shaders
#define PIPELINE_HYBRID_3DGUT 5  // Hybrid rendering: raster primary rays (3DGUT), raytrace secondary rays (3DGRT)

// visualization mode
#define VISUALIZE_FINAL 0
#define VISUALIZE_CLOCK 1
#define VISUALIZE_RAYHITS 2
#define VISUALIZE_DEPTH 3              // depth of closest intersection
#define VISUALIZE_DEPTH_INTEGRATED 4   // depth reconstructed
#define VISUALIZE_DEPTH_FOR_DLSS 5     // depth reconstructed for DLSS coherent with Albeo Guide
#define VISUALIZE_NORMAL 6             // normal of closest intersection
#define VISUALIZE_NORMAL_INTEGRATED 7  // normal reconstructed
#define VISUALIZE_NORMAL_FOR_DLSS 8    // normal reconstructed for DLSS coherent with Albeo Guide
#define VISUALIZE_DLSS_INPUT 9         // DLSS input image (before denoising/upscaling)
#define VISUALIZE_DLSS_ALBEDO 10       // show raw buffer values
#define VISUALIZE_DLSS_SPECULAR 11     // show raw buffer values
#define VISUALIZE_DLSS_NORMAL 12       // show raw buffer values
#define VISUALIZE_DLSS_MOTION 13       // show raw buffer values
#define VISUALIZE_DLSS_DEPTH 14        // show raw buffer values
#define VISUALIZE_SPLAT_ID 15          // splat ID as harlequin/false color

// type of frustum culling
#define FRUSTUM_CULLING_NONE 0
#define FRUSTUM_CULLING_AT_DIST 1
#define FRUSTUM_CULLING_AT_RASTER 2

// size culling (cull splats whose projected screen size is below threshold)
#define SIZE_CULLING_DISABLED 0
#define SIZE_CULLING_ENABLED 1

// method used to compute the 2D extent projection from the 3D covariance
#define EXTENT_EIGEN 0  // basis aligned rectangular extent
#define EXTENT_CONIC 1  // axis aligned rectangular extend as in original INRIA

// type of camera
#define CAMERA_PINHOLE 0
#define CAMERA_FISHEYE 1

// FTB synchronization mode for depth buffer access
#define FTB_SYNC_DISABLED 0   // No synchronization (fastest, may have artifacts)
#define FTB_SYNC_INTERLOCK 1  // Use fragment shader interlock (correct, slower)

// particle format (PF), RTX
// not used in shaders (using RTX_USE_AABBS compiler defined instead)
// used only by UI but here to be easier to find
#define PARTICLE_FORMAT_ICOSAHEDRON 0
#define PARTICLE_FORMAT_PARAMETRIC 1

// degree of the splat kernel, RTX
#define KERNEL_DEGREE_QUINTIC 5
#define KERNEL_DEGREE_TESSERACTIC 4
#define KERNEL_DEGREE_CUBIC 3
#define KERNEL_DEGREE_QUADRATIC 2
#define KERNEL_DEGREE_LAPLACIAN 1
#define KERNEL_DEGREE_LINEAR 0

// Ray tracing trace strategy for gaussian intersection
#define RTX_TRACE_STRATEGY_FULL_ANYHIT 0
#define RTX_TRACE_STRATEGY_PASS_STOCHASTIC 1
#define RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT 2

// Normal computation method (used as compile-time macro NORMAL_METHOD in shaders)
#define NORMAL_METHOD_MAX_DENSITY_PLANE 0  // Max density plane approximation (StochasticSplats)
#define NORMAL_METHOD_ISO_SURFACE 1        // Iso-surface ellipsoid ray intersection

// Lighting mode (used as compile-time macro LIGHTING_MODE in shaders)
#define LIGHTING_DISABLED 0  // No lighting computed
#define LIGHTING_DIRECT 1    // Direct lighting only (one bounce, no reflections/refractions)
#define LIGHTING_INDIRECT 2  // Full lighting with bounces, reflections, refractions

// Depth of Field mode (used as compile-time macro DOF_MODE in shaders)
#define DOF_DISABLED 0     // No depth of field
#define DOF_FIXED_FOCUS 1  // Fixed focus distance (manual)
#define DOF_AUTO_FOCUS 2   // Auto focus using surface distance at cursor position

// Shadows mode (used as compile-time macro SHADOWS_MODE in shaders)
#define SHADOWS_DISABLED 0  // No shadow rays traced
#define SHADOWS_HARD 1      // Hard shadows (point-sampled)
#define SHADOWS_SOFT 2      // Soft shadows (disk-sampled around lights)

// bindings for set 0 (common to Raster and RTX)
#define BINDING_FRAME_INFO_UBO 0
#define BINDING_CENTERS_TEXTURE 1
#define BINDING_COLORS_TEXTURE 2
#define BINDING_COVARIANCES_TEXTURE 3
#define BINDING_SH_TEXTURE 4
// BINDING_DISTANCES_BUFFER 5 - DEPRECATED (now accessed via SceneAssets.splatSortingDistancesAddress)
// BINDING_INDICES_BUFFER 6 - DEPRECATED (now accessed via SceneAssets.splatSortingIndicesAddress)
#define BINDING_INDIRECT_BUFFER 7
#define BINDING_CENTERS_BUFFER 8
#define BINDING_COLORS_BUFFER 9
#define BINDING_COVARIANCES_BUFFER 10
#define BINDING_SH_BUFFER 11
#define BINDING_SCALES_TEXTURE 12
#define BINDING_ROTATIONS_TEXTURE 13
#define BINDING_SCALES_BUFFER 14
#define BINDING_ROTATIONS_BUFFER 15
#define BINDING_OPACITY_TEXTURE 16
#define BINDING_OPACITY_BUFFER 17
// BINDING_RTX_PAYLOAD_BUFFER 18 - REMOVED (no longer used)
// BINDING_MESH_DESCRIPTORS 19 - REMOVED (now in SceneAssets via BINDING_ASSETS)
// BINDING_LIGHT_SET 20 - REMOVED (now in SceneAssets via BINDING_ASSETS)
// BINDING_SPLAT_MATERIAL 21 - REMOVED (now per-instance in SplatSetDesc.material)
// NEW: Bindless assets structure (replaces multiple bindings)
#define BINDING_ASSETS 22
// Bindless texture array for STORAGE_TEXTURES mode
#define BINDING_SPLAT_TEXTURES 23  // Unbounded array: Sampler2D allSplatTextures[]

// Rasterization surface reconstruction buffers (set 0)
#define BINDING_RASTER_NORMAL 24        // Rasterization: integrated normals output (RGB16F)
#define BINDING_RASTER_DEPTH 25         // Rasterization: integrated depth (R) + transmittance (G)
#define BINDING_RASTER_SPLATID 26       // Rasterization: global splat ID (R32_UINT)
#define BINDING_RASTER_COLOR 27         // Rasterization: main color output (for deferred shading input)
#define BINDING_DEFERRED_OUTPUT 28      // Deferred shading: output image
#define BINDING_RASTER_COLOR_AUX 29     // Rasterization: aux color buffer (for temporal accumulation)
#define BINDING_DEFERRED_OUTPUT_AUX 30  // Deferred shading: aux output image (for temporal accumulation)

// bindings for set 1 of RTX
#define RTX_BINDING_OUTIMAGE 0     // Ray tracer output image
#define RTX_BINDING_TLAS_SPLATS 1  // Top-level acceleration structure for splats
#define RTX_BINDING_TLAS_MESH 2    // Top-level acceleration structure for meshes
// RTX_BINDING_PAYLOAD_BUFFER 3 - REMOVED (no longer used)
#define RTX_BINDING_AUX1 4      // Ray tracer auxiliary output image, when using hybrid mode + temporal sampling
#define RTX_BINDING_OUTDEPTH 5  // depth buffer
// Array of DLSS output images (albedo, specular, normal/roughness, motion, depth)
#define RTX_BINDING_DLSS_OUT_IMAGES 6

// Temporal sampling mode
#define TEMPORAL_SAMPLING_AUTO 0  // Detects automatically if TS is needed for best visual results (e.g. if DoF is on)
#define TEMPORAL_SAMPLING_ENABLED 1   // Force enabled
#define TEMPORAL_SAMPLING_DISABLED 2  // Force disabled

// bindings for set 0 of Post Process (0 is reserved for BINDING_FRAME_INFO_UBO)
#define POST_BINDING_MAIN_IMAGE 1  // the image that is presented
#define POST_BINDING_AUX1_IMAGE 2  // optional aux image to be accumulated (for example)

// location for vertex attributes
// (only for vertex shader mode)
#define ATTRIBUTE_LOC_POSITION 0
#define ATTRIBUTE_LOC_SPLAT_INDEX 1
// used for mesh rasterization
#define ATTRIBUTE_LOC_MESH_POSITION 0
#define ATTRIBUTE_LOC_MESH_NORMAL 1

#ifdef __cplusplus
#include "nvshaders/slang_types.h"
using double3 = glm::dvec3;
#include "wavefront.h"
namespace shaderio {
#else
// we are in Slang here
#include "wavefront.h"
#endif

// Enums to dereference DLSS Images
enum class DlssImages
{
  eDlssInputImage = 0,   // Output image (RGBA32)
  eDlssAlbedo,           // Diffuse albedo (RGBA8)
  eDlssSpecAlbedo,       // Specular albedo (RGBA16F)
  eDlssNormalRoughness,  // Normal and roughness (RGBA16F)
  eDlssMotion,           // Motion vectors (RG16F)
  eDlssDepth,            // Depth (R16F)
};

// Mesh instance type enumeration (available in both C++ and Slang)
// Values match RTX acceleration structure instance masks for efficient ray tracing
enum MeshType : uint32_t
{
  eObject     = 0xFE,  // Standard mesh objects (mask for regular geometry)
  eLightProxy = 0x01,  // Light visualization proxy meshes (mask for light proxies)
};

struct FrameInfo
{
  float3   cameraPosition;  // position in world space
  float4   viewQuat;        // quaternion storing the rotation part of the view matrix
  float3   viewTrans;       // translation part of the view matrix
  float4x4 viewMatrix;
  float4x4 viewInverse;  // Camera inverse view matrix

  float4x4 projectionMatrix;
  float4x4 projInverse;  // Camera inverse projection matrix
  float2   nearFar;
  float2   focal;
  float2   viewport;
  float2   basisViewport;

  float fovRad                 = 0.009f;  // Field of view in radians for fisheye camera
  float inverseFocalAdjustment = 1.0f;

  // Ortho is not fully implemented
  float   orthoZoom        = 1.0f;  //
  int32_t orthographicMode = 0;     // disabled, in [0,1]

  int32_t splatCount = 0;     //
  float   splatScale = 1.0f;  // in {0.1, 2.0}
  uint    shDegree   = 3;     // in [0,1,2,3] max sh degree to render if available

  float frustumDilation      = 0.2f;           // for frustum culling, 2% scale
  float alphaCullThreshold   = 1.0f / 255.0f;  // for alpha culling
  float sizeCullingMinPixels = 1.0f;           // for size culling, minimum projected pixel coverage

  int2    cursor    = int2(0, 0);  // position of the mouse cursor for debug
  int32_t maxPasses = 200;         // RTX maximum hits during marching

  float alphaClamp = 0.99f;        // 0.99 in original paper
  float minTransmittance = 0.01f;  // 0.1  in original paper ? transmittance value under which particle marching loop stops
  int32_t rtxMaxBounces = 3;

  int32_t frameSampleId  = 0;    // the frame sample index since last frame sampling reset
  int32_t frameSampleMax = 200;  // maximum number of frame after which we stop accumulating frames samples

  float focusDist = 1.3f;    // focus distance to compute depth of field
  float aperture  = 0.001f;  // aperture distance to compute depth of field, 0 does no DOF effect

  // threshold under which meshes are not composited in RTX pipelines,
  // prevents to see meshes through semi transparent splat sets
  // this threshold generaly needs to be raised when using stochastic
  // pass since transmittance is not fully evaluated for some paths
  float minMeshCompositeTransmittance = 0.0;

  // DLSS - Previous frame's view-projection matrix for motion vectors (uses unjittered projection)
  float4x4 prevViewProjMatrix;
  // DLSS - Jittered projection matrix for rasterization when DLSS is enabled
  // projectionMatrix remains unjittered (used by raygen, motion vectors, etc.)
  // projectionMatrixJittered is used by raster pipelines for DLSS sub-pixel sampling
  float4x4 projectionMatrixJittered;

  // for alternative visualization modes

  float  visuShift  = 0.0;
  float2 visuMinMax = float2(0, 100);

  // Ray tracing mask for mesh visibility (computed on CPU)
  // 0xFF = show all meshes (including light proxies)
  // 0xFE = hide light proxies (exclude mask 0x01)
  uint32_t rayMask = 0xFF;

  // Particle shadow parameters
  float particleShadowOffset                 = 0.2f;  // Shadow ray offset for particles
  float particleShadowTransmittanceThreshold = 0.8f;  // Transmittance threshold for particle shadows
  float particleShadowColorStrength = 0.0f;  // Per-channel absorption from particle color [0=mono, 1=fully colored]

  // Depth iso threshold: transmittance threshold for depth picking
  // Depth is picked when transmittance drops below this threshold
  float depthIsoThreshold    = 0.7f;  // For rasterization
  float depthIsoThresholdRTX = 0.7f;  // For ray tracing

  // Thin particle threshold: scale below which a particle axis is considered degenerate
  // Used by normal computation to detect flat (disk) or line/point particles
  float thinParticleThreshold = 1e-6f;
};

// Push constant for raster
struct PushConstant
{
  // Mesh transforms now come from MeshDesc in bindless assets buffer
  // Only objIndex remains to index into the assets.meshes[] array
  uint32_t objIndex;      // index of the mesh being rendered
  uint32_t ftbColorPass;  // FTB mode: 0=depth pre-pass, 1=color pass (read transmittance)
};

// indirect parameters for
// - vkCmdDrawIndexedIndirect (first 6 attr)
// - vkCmdDrawMeshTasksIndirectEXT (last 3 attr)
// - shader feedback readback (extra fields)
struct HitProfile
{
  float    dist               = 0.0f;          // hit distance along the ray
  uint32_t splatId            = 0;             // global splat ID
  float    alpha              = 0.0f;          // splat alpha response
  float    _pad0              = 0.0f;          // padding to 16-byte alignment
  float4   color              = float4(0.0f);  // rgb = particle radiance/color, a unused
  float4   transmittance      = float4(1.0f);  // rgb = transmittance after integration, a unused
  float4   integratedRadiance = float4(0.0f);  // rgb = accumulated pixel radiance after integration, a unused
};

struct IndirectParams
{
  // for vkCmdDrawIndexedIndirect
  uint32_t indexCount    = 6;  // allways = 6 indices for the quad (2 triangles)
  uint32_t instanceCount = 0;  // will be incremented by the distance compute shader
  uint32_t firstIndex    = 0;  // allways zero
  uint32_t vertexOffset  = 0;  // allways zero
  uint32_t firstInstance = 0;  // allways zero

  // for vkCmdDrawMeshTasksIndirectEXT
  uint32_t groupCountX = 0;  // Will be incremented by the distance compute shader
  uint32_t groupCountY = 1;  // Allways one workgroup on Y
  uint32_t groupCountZ = 1;  // Allways one workgroup on Z

  // for info readback, TODO shall be in some other buffer
  int32_t particleGlobalId         = -1;  // Global splat ID (across all instances)
  int32_t particleId               = -1;  // Local splat index within the splat set
  int32_t splatSetId               = -1;  // Which splat set instance (descriptor array index)
  int32_t particleHitCount         = 0;
  float   particleDist             = 0.0;  // Will be set by the dist to the nearest splat on the ray path
  float3  particleNormal           = float3(0.0);
  float   particleIntegratedDist   = 0.0;  // Will be set by the dist to the nearest splat on the ray path
  float3  particleIntegratedNormal = float3(0.0);
  float3  particleRadiance         = float3(0.0);
  double3 particleTransmittance    = double3(0.0);

  float closestParticleAlpha = 0.0;
  //float3 closestParticleNormal      = float3(0.0); // see patrticleNormal
  float3 closestParticleWeight        = float3(0.0);
  float3 closestParticleTransmittance = float3(0.0);

  // for debug info, TODO shall be in some other buffer
  float val1 = 0.0;
  float val2 = 0.0;
  float val3 = 0.0;
  float val4 = 0.0;
  float val5 = 0.0;
  float val6 = 0.0;
  float val7 = 0.0;
  float val8 = 0.0;

  // --------------------------------------------------------------------------
  // Trace profile (shader feedback extension)
  // Stored in the same indirect buffer for CPU readback.
  //
  // NOTE:
  // - The storage is always present so host/device layouts match.
  // - The shader-side collection and writes are guarded by compile-time macro
  //   TRACE_PROFILE (0/1) so this feature has zero cost when disabled.
  // --------------------------------------------------------------------------
  uint32_t traceProfileHitCount = 0;  // total intersections recorded (may exceed max storage)
  uint32_t _traceProfilePad0    = 0;
  uint32_t _traceProfilePad1    = 0;
  uint32_t _traceProfilePad2    = 0;

  // Fixed-capacity array (max 200 hits)
  HitProfile traceProfileHits[200] = {};
};


// Push constant specific to raytracing
struct PushConstantRay
{
  // Splat/mesh transforms now come from SplatSetDesc/MeshDesc in bindless assets buffer
  // Vertex/index addresses now come from MeshDesc in bindless assets buffer
  // Only rendering flags and DLSS parameters remain

  // set to true will raytrace the mesh depth as a pre-pass
  bool meshDepthOnly;
  // #DLSS - DLSS related fields
  int32_t useDlss = 0;                   // Use DLSS (0: no, 1: yes)
  float2  jitter  = float2(0.0f, 0.0f);  // Camera jitter for DLSS
};

// ============================================================================
// Bindless Architecture Structures
// ============================================================================

// MeshDesc: Per-instance mesh descriptor (similar to ObjDesc but with per-instance data)
// Note: This will eventually replace ObjDesc in the bindless architecture
struct MeshDesc
{
  // Geometry buffer addresses (shared across instances)
  ObjVertex*   vertexAddress;         // Address of the Vertex buffer
  uint32_t*    indexAddress;          // Address of the index buffer
  ObjMaterial* materialAddress;       // Address of the material buffer
  uint32_t*    materialIndexAddress;  // Address of the triangle material index buffer

  // Per-instance transform
  float4x4 transform;
  float4x4 transformInverse;
  float3x3 transformRotScaleInverse;
};

// SplatSetDesc: Per-instance Gaussian splat set descriptor
// Contains addresses to splat data buffers/textures and per-instance transform/material
struct SplatSetDesc
{
  // Data buffer addresses (for STORAGE_BUFFERS mode) - using pointers for bindless access
  float*   centersAddress;      // float[splatCount * 3]
  uint64_t colorsAddress;       // Device address to RGBA data (typed by desc.rgbaFormat: float/float16_t/uint8_t)
  float*   scalesAddress;       // float[splatCount * 3]
  float*   rotationsAddress;    // float[splatCount * 4] (quaternion)
  float*   covariancesAddress;  // float[splatCount * 6] (precomputed covariances for raster)
  uint64_t shAddress;           // Device address to SH data (typed by desc.format: float/float16_t/uint8_t)
  uint64_t blasAddress;         // Per-splat-set BLAS address (for non-instance TLAS mode)

  // Texture handles (for STORAGE_TEXTURES mode)
  uint32_t centersTexture;
  uint32_t colorsTexture;
  uint32_t scalesTexture;
  uint32_t rotationsTexture;
  uint32_t covariancesTexture;
  uint32_t shTexture;

  // Metadata
  uint32_t splatCount;
  uint32_t shDegree;  // from the splat set

  // Splat index bases (used when split-BLAS creates one TLAS instance per chunk)
  // splatBase: local base within the source splat set (chunk offset).
  // globalSplatBase: base in the global splat ID space (all instances).
  uint32_t splatBase;
  uint32_t globalSplatBase;

  uint32_t storage;     // STORAGE_BUFFERS or STORAGE_TEXTURES
  uint32_t format;      // FORMAT_FLOAT32, FORMAT_FLOAT16, FORMAT_UINT8 (for SH)
  uint32_t rgbaFormat;  // FORMAT_FLOAT32, FORMAT_FLOAT16, FORMAT_UINT8 (for RGBA colors)

  // Per-instance transform
  float4x4 transform;
  float4x4 transformInverse;
  float3x3 transformRotScaleInverse;

  // Per-instance material (each splat instance has its own material)
  ObjMaterial material;
};

// SceneAssets: Unified bindless asset structure (single binding point)
// This structure provides access to all scene assets through device addresses
struct SceneAssets
{
  // Mesh geometry instances (includes both regular objects and light proxies)
  // Light proxies are differentiated by MeshType::eLightProxy
  MeshDesc* meshes          = nullptr;
  uint32_t  meshCount       = 0;
  uint32_t  _pad0           = 0;
  uint64_t  meshTlasAddress = 0;  // TLAS device address for meshes

  // Light sources
  LightSource* lights     = nullptr;
  uint32_t     lightCount = 0;
  uint32_t     _pad1      = 0;

  // Gaussian splat set instances
  // splatSetDescriptors: per-instance array used by raster/hybrid and global index table.
  // splatSetDescriptorsRtx: optional per-TLAS-instance array used by RTX when BLAS is split.
  SplatSetDesc* splatSetDescriptors       = nullptr;
  SplatSetDesc* splatSetDescriptorsRtx    = nullptr;
  uint32_t      splatSetCount             = 0;
  uint32_t      _pad2                     = 0;
  uint64_t      splatSetTlasArrayAddress  = 0;  // Address of array of TLAS addresses (uint64_t[])
  uint64_t      splatSetTlasOffsetAddress = 0;  // Address of array of instance offsets (uint32_t[])
  uint32_t      splatSetTlasCount         = 0;  // Number of TLAS in the array
  uint32_t      _pad3                     = 0;

  // Global indexing for unified sorting (multi-instance support)
  // These buffers enable unified depth sorting across all splat instances
  uint64_t globalIndexTableAddress         = 0;  // GlobalSplatIndexEntry[] buffer
  uint64_t splatSetGlobalIndexTableAddress = 0;  // uint32_t[] buffer (starting indices)
  uint32_t totalGlobalSplatCount           = 0;  // Total splats across all instances
  uint32_t _pad4                           = 0;

  // Sorting buffer device addresses (bindless, replaces BINDING_DISTANCES_BUFFER and BINDING_INDICES_BUFFER)
  uint32_t* splatSortingDistancesAddress = nullptr;  // Device address to distances buffer (RW)
  uint32_t* splatSortingIndicesAddress   = nullptr;  // Device address to sorted indices buffer (RW)
};

// GlobalSplatIndexEntry: Maps global splat index to (splatSetIndex, splatIndex)
// Used for unified depth sorting across multiple splat set instances
struct GlobalSplatIndexEntry
{
  uint32_t splatSetIndex;  // Which splat set instance
  uint32_t splatIndex;     // Which splat within that instance
};

// Push constants for GPU-built particle TLAS/BLAS instances
struct ParticleAsBuildPushConstants
{
  uint64_t splatSetDescriptorAddress = 0;  // shaderio::SplatSetDesc[]
  uint64_t globalIndexTableAddress   = 0;  // shaderio::GlobalSplatIndexEntry[]
  uint64_t tlasInstanceBufferAddress = 0;  // VkAccelerationStructureInstanceKHR[]
  uint64_t aabbBufferAddress         = 0;  // VkAabbPositionsKHR (single)
  uint64_t blasAddress               = 0;  // BLAS device address for instances
  uint64_t vertexBufferAddress       = 0;  // float3[]
  uint64_t indexBufferAddress        = 0;  // uint32[]
  uint32_t instanceCount             = 0;
  uint32_t instanceBaseIndex         = 0;
  uint32_t splatBaseIndex            = 0;  // Base splat index for geometry generation
  uint32_t geometryType              = 0;  // 0=triangles, 1=aabbs
  uint32_t writeGeometry             = 0;  // 0/1
  uint32_t geometryMode              = 0;  // 0=unit, 1=per-splat (global), 2=per-splat-set
  uint32_t kernelDegree              = 0;
  float    kernelMinResponse         = 0.0f;
  uint32_t kernelAdaptiveClamping    = 0;
  uint32_t instanceMode              = 0;  // 0=per-splat, 1=per-splat-set instance
  uint32_t _pad0                     = 0;
};


#ifdef __cplusplus
}  // namespace shaderio
#endif

#endif
