/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshaders/random.h.slang"
#include "nvshaders/wireframe.h.slang"

#include "shaderio.h"
#include "cameras.h.slang"
#include "color.h.slang"
#include "wavefront.h.slang"
#include "wireframe_extended.h.slang"

#include "threedgs_particle_storage.h.slang"
#include "threedgrt.h.slang"
#include "threedgrt_payload.h.slang"

// #DLSS - DLSS utilities
#include "dlss_util.h"

// clang-format off
[[vk::binding(RTX_BINDING_OUTIMAGE, 1)]] RWTexture2D<float4> image;
[[vk::binding(RTX_BINDING_AUX1, 1)]] RWTexture2D<float4> imageAux;
[[vk::binding(RTX_BINDING_OUTDEPTH, 1)]] RWTexture2D<float> depthBuffer;
[[vk::binding(RTX_BINDING_DLSS_OUT_IMAGES, 1)]] RWTexture2D<float4> outImages[]; // DLSS 

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
#if HYBRID_ENABLED && NEED_SURFACE_INFO
[[vk::binding(BINDING_RASTER_NORMAL, 0)]] RWTexture2D<float4> rasterNormalBuffer;
[[vk::binding(BINDING_RASTER_DEPTH, 0)]] RWTexture2D<float2> rasterDepthBuffer;  // R=depth, G=transmittance
[[vk::binding(BINDING_RASTER_SPLATID, 0)]] RWTexture2D<uint> rasterSplatIdBuffer;
#endif
// TODO to be removed, just for debug readback
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]] RWStructuredBuffer<IndirectParams> indirectBuffer;

// TLAS - accessed via device addresses in SceneAssets (bindless approach)
// No explicit bindings needed - addresses converted using accelerationStructureEXT() in shader

// Bindless assets (contains lights, mesh descriptors, splat descriptors, TLAS addresses)
[[vk::binding(BINDING_ASSETS, 0)]] ConstantBuffer<SceneAssets> assets;

// clang-format on

// Structure holding shadow information (soft shadows with colored transmission)
struct ShadowInfo
{
  float3 transmittance;  // RGB color of light transmission (colored shadows)
};

double maxComponent(in double3 v)
{
  return max(v.x, max(v.y, v.z));
}

float maxComponent(in float3 v)
{
  return max(v.x, max(v.y, v.z));
}

struct PixelData
{
  uint64_t clockStart = clockARB();
  uint     hitCount   = 0;  // number of ray intersections with particles

  bool   meshHit         = false;  // did we have a mesh hit by primary ray
  float  meshHitDist     = PAYLOAD_INF_DISTANCE;
  float3 meshHitWorldPos = {};
  float3 meshHitWorldNrm = {};
  int    meshHitObjId    = PAYLOAD_INVALID_ID;
  int    meshHitMatId    = PAYLOAD_INVALID_ID;


  double3 transmittance = double3(1.0);  // mesh + particle transmittance
  float3  radiance      = float3(0.0f);  // mesh + particle integrated radiance

  float3 unlitAlbedo       = float3(0.0f);  // Base particle albedo before lighting (for DLSS guides)
  float3 integratedNormal  = float3(0.0);   // particle integrated normal (current bounce)
  float  isoSurfDepth      = 0.0;           // particle picked depth, world space (current bounce)
  float3 isoSurfPos        = float3(0.0);   // world position of the picked depth (current bounce)
  int    isoSurfSplatId    = PAYLOAD_INVALID_ID;
  int    isoSurfSplatSetId = PAYLOAD_INVALID_ID;

  // Primary surface depth (bounce 0 only, for depth buffer write)
  float  primaryHitDist = PAYLOAD_INF_DISTANCE;
  float3 primaryHitPos  = float3(0.0);

  bool    closestParticleFound         = false;
  int     closestParticleGlobalId      = PAYLOAD_INVALID_ID;  // Global splat ID across all instances
  int     closestParticleId            = PAYLOAD_INVALID_ID;  // Local splat ID within the splat set
  int     closestSplatSetId            = PAYLOAD_INVALID_ID;  // Splat set index (descriptor array index)
  float   closestParticleDist          = PAYLOAD_INF_DISTANCE;
  float   closestParticleAlpha         = 0.0f;
  float3  closestParticleNormal        = float3(0.0f);
  float3  closestParticleWeight        = float3(0.0f);
  double3 closestParticleTransmittance = float3(1.0f);
  float3  closestParticleRadiance      = float3(0.0f);  // vdp radiance for the particle, no weighting
};

// Structure to accumulate DLSS guide data
struct DlssGuideData
{
  float4 albedo          = float4(0.0, 0.0, 0.0, 1.0);  // Diffuse albedo + opacity
  float3 specularAlbedo  = float3(0.0);                 // Specular reflectance (minimal for Gaussians)
  float4 normalRoughness = float4(0.0, 0.0, 0.0, 0.9);  // World-space normal (xyz) + roughness (w)
  float3 hitPosition     = float3(0.0);                 // World-space hit position for motion vectors
  bool   isValid         = false;                       // Whether we have valid data
};

// -----------------------------------------------------------------------------
// Trace profile collection (shader feedback extension)
// -----------------------------------------------------------------------------
#if TRACE_PROFILE
static const uint TRACE_PROFILE_MAX_HITS = 200;

// Collect per-intersection profile right after integration.
// Only intended to be called for the shader feedback pixel (cursor).
void collectHitProfile(in uint splatId, in float dist, in float alpha, in float3 color, in double3 transmittanceAfter, in float3 integratedRadianceAfter)
{
  const uint idx = indirectBuffer[0].traceProfileHitCount;

  if(idx < TRACE_PROFILE_MAX_HITS)
  {
    indirectBuffer[0].traceProfileHits[idx].dist               = dist;
    indirectBuffer[0].traceProfileHits[idx].splatId            = splatId;
    indirectBuffer[0].traceProfileHits[idx].alpha              = alpha;
    indirectBuffer[0].traceProfileHits[idx].color              = float4(color, 1.0);
    indirectBuffer[0].traceProfileHits[idx].transmittance      = float4(float3(transmittanceAfter), 1.0);
    indirectBuffer[0].traceProfileHits[idx].integratedRadiance = float4(integratedRadianceAfter, 1.0);
  }

  // Count can exceed storage size; CPU can detect overflow.
  indirectBuffer[0].traceProfileHitCount = idx + 1;
}
#endif

// will use Any hit only
static const uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
// will use Closest hit only
static const uint rayFlagsMesh = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

static const float epsT = 1e-9;

[shader("raygeneration")]
void main()
{
  float3 rayOrigin;
  float3 rayDirection;

  const uint2  launchID        = DispatchRaysIndex().xy;
  const uint2  launchSize      = DispatchRaysDimensions().xy;
  const float2 launchIdFloat   = float2(launchID.x, launchID.y);      // notation required to prevent warning
  const float2 launchSizeFloat = float2(launchSize.x, launchSize.y);  // notation required to prevent warning

  // center of the pixel
  float2 subPixelOffset = 0.5;
  // Apply jitter for DLSS
  if(pcRay.useDlss == 1)
  {
    subPixelOffset += pcRay.jitter;
  }

  // generate the ray for this pixel
#if CAMERA_TYPE == CAMERA_PINHOLE
  generatePinholeRay(launchIdFloat, subPixelOffset, launchSizeFloat, frameInfo.viewInverse, frameInfo.projInverse,
                     rayOrigin, rayDirection);
#else
  if(!generateFisheyeRay(launchIdFloat, launchSizeFloat, frameInfo.fovRad, float2(0.0), frameInfo.viewInverse, rayOrigin, rayDirection))
  {
    image[launchID] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }
#endif

  // add Depth-of-Field perturbation to the ray
#if((PIPELINE != PIPELINE_HYBRID) && (DOF_MODE != DOF_DISABLED))
  // Initialize the random number
  uint seed = xxhash32(uint3(launchID, frameInfo.frameSampleId));

  depthOfField(seed, frameInfo.focusDist, frameInfo.aperture, frameInfo.viewInverse, rayOrigin, rayDirection);
#endif

  // Create the payload
  HitPayload payload;  // no default init, will do per trace call
  // where we will store the data computed by the different stages
  // clockStart is performed by constructor
  PixelData pixel = {};

  // Initialize guide data collection
  DlssGuideData dlssGuide = {};

  // Reconstruct the state of ray tracing as if it did trace primary rays
  // We copy the information from the rasterization
#if HYBRID_ENABLED
  initRtxStateFromRasterPass(pixel, launchID, launchIdFloat, launchSizeFloat, rayOrigin, rayDirection);
#endif

#if WIREFRAME
  {
    float3 diffX, diffY;
#if CAMERA_TYPE == CAMERA_PINHOLE
    computeDifferentials(launchIdFloat, launchSizeFloat, frameInfo.viewInverse, frameInfo.projInverse, diffX, diffY);
#else
    computeDifferentialsFisheye(launchIdFloat, launchSizeFloat, frameInfo.fovRad, float2(0.0), frameInfo.viewInverse,
                                rayDirection, diffX, diffY);
#endif
    payload.differentialX.set(diffX);
    payload.differentialY.set(diffY);
  }
#endif

  // Initialize the random number for stochastic ray marching
  uint seedRayPass = xxhash32(uint3(launchID, frameInfo.frameSampleId));

  int bounce = 0;  // id of the bounce
  int done   = 1;  // continue on next bounce or stop if done

  const bool isFeedbackPixel = all(launchID == frameInfo.cursor);

  // Reset trace profile buffer for the feedback pixel (cursor)
#if TRACE_PROFILE
  if(isFeedbackPixel)
  {
    indirectBuffer[0].traceProfileHitCount = 0;
  }
#endif

  // bounce loop
  while(true)
  {
    // each new bounce reinits Tmin/Tmax
    float tMin               = 0.001;
    float tMax               = 10000.0;
    float rayLastHitDistance = max(0.0f, tMin - epsT);

    // trace the mesh with one closest hit if needed
    if(assets.meshCount > 0)
    {
      // if mesh hit, will modify Tmax, so splat tracing if any will be on reduced range later on
      traceBounceRayMeshes(bounce, tMin, tMax, rayOrigin, rayDirection, pcRay.meshDepthOnly, payload, dlssGuide, pixel);

#if HYBRID_ENABLED
      // Early return if meshDepthOnly pre-pass
      if(pcRay.meshDepthOnly)
      {
        if(pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
        {
          const float4 clip = mul(mul(float4(pixel.meshHitWorldPos, 1.0), frameInfo.viewMatrix), frameInfo.projectionMatrix);
          const float3 ndc      = clip.xyz / clip.w;
          depthBuffer[launchID] = ndc.z;
        }
        else
        {
          depthBuffer[launchID] = 1.0;
        }

        // clear image buffer
        if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
          image[launchID] = float4(0.0, 0.0, 0.0, 1.0);
        else
          imageAux[launchID] = float4(0.0, 0.0, 0.0, 1.0);

        return;
      }
#endif
    }

    // when hybrid is enabled, we trace the particles only for secondary rays
#if HYBRID_ENABLED
    const float3 radianceBeforeParticles = (bounce > 0) ? pixel.radiance : float3(0.0);
    if(bounce > 0)
    {
#else
    const float3 radianceBeforeParticles = pixel.radiance;
#endif
      // Check if we have any splat instances to trace
      if(assets.totalGlobalSplatCount > 0)
      {
        // trace the gaussians with multi-pass any hit
        traceBounceRayParticles(bounce, tMin, tMax, rayOrigin, rayDirection, seedRayPass, payload, dlssGuide, pixel, isFeedbackPixel);
      }

#if HYBRID_ENABLED
    }  // End bounce 0 skip
#endif

    evaluateLightingAndShadingForBounce(bounce, pixel, done, rayOrigin, rayDirection, seedRayPass, payload, radianceBeforeParticles);

    if(bounce == 0)
    {
      populateDlssGuides(pixel, dlssGuide);

      // shader feedback (TODO rename attributes for better coherence)
      if(all(launchID == frameInfo.cursor))
      {
        indirectBuffer[0].particleGlobalId             = pixel.closestParticleGlobalId;  // Global ID
        indirectBuffer[0].particleId                   = pixel.closestParticleId;        // Local ID
        indirectBuffer[0].splatSetId                   = pixel.closestSplatSetId;        // Splat set index
        indirectBuffer[0].particleHitCount             = pixel.hitCount;
        indirectBuffer[0].particleRadiance             = pixel.radiance;
        indirectBuffer[0].particleTransmittance        = pixel.transmittance;
        indirectBuffer[0].particleDist                 = pixel.closestParticleDist;
        indirectBuffer[0].particleNormal               = pixel.closestParticleNormal;
        indirectBuffer[0].particleIntegratedDist       = pixel.isoSurfDepth;
        indirectBuffer[0].particleIntegratedNormal     = pixel.integratedNormal;
        indirectBuffer[0].closestParticleAlpha         = pixel.closestParticleAlpha;
        indirectBuffer[0].closestParticleWeight        = pixel.closestParticleWeight;
        indirectBuffer[0].closestParticleTransmittance = float3(pixel.closestParticleTransmittance);
      }
    }

    // continue with secondary rays if needed (decided by evaluateLightingAndShadingForBounce)
    bounce++;
    if(done == 1 || bounce > frameInfo.rtxMaxBounces)
      break;

    // prepare for next iteration
    // new rayOrigin and direction was set by evaluateLightingAndShadingForBounce
    // Next iteration will stop if a reflective/refractive material isn't hit
    done = 1;

  }  // while(true) - end bounce loop

  writeToGBuffers<VISUALIZE, HYBRID_ENABLED>(frameInfo, launchSizeFloat, launchID, pixel, depthBuffer, image, imageAux,
                                             pcRay.useDlss, dlssGuide, outImages);
}

#if HYBRID_ENABLED
// Initialize RTX state from rasterization pass results
// Reconstructs pixel state as if primary rays were traced
void initRtxStateFromRasterPass(inout PixelData pixel, in uint2 launchID, in float2 launchIdFloat, in float2 launchSizeFloat, in float3 rayOrigin, in float3 rayDirection)
{
  // Initialize isoSurfDepth to infinity - will be set to actual value if valid splat data exists
  // This ensures mesh-only pixels correctly fall through to mesh path in comparisons
  pixel.isoSurfDepth = PAYLOAD_INF_DISTANCE;

  float4 rasterPassPixel;
  if(frameInfo.frameSampleId <= 0)  // no temporal or first frame
    rasterPassPixel = image[launchID];
  else
    rasterPassPixel = imageAux[launchID];
  pixel.radiance      = rasterPassPixel.xyz;
  pixel.transmittance = double3(1.0 - rasterPassPixel.w);

  // --- Init depth related fields ---
  // Read NDC depth from appropriate buffer and reconstruct world position

#if NEED_SURFACE_INFO
  // FTB mode: read picked depth from raster depth buffer
  float ndcDepthPicked  = rasterDepthBuffer[launchID].r;
  float ndcDepthClosest = depthBuffer[launchID].x;
  bool  validDepth      = (ndcDepthPicked > 0.0001);
#else
  // BTF mode: read closest depth from hardware depth buffer
  float ndcDepthPicked  = depthBuffer[launchID];
  float ndcDepthClosest = ndcDepthPicked;
  bool  validDepth      = (ndcDepthClosest < 1.0);
#endif

  if(validDepth)
  {
#if CAMERA_TYPE == CAMERA_PINHOLE
    float2 ndc_xy = (launchIdFloat + 0.5) / launchSizeFloat * 2.0 - 1.0;

    // Unproject picked depth to world space
    float4 clipPos = float4(ndc_xy, ndcDepthPicked, 1.0);
    float4 viewPos = mul(clipPos, frameInfo.projInverse);
    viewPos /= viewPos.w;
    float3 worldPos   = mul(viewPos, frameInfo.viewInverse).xyz;
    float  linearDist = length(worldPos - rayOrigin);

    pixel.isoSurfDepth         = linearDist;
    pixel.isoSurfPos           = worldPos;
    pixel.primaryHitDist       = linearDist;
    pixel.primaryHitPos        = worldPos;
    pixel.closestParticleFound = true;

    // Unproject closest depth to world space (may differ in FTB mode)
    float4 clipPosClosest = float4(ndc_xy, ndcDepthClosest, 1.0);
    float4 viewPosClosest = mul(clipPosClosest, frameInfo.projInverse);
    viewPosClosest /= viewPosClosest.w;
    float3 worldPosClosest   = mul(viewPosClosest, frameInfo.viewInverse).xyz;
    float  linearDistClosest = length(worldPosClosest - rayOrigin);

    pixel.closestParticleDist = linearDistClosest;
#else
    // Fisheye: the rasterizer stores NDC depth computed via the pinhole projection matrix.
    // The Z-mapping (NDC depth <-> view-space Z) is independent of XY, so we can extract
    // view-space Z correctly. Then we use the fisheye ray direction to compute the actual
    // linear distance and world position along the fisheye ray.
    float3 rayDirView = mul(float4(rayDirection, 0), frameInfo.viewMatrix).xyz;

    // Picked depth
    float4 clipPosPicked  = float4(0, 0, ndcDepthPicked, 1.0);
    float4 viewPosPicked  = mul(clipPosPicked, frameInfo.projInverse);
    float  viewZPicked    = viewPosPicked.z / viewPosPicked.w;
    float  linearDist     = viewZPicked / rayDirView.z;
    float3 worldPos       = rayOrigin + linearDist * rayDirection;

    pixel.isoSurfDepth         = linearDist;
    pixel.isoSurfPos           = worldPos;
    pixel.primaryHitDist       = linearDist;
    pixel.primaryHitPos        = worldPos;
    pixel.closestParticleFound = true;

    // Closest depth (may differ from picked in FTB mode)
    float4 clipPosClosest    = float4(0, 0, ndcDepthClosest, 1.0);
    float4 viewPosClosest    = mul(clipPosClosest, frameInfo.projInverse);
    float  viewZClosest      = viewPosClosest.z / viewPosClosest.w;
    float  linearDistClosest = viewZClosest / rayDirView.z;

    pixel.closestParticleDist = linearDistClosest;
#endif
  }

  // --- Init normal and splat ID related fields ---
#if NEED_SURFACE_INFO
  uint globalSplatId = rasterSplatIdBuffer[launchID];
  if(globalSplatId != 0xFFFFFFFF)  // Valid splat ID (not cleared value)
  {
    // Use the global index table to resolve to splat set and local index
    // 64-bit byte offset to prevent overflow when globalSplatId > ~537M (buffer > 4 GB)
    uint64_t byteOffset = uint64_t(globalSplatId) * uint64_t(sizeof(GlobalSplatIndexEntry));
    GlobalSplatIndexEntry entry   = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + byteOffset);
    pixel.isoSurfSplatSetId       = int(entry.splatSetIndex);
    pixel.isoSurfSplatId          = int(entry.splatIndex);
    pixel.closestParticleFound    = true;
    pixel.closestParticleGlobalId = int(globalSplatId);
    pixel.closestParticleId       = int(entry.splatIndex);
    pixel.closestSplatSetId       = int(entry.splatSetIndex);
  }

  float4 rasterNormal = rasterNormalBuffer[launchID];
  if(rasterNormal.w > 0.001)  // Has valid alpha
  {
    pixel.integratedNormal = normalize(rasterNormal.xyz);
  }

  // Final filtering for hybrid mode: if depth or normal is invalid, discard splat contribution
  if(!validDepth || length(pixel.integratedNormal) < 0.001)
  {
    discardSplatContribution(pixel);
  }
#endif
}
#endif  // HYBRID_ENABLED

#if NEED_SURFACE_INFO

// Discard splat contribution when surface data is invalid
// Resets depth, normal, splat IDs,
// This prevents unlit border artifacts in lighting mode
void discardSplatContribution(inout PixelData pixel)
{
  pixel.isoSurfDepth      = PAYLOAD_INF_DISTANCE;
  pixel.integratedNormal  = float3(0.0);
  pixel.isoSurfSplatId    = PAYLOAD_INVALID_ID;
  pixel.isoSurfSplatSetId = PAYLOAD_INVALID_ID;

  // Discard splat radiance to avoid unlit border artifacts
  pixel.radiance      = float3(0.0);
  pixel.transmittance = double3(1.0);  // Fully transparent - mesh/background shows through
  pixel.unlitAlbedo   = float3(0.0);

  // Reset closest particle debug fields for consistency
  pixel.closestParticleFound         = false;
  pixel.closestParticleGlobalId      = PAYLOAD_INVALID_ID;
  pixel.closestParticleId            = PAYLOAD_INVALID_ID;
  pixel.closestSplatSetId            = PAYLOAD_INVALID_ID;
  pixel.closestParticleDist          = PAYLOAD_INF_DISTANCE;
  pixel.closestParticleAlpha         = 0.0f;
  pixel.closestParticleNormal        = float3(0.0f);
  pixel.closestParticleWeight        = float3(0.0f);
  pixel.closestParticleTransmittance = double3(1.0f);
  pixel.closestParticleRadiance      = float3(0.0f);
}

#endif

void traceBounceRayMeshes(in int              bounce,
                          in float            tMin,
                          inout float         tMax,
                          in float3           rayOrigin,
                          in float3           rayDirection,
                          in bool             meshDepthOnly,
                          inout HitPayload    payload,
                          inout DlssGuideData dlssGuide,
                          inout PixelData     pixel)
{
  if(bounce > 0)
  {
    pixel.meshHitDist     = PAYLOAD_INF_DISTANCE;
    pixel.meshHitWorldPos = {};
    pixel.meshHitWorldNrm = {};
    pixel.meshHitObjId    = PAYLOAD_INVALID_ID;
    pixel.meshHitMatId    = PAYLOAD_INVALID_ID;
  }

  payload.dist[0] = PAYLOAD_INF_DISTANCE;
  payload.id[0]   = PAYLOAD_INVALID_ID;
  payload.id[1]   = PAYLOAD_INVALID_ID;

  RayDesc rayMesh;
  rayMesh.Origin    = rayOrigin;
  rayMesh.Direction = rayDirection;
  rayMesh.TMin      = tMin + epsT;
  rayMesh.TMax      = tMax + epsT;

  TraceRay((RaytracingAccelerationStructure)assets.meshTlasAddress,  // Cast device address to AS
           rayFlagsMesh,                                             // rayFlags
           frameInfo.rayMask,                                        // cullMask (computed on CPU)
           0,                                                        // sbtRecordOffset
           0,                                                        // sbtRecordStride
           0,                                                        // missIndex
           rayMesh,                                                  // ray
           payload                                                   // payload (location = 0)
  );

  pixel.meshHitDist = payload.dist[0];

  // store hit results
  if(pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
  {
    tMax = pixel.meshHitDist;
    // collect info to compute shading and potential new ray later on
    pixel.meshHit         = true;
    pixel.meshHitObjId    = payload.id[0];
    pixel.meshHitMatId    = payload.id[1];
    pixel.meshHitWorldPos = float3(payload.dist[1], payload.dist[2], payload.dist[3]);
    pixel.meshHitWorldNrm = float3(payload.dist[4], payload.dist[5], payload.dist[6]);
    //
    if(bounce == 0)
    {
      pixel.primaryHitDist = tMax;
      pixel.primaryHitPos  = rayOrigin + tMax * rayDirection;
    }
  }
}

void traceBounceRayParticles(in int              bounce,
                             in float            tMin,
                             in float            tMax,
                             in float3           rayOrigin,
                             in float3           rayDirection,
                             inout uint          seedRayPass,
                             inout HitPayload    payload,
                             inout DlssGuideData dlssGuide,
                             inout PixelData     pixel,
                             in bool             isFeedbackPixel)
{

  // For bounce > 0, reset integration tracking (primary depth is saved in PixelData for final write)
  if(bounce > 0)
  {
    pixel.isoSurfDepth     = 0.0;
    pixel.isoSurfPos       = float3(0.0);
    pixel.integratedNormal = float3(0.0);
  }

  // perform the evaluation of ray particle intersections depending on sorting strategy
  traceRayParticles<RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT, RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_PASS_STOCHASTIC>(
      bounce, tMin, tMax, rayOrigin, rayDirection, seedRayPass, payload, pixel, isFeedbackPixel);

#if NEED_SURFACE_INFO
  // Final filtering after integration
  if(bounce == 0)
  {
    surfaceFinalFiltering(pixel, rayDirection);
  }
#endif
}

// perform the evaluation of ray particle intersections depending on sorting strategy
void traceRayParticles<let kStochasticSort : bool, let kStochasticPass : bool>(in int           bounce,
                                                                               in float         tMin,
                                                                               in float         tMax,
                                                                               in float3        rayOrigin,
                                                                               in float3        rayDirection,
                                                                               inout uint       seedRayPass,
                                                                               inout HitPayload payload,
                                                                               inout PixelData  pixel,
                                                                               in bool          isFeedbackPixel)
{
// This compile conditional is not nice, but is due to the fact that
// stochastic Sort uses compile dependent attributes in the payload
// We shall use the optional attribute mechanism of slang, TODO
#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
  if(kStochasticSort)
  {
    traceRayParticlesStochasticSort(bounce, tMin, tMax, rayOrigin, rayDirection, seedRayPass, payload, pixel);
  }
  else
#endif
  {
    traceRayParticlesInsertionSort<kStochasticPass>(bounce, tMin, tMax, rayOrigin, rayDirection, seedRayPass, payload, pixel, isFeedbackPixel);
  }
};

// version that sorts particle intersection inside the anyhit shader
void traceRayParticlesInsertionSort<let kStochasticPass : bool>(in int           bounce,
                                                                in float         tMin,
                                                                in float         tMax,
                                                                in float3        rayOrigin,
                                                                in float3        rayDirection,
                                                                inout uint       seedRayPass,
                                                                inout HitPayload payload,
                                                                inout PixelData  pixel,
                                                                in bool          isFeedbackPixel)
{
#if WIREFRAME
  bool wireframeDone = false;
#endif

  // If Monte Carlo on Passes: do we compute next pass or stop
  bool abortedTracing = false;

  int passIdx = 0;

  while(!abortedTracing && (passIdx < frameInfo.maxPasses) && (tMin <= tMax)
        && (maxComponent(pixel.transmittance) > double(frameInfo.minTransmittance)))
  {
    // prepare the payload
    [[unroll]]
    for(int i = 0; i < PARTICLES_SPP; ++i)
    {
      payload.id[i]   = PAYLOAD_INVALID_ID;
      payload.dist[i] = PAYLOAD_INF_DISTANCE;
    }

#if WIREFRAME
    // Infinite means no wireframe hit by the ray
    payload.wireframeDist.set(PAYLOAD_INF_DISTANCE);
    payload.wireframeAlpha.set(0.0);
#endif

    // trace the PARTICLES_SPP any hits
    RayDesc raySplat;
    raySplat.Origin    = rayOrigin;
    raySplat.Direction = rayDirection;
    raySplat.TMin      = tMin + epsT;
    raySplat.TMax      = tMax + epsT;

    // Trace against all splat TLAS structures (multi-TLAS support for large scenes)
    traceRayAllSplatTlas(raySplat, rayFlags, payload);

    const int firstId = payload.id[0];

    // no more hits found
    if(firstId == PAYLOAD_INVALID_ID)
    {
      break;
    }

#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_PASS_STOCHASTIC
    // this is certainly not optimal
    const PixelData oldPixel        = pixel;
    int             lastValidHitIdx = 0;
#endif

    // evaluate the sorted hits
    [[unroll]]
    for(int i = 0; i < PARTICLES_SPP; ++i)
    {
      const uint  globalSplatID = payload.id[i];
      const float dist          = payload.dist[i];

      if((globalSplatID != PAYLOAD_INVALID_ID) && (maxComponent(pixel.transmittance) > double(frameInfo.minTransmittance)))
      {
#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_PASS_STOCHASTIC
        // Track the farthest valid payload slot seen so far.
        lastValidHitIdx = i;
#endif

        // 64-bit byte offset to prevent overflow when globalSplatID > ~537M (buffer > 4 GB)
        uint64_t              gitByteOffset    = uint64_t(globalSplatID) * uint64_t(sizeof(GlobalSplatIndexEntry));
        GlobalSplatIndexEntry entry            = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + gitByteOffset);
        SplatSetDesc          hitSplatSetDesc = assets.splatSetDescriptors[entry.splatSetIndex];
        const int             splatId         = int(entry.splatIndex);

        // rays are in world space
        // We need to transform to the splat set's model space for particleProcessHit
        const float3 hitSplatSetModelRayOrigin = mul(float4(rayOrigin, 1.0), hitSplatSetDesc.transformInverse).xyz;
        const float3 hitSplatSetModelRayDirection = normalize(mul(rayDirection, hitSplatSetDesc.transformRotScaleInverse));

        float3 normalWorld;
        float  alpha            = 0.0;
        float3 weight           = float3(0.0);
        float3 particleRadiance = float3(0.0);

        const bool acceptedHit =
            particleProcessHit<(NEED_SURFACE_INFO != 0)>(frameInfo, hitSplatSetDesc, hitSplatSetModelRayOrigin,
                                                         hitSplatSetModelRayDirection, hitSplatSetDesc.transformRotScaleInverse,
                                                         splatId, dist, alpha, particleRadiance, normalWorld);

        if(acceptedHit)
        {
          particleIntegrate(alpha, particleRadiance, pixel.transmittance, pixel.radiance, weight);
          pixel.hitCount++;

#if TRACE_PROFILE
          if(isFeedbackPixel)
          {
            collectHitProfile(globalSplatID, dist, alpha, particleRadiance, pixel.transmittance, float3(pixel.radiance));
          }
#endif

#if NEED_SURFACE_INFO
          // Accumulate weighted normal over all valid intersection (for all passes)
          // we could certainly do better, we use the normal to filter depth later on.
          pixel.integratedNormal = pixel.integratedNormal + normalWorld * weight;
#endif

          // only update when transmittance drops below threshold
          bool shouldUpdateDepth = (pixel.isoSurfDepth == 0.0 && pixel.transmittance.x < double(frameInfo.depthIsoThresholdRTX));
          if(shouldUpdateDepth)
          {
            pixel.isoSurfSplatSetId = entry.splatSetIndex;
            pixel.isoSurfSplatId    = splatId;
            pixel.isoSurfDepth      = dist;
            pixel.isoSurfPos        = rayOrigin + dist * rayDirection;
            if((bounce == 0) && (dist < pixel.primaryHitDist))
            {
              pixel.primaryHitDist = dist;
              pixel.primaryHitPos  = rayOrigin + dist * rayDirection;
            }
          }

          //
          if(bounce == 0 && !pixel.closestParticleFound)
          {
            // store info for debug feedback and alternative visualizations
            pixel.closestParticleFound         = true;
            pixel.closestParticleGlobalId      = globalSplatID;        // Global ID
            pixel.closestParticleId            = splatId;              // Local ID (entry.splatIndex)
            pixel.closestSplatSetId            = entry.splatSetIndex;  // Splat set index
            pixel.closestParticleDist          = dist;
            pixel.closestParticleNormal        = normalWorld;
            pixel.closestParticleAlpha         = alpha;
            pixel.closestParticleWeight        = weight;
            pixel.closestParticleTransmittance = pixel.transmittance;
            pixel.closestParticleRadiance      = particleRadiance;
          }
        }

        // we move on in any case
        tMin = max(tMin, dist);
      }
    }

#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_PASS_STOCHASTIC
    {
      float opacity = float(1.0 - maxComponent(pixel.transmittance));
      if(rand(seedRayPass) < opacity)
      {
        pixel.radiance = pixel.radiance / opacity;
        // if iso density surface not found we use farthest point, we do refetch the entry, not optimal
        if(pixel.isoSurfDepth == 0.0)
        {
          const uint            globalSplatID = payload.id[lastValidHitIdx];
          const float           dist          = payload.dist[lastValidHitIdx];
          // 64-bit byte offset to prevent overflow when globalSplatID > ~537M (buffer > 4 GB)
          uint64_t              gitByteOffset = uint64_t(globalSplatID) * uint64_t(sizeof(GlobalSplatIndexEntry));
          GlobalSplatIndexEntry entry         = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + gitByteOffset);
          SplatSetDesc          hitSplatSetDesc = assets.splatSetDescriptors[entry.splatSetIndex];
          const int             splatId         = int(entry.splatIndex);
          pixel.isoSurfSplatSetId               = entry.splatSetIndex;
          pixel.isoSurfSplatId                  = splatId;
          pixel.isoSurfDepth                    = dist;
          pixel.isoSurfPos                      = rayOrigin + dist * rayDirection;
          if((bounce == 0) && (dist < pixel.primaryHitDist))
          {
            pixel.primaryHitDist = dist;
            pixel.primaryHitPos  = rayOrigin + dist * rayDirection;
          }
        }
        // Terminate path, use current pass result
        pixel.transmittance = double3(0.0);  // fully opaque (matches raster stochastic splat)
        abortedTracing      = true;
      }
      else
      {
        // discard current pass, keep previous
        pixel = oldPixel;
      }
    }
#endif

#if WIREFRAME
    if(bounce == 0 && !wireframeDone
#if DISABLE_OPACITY_GAUSSIAN
       && payload.wireframeDist.get().value <= pixel.closestParticleDist
#endif
    )
    {
      pixel.radiance      = lerp(pixel.radiance, float3(1.0, 0.0, 0.0), payload.wireframeAlpha.get().value);
      pixel.transmittance = double3(0.0);  // opaque
      wireframeDone       = true;
    }
#endif

    //
    passIdx++;
  }  // End pass loop
}

#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
// version that uses stochastic sort (no "real" sort) of particle intersection inside the anyhit shader
void traceRayParticlesStochasticSort(in int           bounce,
                                     in float         tMin,
                                     in float         tMax,
                                     in float3        rayOrigin,
                                     in float3        rayDirection,
                                     inout uint       seedRayPass,
                                     inout HitPayload payload,
                                     inout PixelData  pixel)
{
  float3 totalRadianceSum = float3(0.0);
  float3 totalNormalSum   = float3(0.0);
  float  totalAlphaSum    = 0.0f;
  int    totalSampleCount = 0;

  // prepare the payload
  [[unroll]]
  for(int i = 0; i < PARTICLES_SPP; ++i)
  {
    payload.id[i]   = PAYLOAD_INVALID_ID;
    payload.dist[i] = PAYLOAD_INF_DISTANCE;
    payload.color[i].set(float4(0.0f));
    payload.normal[i].set(float3(0.0f));
  }

  const uint2 launchId = DispatchRaysIndex().xy;
  payload.rngSeed.set(xxhash32(uint3(launchId.x, launchId.y, frameInfo.frameSampleId)));

  // trace the PARTICLES_SPP any hits
  RayDesc raySplat;
  raySplat.Origin    = rayOrigin;
  raySplat.Direction = rayDirection;
  raySplat.TMin      = tMin + epsT;
  raySplat.TMax      = tMax + epsT;

  // Trace against all splat TLAS structures (multi-TLAS support for large scenes)
  traceRayAllSplatTlas(raySplat, rayFlags, payload);

  float3 passRadianceSum = float3(0.0);
  float3 passNormalSum   = float3(0.0);
  float  passAlphaSum    = 0.0f;
  int    passSampleCount = 0;

  float  closestDist          = PAYLOAD_INF_DISTANCE;
  int    closestGlobalSplatId = PAYLOAD_INVALID_ID;
  float3 closestNormal        = float3(0.0);
  float3 closestPos           = float3(0.0);
  float3 closestUnlit         = float3(0.0);

  [[unroll]]
  for(int i = 0; i < PARTICLES_SPP; ++i)
  {
    const int   globalSplatID = payload.id[i];
    const float dist          = payload.dist[i];

    // This would never occur
    if(globalSplatID == PAYLOAD_INVALID_ID)
    {
      continue;
    }

    // Skip samples behind the closest mesh hit
    if(dist >= pixel.meshHitDist)
    {
      continue;
    }

    float4 sampleColor    = payload.color[i].get().value;
    float3 sampleRadiance = sampleColor.rgb;
    float3 sampleNormal   = payload.normal[i].get().value;

    passRadianceSum += sampleRadiance;
    passNormalSum += sampleNormal;
    passAlphaSum += sampleColor.a;
    passSampleCount++;
    pixel.hitCount++;

    if(dist < closestDist)
    {
      closestDist          = dist;
      closestGlobalSplatId = globalSplatID;
      closestNormal        = sampleNormal;
      closestPos           = rayOrigin + dist * rayDirection;
      closestUnlit         = sampleRadiance;
    }
  }

  if(passSampleCount > 0)
  {
    totalRadianceSum += passRadianceSum;
    totalNormalSum += passNormalSum;
    totalAlphaSum += passAlphaSum;
    totalSampleCount += passSampleCount;
  }

  if(closestGlobalSplatId != PAYLOAD_INVALID_ID)
  {
    // Note: this re-fetches GlobalSplatIndexEntry that was already fetched in the anyhit.
    // We accept this single cached buffer load per pixel rather than enlarging the payload
    // with extra fields (which would increase register pressure for all rays including shadows).
    // 64-bit byte offset to prevent overflow when closestGlobalSplatId > ~537M (buffer > 4 GB)
    uint64_t              closestByteOffset = uint64_t(closestGlobalSplatId) * uint64_t(sizeof(GlobalSplatIndexEntry));
    GlobalSplatIndexEntry closestEntry      = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + closestByteOffset);
    pixel.isoSurfSplatSetId            = closestEntry.splatSetIndex;
    pixel.isoSurfSplatId               = int(closestEntry.splatIndex);
    pixel.isoSurfDepth                 = closestDist;
    pixel.isoSurfPos                   = closestPos;

    if((bounce == 0) && (closestDist < pixel.primaryHitDist))
    {
      pixel.primaryHitDist = closestDist;
      pixel.primaryHitPos  = closestPos;
    }

    if(bounce == 0)
    {
      pixel.closestParticleFound         = true;
      pixel.closestParticleGlobalId      = closestGlobalSplatId;
      pixel.closestParticleId            = pixel.isoSurfSplatId;
      pixel.closestSplatSetId            = pixel.isoSurfSplatSetId;
      pixel.closestParticleDist          = closestDist;
      pixel.closestParticleNormal        = closestNormal;
      pixel.closestParticleAlpha         = 1.0f;
      pixel.closestParticleWeight        = float3(1.0f);
      pixel.closestParticleTransmittance = pixel.transmittance;
      pixel.closestParticleRadiance      = closestUnlit;
    }
  }

  if(totalSampleCount > 0)
  {
    const float3 stochasticRadiance = totalRadianceSum / float(totalSampleCount);
    const float  avgAlpha           = totalAlphaSum / float(totalSampleCount);
    pixel.radiance      += float3(pixel.transmittance) * stochasticRadiance;
    pixel.transmittance *= double3(1.0 - avgAlpha);
    pixel.unlitAlbedo    = stochasticRadiance;
    const float3 avgNormal = totalNormalSum / float(totalSampleCount);
    pixel.integratedNormal = (length(avgNormal) > 0.0) ? normalize(avgNormal) : float3(0.0);
  }
}
#endif

// Helper to trace against all splat TLAS in the array (multi-TLAS support for large scenes)
// Traces against all TLAS structures until payload is full or all TLAS have been checked
void traceRayAllSplatTlas(in RayDesc ray, in uint rayFlags, inout HitPayload payload)
{
  // Get TLAS address and offset arrays
  uint64_t* tlasAddresses = (uint64_t*)assets.splatSetTlasArrayAddress;
  uint32_t* tlasOffsets   = (uint32_t*)assets.splatSetTlasOffsetAddress;

  // Trace against each TLAS in the array
  // Note: instanceCustomIndex is limited to 24 bits, so we use offsets for large scenes
  for(uint tlasIdx = 0; tlasIdx < assets.splatSetTlasCount; ++tlasIdx)
  {
    RaytracingAccelerationStructure currentTlas = (RaytracingAccelerationStructure)tlasAddresses[tlasIdx];

    // Set the base offset for this TLAS
    // Any-hit shader will add this to InstanceIndex() to get global splat ID
    payload.tlasInstanceOffset = tlasOffsets[tlasIdx];

    // Trace against this TLAS
    TraceRay(currentTlas, rayFlags, 0xFF, 0, 0, 0, ray, payload);
  }
}

#if NEED_SURFACE_INFO
// Final filtering for surface data after integration
// Fixes invalid normals and discards contribution if depth is invalid
// Call after particle integration (RTX mode) or after reading from raster pass (hybrid mode)
void surfaceFinalFiltering(inout PixelData pixel, in float3 rayDirection)
{
  // If integrated normal is invalid, force one pointing toward the camera
  if(length(pixel.integratedNormal) <= 0.2)
  {
    pixel.integratedNormal = -rayDirection;
  }

  // Check if we found a proper depth for the particle
  if(pixel.isoSurfDepth > 0.0 && pixel.isoSurfDepth < PAYLOAD_INF_DISTANCE)
  {
    pixel.integratedNormal = normalize(pixel.integratedNormal);
  }
  else
  {
    // No valid depth found - discard splat contribution
    discardSplatContribution(pixel);
  }
}
#endif

void populateDlssGuides(in PixelData pixel, inout DlssGuideData dlssGuide)
{
  // In hybrid mode, isoSurfDepth is initialized to INFINITY if no valid splat
  // So this comparison correctly handles: splat closer, mesh closer, or no hit cases
  if(pixel.isoSurfDepth < pixel.meshHitDist)
  {
    dlssGuide.isValid     = true;
    dlssGuide.hitPosition = pixel.isoSurfPos;

    dlssGuide.specularAlbedo  = {};
    dlssGuide.albedo          = float4(pixel.unlitAlbedo, 1.0);
    dlssGuide.normalRoughness = float4(pixel.integratedNormal, 0.9);
  }
  else if(pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
  {
    dlssGuide.isValid     = true;
    dlssGuide.hitPosition = pixel.meshHitWorldPos;

    const ObjMaterial meshMaterial = assets.meshes[pixel.meshHitObjId].materialAddress[pixel.meshHitMatId];
    dlssGuide.albedo               = float4(meshMaterial.diffuse.xyz, 1.0);
    dlssGuide.specularAlbedo       = meshMaterial.specular.xyz;
    dlssGuide.normalRoughness      = float4(pixel.meshHitWorldNrm, 1.0 - meshMaterial.shininess / 255.0);
  }
}

void evaluateLightingAndShadingForBounce(in int           bounce,
                                         inout PixelData  pixel,
                                         inout int        done,
                                         inout float3     rayOrigin,
                                         inout float3     rayDirection,
                                         inout uint       seedRayPass,
                                         inout HitPayload payload,
                                         in float3        radianceBeforeParticles)
{
#if LIGHTING_MODE != LIGHTING_DISABLED
  const bool splatIsClosest = (pixel.isoSurfDepth > 0.0) && (pixel.isoSurfDepth < pixel.meshHitDist);

  if(splatIsClosest && pixel.isoSurfSplatSetId != PAYLOAD_INVALID_ID)
  {
    // Isolate particle-only radiance: remove contributions from previous bounces
    // so evaluateLightingAndShadingParticles sees only the particle color as base
    pixel.radiance -= radianceBeforeParticles;
    evaluateLightingAndShadingParticles(pixel, rayOrigin, rayDirection, seedRayPass, payload);
    pixel.radiance += radianceBeforeParticles;
  }

  if((pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
     && (maxComponent(pixel.transmittance) > double(frameInfo.minMeshCompositeTransmittance)))
  {
    evaluateLightingAndShadingMeshes(pixel, done, rayOrigin, rayDirection, seedRayPass, payload);
  }
#else
  // Lighting disabled - process mesh shading if needed
  if((pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
     && (maxComponent(pixel.transmittance) > double(frameInfo.minMeshCompositeTransmittance)))
  {
    // Regular mesh - retrieve the material for this hit
    ObjMaterial meshHitMaterial = assets.meshes[pixel.meshHitObjId].materialAddress[pixel.meshHitMatId];

    // Display unlit material color (emission + ambient + diffuse)
    // Apply current transmittance so splats in front correctly occlude meshes
    const float3 meshColor = meshHitMaterial.emission + meshHitMaterial.ambient + meshHitMaterial.diffuse;
    pixel.radiance += float3(pixel.transmittance) * meshColor;
  }
#endif
}

#if LIGHTING_MODE != LIGHTING_DISABLED
// Evaluate lighting and shading for particles (splats)
// Uses global loop over lights with inline shadow tracing
void evaluateLightingAndShadingParticles(inout PixelData pixel, in float3 rayOrigin, in float3 rayDirection, inout uint seedRayPass, inout HitPayload payload)
{
  SplatSetDesc splatSetDesc = assets.splatSetDescriptors[pixel.isoSurfSplatSetId];

  // Start with particle radiance modulated by material emission
  float3 emissiveContribution = pixel.radiance * splatSetDesc.material.emission;
  //pixel.unlitAlbedo = pixel.radiance;

  // Early exit for pure emissive materials - no lighting interaction
  if(splatSetDesc.material.needShading != 1)
  {
    pixel.radiance = emissiveContribution;
    return;
  }

  // Modulate particle radiance by diffuse material
  const float3 baseRadiance = pixel.radiance;

  // Set up material for lighting computation
  ObjMaterial mat = {};
  mat.diffuse     = baseRadiance * splatSetDesc.material.diffuse;
  mat.ambient     = baseRadiance * splatSetDesc.material.ambient;
  mat.specular    = baseRadiance * splatSetDesc.material.specular;
  mat.shininess   = splatSetDesc.material.shininess;

  // Reset radiance to only emissive, lighting will add on top
  pixel.radiance = emissiveContribution;

  float3 shadowOrigin = rayOrigin + pixel.isoSurfDepth * rayDirection;

  // Global loop over lights with inline shadow tracing
  for(int i = 0; i < assets.lightCount; ++i)
  {
    LightSource light = assets.lights[i];

    // Compute light-to-surface vector, check if light is in range
    float3 lightDir;
    float  lightDist;
    if(!computeLightToSurfaceVector(light, pixel.isoSurfPos, seedRayPass, lightDir, lightDist))
      continue;  // Light culled (out of range)

#if SHADOWS_MODE != SHADOWS_DISABLED
    // Trace shadow ray inline for this light
    ShadowInfo shadow = traceShadowRayForLight(frameInfo, assets, shadowOrigin, lightDir, lightDist, payload);
    light.color *= shadow.transmittance;
    bool inShadow = (maxComponent(shadow.transmittance) < 0.001);
#else
    bool inShadow = false;
#endif

    wavefrontComputeShadingDirectOnly(light, pixel.isoSurfPos, pixel.integratedNormal, mat, rayDirection, inShadow,
                                      double3(1.0), pixel.radiance);
  }

  // Headlight fallback when no lights in scene
  if(assets.lightCount == 0)
  {
    const LightSource headlight = createHeadlight(frameInfo.cameraPosition);
    bool              inShadow  = false;
    wavefrontComputeShadingDirectOnly(headlight, pixel.isoSurfPos, pixel.integratedNormal, mat, rayDirection, inShadow,
                                      double3(1.0), pixel.radiance);
  }
}

// Evaluate lighting and shading for meshes
// Uses global loop over lights with inline shadow tracing
void evaluateLightingAndShadingMeshes(inout PixelData  pixel,
                                      inout int        done,
                                      inout float3     rayOrigin,
                                      inout float3     rayDirection,
                                      inout uint       seedRayPass,
                                      inout HitPayload payload)
{
  // Regular mesh - retrieve the material for this hit
  ObjMaterial meshMaterial = assets.meshes[pixel.meshHitObjId].materialAddress[pixel.meshHitMatId];

  // Add emissive contribution first (for light proxies and other emissive materials)
  pixel.radiance += meshMaterial.emission;

  // Early exit for pure emissive materials (light proxies) - no lighting interaction
  if(meshMaterial.needShading != 1)
  {
    return;
  }

  float3 shadowOrigin = pixel.meshHitWorldPos;

#if LIGHTING_MODE == LIGHTING_INDIRECT
  // Indirect mode: full path tracing with state save/restore per light and secondary bounces

  // Save initial state before light loop (wavefrontComputeShading modifies these)
  double3 initialTransmittance = pixel.transmittance;
  float3  initialRayOrigin     = rayOrigin;
  float3  initialRayDirection  = rayDirection;
  int     initialDone          = done;

  // Global loop over lights with inline shadow tracing
  for(int i = 0; i < assets.lightCount; ++i)
  {
    LightSource light = assets.lights[i];

    // Compute light-to-surface vector, check if light is in range
    float3 lightDir;
    float  lightDist;
    if(!computeLightToSurfaceVector(light, pixel.meshHitWorldPos, seedRayPass, lightDir, lightDist))
      continue;  // Light culled (out of range)

    // Restore state for this light (except first iteration)
    // Last light keeps modifications for secondary rays
    if(i > 0)
    {
      pixel.transmittance = initialTransmittance;
      rayOrigin           = initialRayOrigin;
      rayDirection        = initialRayDirection;
      done                = initialDone;
    }

#if SHADOWS_MODE != SHADOWS_DISABLED
    // Trace shadow ray inline for this light
    ShadowInfo shadow = traceShadowRayForLight(frameInfo, assets, shadowOrigin, lightDir, lightDist, payload);
    light.color *= shadow.transmittance;
    bool inShadow = (maxComponent(shadow.transmittance) < 0.001);
#else
    bool inShadow = false;
#endif

    wavefrontComputeShading(light, pixel.meshHitWorldPos, pixel.meshHitWorldNrm, meshMaterial, rayDirection, inShadow,
                            pixel.transmittance, pixel.radiance, done, rayOrigin, rayDirection);
  }

  // Headlight fallback when no lights in scene
  if(assets.lightCount == 0)
  {
    const LightSource headlight = createHeadlight(frameInfo.cameraPosition);
    bool              inShadow  = false;
    wavefrontComputeShading(headlight, pixel.meshHitWorldPos, pixel.meshHitWorldNrm, meshMaterial, rayDirection,
                            inShadow, pixel.transmittance, pixel.radiance, done, rayOrigin, rayDirection);
  }

#else
  // Direct mode: simple lighting loop, no secondary bounces, no state management

  for(int i = 0; i < assets.lightCount; ++i)
  {
    LightSource light = assets.lights[i];

    float3 lightDir;
    float  lightDist;
    if(!computeLightToSurfaceVector(light, pixel.meshHitWorldPos, seedRayPass, lightDir, lightDist))
      continue;

#if SHADOWS_MODE != SHADOWS_DISABLED
    ShadowInfo shadow = traceShadowRayForLight(frameInfo, assets, shadowOrigin, lightDir, lightDist, payload);
    light.color *= shadow.transmittance;
    bool inShadow = (maxComponent(shadow.transmittance) < 0.001);
#else
    bool inShadow = false;
#endif

    wavefrontComputeShadingDirectOnly(light, pixel.meshHitWorldPos, pixel.meshHitWorldNrm, meshMaterial, rayDirection,
                                      inShadow, pixel.transmittance, pixel.radiance);
  }

  // Headlight fallback when no lights in scene
  if(assets.lightCount == 0)
  {
    const LightSource headlight = createHeadlight(frameInfo.cameraPosition);
    bool              inShadow  = false;
    wavefrontComputeShadingDirectOnly(headlight, pixel.meshHitWorldPos, pixel.meshHitWorldNrm, meshMaterial,
                                      rayDirection, inShadow, pixel.transmittance, pixel.radiance);
  }

  // No bounces — stop tracing
  pixel.transmittance = double3(0.0);

#endif  // LIGHTING_MODE == LIGHTING_INDIRECT
}
#endif  // LIGHTING_MODE != LIGHTING_DISABLED

// Trace shadow ray for a single light and compute transmission
ShadowInfo traceShadowRayForLight(in FrameInfo     frameInfo,
                                  in SceneAssets   assets,
                                  in float3        shadowOrigin,
                                  in float3        lightDir,
                                  in float         lightDist,
                                  inout HitPayload payload)
{
  ShadowInfo result;
  result.transmittance = float3(1.0);

  // Trace through meshes first (typically fewer, faster to reject)
  ShadowInfo meshShadow;
  meshShadow.transmittance = float3(1.0);
  if(assets.meshCount > 0)
  {
    meshShadow = traceShadowRayMesh(assets, shadowOrigin, lightDir, lightDist, payload);
  }

  // Trace through particles (only if mesh didn't fully occlude)
  // Skip for stochastic anyhit: alpha=1 in anyhit makes shadow transmittance always 0
  ShadowInfo particleShadow;
  particleShadow.transmittance = float3(1.0);

  if(maxComponent(meshShadow.transmittance) > 0.001 && assets.totalGlobalSplatCount > 0)
  {
    particleShadow = traceShadowRayParticle(frameInfo, assets, shadowOrigin, lightDir, lightDist, payload);
  }
  result.transmittance = meshShadow.transmittance * particleShadow.transmittance;

  return result;
}

// Test if a shadow ray hits a mesh and compute shadow transmission (soft shadows with colored transmission)
ShadowInfo traceShadowRayMesh(in SceneAssets assets, in float3 shadowOrigin, in float3 shadowDir, in float lightDist, inout HitPayload payload)
{
  ShadowInfo result;
  result.transmittance = float3(1.0);  // Start fully transmissive (white)

  payload.dist[0] = PAYLOAD_INF_DISTANCE;
  payload.id[0]   = PAYLOAD_INVALID_ID;
  payload.id[1]   = PAYLOAD_INVALID_ID;

  RayDesc shadowRay;
  shadowRay.Origin    = shadowOrigin;
  shadowRay.Direction = shadowDir;
  shadowRay.TMin      = 0.001;
  shadowRay.TMax      = lightDist - 0.001;

  // Shadow ray flags: terminate on first hit (single transmissive surface)
  const uint shadowRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

  // Trace with cull mask 0xFE (excludes light proxies with mask 0x01)
  TraceRay((RaytracingAccelerationStructure)assets.meshTlasAddress, shadowRayFlags,
           0xFE,  // Cull mask: only hit regular meshes
           0, 0, 0, shadowRay, payload);

  // Check if mesh hit
  if(payload.dist[0] < lightDist && payload.id[0] != PAYLOAD_INVALID_ID)
  {
    // Lookup mesh material to get transmittance (for colored shadows)
    int meshId = payload.id[0];
    int matId  = payload.id[1];

    if(matId != PAYLOAD_INVALID_ID)
    {
      // Get material from mesh descriptor
      ObjMaterial material = assets.meshes[meshId].materialAddress[matId];

      // Apply material transmittance (colored shadows from glass, etc.)
      result.transmittance = material.transmittance;
    }
    else
    {
      // No material ID - assume opaque
      result.transmittance = float3(0.0);
    }
  }

  return result;
}

// Test if a shadow ray hits particles and compute shadow transmission (soft shadows with colored transmission)
ShadowInfo traceShadowRayParticle(in FrameInfo     frameInfo,
                                  in SceneAssets   assets,
                                  in float3        shadowOrigin,
                                  in float3        shadowDir,
                                  in float         lightDist,
                                  inout HitPayload payload)
{
  ShadowInfo result;
  result.transmittance = float3(1.0);  // Start fully transmissive

  // Add offset to shadow origin to prevent self-shadowing from layers of splats
  // Use a larger offset for particles due to their volumetric nature
  const float3 offsetShadowOrigin = shadowOrigin + shadowDir * frameInfo.particleShadowOffset;

  // Prepare payload for particle hits
  [[unroll]]
  for(int j = 0; j < PARTICLES_SPP; ++j)
  {
    payload.id[j]   = PAYLOAD_INVALID_ID;
    payload.dist[j] = PAYLOAD_INF_DISTANCE;
#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
    payload.color[j].set(float4(0.0f));
    payload.normal[j].set(float3(0.0f));
#endif
  }

  RayDesc splatShadowRay;
  splatShadowRay.Origin    = offsetShadowOrigin;  // World space origin with offset
  splatShadowRay.Direction = shadowDir;           // World space direction
  splatShadowRay.TMin      = 0.0;
  splatShadowRay.TMax      = lightDist - 0.001;

  // Use any-hit shader (same as primary ray particle tracing)
  const uint splatShadowRayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

  // Trace against all particle TLAS structures (multi-TLAS support for large scenes)
  traceRayAllSplatTlas(splatShadowRay, splatShadowRayFlags, payload);

#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
  // Stochastic path: alpha and radiance were already computed in the anyhit shader
  // and stored in payload.color[]. Use averaging (not integration), consistent with
  // the stochastic primary ray approach in traceRayParticlesStochasticSort.
  float shadowAlphaSum    = 0.0f;
  int   shadowSampleCount = 0;

  [[unroll]]
  for(int j = 0; j < PARTICLES_SPP; ++j)
  {
    const int   globalSplatID = payload.id[j];
    const float dist          = payload.dist[j];

    if(globalSplatID == PAYLOAD_INVALID_ID || dist >= lightDist)
      continue;

    float4 precomputed = payload.color[j].get().value;
    shadowAlphaSum += precomputed.a;
    shadowSampleCount++;
  }

  if(shadowSampleCount > 0)
  {
    const float avgAlpha = shadowAlphaSum / float(shadowSampleCount);
    result.transmittance = float3(saturate(1.0 - avgAlpha));
  }
#else
  double3 shadowTransmittance = double3(1.0, 1.0, 1.0);
  float3  shadowRadiance      = float3(0.0, 0.0, 0.0);
  float3  dummyNormal         = float3(0.0, 0.0, 0.0);

  const double TRANSMITTANCE_THRESHOLD = double(frameInfo.particleShadowTransmittanceThreshold);
  const float  coloredShadowStrength   = frameInfo.particleShadowColorStrength;

  [[unroll]]
  for(int j = 0; j < PARTICLES_SPP; ++j)
  {
    const int   globalSplatID = payload.id[j];
    const float dist          = payload.dist[j];

    if(globalSplatID == PAYLOAD_INVALID_ID || dist >= lightDist)
      continue;

    // 64-bit byte offset to prevent overflow when globalSplatID > ~537M (buffer > 4 GB)
    uint64_t              gitByteOffset = uint64_t(globalSplatID) * uint64_t(sizeof(GlobalSplatIndexEntry));
    GlobalSplatIndexEntry entry         = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + gitByteOffset);
    SplatSetDesc          shadowHitDesc = assets.splatSetDescriptors[entry.splatSetIndex];
    const int             splatId       = int(entry.splatIndex);

    const float3 splatModelShadowOrigin = mul(float4(offsetShadowOrigin, 1.0), shadowHitDesc.transformInverse).xyz;
    const float3 splatModelShadowDir    = normalize(mul(shadowDir, shadowHitDesc.transformRotScaleInverse));

    float3 weight, particleRad;
    float  alpha;

    bool hit = particleProcessHit<false>(frameInfo, shadowHitDesc, splatModelShadowOrigin, splatModelShadowDir,
                                         shadowHitDesc.transformRotScaleInverse, splatId, dist, alpha, particleRad, dummyNormal);
    if(hit)
    {
      particleIntegrate(alpha, particleRad, shadowTransmittance, shadowRadiance, weight);
    }

    if(maxComponent(shadowTransmittance) < TRANSMITTANCE_THRESHOLD)
    {
      shadowTransmittance = double3(0.0);
      break;
    }
  }

  // Post-loop: threshold is a hard cutoff for black shadows.
  // T in [0, threshold] → black.  T in (threshold, 1) → color transmission zone.  T = 1 → no shadow.
  float  T               = saturate(float(shadowTransmittance.x));
  float  threshold       = float(TRANSMITTANCE_THRESHOLD);
  float  scaledT         = saturate((T - threshold) / (1.0 - threshold));
  float  maxRad          = maxComponent(shadowRadiance);
  float3 normalizedColor = maxRad > 0.001 ? shadowRadiance / maxRad : float3(1.0);
  // Per-channel color tinting: dark color channels stay darker, bright channels ramp faster.
  // Effect fades to zero at scaledT=1 (no shadow) to avoid tinting fully lit areas.
  result.transmittance = saturate(scaledT * lerp(float3(1.0), normalizedColor, coloredShadowStrength * (1.0 - scaledT)));
#endif

  return result;
}

void writeToGBuffers<let kVisualizeMode : uint, let kHybridEnabled : uint>(in FrameInfo frameInfo,
                                                                           in float2    resolution,
                                                                           in uint2     pixelId,
                                                                           in PixelData pixel,
                                                                           in RWTexture2D<float> depthBuffer,
                                                                           in RWTexture2D<float4> image,
                                                                           in RWTexture2D<float4> imageAux,
                                                                           in int                 useDlss,
                                                                           in DlssGuideData       dlssGuide,
                                                                           in RWTexture2D<float4> outImages[])
{

//#define SHOW_MISS_AS_RED  // uncomment for debug
#ifdef SHOW_MISS_AS_RED
  // Miss color as red
  float3 pixelColor = float3(1.0, 0.0, 0.0);
#else
  // miss color is black. it is not background color, background color is
  // set at pixel parameter initialization through default pixel.radiance value
  float3 pixelColor = float3(0.0, 0.0, 0.0);
#endif

  // Write depth for additional passes (like 3D helpers)
  // Use PRIMARY depth (bounce 0)
  if(pixel.primaryHitDist < PAYLOAD_INF_DISTANCE)
  {
    const float4 clip    = mul(mul(float4(pixel.primaryHitPos, 1.0), frameInfo.viewMatrix), frameInfo.projectionMatrix);
    const float3 ndc     = clip.xyz / clip.w;
    depthBuffer[pixelId] = ndc.z;
  }
#if !HYBRID_ENABLED
  // In non-hybrid mode, write far plane for misses
  // In hybrid mode, preserve depth from raster pass (splats already wrote their depth)
  else
  {
    depthBuffer[pixelId] = 1.0;  // Far plane in NDC - no hit
  }
#endif

  //
  switch(kVisualizeMode)
  {
    case VISUALIZE_DLSS_INPUT:
    case VISUALIZE_FINAL:
      // the final pixel color is the integrated radiance
      pixelColor = pixel.radiance;
      if(kHybridEnabled == 1)
      {
        const bool generateSurface = (NEED_SURFACE_INFO != 0);
        if(pixel.meshHit || generateSurface)
        {
          if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
            image[pixelId] = float4(pixelColor, 1.0);
          else
            imageAux[pixelId] = float4(pixelColor, 1.0);
        }
        // For hybrid mode without DLSS, early return (DLSS output is handled below)
        if(useDlss == 0)
          return;
      }
      else
      {
        // Perform temporal accumulation temporal sampling is enabled
        // DLSS handles its own temporal accumulation, so we do not accumulate for DLSS
        if(frameInfo.frameSampleId > 0 && pcRay.useDlss == 0)
        {
          // Do accumulation over time (only when DLSS is disabled)
          const float  a        = 1.0F / float(frameInfo.frameSampleId + 1);
          const float4 oldColor = image[pixelId];
          pixelColor            = lerp(oldColor, float4(pixelColor, 1.0), a).xyz;
        }
      }
      break;
    case VISUALIZE_CLOCK:
      uint64_t clock = clockARB() - pixel.clockStart;
      pixelColor = computeVisualizationColor(0.1 * float(clock) / float(1 << 20), frameInfo.visuMinMax, frameInfo.visuShift, false);
      break;
    case VISUALIZE_RAYHITS:
      pixelColor = computeVisualizationColor(float(pixel.hitCount), frameInfo.visuMinMax, frameInfo.visuShift, false);
      break;
    case VISUALIZE_DEPTH:
      if(pixel.closestParticleFound)
      {
        pixelColor = computeVisualizationColor(pixel.closestParticleDist, frameInfo.visuMinMax, frameInfo.visuShift, true);
      }
      break;
    case VISUALIZE_DEPTH_INTEGRATED:
      if(pixel.isoSurfDepth < PAYLOAD_INF_DISTANCE)
      {
        pixelColor = computeVisualizationColor(pixel.isoSurfDepth, frameInfo.visuMinMax, frameInfo.visuShift, true);
      }
      break;
    case VISUALIZE_DEPTH_FOR_DLSS:
      if(dlssGuide.isValid)
      {
        const float4 posView     = mul(float4(dlssGuide.hitPosition, 1.0), frameInfo.viewMatrix);
        const float  linearDepth = -posView.z;  // Actual distance from camera
        pixelColor = computeVisualizationColor(linearDepth, frameInfo.visuMinMax, frameInfo.visuShift, true);
      }
      break;
    case VISUALIZE_NORMAL:
      if(pixel.closestParticleFound)
      {
        const float3 normal = pixel.closestParticleNormal;
        pixelColor          = float3(normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f);
      }
      break;
    case VISUALIZE_NORMAL_INTEGRATED:
      if(length(pixel.integratedNormal) != 0.0)
      {
        const float3 normal = pixel.integratedNormal;
        pixelColor          = float3(normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f);
      }
      break;
    case VISUALIZE_NORMAL_FOR_DLSS:
      if(dlssGuide.isValid)
      {
        const float3 normal = dlssGuide.normalRoughness.xyz;
        if(length(normal) != 0.0)
          pixelColor = float3(normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f);
      }
      break;
    case VISUALIZE_SPLAT_ID:
      // Harlequin/false color visualization of splat ID
      if(pixel.isoSurfSplatId != PAYLOAD_INVALID_ID)
      {
        // Hash the splat ID to generate a pseudo-random color
        // Using a simple but effective hash for visual distinctness
        uint id = uint(pixel.isoSurfSplatId);
        // Robert Jenkins' 32-bit integer hash
        id = (id + 0x7ed55d16u) + (id << 12);
        id = (id ^ 0xc761c23cu) ^ (id >> 19);
        id = (id + 0x165667b1u) + (id << 5);
        id = (id + 0xd3a2646cu) ^ (id << 9);
        id = (id + 0xfd7046c5u) + (id << 3);
        id = (id ^ 0xb55a4f09u) ^ (id >> 16);
        // Convert to RGB
        float r = float((id >> 0) & 0xFF) / 255.0;
        float g = float((id >> 8) & 0xFF) / 255.0;
        float b = float((id >> 16) & 0xFF) / 255.0;
        // Boost saturation for better visibility
        float maxC = max(r, max(g, b));
        float minC = min(r, min(g, b));
        if(maxC > minC)
        {
          float scale = 1.0 / (maxC - minC + 0.001);
          r           = (r - minC) * scale * 0.8 + 0.1;
          g           = (g - minC) * scale * 0.8 + 0.1;
          b           = (b - minC) * scale * 0.8 + 0.1;
        }
        pixelColor = float3(r, g, b);
      }
      break;
  }

  // Write to GBuffers

  if(useDlss == 0)
  {
    // Normal rendering without DLSS - write to appropriate buffer based on temporal accumulation state
    // For hybrid mode, must match the buffer selection used by rasterization
    if(kHybridEnabled == 1 && frameInfo.frameSampleId > 0)
      imageAux[pixelId] = float4(pixelColor, 1.0);
    else
      image[pixelId] = float4(pixelColor, 1.0);
  }
  else
  {
    // Write rendered image (to upscale or denoise) to DLSS input (eResultImage in DLSS G-buffers)
    outImages[int(DlssImages::eDlssInputImage)][pixelId] = float4(pixelColor, 1.0);

    // Write guide images if we have valid first hit data
    if(dlssGuide.isValid)
    {
      // Calculate view-space depth
      const float4x4 viewProjMatrix = mul(frameInfo.viewMatrix, frameInfo.projectionMatrix);
      const float4   posClip        = mul(float4(dlssGuide.hitPosition, 1.0), viewProjMatrix);
      const float    viewZ          = posClip.z / posClip.w;
      const float4   depthValue     = float4(abs(viewZ), 0.0, 0.0, 0.0);

      // Calculate motion vectors
      const float2 motion = calculateMotionVector(dlssGuide.hitPosition, frameInfo.prevViewProjMatrix, viewProjMatrix, resolution);

      // Write to DLSS guide images
      outImages[int(DlssImages::eDlssAlbedo)][pixelId]          = dlssGuide.albedo;
      outImages[int(DlssImages::eDlssSpecAlbedo)][pixelId]      = float4(dlssGuide.specularAlbedo, 1.0);
      outImages[int(DlssImages::eDlssNormalRoughness)][pixelId] = dlssGuide.normalRoughness;
      outImages[int(DlssImages::eDlssMotion)][pixelId]          = float4(motion, 0.0, 0.0);
      outImages[int(DlssImages::eDlssDepth)][pixelId]           = depthValue;
    }
    else
    {
      // No hit - write default/background values for guide images
      outImages[int(DlssImages::eDlssAlbedo)][pixelId]          = float4(0.0, 0.0, 0.0, 0.0);
      outImages[int(DlssImages::eDlssSpecAlbedo)][pixelId]      = float4(0.0, 0.0, 0.0, 0.0);
      outImages[int(DlssImages::eDlssNormalRoughness)][pixelId] = float4(0.0, 0.0, 0.0, 0.0);
      outImages[int(DlssImages::eDlssMotion)][pixelId]          = float4(0.0, 0.0, 0.0, 0.0);
      outImages[int(DlssImages::eDlssDepth)][pixelId]           = float4(0.0, 0.0, 0.0, 0.0);
    }
  }
}

float3 computeVisualizationColor(in float value, in float2 minMax, in float shift, bool flip)
{
  if(flip)
    return turboColormap(clamp(shift + 1.0 - (value - minMax[0]) / (minMax[1] - minMax[0]), 0.0, 1.0));
  else
    return turboColormap(clamp(shift + (value - minMax[0]) / (minMax[1] - minMax[0]), 0.0, 1.0));
}
