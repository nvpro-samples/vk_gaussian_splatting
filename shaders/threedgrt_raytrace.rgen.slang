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
#include "shading.h.slang"
#include "wireframe_extended.h.slang"

#include "threedgs_particle_storage.h.slang"
#include "threedgrt.h.slang"
#include "threedgrt_payload.h.slang"
#if RTX_HAS_MESHES
#include "mesh_hit_packing.h.slang"
#endif

// clang-format off
[[vk::binding(RTX_BINDING_OUTIMAGE, 1)]] RWTexture2D<float4> image;
[[vk::binding(RTX_BINDING_AUX1, 1)]] RWTexture2D<float4> imageAux;
[[vk::binding(RTX_BINDING_OUTDEPTH, 1)]] RWTexture2D<float> depthBuffer;
#if DLSS_ENABLED
#include "dlss_util.h"
[[vk::binding(RTX_BINDING_DLSS_OUT_IMAGES, 1)]] RWTexture2D<float4> outImages[];
#endif

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

// Environment equirectangular texture (sky or HDR IBL)
[[vk::binding(RTX_BINDING_SKY_TEXTURE, 1)]] Sampler2D skyTexture;

#if LIGHTING_MODE != LIGHTING_DISABLED
// HDR importance sampling acceleration buffer
[[vk::binding(RTX_BINDING_ENV_ACCEL, 1)]] StructuredBuffer<EnvAccel> envAccelBuffer;
#endif

#if RTX_HAS_MESHES
// Mesh material textures (base color, metallic-roughness, normal, emissive, occlusion)
[[vk::binding(BINDING_MESH_TEXTURES, 0)]] Sampler2D meshTextures[];
#endif

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
  float2 meshHitUV       = {};
  float2 meshHitUV1      = {};
  float4 meshHitTangent  = {};


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

  // Mesh wireframe (saved from mesh trace before particle passes overwrite payload)
  float meshWireframeDist  = PAYLOAD_INF_DISTANCE;
  float meshWireframeAlpha = 0.0;
};

#if DLSS_ENABLED
struct DlssGuideData
{
  float4 albedo          = float4(0.0, 0.0, 0.0, 1.0);
  float3 specularAlbedo  = float3(0.0);
  float4 normalRoughness = float4(0.0, 0.0, 0.0, 0.9);
  float3 hitPosition     = float3(0.0);
  float3 prevHitPosition = float3(0.0);  // Previous frame's world position (for object motion vectors)
  bool   isValid         = false;
};
#endif

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
#if DLSS_ENABLED
  subPixelOffset += pcRay.jitter;
#endif

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

#if DLSS_ENABLED
  DlssGuideData dlssGuide = {};
#endif

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

  int   bounce      = 0;      // id of the bounce
  int   done        = 1;      // continue on next bounce or stop if done
  float lastBrdfPdf = DIRAC;  // BRDF sampling PDF from last bounce (for env miss MIS)

  const bool isFeedbackPixel = all(launchID == frameInfo.cursor);
#if DLSS_ENABLED
  const float3 cameraOrigin = rayOrigin;  // Save before bounce loop modifies rayOrigin
#endif

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
    float tMin               = (bounce == 0) ? frameInfo.nearFar.x : frameInfo.rtxSecondaryRayOffset;
    float tMax               = frameInfo.nearFar.y;
    float rayLastHitDistance = max(0.0f, tMin - epsT);

#if RTX_HAS_MESHES
    // trace the mesh with one closest hit if needed
    if(assets.meshCount > 0)
    {
      // if mesh hit, will modify Tmax, so splat tracing if any will be on reduced range later on
      traceBounceRayMeshes(bounce, tMin, tMax, rayOrigin, rayDirection, pcRay.meshDepthOnly, payload, pixel);

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
#endif  // RTX_HAS_MESHES

#if RTX_HAS_PARTICLES
    // when hybrid is enabled, we trace the particles only for secondary rays
#if HYBRID_ENABLED
    const float3 radianceBeforeParticles = (bounce > 0) ? pixel.radiance : float3(0.0);
    if(bounce > 0)
    {
#else
    const float3 radianceBeforeParticles = pixel.radiance;
#endif
      if(assets.totalGlobalSplatCount > 0)
      {
        traceBounceRayParticles(bounce, tMin, tMax, rayOrigin, rayDirection, seedRayPass, payload, pixel, isFeedbackPixel);
      }

#if HYBRID_ENABLED
    }  // End bounce 0 skip
#endif
#else
    const float3 radianceBeforeParticles = pixel.radiance;
#endif  // RTX_HAS_PARTICLES

    evaluateLightingAndShadingForBounce(bounce, pixel, done, rayOrigin, rayDirection, seedRayPass, payload,
                                        radianceBeforeParticles, lastBrdfPdf);

    // Mesh wireframe overlay — applied after mesh shading so we blend with the shaded mesh color.
    // Particle wireframe is handled separately inside the particle pass loop (before this point).
    // Only show mesh wireframe when the mesh is in front of (or there is no) particle surface.
#if WIREFRAME
    if(bounce == 0 && pixel.meshWireframeAlpha > 0.0 && (pixel.isoSurfDepth <= 0.0 || pixel.meshHitDist <= pixel.isoSurfDepth))
    {
      pixel.radiance      = lerp(pixel.radiance, float3(1.0, 0.0, 0.0), pixel.meshWireframeAlpha);
      pixel.transmittance = double3(0.0);
    }
#endif

    if(bounce == 0)
    {
#if DLSS_ENABLED
      populateDlssGuides(pixel, cameraOrigin, dlssGuide);
#endif

      // shader feedback (TODO rename attributes for better coherence)
      if(all(launchID == frameInfo.cursor))
      {
        indirectBuffer[0].particleGlobalId             = pixel.closestParticleGlobalId;  // Global ID
        indirectBuffer[0].particleId                   = pixel.closestParticleId;        // Local ID
        indirectBuffer[0].splatSetId                   = pixel.closestSplatSetId;        // Splat set index
        indirectBuffer[0].particleHitCount             = pixel.hitCount;
        indirectBuffer[0].particleRadiance             = pixel.radiance;
        indirectBuffer[0].particleTransmittance        = float3(pixel.transmittance);
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

    // Russian roulette: probabilistically terminate low-energy paths after minimum depth
    if(bounce >= 2)
    {
      float rrThroughput = float(maxComponent(pixel.transmittance));
      float rrProb       = min(rrThroughput + 0.001, 0.95);
      if(rand(seedRayPass) >= rrProb)
        break;
      pixel.transmittance /= double(rrProb);
    }

    // prepare for next iteration
    done = 1;

  }  // while(true) - end bounce loop

  // Add environment background contribution through remaining transmittance.
  // Covers both full miss and partial transparency from semi-transparent splats.
  // When the last bounce used BSDF sampling, apply MIS weight to avoid
  // double-counting with the NEE environment sample from that bounce.
  if(frameInfo.envMode != ENV_MODE_NONE && maxComponent(pixel.transmittance) > double(frameInfo.minSplatSetCompositeTransmittance))
  {
    float3 envDir = rayDirection;
    if(frameInfo.envMode == ENV_MODE_HDR)
      envDir = applyEnvRotation(envDir, frameInfo.envRotation);

    float2 skyUV = (frameInfo.envMode == ENV_MODE_HDR) ? directionToHdrEquirectUV(envDir) : directionToEquirectUV(envDir);
    float3 skyColor = skyTexture.SampleLevel(skyUV, 0).rgb;
    if(frameInfo.envMode == ENV_MODE_HDR)
      skyColor *= frameInfo.envIntensity;

    float misWeight   = 1.0;
    bool  envLighting = (frameInfo.envEnabled != 0);
    if(envLighting && lastBrdfPdf > 0.0)
    {
      float envPdf = 0.0;
      if(frameInfo.envMode == ENV_MODE_SKY)
      {
        float3 dirZUp = float3(rayDirection.x, rayDirection.z, rayDirection.y);
        envPdf        = samplePhysicalSkyPDF(frameInfo.skyParams, dirZUp);
      }
      else
      {
        // PDF is stored in alpha channel of the HDR texture by HdrIbl
        float4 texVal = skyTexture.SampleLevel(skyUV, 0.0);
        envPdf        = texVal.a;
      }
      misWeight = lastBrdfPdf / (lastBrdfPdf + envPdf);
    }

    pixel.radiance += float3(pixel.transmittance) * skyColor * misWeight;
  }

#if DLSS_ENABLED
  writeToGBuffers<VISUALIZE, HYBRID_ENABLED>(frameInfo, launchSizeFloat, launchID, pixel, depthBuffer, image, imageAux,
                                             dlssGuide, outImages);
#else
  writeToGBuffers<VISUALIZE, HYBRID_ENABLED>(frameInfo, launchSizeFloat, launchID, pixel, depthBuffer, image, imageAux);
#endif
}

#if HYBRID_ENABLED
// Initialize RTX state from rasterization pass results
// Reconstructs pixel state as if primary rays were traced
void initRtxStateFromRasterPass(inout PixelData pixel,
                                in uint2        launchID,
                                in float2       launchIdFloat,
                                in float2       launchSizeFloat,
                                in float3       rayOrigin,
                                in float3       rayDirection)
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
  pixel.unlitAlbedo   = rasterPassPixel.xyz;  // SH-evaluated color serves as albedo for DLSS guides

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
    float4 clipPosPicked = float4(0, 0, ndcDepthPicked, 1.0);
    float4 viewPosPicked = mul(clipPosPicked, frameInfo.projInverse);
    float  viewZPicked   = viewPosPicked.z / viewPosPicked.w;
    float  linearDist    = viewZPicked / rayDirView.z;
    float3 worldPos      = rayOrigin + linearDist * rayDirection;

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
    uint64_t              byteOffset = uint64_t(globalSplatId) * uint64_t(sizeof(GlobalSplatIndexEntry));
    GlobalSplatIndexEntry entry      = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + byteOffset);
    pixel.isoSurfSplatSetId          = int(entry.splatSetIndex);
    pixel.isoSurfSplatId             = int(entry.splatIndex);
    pixel.closestParticleFound       = true;
    pixel.closestParticleGlobalId    = int(globalSplatId);
    pixel.closestParticleId          = int(entry.splatIndex);
    pixel.closestSplatSetId          = int(entry.splatSetIndex);
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

#if RTX_HAS_MESHES
void traceBounceRayMeshes(in int           bounce,
                          in float         tMin,
                          inout float      tMax,
                          in float3        rayOrigin,
                          in float3        rayDirection,
                          in bool          meshDepthOnly,
                          inout HitPayload payload,
                          inout PixelData  pixel)
{
  if(bounce > 0)
  {
    pixel.meshHitDist     = PAYLOAD_INF_DISTANCE;
    pixel.meshHitWorldPos = {};
    pixel.meshHitWorldNrm = {};
    pixel.meshHitObjId    = PAYLOAD_INVALID_ID;
    pixel.meshHitMatId    = PAYLOAD_INVALID_ID;
    pixel.meshHitUV       = {};
    pixel.meshHitTangent  = {};
  }

  payload.dist[0] = PAYLOAD_INF_DISTANCE;
  payload.id[0]   = PAYLOAD_INVALID_ID;
  payload.id[1]   = PAYLOAD_INVALID_ID;

  // Init wireframe fields before mesh trace (rchit will write them if it hits an edge)
#if WIREFRAME
  payload.wireframeDist.set(PAYLOAD_INF_DISTANCE);
  payload.wireframeAlpha.set(0.0);
#endif

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

  // Save mesh wireframe before particle passes overwrite the payload fields
#if WIREFRAME
  pixel.meshWireframeDist  = payload.wireframeDist.get().value;
  pixel.meshWireframeAlpha = payload.wireframeAlpha.get().value;
#endif

  // store hit results
  if(pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
  {
    int         objId, matId;
    MeshHitData hit = unpackMeshHitFromPayload(payload.dist, payload.id, objId, matId);

    tMax                  = hit.hitT;
    pixel.meshHit         = true;
    pixel.meshHitObjId    = objId;
    pixel.meshHitMatId    = matId;
    pixel.meshHitWorldPos = hit.worldPos;
    pixel.meshHitWorldNrm = hit.worldNrm;
    pixel.meshHitUV       = hit.uv0;
    pixel.meshHitTangent  = float4(hit.tangentDir, hit.tangentW);
    pixel.meshHitUV1      = hit.uv1;

    if(bounce == 0)
    {
      pixel.primaryHitDist = tMax;
      pixel.primaryHitPos  = rayOrigin + tMax * rayDirection;
    }
  }
}
#endif  // RTX_HAS_MESHES

#if RTX_HAS_PARTICLES
void traceBounceRayParticles(in int           bounce,
                             in float         tMin,
                             in float         tMax,
                             in float3        rayOrigin,
                             in float3        rayDirection,
                             inout uint       seedRayPass,
                             inout HitPayload payload,
                             inout PixelData  pixel,
                             in bool          isFeedbackPixel)
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
    traceRayParticlesInsertionSort<kStochasticPass>(bounce, tMin, tMax, rayOrigin, rayDirection, seedRayPass, payload,
                                                    pixel, isFeedbackPixel);
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
      payload.id[i]          = PAYLOAD_INVALID_ID;
      payload.splatSetIdx[i] = PAYLOAD_INVALID_ID;
      payload.dist[i]        = PAYLOAD_INF_DISTANCE;
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
    payload.rayBounce = uint(bounce);
#if RTX_PARTICLE_DEPTH == PARTICLE_DEPTH_BILLBOARD && RTX_USE_AABBS
    const uint particleSbtOffset = (bounce == 0) ? 1 : 0;
#else
    const uint particleSbtOffset = 0;
#endif
    traceRayAllSplatTlas(raySplat, rayFlags, payload, particleSbtOffset);

    const int firstId = payload.id[0];

    // no particle hits found
    if(firstId == PAYLOAD_INVALID_ID)
    {
#if WIREFRAME
      // In AABB billboard mode many rays hit only the AABB edge (for wireframe) but not the
      // particle acceptance test. Those rays still carry a valid wireframe alpha from any-hit.
      // Blend it before leaving the pass; otherwise only rays with particle hits show edges.
      if(bounce == 0 && !wireframeDone && payload.wireframeAlpha.get().value > 0.0)
      {
        pixel.radiance      = lerp(pixel.radiance, float3(1.0, 0.0, 0.0), payload.wireframeAlpha.get().value);
        pixel.transmittance = double3(0.0);  // opaque
        wireframeDone       = true;
      }
#endif
      break;
    }

#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_PASS_STOCHASTIC
    // this is certainly not optimal
    const PixelData oldPixel        = pixel;
    int             lastValidHitIdx = 0;
#endif

    // evaluate the sorted hits
    SplatSetDesc* rtxDescs = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;

    [[unroll]]
    for(int i = 0; i < PARTICLES_SPP; ++i)
    {
      const int   localSplatID = payload.id[i];
      const int   descIdx      = payload.splatSetIdx[i];
      const float dist         = payload.dist[i];

      if((localSplatID != PAYLOAD_INVALID_ID) && (maxComponent(pixel.transmittance) > double(frameInfo.minTransmittance)))
      {
#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_PASS_STOCHASTIC
        // Track the farthest valid payload slot seen so far.
        lastValidHitIdx = i;
#endif

        SplatSetDesc hitSplatSetDesc = rtxDescs[descIdx];
        const int    splatId         = localSplatID;

        // rays are in world space
        // We need to transform to the splat set's model space for threedgrtProcessHit
        // Direction is intentionally NOT normalized: with D_local = M_inv_3x3 * D_world,
        // the parametric t from billboard mode directly equals world-space distance,
        // keeping hits from differently-transformed sets directly comparable.
        const float3 hitSplatSetModelRayOrigin = mul(float4(rayOrigin, 1.0), hitSplatSetDesc.transformInverse).xyz;
        const float3 hitSplatSetModelRayDirection = mul(rayDirection, hitSplatSetDesc.transformRotScaleInverse);

        float3 normalWorld;
        float  alpha            = 0.0;
        float3 weight           = float3(0.0);
        float3 particleRadiance = float3(0.0);

        bool acceptedHit;
#if RTX_PARTICLE_DEPTH == PARTICLE_DEPTH_BILLBOARD
        if(bounce == 0)
        {
          acceptedHit = threedgsProcessHit<(NEED_SURFACE_INFO != 0)>(frameInfo, hitSplatSetDesc, hitSplatSetModelRayOrigin,
                                                                     hitSplatSetModelRayDirection,
                                                                     hitSplatSetDesc.transformRotScaleInverse, splatId,
                                                                     dist, alpha, particleRadiance, normalWorld);
        }
        else
#endif
        {
          acceptedHit = threedgrtProcessHit<(NEED_SURFACE_INFO != 0)>(frameInfo, hitSplatSetDesc, hitSplatSetModelRayOrigin,
                                                                      hitSplatSetModelRayDirection,
                                                                      hitSplatSetDesc.transformRotScaleInverse, splatId,
                                                                      alpha, particleRadiance, normalWorld);
        }

        if(acceptedHit)
        {
          threedgrtIntegrate(alpha, particleRadiance, pixel.transmittance, pixel.radiance, weight);
          pixel.hitCount++;

#if TRACE_PROFILE
          if(isFeedbackPixel)
          {
#if RTX_USE_INSTANCES
            collectHitProfile(uint(int(hitSplatSetDesc.globalSplatBase) + splatId), dist, alpha, particleRadiance,
                              pixel.transmittance, float3(pixel.radiance));
#else
            collectHitProfile(uint(int(hitSplatSetDesc.globalSplatBase) + splatId - int(hitSplatSetDesc.splatBase)),
                              dist, alpha, particleRadiance, pixel.transmittance, float3(pixel.radiance));
#endif
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
            pixel.isoSurfSplatSetId = descIdx;
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
            pixel.closestParticleFound = true;
#if RTX_USE_INSTANCES
            pixel.closestParticleGlobalId = int(hitSplatSetDesc.globalSplatBase) + splatId;
#else
            // Per-model chunks: globalSplatBase already includes splatBase, so subtract to avoid double-counting
            pixel.closestParticleGlobalId = int(hitSplatSetDesc.globalSplatBase) + splatId - int(hitSplatSetDesc.splatBase);
#endif
            pixel.closestParticleId            = splatId;
            pixel.closestSplatSetId            = descIdx;
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
        // if iso density surface not found we use farthest point
        if(pixel.isoSurfDepth == 0.0)
        {
          const int   lastDescIdx = payload.splatSetIdx[lastValidHitIdx];
          const int   lastSplatId = payload.id[lastValidHitIdx];
          const float dist        = payload.dist[lastValidHitIdx];
          pixel.isoSurfSplatSetId = lastDescIdx;
          pixel.isoSurfSplatId    = lastSplatId;
          pixel.isoSurfDepth      = dist;
          pixel.isoSurfPos        = rayOrigin + dist * rayDirection;
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
    // Particle wireframe overlay (blends with accumulated particle radiance)
    // Mesh wireframe is handled separately after mesh shading in the bounce loop.
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
    payload.id[i]          = PAYLOAD_INVALID_ID;
    payload.splatSetIdx[i] = PAYLOAD_INVALID_ID;
    payload.dist[i]        = PAYLOAD_INF_DISTANCE;
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
  payload.rayBounce = uint(bounce);
#if RTX_PARTICLE_DEPTH == PARTICLE_DEPTH_BILLBOARD && RTX_USE_AABBS
  const uint particleSbtOffset = (bounce == 0) ? 1 : 0;
#else
  const uint particleSbtOffset = 0;
#endif
  traceRayAllSplatTlas(raySplat, rayFlags, payload, particleSbtOffset);

  float3 passRadianceSum = float3(0.0);
  float3 passNormalSum   = float3(0.0);
  float  passAlphaSum    = 0.0f;
  int    passSampleCount = 0;

  float  closestDist    = PAYLOAD_INF_DISTANCE;
  int    closestDescIdx = PAYLOAD_INVALID_ID;
  int    closestLocalId = PAYLOAD_INVALID_ID;
  float3 closestNormal  = float3(0.0);
  float3 closestPos     = float3(0.0);
  float3 closestUnlit   = float3(0.0);

  [[unroll]]
  for(int i = 0; i < PARTICLES_SPP; ++i)
  {
    const int   localSplatID = payload.id[i];
    const float dist         = payload.dist[i];

    if(localSplatID == PAYLOAD_INVALID_ID)
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
      closestDist    = dist;
      closestDescIdx = payload.splatSetIdx[i];
      closestLocalId = localSplatID;
      closestNormal  = sampleNormal;
      closestPos     = rayOrigin + dist * rayDirection;
      closestUnlit   = sampleRadiance;
    }
  }

  if(passSampleCount > 0)
  {
    totalRadianceSum += passRadianceSum;
    totalNormalSum += passNormalSum;
    totalAlphaSum += passAlphaSum;
    totalSampleCount += passSampleCount;
  }

  if(closestDescIdx != PAYLOAD_INVALID_ID)
  {
    SplatSetDesc* rtxDescs = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;

    pixel.isoSurfSplatSetId = closestDescIdx;
    pixel.isoSurfSplatId    = closestLocalId;
    pixel.isoSurfDepth      = closestDist;
    pixel.isoSurfPos        = closestPos;

    if((bounce == 0) && (closestDist < pixel.primaryHitDist))
    {
      pixel.primaryHitDist = closestDist;
      pixel.primaryHitPos  = closestPos;
    }

    if(bounce == 0)
    {
      pixel.closestParticleFound = true;
#if RTX_USE_INSTANCES
      pixel.closestParticleGlobalId = int(rtxDescs[closestDescIdx].globalSplatBase) + closestLocalId;
#else
      pixel.closestParticleGlobalId =
          int(rtxDescs[closestDescIdx].globalSplatBase) + closestLocalId - int(rtxDescs[closestDescIdx].splatBase);
#endif
      pixel.closestParticleId            = closestLocalId;
      pixel.closestSplatSetId            = closestDescIdx;
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
    pixel.radiance += float3(pixel.transmittance) * stochasticRadiance;
    pixel.transmittance *= double3(1.0 - avgAlpha);
    pixel.unlitAlbedo      = stochasticRadiance;
    const float3 avgNormal = totalNormalSum / float(totalSampleCount);
    pixel.integratedNormal = (length(avgNormal) > 0.0) ? normalize(avgNormal) : float3(0.0);
  }
}
#endif

// Helper to trace against all splat TLAS in the array (multi-TLAS support for large scenes)
// Traces against all TLAS structures until payload is full or all TLAS have been checked
void traceRayAllSplatTlas(in RayDesc ray, in uint rayFlags, inout HitPayload payload, uint sbtRecordOffset = 0)
{
#if RTX_USE_INSTANCES
  // Per-particle instanced mode: iterate over per-chunk descriptors (one per instance × TLAS chunk).
  // Each descriptor carries per-instance transform + per-chunk TLAS address.
  // One shared TLAS per unique asset; the per-instance global transform is applied to the ray.
  SplatSetDesc* rtxDescs = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;
  uint rtxCount = (assets.splatSetRtxCount > 0) ? assets.splatSetRtxCount : assets.splatSetCount;
  for(uint instIdx = 0; instIdx < rtxCount; ++instIdx)
  {
    SplatSetDesc desc = rtxDescs[instIdx];
    if(desc.blasAddress == 0 || desc.show == 0)
      continue;

    // Transform ray into this instance's local space.
    // The direction is intentionally NOT normalized: with D_local = M_inv_3x3 * D_world,
    // the parametric ray equation P_local = O_local + t * D_local yields t = t_world,
    // i.e. the parametric value directly equals the world-space distance regardless of
    // scale in the instance transform. This means:
    //   - RayTCurrent() returns world-space distances, making hits from differently-scaled
    //     instances directly comparable in the payload's insertion sort.
    //   - TMin/TMax remain in world-space units with no per-instance adaptation needed.
    //   - The intersection shader (.rint) and any-hit already normalize the direction
    //     internally before particle evaluation (see particleDensityHitInstance and
    //     the modelRayDir normalization in the stochastic any-hit path).
    // Supports rotation, translation, and uniform/non-uniform scale.
    RayDesc localRay;
    localRay.Origin    = mul(float4(ray.Origin, 1.0), desc.transformInverse).xyz;
    localRay.Direction = mul(ray.Direction, desc.transformRotScaleInverse);
    localRay.TMin      = ray.TMin;
    localRay.TMax      = ray.TMax;

    payload.currentSplatSetInstance = instIdx;

    RaytracingAccelerationStructure tlas = (RaytracingAccelerationStructure)desc.blasAddress;
    TraceRay(tlas, rayFlags, 0xFF, sbtRecordOffset, 0, 0, localRay, payload);
  }
#else
  // Non-instanced (large BLAS) mode: trace against multi-TLAS array.
  // Per-instance visibility is handled via TLAS instance masks (mask=0x00 for hidden),
  // set by the particle_as_build compute shader from desc.show.
  uint64_t* tlasAddresses = (uint64_t*)assets.splatSetTlasArrayAddress;

  for(uint tlasIdx = 0; tlasIdx < assets.splatSetTlasCount; ++tlasIdx)
  {
    RaytracingAccelerationStructure currentTlas = (RaytracingAccelerationStructure)tlasAddresses[tlasIdx];
    TraceRay(currentTlas, rayFlags, 0xFF, sbtRecordOffset, 0, 0, ray, payload);
  }
#endif
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
#endif  // RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
#endif  // RTX_HAS_PARTICLES

#if DLSS_ENABLED
void populateDlssGuides(in PixelData pixel, in float3 rayOrigin, inout DlssGuideData dlssGuide)
{
  // isoSurfDepth == 0.0 means no splat was hit (default init value in pure RTX mode)
  // In hybrid mode, isoSurfDepth is initialized to INFINITY if no valid splat
  // So this comparison correctly handles: splat closer, mesh closer, or no hit cases
#if RTX_HAS_PARTICLES && RTX_HAS_MESHES
  if(pixel.isoSurfDepth > 0.0 && pixel.isoSurfDepth < pixel.meshHitDist)
  {
    dlssGuide.isValid     = true;
    dlssGuide.hitPosition = pixel.isoSurfPos;

    // Compute previous world position for object motion vectors
    SplatSetDesc* splatDescs    = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;
    SplatSetDesc  splatDesc     = splatDescs[pixel.isoSurfSplatSetId];
    float3        localPos      = mul(float4(pixel.isoSurfPos, 1.0), splatDesc.transformInverse).xyz;
    dlssGuide.prevHitPosition   = mul(float4(localPos, 1.0), splatDesc.prevTransform).xyz;

    dlssGuide.specularAlbedo  = {};
#if VISUALIZE == VISUALIZE_CLAY
    dlssGuide.albedo          = float4(frameInfo.clayColor.xyz, 1.0);
#else
    dlssGuide.albedo          = float4(pixel.unlitAlbedo, 1.0);
#endif
    dlssGuide.normalRoughness = float4(pixel.integratedNormal, 1.0);
  }
  else if(pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
  {
    dlssGuide.isValid     = true;
    dlssGuide.hitPosition = pixel.meshHitWorldPos;

    // Compute previous world position for object motion vectors
    MeshDesc meshDescForMotion    = assets.meshes[pixel.meshHitObjId];
    float3   localPos             = mul(float4(pixel.meshHitWorldPos, 1.0), meshDescForMotion.transformInverse).xyz;
    dlssGuide.prevHitPosition     = mul(float4(localPos, 1.0), meshDescForMotion.prevTransform).xyz;

    Material meshMaterial = assets.meshes[pixel.meshHitObjId].materialAddress[pixel.meshHitMatId];
    float3   dlssNrm      = pixel.meshHitWorldNrm;
    applyMaterialTextures(meshMaterial, meshTextures, pixel.meshHitUV, pixel.meshHitUV1, dlssNrm, pixel.meshHitTangent);

#if VISUALIZE == VISUALIZE_CLAY
    dlssGuide.albedo          = float4(frameInfo.clayColor.xyz, 1.0);
    dlssGuide.normalRoughness = float4(dlssNrm, 1.0);
    dlssGuide.specularAlbedo  = {};
#else
    dlssGuide.albedo          = float4(meshMaterial.baseColor.xyz, 1.0);
    dlssGuide.normalRoughness = float4(dlssNrm, meshMaterial.roughness);

    float  f0_from_ior          = pow((meshMaterial.ior - 1.0) / (meshMaterial.ior + 1.0), 2.0);
    float3 specularColorClamped = min(f0_from_ior * meshMaterial.specularColorFactor, float3(1.0));
    float  alpha                = meshMaterial.roughness * meshMaterial.roughness;
    float3 viewDir              = normalize(pixel.meshHitWorldPos - rayOrigin);
    float  NoV                  = dot(dlssNrm, -viewDir);
    dlssGuide.specularAlbedo    = EnvBRDFApprox2(specularColorClamped, alpha, NoV);
#endif
  }
#elif RTX_HAS_PARTICLES
  if(pixel.isoSurfDepth > 0.0 && pixel.isoSurfDepth < pixel.meshHitDist)
  {
    dlssGuide.isValid     = true;
    dlssGuide.hitPosition = pixel.isoSurfPos;

    // Compute previous world position for object motion vectors
    SplatSetDesc* splatDescs    = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;
    SplatSetDesc  splatDesc     = splatDescs[pixel.isoSurfSplatSetId];
    float3        localPos      = mul(float4(pixel.isoSurfPos, 1.0), splatDesc.transformInverse).xyz;
    dlssGuide.prevHitPosition   = mul(float4(localPos, 1.0), splatDesc.prevTransform).xyz;

    dlssGuide.specularAlbedo  = {};
#if VISUALIZE == VISUALIZE_CLAY
    dlssGuide.albedo          = float4(frameInfo.clayColor.xyz, 1.0);
#else
    dlssGuide.albedo          = float4(pixel.unlitAlbedo, 1.0);
#endif
    dlssGuide.normalRoughness = float4(pixel.integratedNormal, 1.0);
  }
#elif RTX_HAS_MESHES
  if(pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
  {
    dlssGuide.isValid     = true;
    dlssGuide.hitPosition = pixel.meshHitWorldPos;

    // Compute previous world position for object motion vectors
    MeshDesc meshDescForMotion    = assets.meshes[pixel.meshHitObjId];
    float3   localPos             = mul(float4(pixel.meshHitWorldPos, 1.0), meshDescForMotion.transformInverse).xyz;
    dlssGuide.prevHitPosition     = mul(float4(localPos, 1.0), meshDescForMotion.prevTransform).xyz;

    Material meshMaterial = assets.meshes[pixel.meshHitObjId].materialAddress[pixel.meshHitMatId];
    float3   dlssNrm      = pixel.meshHitWorldNrm;
    applyMaterialTextures(meshMaterial, meshTextures, pixel.meshHitUV, pixel.meshHitUV1, dlssNrm, pixel.meshHitTangent);

#if VISUALIZE == VISUALIZE_CLAY
    dlssGuide.albedo          = float4(frameInfo.clayColor.xyz, 1.0);
    dlssGuide.normalRoughness = float4(dlssNrm, 1.0);
    dlssGuide.specularAlbedo  = {};
#else
    dlssGuide.albedo          = float4(meshMaterial.baseColor.xyz, 1.0);
    dlssGuide.normalRoughness = float4(dlssNrm, meshMaterial.roughness);

    float  f0_from_ior          = pow((meshMaterial.ior - 1.0) / (meshMaterial.ior + 1.0), 2.0);
    float3 specularColorClamped = min(f0_from_ior * meshMaterial.specularColorFactor, float3(1.0));
    float  alpha                = meshMaterial.roughness * meshMaterial.roughness;
    float3 viewDir              = normalize(pixel.meshHitWorldPos - rayOrigin);
    float  NoV                  = dot(dlssNrm, -viewDir);
    dlssGuide.specularAlbedo    = EnvBRDFApprox2(specularColorClamped, alpha, NoV);
#endif
  }
#endif
}
#endif  // DLSS_ENABLED

void evaluateLightingAndShadingForBounce(in int           bounce,
                                         inout PixelData  pixel,
                                         inout int        done,
                                         inout float3     rayOrigin,
                                         inout float3     rayDirection,
                                         inout uint       seedRayPass,
                                         inout HitPayload payload,
                                         in float3        radianceBeforeParticles,
                                         inout float      lastBrdfPdf)
{
#if LIGHTING_MODE != LIGHTING_DISABLED

#if RTX_HAS_MESHES && RTX_HAS_PARTICLES
  const bool splatIsClosest = (pixel.isoSurfDepth > 0.0) && (pixel.isoSurfDepth < pixel.meshHitDist);
#elif RTX_HAS_PARTICLES
  const bool splatIsClosest = true;
#else
  const bool splatIsClosest = false;
#endif

#if RTX_HAS_PARTICLES
  if(splatIsClosest && pixel.isoSurfSplatSetId != PAYLOAD_INVALID_ID)
  {
    pixel.radiance -= radianceBeforeParticles;
    evaluateLightingAndShadingParticles(bounce, pixel, done, rayOrigin, rayDirection, seedRayPass, payload, lastBrdfPdf);
    pixel.radiance += radianceBeforeParticles;
  }
#endif

#if RTX_HAS_MESHES
  if((pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
     && (maxComponent(pixel.transmittance) > double(frameInfo.minSplatSetCompositeTransmittance)) && (done != 0 || !splatIsClosest))
  {
    evaluateLightingAndShadingMeshes(bounce, pixel, done, rayOrigin, rayDirection, seedRayPass, payload, lastBrdfPdf);
  }
#endif

#else  // LIGHTING_DISABLED

#if RTX_HAS_MESHES
  // Lighting disabled - process mesh shading if needed
  if((pixel.meshHitDist < PAYLOAD_INF_DISTANCE)
     && (maxComponent(pixel.transmittance) > double(frameInfo.minSplatSetCompositeTransmittance)))
  {
    Material meshHitMaterial = assets.meshes[pixel.meshHitObjId].materialAddress[pixel.meshHitMatId];

    float3 unlitNrm = pixel.meshHitWorldNrm;
    applyMaterialTextures(meshHitMaterial, meshTextures, pixel.meshHitUV, pixel.meshHitUV1, unlitNrm, pixel.meshHitTangent);

    const float3 meshColor = meshHitMaterial.emissive * meshHitMaterial.emissiveStrength + meshHitMaterial.baseColor;
    pixel.radiance += float3(pixel.transmittance) * meshColor;
    pixel.transmittance = double3(0.0);
  }
#endif

#endif  // LIGHTING_MODE
}

#if LIGHTING_MODE != LIGHTING_DISABLED && RTX_HAS_PARTICLES
// Evaluate lighting and shading for particles (splats)
void evaluateLightingAndShadingParticles(in int           bounce,
                                         inout PixelData  pixel,
                                         inout int        done,
                                         inout float3     rayOrigin,
                                         inout float3     rayDirection,
                                         inout uint       seedRayPass,
                                         inout HitPayload payload,
                                         inout float      lastBrdfPdf)
{
  SplatSetDesc* descs = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;
  SplatSetDesc splatSetDesc = descs[pixel.isoSurfSplatSetId];

  // Splat radiance is trained against sRGB display — convert to linear for PBR shading
  // and correct light source intensity when used by bounce rays.
  const float3 baseRadiance = srgbToLinear(pixel.radiance);

  // Emissive contribution in linear space.
  // On bounce 0 (direct display), emissiveStrength is not applied — the trained
  // radiance is the ground truth. On secondary bounces (light source for other
  // surfaces), emissiveStrength scales the emitted energy.
  float  strength             = (bounce == 0) ? 1.0 : splatSetDesc.material.emissiveStrength;
  float3 emissiveContribution = baseRadiance * splatSetDesc.material.emissive * strength;

  // Save volumetric transmittance from splat tracing before overriding for shading
  double3 savedTransmittance = pixel.transmittance;

  // AO for splat emissive: attenuate emissive contribution by mesh proximity.
  // Applies to both pure emissive and shaded splat sets — the PBR path handles
  // light occlusion via NEE shadow rays, but the emissive term has no other occlusion.
#if PARTICLE_AO_ENABLED
  {
    float aoFactor = computeSplatEmissiveAO(pixel.isoSurfPos, pixel.integratedNormal,
                                            frameInfo.particleEmissiveAoRadius, seedRayPass, payload);
    emissiveContribution *= aoFactor;
  }
#endif

  // GS shadow mask: shadow-only (gs-shadow) lights darken the emissive term at the
  // surfel. Same rationale as the AO block above: the emissive term has no other
  // occlusion, while shaded contributions get their shadows from NEE shadow rays
  // (gs-shadow lights are skipped by NEE, so nothing is counted twice).
#if GS_SHADOW_MASK
  emissiveContribution *= computeGsShadowMask(pixel.isoSurfPos, pixel.integratedNormal, seedRayPass, payload);
#endif

  // Early exit for pure emissive materials - no lighting interaction
  // No clay mode in pure emissive
  if(splatSetDesc.material.needShading != 1)
  {
    pixel.radiance      = emissiveContribution;
    pixel.transmittance = savedTransmittance;
    return;
  }

  // Set up material for lighting computation
  Material mat = splatSetDesc.material;

  // Clay mode: override material with uniform matte clay
#if VISUALIZE == VISUALIZE_CLAY
  mat.baseColor       = frameInfo.clayColor;
  mat.metallic        = 0.0;
  mat.roughness       = 1.0;
  mat.ior             = 1.0;
  mat.specularFactor  = 0;
  mat.clearcoatFactor = 0.0;
  pixel.radiance      = float3(0.0);
#else
  mat.baseColor = baseRadiance * splatSetDesc.material.baseColor;
  // Reset radiance to only emissive, lighting will add on top
  pixel.radiance = emissiveContribution;
#endif

  float3 shadowOrigin = rayOrigin + pixel.isoSurfDepth * rayDirection;

  // Splat color (baseRadiance) already encodes Gaussian blending, so lighting uses
  // unit transmittance — not the volumetric compositing transmittance from splat tracing.
  pixel.transmittance = double3(1.0);

  // Path tracing with NEE + BSDF MIS (GGX)
  PbrMaterial pbr = toPbrMaterial(mat, pixel.integratedNormal, pixel.integratedNormal);

  // 1. NEE: sample one light (punctual or environment)
  bool      envLighting = (frameInfo.envMode != ENV_MODE_NONE && frameInfo.envEnabled != 0);
  NEESample nee = sampleOneLightNEE(pixel.isoSurfPos, assets.lightCount, assets.lights, envLighting, frameInfo.envMode,
                                    frameInfo.skyParams, skyTexture, envAccelBuffer, frameInfo.envRotation,
                                    frameInfo.envIntensity, seedRayPass);
  float3    neeContribution = float3(0.0);
  bool      neeValid        = (nee.pdf != 0.0);
  bool      neeFrontFace    = dot(nee.direction, pixel.integratedNormal) > 0.0;
  if(!neeFrontFace && mat.transmission == 0.0)
    neeValid = false;

  if(neeValid)
  {
    BsdfEvaluateData evalData;
    evalData.k1 = -rayDirection;
    evalData.k2 = nee.direction;
    evalData.xi = float3(rand(seedRayPass), rand(seedRayPass), rand(seedRayPass));
    bsdfEvaluate(evalData, pbr);

    if(evalData.pdf > 0.0)
    {
      float misWeight = (nee.pdf == DIRAC) ? 1.0 : nee.pdf / (nee.pdf + evalData.pdf);
      neeContribution = nee.radianceOverPdf * misWeight * (evalData.bsdf_diffuse + evalData.bsdf_glossy);
    }
  }

  // 2. Shadow ray for NEE
  if(neeValid && luminance(neeContribution) > 0.001)
  {
#if SHADOWS_MODE != SHADOWS_DISABLED
    ShadowInfo shadow = traceShadowRayForLight(frameInfo, assets, shadowOrigin, nee.direction, nee.distance, payload);
    neeContribution *= shadow.transmittance;
#endif
  }
  pixel.radiance += float3(pixel.transmittance) * neeContribution;

  // 3. BSDF sample for bounce direction (gated by per-material maxBounces)
  if(bounce < mat.maxBounces)
  {
    BsdfSampleData sampleData;
    sampleData.k1 = -rayDirection;
    sampleData.xi = float3(rand(seedRayPass), rand(seedRayPass), rand(seedRayPass));
    bsdfSample(sampleData, pbr);

    if(sampleData.event_type != BSDF_EVENT_ABSORB && luminance(sampleData.bsdf_over_pdf) > 0.001)
    {
      done                = 0;
      rayOrigin           = pixel.isoSurfPos;
      rayDirection        = sampleData.k2;
      pixel.transmittance = double3(sampleData.bsdf_over_pdf);
      lastBrdfPdf         = sampleData.pdf;
    }
  }

  // For opaque splats (no bounce), restore the volumetric transmittance so
  // the mesh behind can still composite through semi-transparent splats.
  if(done != 0)
    pixel.transmittance = savedTransmittance;
}
#endif  // LIGHTING_MODE && RTX_HAS_PARTICLES

#if LIGHTING_MODE != LIGHTING_DISABLED && RTX_HAS_MESHES
// Evaluate lighting and shading for meshes
void evaluateLightingAndShadingMeshes(in int           bounce,
                                      inout PixelData  pixel,
                                      inout int        done,
                                      inout float3     rayOrigin,
                                      inout float3     rayDirection,
                                      inout uint       seedRayPass,
                                      inout HitPayload payload,
                                      inout float      lastBrdfPdf)
{
  // Regular mesh - retrieve the material for this hit
  Material meshMaterial = assets.meshes[pixel.meshHitObjId].materialAddress[pixel.meshHitMatId];

  // Apply material textures (modifies material properties and normal in-place)
  float3 meshShadingNrm = pixel.meshHitWorldNrm;
  applyMaterialTextures(meshMaterial, meshTextures, pixel.meshHitUV, pixel.meshHitUV1, meshShadingNrm, pixel.meshHitTangent);
  pixel.meshHitWorldNrm = meshShadingNrm;

  // Clay mode: override material with uniform matte clay
#if VISUALIZE == VISUALIZE_CLAY
  meshMaterial.baseColor       = frameInfo.clayColor;
  meshMaterial.metallic        = 0.0;
  meshMaterial.roughness       = 1.0;
  meshMaterial.transmission    = 0.0;
  meshMaterial.ior             = 1.0;
  meshMaterial.specularFactor  = 0;
  meshMaterial.clearcoatFactor = 0.0;

  //meshMaterial.emissive         = float3(0);
  //meshMaterial.emissiveStrength = 0.0;
#endif

  // Add emissive contribution (weighted by path throughput for secondary bounces)
  pixel.radiance += float3(pixel.transmittance) * meshMaterial.emissive * meshMaterial.emissiveStrength;

  // Early exit for pure emissive materials (light proxies) - no lighting interaction
  if(meshMaterial.needShading != 1)
  {
    pixel.transmittance = double3(0.0);
    return;
  }

  float3 shadowOrigin = pixel.meshHitWorldPos;

  // Path tracing with NEE + BSDF MIS (GGX)
  PbrMaterial pbr = toPbrMaterial(meshMaterial, pixel.meshHitWorldNrm, pixel.meshHitWorldNrm);

  // 1. NEE: sample one light (punctual or environment)
  bool      envLighting     = (frameInfo.envMode != ENV_MODE_NONE && frameInfo.envEnabled != 0);
  NEESample nee             = sampleOneLightNEE(pixel.meshHitWorldPos, assets.lightCount, assets.lights, envLighting,
                                                frameInfo.envMode, frameInfo.skyParams, skyTexture, envAccelBuffer,
                                                frameInfo.envRotation, frameInfo.envIntensity, seedRayPass);
  float3    neeContribution = float3(0.0);
  bool      neeValid        = (nee.pdf != 0.0);
  bool      neeFrontFace    = dot(nee.direction, pixel.meshHitWorldNrm) > 0.0;
  if(!neeFrontFace && meshMaterial.transmission == 0.0)
    neeValid = false;

  if(neeValid)
  {
    BsdfEvaluateData evalData;
    evalData.k1 = -rayDirection;
    evalData.k2 = nee.direction;
    evalData.xi = float3(rand(seedRayPass), rand(seedRayPass), rand(seedRayPass));
    bsdfEvaluate(evalData, pbr);

    if(evalData.pdf > 0.0)
    {
      float misWeight = (nee.pdf == DIRAC) ? 1.0 : nee.pdf / (nee.pdf + evalData.pdf);
      neeContribution = nee.radianceOverPdf * misWeight * (evalData.bsdf_diffuse + evalData.bsdf_glossy);
    }
  }

  // 2. Shadow ray for NEE
  if(neeValid && luminance(neeContribution) > 0.001)
  {
#if SHADOWS_MODE != SHADOWS_DISABLED
    ShadowInfo shadow = traceShadowRayForLight(frameInfo, assets, shadowOrigin, nee.direction, nee.distance, payload);
    neeContribution *= shadow.transmittance;
#endif
  }
  pixel.radiance += float3(pixel.transmittance) * neeContribution;

  // 3. BSDF sample for bounce direction (gated by per-material maxBounces)
  if(bounce < meshMaterial.maxBounces)
  {
    BsdfSampleData sampleData;
    sampleData.k1 = -rayDirection;
    sampleData.xi = float3(rand(seedRayPass), rand(seedRayPass), rand(seedRayPass));
    bsdfSample(sampleData, pbr);

    if(sampleData.event_type != BSDF_EVENT_ABSORB && luminance(sampleData.bsdf_over_pdf) > 0.001)
    {
      done         = 0;
      rayOrigin    = pixel.meshHitWorldPos;
      rayDirection = sampleData.k2;
      pixel.transmittance *= double3(sampleData.bsdf_over_pdf);
      lastBrdfPdf = sampleData.pdf;
      return;
    }
  }

  // No bounce continuation — opaque surface blocks the environment
  pixel.transmittance = double3(0.0);
}
#endif  // LIGHTING_MODE && RTX_HAS_MESHES

// Compute ambient occlusion for emissive splat sets against mesh geometry only.
// Returns 1.0 (unoccluded) or <1.0 (occluded) for a single stochastic hemisphere sample.
// Converges via temporal accumulation over multiple frames.
#if PARTICLE_AO_ENABLED
float computeSplatEmissiveAO(in float3 surfacePos, in float3 surfaceNormal, in float aoRadius, inout uint seed, inout HitPayload payload)
{
  float3 N = normalize(surfaceNormal);

  // Frisvad orthonormal basis
  float3 T, B;
  if(N.z < -0.9999999f)
  {
    T = float3(0.0f, -1.0f, 0.0f);
    B = float3(-1.0f, 0.0f, 0.0f);
  }
  else
  {
    float a = 1.0f / (1.0f + N.z);
    float b = -N.x * N.y * a;
    T       = float3(1.0f - N.x * N.x * a, b, -N.x);
    B       = float3(b, 1.0f - N.y * N.y * a, -N.y);
  }

  // Cosine-weighted hemisphere sample (Malley's method)
  float  r1    = rand(seed);
  float  r2    = rand(seed);
  float  sq    = sqrt(r1);
  float  phi   = M_TWO_PI * r2;
  float3 aoDir = normalize(T * cos(phi) * sq + B * sin(phi) * sq + N * sqrt(1.0 - r1));

  // Trace AO ray against mesh TLAS only (excludes light proxies via mask 0xFE)
  payload.dist[0] = PAYLOAD_INF_DISTANCE;
  payload.id[0]   = PAYLOAD_INVALID_ID;

  RayDesc aoRay;
  aoRay.Origin    = surfacePos + N * frameInfo.rtxSecondaryRayOffset;
  aoRay.Direction = aoDir;
  aoRay.TMin      = 0.0;
  aoRay.TMax      = aoRadius;

  const uint aoRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

  TraceRay((RaytracingAccelerationStructure)assets.meshTlasAddress, aoRayFlags, 0xFE, 0, 0, 0, aoRay, payload);

  if(payload.dist[0] < aoRadius && payload.id[0] != PAYLOAD_INVALID_ID)
  {
    // here we allow negative values to overamplify the effect over temporal accumulation
    // so instead of emitting, a value < 0 will absorb energy
    return 1.0 - frameInfo.particleEmissiveAoStrength;
  }
  return 1.0;
}
#endif  // PARTICLE_AO_ENABLED

#if GS_SHADOW_MASK
// GS shadow mask: weighted-average visibility of the surfel toward all shadow-only
// (gs-shadow) lights, remapped through the gsShadowMaskMin floor. Occluders are
// meshes by default; particles opt in via GS_SHADOW_MASK_PARTICLES (the receiving
// splat set then occludes itself — use with care). Baked radiance approximates the
// sum of all incident lighting, so per-light visibilities are averaged weighted by
// light importance (not multiplied): overlapping shadows do not double-darken.
// Returns 1.0 when no shadow-only light reaches the surfel.
float3 computeGsShadowMask(in float3 surfPos, in float3 surfNormal, inout uint seed, inout HitPayload payload)
{
  float3 maskSum   = float3(0.0);
  float  weightSum = 0.0;

  for(uint i = 0; i < assets.lightCount; ++i)
  {
    LightSource light = assets.lights[i];
    // Include shadow-only lights (mask-only) AND lit lights flagged castOnGs (illuminate + cast on GS).
    if(light.enabled == 0 || (light.shadowOnly == 0 && light.castOnGs == 0))
      continue;

    // Direction/distance to the light; with SHADOWS_SOFT and radius > 0 this
    // jitters on the light disk, converging over temporal accumulation.
    float3 lightDir;
    float  lightDist;
    if(!computeLightToSurfaceVector(light, surfPos, seed, lightDir, lightDist))
      continue;  // out of range: the light does not reach this surfel, no vote

    const float3 shadowOrigin = surfPos + surfNormal * frameInfo.rtxSecondaryRayOffset;

    float3 visibility = float3(1.0);
#if RTX_HAS_MESHES
    if(assets.meshCount > 0)
      visibility *= traceShadowRayMesh(assets, shadowOrigin, lightDir, lightDist, payload).transmittance;
#endif
#if GS_SHADOW_MASK_PARTICLES && RTX_HAS_PARTICLES
    if(maxComponent(visibility) > 0.001 && assets.totalGlobalSplatCount > 0)
      visibility *= traceShadowRayParticle(frameInfo, assets, shadowOrigin, lightDir, lightDist, payload).transmittance;
#endif

    const float weight = max(light.intensity, 0.0) * luminance(light.color);
    maskSum += weight * visibility;
    weightSum += weight;
  }

  if(weightSum <= 0.0)
    return float3(1.0);

  const float3 mask = maskSum / weightSum;
  return float3(frameInfo.gsShadowMaskMin) + (1.0 - frameInfo.gsShadowMaskMin) * mask;
}
#endif  // GS_SHADOW_MASK

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
#if RTX_HAS_MESHES
  if(assets.meshCount > 0)
  {
    meshShadow = traceShadowRayMesh(assets, shadowOrigin, lightDir, lightDist, payload);
  }
#endif

  // Trace through particles (only if mesh didn't fully occlude)
  // Skip for stochastic anyhit: alpha=1 in anyhit makes shadow transmittance always 0
  ShadowInfo particleShadow;
  particleShadow.transmittance = float3(1.0);
#if RTX_HAS_PARTICLES
  if(maxComponent(meshShadow.transmittance) > 0.001 && assets.totalGlobalSplatCount > 0)
  {
    particleShadow = traceShadowRayParticle(frameInfo, assets, shadowOrigin, lightDir, lightDist, payload);
  }
#endif
  result.transmittance = meshShadow.transmittance * particleShadow.transmittance;

  return result;
}

#if RTX_HAS_MESHES
// Test if a shadow ray hits a mesh and compute shadow transmission (soft shadows with colored transmission)
ShadowInfo traceShadowRayMesh(in SceneAssets assets, in float3 shadowOrigin, in float3 shadowDir, in float lightDist, inout HitPayload payload)
{
  ShadowInfo result;
  result.transmittance = float3(1.0);

  payload.dist[0] = PAYLOAD_INF_DISTANCE;
  payload.id[0]   = PAYLOAD_INVALID_ID;
  payload.id[1]   = PAYLOAD_INVALID_ID;

  RayDesc shadowRay;
  shadowRay.Origin    = shadowOrigin;
  shadowRay.Direction = shadowDir;
  shadowRay.TMin      = frameInfo.rtxSecondaryRayOffset;
  shadowRay.TMax      = lightDist - frameInfo.rtxSecondaryRayOffset;

  const uint shadowRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

  TraceRay((RaytracingAccelerationStructure)assets.meshTlasAddress, shadowRayFlags, 0xFE, 0, 0, 0, shadowRay, payload);

  if(payload.dist[0] < lightDist && payload.id[0] != PAYLOAD_INVALID_ID)
  {
    int meshId = payload.id[0];
    int matId  = payload.id[1];

    if(matId != PAYLOAD_INVALID_ID)
    {
      Material material    = assets.meshes[meshId].materialAddress[matId];
      result.transmittance = material.transmission > 0.0 ? material.baseColor : float3(0.0);
    }
    else
    {
      result.transmittance = float3(0.0);
    }
  }

  return result;
}
#endif  // RTX_HAS_MESHES

#if RTX_HAS_PARTICLES
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
    payload.id[j]          = PAYLOAD_INVALID_ID;
    payload.splatSetIdx[j] = PAYLOAD_INVALID_ID;
    payload.dist[j]        = PAYLOAD_INF_DISTANCE;
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
  payload.rayBounce = 1;  // non-zero: shadow rays always use 3DGRT kernel
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
    const int   localSplatID = payload.id[j];
    const float dist         = payload.dist[j];

    if(localSplatID == PAYLOAD_INVALID_ID || dist >= lightDist)
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

  SplatSetDesc* shadowRtxDescs = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;

  [[unroll]]
  for(int j = 0; j < PARTICLES_SPP; ++j)
  {
    const int   localSplatID = payload.id[j];
    const int   descIdx      = payload.splatSetIdx[j];
    const float dist         = payload.dist[j];

    if(localSplatID == PAYLOAD_INVALID_ID || dist >= lightDist)
      continue;

    SplatSetDesc shadowHitDesc = shadowRtxDescs[descIdx];
    const int    splatId       = localSplatID;

    const float3 splatModelShadowOrigin = mul(float4(offsetShadowOrigin, 1.0), shadowHitDesc.transformInverse).xyz;
    const float3 splatModelShadowDir    = normalize(mul(shadowDir, shadowHitDesc.transformRotScaleInverse));

    float3 weight, particleRad;
    float  alpha;

    bool hit = threedgrtProcessHit<false>(frameInfo, shadowHitDesc, splatModelShadowOrigin, splatModelShadowDir,
                                          shadowHitDesc.transformRotScaleInverse, splatId, alpha, particleRad, dummyNormal);
    if(hit)
    {
      threedgrtIntegrate(alpha, particleRad, shadowTransmittance, shadowRadiance, weight);
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
#endif  // RTX_HAS_PARTICLES

void writeToGBuffers<let kVisualizeMode : uint, let kHybridEnabled : uint>(in FrameInfo frameInfo,
                                                                           in float2    resolution,
                                                                           in uint2     pixelId,
                                                                           in PixelData pixel,
                                                                           in RWTexture2D<float> depthBuffer,
                                                                           in RWTexture2D<float4> image,
                                                                           in RWTexture2D<float4> imageAux
#if DLSS_ENABLED
                                                                           , in DlssGuideData       dlssGuide,
                                                                           in RWTexture2D<float4> outImages[]
#endif
                                                                           )
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
    case VISUALIZE_CLAY:
#if DLSS_ENABLED
    case VISUALIZE_DLSS_INPUT:
#endif
    case VISUALIZE_FINAL:
      // the final pixel color is the integrated radiance
#if DLSS_ENABLED
      pixelColor = max(float3(frameInfo.dlssMinRadianceThreshold), pixel.radiance);
#else
      pixelColor = pixel.radiance;
#endif

      // Firefly clamp: suppress stochastic outliers that cause temporal streaks
      if(frameInfo.rtxFireflyClampThreshold > 0.0)
      {
        float lum = dot(pixelColor, float3(1.0 / 3.0));
        if(lum > frameInfo.rtxFireflyClampThreshold)
          pixelColor *= frameInfo.rtxFireflyClampThreshold / lum;
      }

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
#if !DLSS_ENABLED
        return;
#endif
      }
      else
      {
#if !DLSS_ENABLED
        // Temporal accumulation (DLSS handles its own when enabled)
        if(frameInfo.frameSampleId > 0)
        {
          const float  a        = 1.0F / float(frameInfo.frameSampleId + 1);
          const float4 oldColor = image[pixelId];
          pixelColor            = lerp(oldColor, float4(pixelColor, 1.0), a).xyz;
        }
#endif
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
#if DLSS_ENABLED
    case VISUALIZE_DEPTH_FOR_DLSS:
      if(dlssGuide.isValid)
      {
        const float4 posView     = mul(float4(dlssGuide.hitPosition, 1.0), frameInfo.viewMatrix);
        const float  linearDepth = -posView.z;
        pixelColor = computeVisualizationColor(linearDepth, frameInfo.visuMinMax, frameInfo.visuShift, true);
      }
      break;
#endif
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
#if DLSS_ENABLED
    case VISUALIZE_NORMAL_FOR_DLSS:
      if(dlssGuide.isValid)
      {
        const float3 normal = dlssGuide.normalRoughness.xyz;
        if(length(normal) != 0.0)
          pixelColor = float3(normal.x * 0.5f + 0.5f, normal.y * 0.5f + 0.5f, normal.z * 0.5f + 0.5f);
      }
      break;
#endif
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

#if DLSS_ENABLED
  // Write rendered image to DLSS input (eResultImage in DLSS G-buffers)
  outImages[int(DlssImages::eDlssInputImage)][pixelId] = float4(pixelColor, 1.0);

  // Write guide images if we have valid first hit data
  const float4x4 viewProjMatrix = mul(frameInfo.viewMatrix, frameInfo.projectionMatrix);
  if(dlssGuide.isValid)
  {
    const float4 posClip    = mul(float4(dlssGuide.hitPosition, 1.0), viewProjMatrix);
    const float4 depthValue = float4(posClip.z / posClip.w, 0.0, 0.0, 0.0);

    const float2 motion = calculateMotionVector(dlssGuide.hitPosition, dlssGuide.prevHitPosition, frameInfo.prevViewProjMatrix, viewProjMatrix, resolution);

    outImages[int(DlssImages::eDlssAlbedo)][pixelId]          = dlssGuide.albedo;
    outImages[int(DlssImages::eDlssSpecAlbedo)][pixelId]      = float4(dlssGuide.specularAlbedo, 1.0);
    outImages[int(DlssImages::eDlssNormalRoughness)][pixelId] = dlssGuide.normalRoughness;
    outImages[int(DlssImages::eDlssMotion)][pixelId]          = float4(motion, 0.0, 0.0);
    outImages[int(DlssImages::eDlssDepth)][pixelId]           = depthValue;
  }
  else
  {
    float2 ndc_xy     = (float2(pixelId) + 0.5) / resolution * 2.0 - 1.0;
    float4 farViewPos = mul(float4(ndc_xy, 1.0, 1.0), frameInfo.projInverse);
    farViewPos /= farViewPos.w;
    float3 farWorldPos = mul(farViewPos, frameInfo.viewInverse).xyz;

    const float2 missMotion = calculateMotionVector(farWorldPos, farWorldPos, frameInfo.prevViewProjMatrix, viewProjMatrix, resolution);

    outImages[int(DlssImages::eDlssAlbedo)][pixelId]          = float4(0.0, 0.0, 0.0, 0.0);
    outImages[int(DlssImages::eDlssSpecAlbedo)][pixelId]      = float4(0.0, 0.0, 0.0, 0.0);
    outImages[int(DlssImages::eDlssNormalRoughness)][pixelId] = float4(0.0, 0.0, 0.0, 0.0);
    outImages[int(DlssImages::eDlssMotion)][pixelId]          = float4(missMotion, 0.0, 0.0);
    outImages[int(DlssImages::eDlssDepth)][pixelId]           = float4(1.0, 0.0, 0.0, 0.0);
  }
#else
  if(kHybridEnabled == 1 && frameInfo.frameSampleId > 0)
    imageAux[pixelId] = float4(pixelColor, 1.0);
  else
    image[pixelId] = float4(pixelColor, 1.0);
#endif
}

float3 computeVisualizationColor(in float value, in float2 minMax, in float shift, bool flip)
{
  if(flip)
    return turboColormap(clamp(shift + 1.0 - (value - minMax[0]) / (minMax[1] - minMax[0]), 0.0, 1.0));
  else
    return turboColormap(clamp(shift + (value - minMax[0]) / (minMax[1] - minMax[0]), 0.0, 1.0));
}
