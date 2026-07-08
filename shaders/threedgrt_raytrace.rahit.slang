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
#include "threedgs_particle_storage.h.slang"
#include "threedgrt_custom_attributes.h.slang"
#include "threedgrt.h.slang"
#include "threedgrt_payload.h.slang"
#include "wireframe_extended.h.slang"

[[vk::binding(BINDING_ASSETS, 0)]]
ConstantBuffer<SceneAssets> assets;
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]
ConstantBuffer<FrameInfo> frameInfo;

SplatSetDesc* getRtxSplatSetDescriptors()
{
  // RTX uses a per-TLAS-instance descriptor array when BLAS is split.
  // Fallback to per-instance descriptors for non-split paths.
  return (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;
}

[shader("anyhit")]
void main(inout HitPayload payload
#if WIREFRAME
          ,
          in CustomIntersectionAttributes attribs
#endif
)
{
#if RTX_USE_INSTANCES
  // Per-particle instanced mode: InstanceIndex() is the chunk-local index (0..chunkSize-1).
  // splatBase carries the chunk's base offset within the asset.
  SplatSetDesc* instanceDescs = (assets.splatSetDescriptorsRtx != nullptr) ? assets.splatSetDescriptorsRtx : assets.splatSetDescriptors;
  const SplatSetDesc splatSetDesc = instanceDescs[payload.getCurrentSplatSetInstance()];  // here we use splatSetDesc to dereference an instanceDesc
  const int descriptorIdx = int(payload.getCurrentSplatSetInstance());
  const int localSplatId  = int(splatSetDesc.splatBase) + int(InstanceIndex());
#elif RTX_USE_AABBS || RTX_USE_SPHERES
  // AABB / Sphere mode: PrimitiveIndex() is local to the chunk's BLAS.
  // splatBase carries the chunk offset within the full set (0 when not split).
  const uint         instanceIdx   = InstanceID();
  const SplatSetDesc splatSetDesc  = getRtxSplatSetDescriptors()[instanceIdx];
  const int          descriptorIdx = int(instanceIdx);
  const int          localSplatId  = int(splatSetDesc.splatBase) + PrimitiveIndex();
#else
  // Non-instanced Icosahedron mode: PrimitiveIndex()/20 is local to the chunk's BLAS.
  // splatBase carries the chunk offset within the full set (0 when not split).
  const int          instanceIdx   = InstanceID();
  const SplatSetDesc splatSetDesc  = getRtxSplatSetDescriptors()[instanceIdx];
  const int          descriptorIdx = int(instanceIdx);
  const int          localSplatId  = int(splatSetDesc.splatBase) + PrimitiveIndex() / 20;
#endif

#if RTX_USE_INSTANCES
  const float3 modelRayOrigin = WorldRayOrigin();
  const float3 modelRayDirRaw = WorldRayDirection();
  const float3 modelRayDir    = normalize(modelRayDirRaw);
#else
  const float3 modelRayOrigin = mul(float4(WorldRayOrigin(), 1.0), splatSetDesc.transformInverse).xyz;
  const float3 modelRayDirRaw = mul(WorldRayDirection(), splatSetDesc.transformRotScaleInverse);
  const float3 modelRayDir    = normalize(modelRayDirRaw);
#endif

#if(RTX_PARTICLE_DEPTH == PARTICLE_DEPTH_BILLBOARD) && !RTX_USE_AABBS
  // No custom intersection shader (icosahedrons, spheres): compute billboard correction in any-hit
  const float rayTCurrentCorrected =
      computeBillboardDist(splatSetDesc, localSplatId, modelRayOrigin, modelRayDirRaw, payload.rayBounce);

  if(rayTCurrentCorrected <= RayTMin())
  {
    IgnoreHit();
    return;
  }
#else
  // AABB custom intersection shader already computed billboard T via ReportHit()
  const float rayTCurrentCorrected = RayTCurrent();
#endif
  const float splatDist = rayTCurrentCorrected;

#if(RTX_PARTICLE_DEPTH != PARTICLE_DEPTH_BILLBOARD) && \
   ((RTX_TRACE_STRATEGY != RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT) || (PARTICLES_SPP == 1))
  // is hit closer than last hit of the array ?
  // In billboard mode, splatDist is the billboard plane distance,
  // not RayTCurrent(), so it doesn't correlate with hardware traversal order.
  if(splatDist < payload.dist[PARTICLES_SPP - 1])
#endif
  {

#if WIREFRAME
#if RTX_USE_AABBS
    const uint kind        = HitKind();
    const bool particleHit = (kind & (1 << 4)) != 0;  // Bit 4: particleHit flag
    processAABBWireframe(attribs.aabbHitT, attribs.barycentrics, payload, kind);
#elif RTX_USE_SPHERES
    processSphereWireframe(splatDist, payload);
#else
    processTriangleWireframe(splatDist, attribs.barycentrics, payload);
#endif
#endif

#if WIREFRAME && RTX_USE_AABBS
    // With AABB mode we may have a box hit but no particle hit
    if(particleHit)
#endif
    {
#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
      processHitStochastic(payload, splatDist, localSplatId, descriptorIdx, splatSetDesc, modelRayOrigin, modelRayDirRaw);
#else
      processHitInsertionSort(payload, splatDist, localSplatId, descriptorIdx);
#endif
    }
  }
}

#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
void processHitStochastic(inout HitPayload payload,
                          float            splatDist,
                          int              localSplatId,
                          int              descriptorIdx,
                          SplatSetDesc     splatSetDesc,
                          float3           modelRayOrigin,
                          float3           modelRayDir)
{

#if(PARTICLES_SPP > 1) || (RTX_PARTICLE_DEPTH == PARTICLE_DEPTH_BILLBOARD)
  // multi-entry reserved for future work. not active.
  bool needFetch = false;
  [unroll]
  for(int i = 0; i < PARTICLES_SPP; ++i)
  {
    if(splatDist < payload.dist[i])
      needFetch = true;
  }
  if(needFetch)
#endif  // if single sample we already did test in the caller
  {
    float  alpha = 0.0;
    float3 particleRad;
    float3 normalWorld;
    bool   acceptedHit;
#if RTX_PARTICLE_DEPTH == PARTICLE_DEPTH_BILLBOARD
    if(payload.rayBounce == 0)
    {
      acceptedHit = threedgsProcessHit<true>(frameInfo, splatSetDesc, modelRayOrigin, modelRayDir, splatSetDesc.transformRotScaleInverse,
                                             localSplatId, splatDist, alpha, particleRad, normalWorld);
    }
    else
#endif
    {
      acceptedHit = threedgrtProcessHit<true>(frameInfo, splatSetDesc, modelRayOrigin, modelRayDir,
                                              splatSetDesc.transformRotScaleInverse, localSplatId, alpha, particleRad, normalWorld);
    }

    if(acceptedHit)
    {

      [unroll]
      for(int i = 0; i < PARTICLES_SPP; ++i)
      {
#if PARTICLES_SPP > 1  // otherwise it is already tested in caller and for loop unroll resolves to simple block
        if(splatDist < payload.dist[i])
#endif
        {
          uint seed = payload.rngSeed.get().value;
          seed      = xxhash32(uint3(seed, uint(descriptorIdx) ^ uint(localSplatId) ^ asuint(splatDist), uint(i)));
          const float randomVal = rand(seed);
          if(randomVal < alpha)
          {
            payload.id[i] = localSplatId;
            payload.setSplatSetIdx(i, descriptorIdx);
            payload.dist[i] = splatDist;
            payload.color[i].set(float4(particleRad, 1.0));
            payload.normal[i].set(normalWorld);
          }
        }
      }
    }
  }

#if RTX_SHORTEN_RAY
  if(payload.dist[PARTICLES_SPP - 1] > RayTCurrent())
#endif
  {
    IgnoreHit();
  }
}

#else  // else all other modes (full pass and stochastic pass)

void processHitInsertionSort(inout HitPayload payload, float splatDist, int localSplatId, int descriptorIdx)
{
  [unroll]
  for(int i = 0; i < PARTICLES_SPP; ++i)
  {
    const float distance = payload.dist[i];
    if(splatDist < distance)
    {
      payload.dist[i] = splatDist;
      splatDist       = distance;

      const int id     = payload.id[i];
      const int setIdx = payload.getSplatSetIdx(i);
      payload.id[i]    = localSplatId;
      payload.setSplatSetIdx(i, descriptorIdx);
      localSplatId  = id;
      descriptorIdx = setIdx;
    }
  }

  if(payload.dist[PARTICLES_SPP - 1] > RayTCurrent())
  {
    IgnoreHit();
  }
}

#endif

#if RTX_PARTICLE_DEPTH == PARTICLE_DEPTH_BILLBOARD
// Compute billboard distance: ray intersection with a plane through the particle center.
// Bounce 0: plane perpendicular to the camera view direction (original billboard behavior).
// Bounce > 0: plane perpendicular to the ray direction. This can produce negative distances
// when the particle center is behind the ray origin; the caller must reject those.
// All inputs must be in model space.
float computeBillboardDist(SplatSetDesc desc, int localSplatId, float3 modelRayOrigin, float3 modelRayDir, uint bounce)
{
  const float3 modelCenter = fetchCenter(desc, uint(localSplatId));

#if RTX_BILLBOARD_FRUSTUM_CULLING
  if(bounce == 0)
  {
    // Coarse frustum culling: reject particles whose center projects outside the NDC frustum.
    // Matches the raster distance compute shader culling to avoid border artifacts.
    // Only for primary rays — secondary bounces travel in arbitrary directions.
    float4      worldPos = mul(float4(modelCenter, 1.0), desc.transform);
    float4      viewPos  = mul(worldPos, frameInfo.viewMatrix);
    float4      clipPos  = mul(viewPos, frameInfo.projectionMatrix);
    float3      ndcPos   = clipPos.xyz / clipPos.w;
    const float clip     = 1.0f + frameInfo.frustumDilation;
    if(abs(ndcPos.x) > clip || abs(ndcPos.y) > clip || ndcPos.z < -frameInfo.frustumDilation || ndcPos.z > 1.0)
      return -1.0;
  }
#endif

  float3 planeNormal;
  if(bounce == 0)
  {
    const float3 cameraViewDirWorld = normalize(mul(float4(0, 0, 1, 0), frameInfo.viewInverse).xyz);
    planeNormal                     = normalize(mul(cameraViewDirWorld, desc.transformRotScaleInverse));
  }
  else
  {
    planeNormal = normalize(modelRayDir);
  }
  return dot(modelCenter - modelRayOrigin, planeNormal) / dot(modelRayDir, planeNormal);
}
#endif


#if WIREFRAME
void processTriangleWireframe(in float triangleHitT, in float2 splatBary, inout HitPayload payload)
{
  if(triangleHitT < payload.wireframeDist.get().value)
  {
    const float3 bary = float3(1.f - splatBary.x - splatBary.y, splatBary.x, splatBary.y);

    // For triangle primitives (icosahedrons), compute proper ray differentials
    const float3 triangleVertexPositions[3] = { HitTriangleVertexPosition(0), HitTriangleVertexPosition(1),
                                                HitTriangleVertexPosition(2) };

    const float3 derivative =
        computeDerivativeHighp(WorldToObject4x3(), WorldRayOrigin(), ObjectRayOrigin(), triangleVertexPositions,
                               payload.differentialX.get().value, payload.differentialY.get().value, bary);

    const float alpha = processWireframe(wireframeDefaultSettings(), bary, derivative);

    if(alpha > 0.0)
    {
      payload.wireframeDist.set(triangleHitT);
      payload.wireframeAlpha.set(alpha);
    }
  }
}

void processAABBWireframe(in float aabbHitT, in float2 splatBary, inout HitPayload payload, uint hitKind)
{
  if(aabbHitT < payload.wireframeDist.get().value)
  {
    const float3 bary = float3(1.f - splatBary.x - splatBary.y, splatBary.x, splatBary.y);

    const double aabbSize = 1.0;  // Must match BLAS AABB half-extent [-1,1]^3

    const float3 derivative = computeAABBFaceDerivativeHighp(WorldToObject4x3(), WorldRayOrigin(), ObjectRayOrigin(),
                                                             aabbSize, payload.differentialX.get().value,
                                                             payload.differentialY.get().value, bary, hitKind);

    const float alpha = processWireframe(wireframeDefaultSettings(), bary, derivative);

    if(alpha > 0.0)
    {
      payload.wireframeDist.set(aabbHitT);
      payload.wireframeAlpha.set(alpha);
    }
  }
}

// Sphere silhouette wireframe: delegates to computeSphereWireframe() for the geometry math,
// then updates the payload if the sphere is closer than previous wireframe hits.
void processSphereWireframe(in float sphereHitT, inout HitPayload payload)
{
  if(sphereHitT >= payload.wireframeDist.get().value)
    return;

  const float alpha = computeSphereWireframe(ObjectRayOrigin(), ObjectRayDirection(), WorldRayOrigin(), WorldToObject4x3(),
                                             payload.differentialX.get().value, payload.differentialY.get().value);
  if(alpha > 0.0)
  {
    payload.wireframeDist.set(sphereHitT);
    payload.wireframeAlpha.set(alpha);
  }
}
#endif
