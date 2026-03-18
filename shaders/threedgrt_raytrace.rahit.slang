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

// Bindless assets (needed for global index table in non-instanced mode)
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
  // instanceCustomIndex is limited to 24 bits (max 16777215), so we use TLAS offset
  // InstanceIndex() returns the instanceCustomIndex (which wraps at 16777215)
  // We need to add the TLAS base offset to get the true global ID
  int globalSplatId = payload.tlasInstanceOffset + InstanceIndex();
#elif RTX_USE_AABBS
  // AABB mode: PrimitiveIndex() is local per TLAS instance
  const uint   instanceIdx  = InstanceID();
  SplatSetDesc splatSetDesc = getRtxSplatSetDescriptors()[instanceIdx];
  // globalSplatBase encodes the TLAS-instance base in global splat space.
  const int globalSplatId = int(splatSetDesc.globalSplatBase) + PrimitiveIndex();
#else
  // Non-instanced Icosahedron mode: compute global splat ID
  int          localSplatId = PrimitiveIndex() / 20;  // 20 triangles per icosahedron
  int          instanceIdx  = InstanceID();           // Returns instanceCustomIndex (descriptor index)
  SplatSetDesc splatSetDesc = getRtxSplatSetDescriptors()[instanceIdx];
  // globalSplatBase encodes the TLAS-instance base in global splat space.
  int globalSplatId = int(splatSetDesc.globalSplatBase) + localSplatId;
#endif

  float splatDist = RayTCurrent();

#if RTX_TRACE_STRATEGY != RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
  // is hit closer than last hit of the array ?
  if(splatDist < payload.dist[PARTICLES_SPP - 1])
#endif
  {

#if WIREFRAME
#if !RTX_USE_AABBS
    processTriangleWireframe(splatDist, attribs.barycentrics, payload);
#else
    const uint kind        = HitKind();
    const bool particleHit = (kind & (1 << 4)) != 0;  // Bit 4: particleHit flag
    processAABBWireframe(attribs.aabbHitT, attribs.barycentrics, payload, kind);
#endif
#endif

#if WIREFRAME && RTX_USE_AABBS
    // With AABB mode we may have a box hit but no particle hit
    if(particleHit)
#endif
    {
#if RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT

      // do we need the particle data to compute alpha ?
      bool needFetch = false;
      [unroll]
      for(int i = 0; i < PARTICLES_SPP; ++i)
      {
        if(splatDist < payload.dist[i])
          needFetch = true;
      }

      if(needFetch)
      {
        // Evaluate hit opacity and apply stochastic accept/reject per payload slot (each slot = a pass).
        // 64-bit byte offset to prevent overflow when globalSplatId > ~537M (buffer > 4 GB)
        uint64_t              gitByteOffset    = uint64_t(globalSplatId) * uint64_t(sizeof(GlobalSplatIndexEntry));
        GlobalSplatIndexEntry entry            = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + gitByteOffset);
        SplatSetDesc          hitSplatSetDesc = assets.splatSetDescriptors[entry.splatSetIndex];
        const int             splatId         = int(entry.splatIndex);

        const float3 modelRayOrigin = mul(float4(WorldRayOrigin(), 1.0), hitSplatSetDesc.transformInverse).xyz;
        const float3 modelRayDir    = normalize(mul(WorldRayDirection(), hitSplatSetDesc.transformRotScaleInverse));

        float      alpha = 0.0;
        float3     particleRad;
        float3     normalWorld;
        const bool acceptedHit = particleProcessHit<true>(frameInfo, hitSplatSetDesc, modelRayOrigin, modelRayDir,
                                                          hitSplatSetDesc.transformRotScaleInverse, splatId, splatDist,
                                                          alpha, particleRad, normalWorld);

        if(acceptedHit)
        {
          [unroll]
          for(int i = 0; i < PARTICLES_SPP; ++i)
          {
            if(splatDist < payload.dist[i])
            {
              uint seed       = payload.rngSeed.get().value;
              seed            = xxhash32(uint3(seed, uint(globalSplatId) ^ asuint(splatDist), uint(i)));
              const float randomVal = rand(seed);
              if(randomVal < alpha)
              {
                payload.id[i]   = globalSplatId;
                payload.dist[i] = splatDist;
                payload.color[i].set(float4(particleRad, 1.0));
                payload.normal[i].set(normalWorld);
              }
            }
          }
        }
      }

      // ignore all inserted hits, except if the last one
      if (payload.dist[PARTICLES_SPP - 1] > RayTCurrent())
      {
          IgnoreHit();
      }

#else
      // insert/sorted from min dist to max dist
      [unroll]
      for(int i = 0; i < PARTICLES_SPP; ++i)
      {
        const float distance = payload.dist[i];
        if(splatDist < distance)
        {
          payload.dist[i] = splatDist;
          splatDist       = distance;

          const int id  = payload.id[i];
          payload.id[i] = globalSplatId;
          globalSplatId = id;
        }
      }

      // ignore all inserted hits, except if the last one
      if(payload.dist[PARTICLES_SPP - 1] > RayTCurrent())
      {
        IgnoreHit();
      }
#endif
    }
  }
}

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

    const double aabbSize = 1.618033988749895f;  // Must match intersection shader (Golden Ratio)

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
#endif
