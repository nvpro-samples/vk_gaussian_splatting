/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshaders/wireframe.h.slang"

#include "shaderio.h"
#include "threedgrt_custom_attributes.h.slang"
#include "threedgrt_payload.h.slang"
#include "wireframe_extended.h.slang"

[shader("anyhit")]
void main(inout HitPayload payload
#if WIREFRAME
          ,
          in CustomIntersectionAttributes attribs
#endif
)
{
#if RTX_USE_INSTANCES
  int splatId = InstanceIndex();
#elif RTX_USE_AABBS
  int splatId = PrimitiveIndex();  // aabbox per splat
#else
  int splatId = PrimitiveIndex() / 20;  // 20 triangles per icosahedron
#endif

  float splatDist = RayTCurrent();

  // is hit closer than last hit of the array ?
  if(splatDist < payload.dist[PAYLOAD_ARRAY_SIZE - 1])
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
      // insert/sorted from min dist to max dist
      [unroll]
      for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
      {
        const float distance = payload.dist[i];
        if(splatDist < distance)
        {
          payload.dist[i] = splatDist;
          splatDist       = distance;

          const int id  = payload.id[i];
          payload.id[i] = splatId;
          splatId       = id;
        }
      }

      // ignore all inserted hits, except if the last one
      if(payload.dist[PAYLOAD_ARRAY_SIZE - 1] > RayTCurrent())
      {
        IgnoreHit();
      }
    }
  }
}

#if WIREFRAME
void processTriangleWireframe(in float triangleHitT, in float2 splatBary, inout HitPayload payload)
{
  if(triangleHitT < payload.wireframeDist)
  {
    const float3 bary = float3(1.f - splatBary.x - splatBary.y, splatBary.x, splatBary.y);

    // For triangle primitives (icosahedrons), compute proper ray differentials
    const float3 triangleVertexPositions[3] = { HitTriangleVertexPosition(0), HitTriangleVertexPosition(1),
                                                HitTriangleVertexPosition(2) };

    const float3 derivative = computeDerivativeHighp(WorldToObject4x3(), WorldRayOrigin(), ObjectRayOrigin(), triangleVertexPositions,
                                                     payload.differentialX, payload.differentialY, bary);

    const float alpha = processWireframe(wireframeDefaultSettings(), bary, derivative);

    if(alpha > 0.0)
    {
      payload.wireframeDist  = triangleHitT;
      payload.wireframeAlpha = alpha;
    }
  }
}

void processAABBWireframe(in float aabbHitT, in float2 splatBary, inout HitPayload payload, uint hitKind)
{
  if(aabbHitT < payload.wireframeDist)
  {
    const float3 bary = float3(1.f - splatBary.x - splatBary.y, splatBary.x, splatBary.y);

    const double aabbSize = 1.618033988749895f;  // Must match intersection shader (Golden Ratio)

    const float3 derivative = computeAABBFaceDerivativeHighp(WorldToObject4x3(), WorldRayOrigin(), ObjectRayOrigin(), aabbSize,
                                                             payload.differentialX, payload.differentialY, bary, hitKind);

    const float alpha = processWireframe(wireframeDefaultSettings(), bary, derivative);

    if(alpha > 0.0)
    {
      payload.wireframeDist  = aabbHitT;
      payload.wireframeAlpha = alpha;
    }
  }
}
#endif
