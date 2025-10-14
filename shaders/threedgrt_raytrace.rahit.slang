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

#include "shaderio.h"
#include "threedgrt_payload.h.slang"

[shader("anyhit")]
void main(inout HitPayload payload
#if WIREFRAME
          , in BuiltInTriangleIntersectionAttributes attribs
#endif
)
{
#if RTX_USE_INSTANCES
  int splatId = InstanceIndex();
#else
#if RTX_USE_AABBS
  int splatId = PrimitiveIndex();  // aabbox per splat
#else
  int splatId = PrimitiveIndex() / 20;  // 20 triangles per icosahedron
#endif
#endif

  float splatDist = RayTCurrent();
#if WIREFRAME
  float2 splatBary = attribs.barycentrics;
#endif
  //prd.hitCount += 1;

  if (splatDist < payload.dist[PAYLOAD_ARRAY_SIZE - 1])
  {
    // insert/sorted from min dist to max dist
    [unroll]
    for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
    {
      const float distance = payload.dist[i];
      if(splatDist < distance)
      {
        payload.dist[i] = splatDist;
        splatDist = distance;

        const int id = payload.id[i];
        payload.id[i] = splatId;
        splatId = id;

#if WIREFRAME
        const float2 bary = payload.bary[i];
        payload.bary[i] = splatBary;
        splatBary = bary;
#endif
      }
    }

    // ignore all inserted hits, except if the last one
    if(payload.dist[PAYLOAD_ARRAY_SIZE - 1] > RayTCurrent())
    {
      IgnoreHit();
    }
  }
}
