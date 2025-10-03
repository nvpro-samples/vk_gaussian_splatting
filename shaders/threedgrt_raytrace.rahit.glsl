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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require  // for [[unroll]]
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_GOOGLE_include_directive : require

#include "shaderio.h"
#include "threedgrt_payload.glsl"

// Hit attributes
#if WIREFRAME
hitAttributeEXT vec2 attrib;
#endif

void main()
{
#if RTX_USE_INSTANCES
  int splatId = gl_InstanceID;
#else
#if RTX_USE_AABBS
  int splatId = gl_PrimitiveID;  // aabbox per splat
#else
  int splatId = gl_PrimitiveID / 20;  // 20 triangles per icosahedron
#endif
#endif

  float splatDist = gl_HitTEXT;
#if WIREFRAME
  vec2 splatBary = attrib;
#endif
  //prd.hitCount += 1;

  if(splatDist < readDist(PAYLOAD_ARRAY_SIZE - 1))
  {
    // insert/sorted from min dist to max dist
    [[unroll]] for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
    {
      const float distance = readDist(i);
      if(splatDist < distance)
      {
        writeDist(i, splatDist);
        splatDist = distance;

        const int id = readId(i);
        writeId(i, splatId);
        splatId = id;

#if WIREFRAME
        const vec2 bary = readBary(i);
        writeBary(i, splatBary);
        splatBary = bary;
#endif
      }
    }

    // ignore all inserted hits, except if the last one
    if(readDist(PAYLOAD_ARRAY_SIZE - 1) > gl_RayTmaxEXT)
    {
      ignoreIntersectionEXT;
    }
  }
}
