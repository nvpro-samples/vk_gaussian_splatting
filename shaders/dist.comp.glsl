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

#version 460

#extension GL_GOOGLE_include_directive : enable
#include "shaderio.h"
#include "common.glsl"

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(local_size_x = DISTANCE_COMPUTE_WORKGROUP_SIZE) in;

layout(set = 0, binding = BINDING_DISTANCES_BUFFER, scalar) writeonly buffer _distances
{
  uint32_t distances[];
};
layout(set = 0, binding = BINDING_INDICES_BUFFER, scalar) writeonly buffer _indices
{
  uint32_t indices[];
};
layout(set = 0, binding = BINDING_INDIRECT_BUFFER, scalar) writeonly buffer _indirect
{
  IndirectParams indirect;
};

// encodes an fp32 into a uint32 that can be ordered
uint encodeMinMaxFp32(float val)
{
  uint bits = floatBitsToUint(val);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  return bits;
}

void main()
{
  const uint id = gl_GlobalInvocationID.x;
  // each workgroup (but the last one if splat count is not a multiple)
  // processes DISTANCE_COMPUTE_WORKGROUP_SIZE points
  if(id >= frameInfo.splatCount)
    return;

  vec4 pos          = vec4(fetchCenter(id), 1.0);
  pos               = frameInfo.projectionMatrix * frameInfo.viewMatrix * pos;
  pos               = pos / pos.w;
  const float depth = pos.z;

  // valid only when center is inside NDC clip space.
  // Note: when culling between x=[-1,1] y=[-1,1], which is NDC extent,
  // the culling is not good since we only take into account
  // the center of each splat instead of its extent.
#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
  const float clip = 1.0f + frameInfo.frustumDilation;
  if(abs(pos.x) > clip || abs(pos.y) > clip || pos.z < 0.f - frameInfo.frustumDilation || pos.z > 1.0)
    return;
#endif

  // increments the visible splat counter in the indirect buffer 
  const uint instance_index = atomicAdd(indirect.instanceCount, 1);
  // stores the distance
  distances[instance_index] = encodeMinMaxFp32(-depth);
  // stores the base index
  indices[instance_index] = id;
  // set the workgroup count for the mesh shading pipeline
  if(instance_index % RASTER_MESH_WORKGROUP_SIZE == 0)
  {
    atomicAdd(indirect.groupCountX, 1);
  }
}
