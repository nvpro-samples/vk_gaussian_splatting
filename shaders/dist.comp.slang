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

#include "shaderio.h"
#include "threedgs_particles_storage.h.slang"
#include "threedgut_camera_models.h.slang"
#include "threedgut_camera_projections.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]   ConstantBuffer<FrameInfo>    frameInfo;
[[vk::push_constant]]                        ConstantBuffer<PushConstant> pcRaster;
[[vk::binding(BINDING_DISTANCES_BUFFER, 0)]] RWStructuredBuffer<uint32_t> distances;
[[vk::binding(BINDING_INDICES_BUFFER, 0)]]   RWStructuredBuffer<uint32_t> indices;
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]]  RWStructuredBuffer<IndirectParams> indirectBuffer;
// clang-format on

// encodes an fp32 into a uint32 that can be ordered
uint encodeMinMaxFp32(float val)
{
  uint bits = asuint(val);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  return bits;
}

[numthreads(DISTANCE_COMPUTE_WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const uint id = dispatchThreadID.x;
  // each workgroup (but the last one if splat count is not a multiple)
  // processes DISTANCE_COMPUTE_WORKGROUP_SIZE points
  if(id >= frameInfo.splatCount)
    return;

  float4 splatPos = float4(fetchCenter(id), 1.0);

  // Row-major: multiply in reverse order (vector * matrix * matrix)
  float4 viewPos      = mul(mul(splatPos, pcRaster.modelMatrix), frameInfo.viewMatrix);
  // TODO: projection matrix is not correct with fisheye, this is a coarse aprox to compute depth
  float4 ndcPos       = mul(viewPos, frameInfo.projectionMatrix);
  ndcPos            = ndcPos / ndcPos.w;
  const float depth = ndcPos.z;

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
#if CAMERA_TYPE == CAMERA_PINHOLE
  // valid only when center is inside NDC clip space.
  // Note: when culling between x=[-1,1] y=[-1,1], which is NDC extent,
  // the culling is not good since we only take into account
  // the center of each splat instead of its extent.

  const float clip = 1.0f + frameInfo.frustumDilation;
  if(abs(ndcPos.x) > clip || abs(ndcPos.y) > clip || ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
    return;

#else

  // Fisheye camera
  CameraModelParameters sensorModel = initPerfectFisheyeCamera(frameInfo.viewport, frameInfo.focal);

  // Fisheye projection to perform culling
  float2 projected;
  bool isValid = projectPointFisheye(sensorModel.ocvFisheyeParams, frameInfo.viewport,
                                     float3(1.0, 1.0, -1.0) * viewPos.xyz, GUT_IN_IMAGE_MARGIN_FACTOR, projected);

  if(!isValid)
    return;

  if(ndcPos.z < 0.f - frameInfo.frustumDilation || ndcPos.z > 1.0)
    return;
#endif
#endif

  // increments the visible splat counter in the indirect buffer
  uint original;
  InterlockedAdd(indirectBuffer[0].instanceCount, 1, original);
  const uint instance_index = original;
  // stores the distance
  distances[instance_index] = encodeMinMaxFp32(-depth);
  // stores the base index
  indices[instance_index] = id;
  // set the workgroup count for the mesh shading pipeline
  if(instance_index % RASTER_MESH_WORKGROUP_SIZE == 0)
  {
    InterlockedAdd(indirectBuffer[0].groupCountX, 1);
  }
}
