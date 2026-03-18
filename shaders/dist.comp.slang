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

#include "shaderio.h"
#include "threedgs_particle_storage.h.slang"
#include "threedgut_camera_models.h.slang"
#include "threedgut_camera_projections.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]   ConstantBuffer<FrameInfo>    frameInfo;
[[vk::binding(BINDING_ASSETS, 0)]]           ConstantBuffer<SceneAssets> assets;
[[vk::push_constant]]                        ConstantBuffer<PushConstant> pcRaster;
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
  const uint globalSplatID = dispatchThreadID.x;
  // each workgroup (but the last one if splat count is not a multiple)
  // processes DISTANCE_COMPUTE_WORKGROUP_SIZE points
  if(globalSplatID >= frameInfo.splatCount)
    return;

  // Resolve global splat ID to correct descriptor and local index
  SplatSetDesc splatSetDesc;
  uint splatIndex;
  resolveGlobalSplatID(assets, globalSplatID, splatSetDesc, splatIndex);
  
  float4 splatPos = float4(fetchCenter(splatSetDesc, splatIndex), 1.0);

  // Use transform from descriptor, not push constants
  float4 viewPos      = mul(mul(splatPos, splatSetDesc.transform), frameInfo.viewMatrix);
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

#if SIZE_CULLING_MODE == SIZE_CULLING_ENABLED
  // Size culling: cull splats whose projected bounding sphere is smaller than a threshold
  {
    // Fetch the scale of the splat (3D scale factors define the ellipsoid extents)
    // Note: scales are stored in log-space, so we need to exponentiate them
    float3 scale = exp(fetchScale(splatSetDesc, splatIndex)) * frameInfo.splatScale;
    
    // Use the maximum scale as the bounding sphere radius (1 standard deviation)
    float radius = max(scale.x, max(scale.y, scale.z));
    
    // Splats are rendered with sqrt(8) standard deviations extent (see raster shaders)
    // Multiply by 2 to get diameter for "pixel coverage" comparison
    const float sqrt8 = 2.8284271247f;  // sqrt(8)
    float extent = radius * sqrt8 * 2.0f;  // diameter in world units
    
    // Transform extent to view space (account for instance transform scaling)
    // Use the maximum scale factor from the instance transform
    float3 transformScale = float3(
      length(splatSetDesc.transform[0].xyz),
      length(splatSetDesc.transform[1].xyz),
      length(splatSetDesc.transform[2].xyz)
    );
    extent *= max(transformScale.x, max(transformScale.y, transformScale.z));
    
    // Project extent to screen space (perspective projection)
    // The focal length is already in pixel units (fx, fy from camera intrinsics)
    // Note: focal.y may be negative due to Vulkan's coordinate system convention
    // projectedPixels = extent * focal / depth
    float viewDist = abs(viewPos.z);
    if(viewDist > 0.0001f)  // Avoid division by zero
    {
      // Use max of absolute focal lengths for conservative projection
      float maxFocal = max(abs(frameInfo.focal.x), abs(frameInfo.focal.y));
      float projectedPixels = (extent * maxFocal) / viewDist;
      
      // Cull if projected size is below threshold
      if(projectedPixels < frameInfo.sizeCullingMinPixels)
        return;
    }
    
  }
#endif

  // increments the visible splat counter in the indirect buffer
  uint original;
  InterlockedAdd(indirectBuffer[0].instanceCount, 1, original);
  const uint instanceIndex = original;

  // stores the global splat ID (used by raster shaders to resolve descriptor + local index)
  uint32_t *indices = assets.splatSortingIndicesAddress;
  indices[instanceIndex] = globalSplatID;

  // set the workgroup count for the mesh shading pipeline
  if (instanceIndex % RASTER_MESH_WORKGROUP_SIZE == 0)
  {
      InterlockedAdd(indirectBuffer[0].groupCountX, 1);
  }

#if !STOCHASTIC_SPLAT
  // stores the distance
  uint32_t *distances = assets.splatSortingDistancesAddress;
  // For back-to-front (default): negate depth so ascending radix sort yields farthest-first
  // For front-to-back: use positive depth so ascending radix sort yields nearest-first
#if FRONT_TO_BACK
  distances[instanceIndex] = encodeMinMaxFp32(depth);
#else
  distances[instanceIndex] = encodeMinMaxFp32(-depth);
#endif
#endif

}
