/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvshaders/random.h.slang"
#include "nvshaders/fwidth.h.slang"
#include "nvshaders/wireframe.h.slang"

#include "shaderio.h"
#include "cameras.h.slang"
#include "threedgrt.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

struct FragmentInput
{
  [[vk::location(0)]] nointerpolation uint splatId : SPLATID;
  [[vk::location(1)]] nointerpolation float4 splatCol : COLOR0;
  [[vk::location(2)]] nointerpolation float3 splatPosition : POSITION0;
  [[vk::location(3)]] nointerpolation float3 splatScale : SCALE0;
  [[vk::location(4)]] nointerpolation float4 splatRotation : ROTATION0;  // a quaternion
  float4 position : SV_Position;
#if WIREFRAME
  float3 barycentrics : SV_Barycentrics;
#endif
};

struct FragmentOutput
{
  [[vk::location(0)]] float4 color : SV_Target0;
};
// clang-format on

[shader("fragment")]
FragmentOutput main(FragmentInput input)
{
  FragmentOutput output;

  // use this to visualize full splat rectangular extent.
#if 0
  {
    float4 color = input.splatCol;
    output.color = float4(color.rgb, 1.0);
    return output;
  }
#endif

  // prepare a ray the evaluation of the particle response
  float3 rayOrigin;
  float3 rayDirection;

  // generate the ray for this pixel
#if CAMERA_TYPE == CAMERA_PINHOLE
  generatePinholeRay(input.position.xy, frameInfo.viewport, frameInfo.viewInverse, frameInfo.projInverse, rayOrigin, rayDirection);
#else
  if(!generateFisheyeRay(input.position.xy, frameInfo.viewport, frameInfo.fovRad, float2(0.0), frameInfo.viewInverse,
                         rayOrigin, rayDirection))
  {
    discard;
  }
#endif

  // add Depth-of-Field perturbation to the ray
#if RTX_DOF_ENABLED
  // Initialize the random number
  uint seed = xxhash32(uint3(int(input.position.x), int(input.position.y), frameInfo.frameSampleId));

  depthOfField(seed, frameInfo.focusDist, frameInfo.aperture, frameInfo.viewInverse, rayOrigin, rayDirection);
#endif

  // The two following transformations are to compute processHit with transformed splat set model
  const float3 splatSetModelRayOrigin = mul(float4(rayOrigin, 1.0), pcRaster.modelMatrixInverse).xyz;
  // Since the ray direction should not be affected by translation,
  // uses the inverse of the rotation-scale part of the model matrix.
  const float3 splatSetModelRayDirection = normalize(mul(rayDirection, float3x3(pcRaster.modelMatrixRotScaleInverse)));

  float opacity;
  bool acceptedHit = particleProcessHitGut(frameInfo, splatSetModelRayOrigin, splatSetModelRayDirection, int(input.splatId),
                                           input.splatCol, input.splatPosition, input.splatScale, input.splatRotation, opacity);

  if(!acceptedHit)
#if !WIREFRAME
    discard;
#else
    output.color = float4(0.0, 0.0, 0.0, 0.0);
#endif
  else
    output.color = float4(input.splatCol.rgb, opacity);

#if WIREFRAME
  const float wireframeAlpha = processWireframe(wireframeDefaultSettings(), input.barycentrics, fwidth(input.barycentrics));
  output.color = lerp(output.color, float4(1.0, 0.0, 0.0, 1.0), wireframeAlpha);
#endif

  return output;
}
