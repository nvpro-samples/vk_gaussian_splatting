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

#version 450

#extension GL_EXT_mesh_shader : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_fragment_shader_barycentric : require

#include "nvshaders/slang_types.h"
#include "nvshaders/random.h.slang"

#include "shaderio.h"
#include "wireframe.glsl"
#include "cameras.glsl"

#include "threedgrt.glsl"

precision highp float;

layout(location = 0) perprimitiveEXT in flat uint inSplatId;
layout(location = 1) perprimitiveEXT in flat vec4 inSplatCol;
layout(location = 2) perprimitiveEXT in flat vec3 inSplatPosition;
layout(location = 3) perprimitiveEXT in flat vec3 inSplatScale;
layout(location = 4) perprimitiveEXT in flat vec4 inSplatRotation;  // a quaternion

layout(location = 0) out vec4 outColor;

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(push_constant) uniform _PushConstantRaster
{
  PushConstant pcRaster;
};

void main()
{

#if WIREFRAME
  if(processWireframe(wireframeDefaultSettings(), gl_BaryCoordEXT) > 0.0)
  {
    outColor = vec4(1.0, 0.0, 0.0, 1.0);  // wireframe color, no blending
    return;
  }
#endif

  // use this to visualize full splat rectangular extent.
#if false
  {
    vec4 color = inSplatCol;
    outColor   = vec4(color.rgb, 1.0);
    return;
  }
#endif

  // prepare a ray the evaluation of the particle response
  vec3 rayOrigin;
  vec3 rayDirection;

  // generate the ray for this pixel
#if CAMERA_TYPE == CAMERA_PINHOLE
  generatePinholeRay(gl_FragCoord.xy, frameInfo.viewport, frameInfo.viewInverse, frameInfo.projInverse, rayOrigin, rayDirection);
#else
  if(!generateFisheyeRay(gl_FragCoord.xy, frameInfo.viewport, frameInfo.fovRad, vec2(0.0), frameInfo.viewInverse,
                         rayOrigin, rayDirection))
  {
    discard;
  }
#endif

  // add Depth-of-Field perturbation to the ray
#if RTX_DOF_ENABLED
  // Initialize the random number
  uint seed = xxhash32(uvec3(gl_FragCoord.xy, frameInfo.frameSampleId));

  depthOfField(seed, frameInfo.focusDist, frameInfo.aperture, frameInfo.viewInverse, rayOrigin, rayDirection);
#endif

  // The two following transformations are to compute processHit with transformed splat set model
  const vec3 splatSetModelRayOrigin = vec3(pcRaster.modelMatrixInverse * vec4(rayOrigin, 1.0));
  // Since the ray direction should not be affected by translation,
  // uses the inverse of the rotation-scale part of the model matrix.
  const vec3 splatSetModelRayDirection = normalize(mat3(pcRaster.modelMatrixRotScaleInverse) * rayDirection);

  float opacity;
  bool acceptedHit = particleProcessHitGut(frameInfo, splatSetModelRayOrigin, splatSetModelRayDirection, int(inSplatId),
                                           inSplatCol, inSplatPosition, inSplatScale, inSplatRotation, opacity);

  if(!acceptedHit)
    discard;

  outColor = vec4(inSplatCol.rgb, opacity);
}
