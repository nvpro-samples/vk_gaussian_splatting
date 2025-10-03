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

/*
* The code has been adapted to Vulkan from the WebGL-based implementation 
* https://github.com/mkkellogg/GaussianSplats3D. Some mathematical formulations 
* and comments have been directly retained from this source. Original source code  
* licence hereafter.
* ----------------------------------
* The MIT License (MIT)
* 
* Copyright (c) 2023 Mark Kellogg
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : require

#include "shaderio.h"
#include "threedgs.glsl"
#include "threedgs_particles_storage.glsl"

layout(location = ATTRIBUTE_LOC_POSITION) in vec3 inPosition;
layout(location = ATTRIBUTE_LOC_SPLAT_INDEX) in uint32_t inSplatIndex;

layout(location = 0) out vec4 outFragCol;
#if !USE_BARYCENTRIC
layout(location = 1) out vec2 outFragPos;
#endif

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform _frameInfo
{
  FrameInfo frameInfo;
};
layout(push_constant) uniform _PushConstantRaster
{
  PushConstant pcRaster;
};

void main()
{
  const uint splatIndex = inSplatIndex;

  const mat4 modelViewMatrix = frameInfo.viewMatrix * pcRaster.modelMatrix;

  // Fetch data as early as possible
  const mat3 cov3Dm      = fetchCovariance(splatIndex);
  vec4       splatColor  = fetchColor(splatIndex);
  const vec3 splatCenter = fetchCenter(splatIndex);

  const vec4 viewCenter = modelViewMatrix * vec4(splatCenter, 1.0);
  const vec4 clipCenter = frameInfo.projectionMatrix * viewCenter;

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_RASTER
  const float clip = (1.0 + frameInfo.frustumDilation) * clipCenter.w;
  if(abs(clipCenter.x) > clip || abs(clipCenter.y) > clip
     || clipCenter.z < (0.f - frameInfo.frustumDilation) * clipCenter.w || clipCenter.z > clipCenter.w)
  {
    // emit same vertex to get degenerate triangle
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
#endif

  const vec2 fragPos = inPosition.xy;
#if !USE_BARYCENTRIC
  // emit as early as possible
  // Scale the position data we send to the fragment shader
  outFragPos = fragPos * sqrt8;
#endif

  // alpha based culling
  if(splatColor.a < frameInfo.alphaCullThreshold)
  {
    // emit same vertex to get degenerate triangle
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

#if SHOW_SH_ONLY == 1
  splatColor.rgb = vec3(0.5);
#endif

  // fetch radiance from SH coefs > degree 0
  // const vec3 worldViewDir = normalize(splatCenter - frameInfo.cameraPosition);
  const vec3 worldViewDir = normalize(splatCenter - vec3(pcRaster.modelMatrixInverse * vec4(frameInfo.cameraPosition, 1.0)));

  splatColor.rgb += fetchViewDependentRadiance(splatIndex, worldViewDir);

  // emit as early as possible for perf reasons, only for original 3DGS,
  // see later on for MipSplatting = on
#if MS_ANTIALIASING == 0
  outFragCol = splatColor;
#endif

  // Computes the projected covariance
  const vec3 cov2Dv = threedgsCovarianceProjection(cov3Dm, viewCenter, frameInfo.focal, modelViewMatrix);

  // computes the basis vectors of the extent of the projected covariance
  // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
  vec2 basisVector1, basisVector2;
  if(!threedgsProjectedExtentBasis(cov2Dv, sqrt8, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
  {
    // emit same vertex to get degenerate triangle
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

#if MS_ANTIALIASING == 1
  // emit the color with alpha compensation
  outFragCol = splatColor;
#endif

  const vec3 ndcCenter = clipCenter.xyz / clipCenter.w;

  const vec2 ndcOffset = vec2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                         * frameInfo.inverseFocalAdjustment;

  const vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

  // Emit the vertex position
  gl_Position = quadPos;
}
