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

#extension GL_EXT_mesh_shader : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_fragment_shader_barycentric : require
#include "shaderio.h"

precision highp float;

layout(location = 0) perprimitiveEXT in flat vec4 inSplatCol;
#if !USE_BARYCENTRIC
layout(location = 1) in vec2 inFragPos;
#endif

layout(location = 0) out vec4 outColor;

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

void main()
{

#if USE_BARYCENTRIC
  // Use barycentric extension to find the position of the fragment
  const float sqrt8 = sqrt(8.0);
  vec2 inFragPos = (gl_BaryCoordEXT.x * vec2(-1,-1) +  gl_BaryCoordEXT.y * vec2(1,1) + gl_BaryCoordEXT.z * vec2(-1,1)) * sqrt8;
#endif

  // Compute the positional squared distance from the center of the splat to the current fragment.
  const float A = dot(inFragPos, inFragPos);
  // Since the positional data in inFragPos has been scaled by sqrt(8), the squared result will be
  // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
  // defined by the rectangle formed by inFragPos. It also means it's farther
  // away than sqrt(8) standard deviations from the mean.
  if(A > 8.0)
    discard;

#ifdef DISABLE_OPACITY_GAUSSIAN
  const float opacity = 1.0;
#else
  // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
  // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
  // and since 'mean' is zero, we have X * X, which is the same as A:
  const float opacity = exp(-0.5 * A) * inSplatCol.a;
#endif

  outColor = vec4(inSplatCol.rgb, opacity);
}
