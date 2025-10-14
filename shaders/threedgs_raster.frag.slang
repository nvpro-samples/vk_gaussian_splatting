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

#include "wireframe.h.slang"
#include "shaderio.h"

// clang-format off

// Per-vertex input (from vertex shader)
struct VertexInput
{
  [[vk::location(0)]] float4 inSplatCol : COLOR0;
#if !USE_BARYCENTRIC
  [[vk::location(1)]] float2 inFragPos : TEXCOORD0;
#endif
};

// Per-primitive input (from mesh shader)
struct MeshPrimitiveInput
{
  [[vk::location(0)]] float4 inSplatCol : COLOR0;
};

// Per-vertex input (from mesh shader)
struct MeshVertexInput
{
#if !USE_BARYCENTRIC
  [[vk::location(1)]] float2 inFragPos : TEXCOORD0;
#endif
};

// Fragment output
struct FragmentOutput
{
  [[vk::location(0)]] float4 outColor : SV_Target0;
};

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;

// clang-format on

///////////////
// main fragment function when using Vertex shader pipeline

[shader("fragment")]
FragmentOutput main(VertexInput vert
#if USE_BARYCENTRIC || WIREFRAME
                    ,
                    float3 baryCoord: SV_Barycentrics
#endif
)
{
#if USE_BARYCENTRIC
  const float2 inFragPos = float(0.0);  // placeholder
#else
  const float2 inFragPos = vert.inFragPos;
#endif

#if !USE_BARYCENTRIC && !WIREFRAME
  const float3 baryCoord = float3(0.0);  // placeholder
#endif

  return processFragment(inFragPos, vert.inSplatCol, baryCoord);
}

///////////////
// main fragment function when using Mesh shader pipeline

[shader("fragment")]
FragmentOutput main_mesh(MeshVertexInput                    vert,
                         nointerpolation MeshPrimitiveInput prim
#if USE_BARYCENTRIC || WIREFRAME
                         ,
                         float3 baryCoord: SV_Barycentrics
#endif
)
{
#if USE_BARYCENTRIC
  const float2 inFragPos = float(0.0);  // placeholder
#else
  const float2 inFragPos = vert.inFragPos;
#endif

#if !USE_BARYCENTRIC && !WIREFRAME
  const float3 baryCoord = float3(0.0);  // placeholder
#endif

  return processFragment(inFragPos, prim.inSplatCol, baryCoord);
}

///////////////
// common processing function (the hart of the shader)
// #if USE_BARYCENTRIC : inFragPos is a placeholder (not used)
// #if !USE_BARYCENTRIC && !WIREFRAME : baryCoord is a placeholder (not used)
FragmentOutput processFragment(float2 inFragPos, float4 inSplatCol, float3 baryCoord)
{
  FragmentOutput output;

#if WIREFRAME
  if(processWireframe(wireframeDefaultSettings(), baryCoord) > 0.0)
  {
    output.outColor = float4(1.0, 0.0, 0.0, 1.0);  // wireframe color, no blending
    return output;
  }
#endif

#if USE_BARYCENTRIC
  // Use barycentric extension to find the position of the fragment
  const float sqrt8 = sqrt(8.0);
  const float2 inFragPos = (baryCoord.x * float2(-1, -1) + baryCoord.y * float2(1, 1) + baryCoord.z * float2(-1, 1)) * sqrt8;
#else
  const float2 inFragPos = inFragPos;
#endif

  // Compute the positional squared distance from the center of the splat to the current fragment.
  const float A = dot(inFragPos, inFragPos);
  // Since the positional data in inFragPos has been scaled by sqrt(8), the squared result will be
  // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
  // defined by the rectangle formed by inFragPos. It also means it's farther
  // away than sqrt(8) standard deviations from the mean.
  if(A > 8.0)
    discard;

#if DISABLE_OPACITY_GAUSSIAN
  const float opacity = 1.0;
#else
  // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
  // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
  // and since 'mean' is zero, we have X * X, which is the same as A:
  const float opacity = exp(-0.5 * A) * inSplatCol.a;
#endif

  output.outColor = float4(inSplatCol.rgb, opacity);

  return output;
}
