/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include "nvshaders/fwidth.h.slang"
#include "nvshaders/wireframe.h.slang"
#include "nvshaders/random.h.slang"

#include "shaderio.h"

#if NEED_SURFACE_INFO && QUANTIZE_NORMALS
#include "octahedral_normal.h.slang"
#endif

// clang-format off

// Per-vertex input (from vertex shader)
struct VertexInput
{
  [[vk::location(0)]] float4 inSplatCol : COLOR0;
#if !USE_BARYCENTRIC
  [[vk::location(1)]] float2 inFragPos : TEXCOORD0;
#endif
#if STOCHASTIC_SPLAT || NEED_SURFACE_INFO
  [[vk::location(3)]] nointerpolation uint inSplatId : SPLATID;
#endif
};

// Per-primitive input (from mesh shader)
struct MeshPrimitiveInput
{
  [[vk::location(0)]] float4 inSplatCol : COLOR0;
#if STOCHASTIC_SPLAT || NEED_SURFACE_INFO
  [[vk::location(3)]] nointerpolation uint inSplatId : SPLATID;
#endif
#if NEED_SURFACE_INFO
#if QUANTIZE_NORMALS
  // Octahedral encoding: unit normal packed into 32-bit uint (16 bits per component)
  [[vk::location(4)]] nointerpolation uint inSplatNormalPacked : NORMAL0;
#else
  [[vk::location(4)]] nointerpolation float3 inSplatNormal : NORMAL0;
#endif
#endif
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
#if NEED_SURFACE_INFO
  [[vk::location(1)]] float4 outNormal : SV_Target1;  // Normals (integrated mode) or zeros (reconstructed mode)
#if !FRONT_TO_BACK
  [[vk::location(2)]] float4 outDepth : SV_Target2;   // Integrated depth (R=depth, A=opacity for blending) - BTF only
  [[vk::location(3)]] uint   outSplatId : SV_Target3; // Global splat ID for material lookup
#else
  [[vk::location(2)]] uint   outSplatId : SV_Target2; // Global splat ID for material lookup (shifted for FTB)
  // Note: In FTB mode, picked depth is stored in storage image (depthTransmittanceBuffer.r)
  // Meshes do manual depth comparison in their fragment shader
#endif
#endif
};

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;

#if FRONT_TO_BACK && NEED_SURFACE_INFO
// For FTB: depth buffer accessed as storage image (R=depth, G=transmittance)
// globallycoherent ensures memory coherency between interlock-protected fragment invocations
[[vk::binding(BINDING_RASTER_DEPTH, 0)]] globallycoherent RWTexture2D<float2> depthTransmittanceBuffer;
#endif

// clang-format on

///////////////
// main fragment function when using Vertex shader pipeline

[shader("fragment")]
FragmentOutput main(VertexInput vert
#if USE_BARYCENTRIC || WIREFRAME
                    ,
                    float3 baryCoord: SV_Barycentrics
#endif
#if STOCHASTIC_SPLAT || NEED_SURFACE_INFO
                    ,
                    float4 fragCoord: SV_Position
#endif
#if STOCHASTIC_SPLAT
                    ,
                    uint primitiveID: SV_PrimitiveID
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

  // Note: Vertex shader pipeline does not compute normals (mesh shader only)
  // Normal will be zero, but splat ID is still valid for material lookup
  const float3 inNormal = float3(0.0);

#if STOCHASTIC_SPLAT
  return processFragment(inFragPos, vert.inSplatCol, baryCoord, fragCoord, vert.inSplatId, primitiveID, inNormal);
#elif NEED_SURFACE_INFO
  return processFragment(inFragPos, vert.inSplatCol, baryCoord, fragCoord, vert.inSplatId, 0, inNormal);
#else
  return processFragment(inFragPos, vert.inSplatCol, baryCoord, float4(0), 0, 0, inNormal);
#endif
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
#if STOCHASTIC_SPLAT || NEED_SURFACE_INFO
                         ,
                         float4 fragCoord: SV_Position
#endif
#if STOCHASTIC_SPLAT
                         ,
                         uint primitiveID: SV_PrimitiveID
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

#if NEED_SURFACE_INFO
#if QUANTIZE_NORMALS
  // Decode octahedral-encoded normal from packed uint
  const float3 inNormal = decodeNormalOctahedral(prim.inSplatNormalPacked);
#else
  const float3 inNormal = prim.inSplatNormal;
#endif
#else
  const float3 inNormal = float3(0.0);  // Normal will be reconstructed from depth in compute pass
#endif

#if STOCHASTIC_SPLAT
  return processFragment(inFragPos, prim.inSplatCol, baryCoord, fragCoord, prim.inSplatId, primitiveID, inNormal);
#elif NEED_SURFACE_INFO
  return processFragment(inFragPos, prim.inSplatCol, baryCoord, fragCoord, prim.inSplatId, 0, inNormal);
#else
  return processFragment(inFragPos, prim.inSplatCol, baryCoord, float4(0), 0, 0, inNormal);
#endif
}

///////////////
// common processing function (the heart of the shader)
// #if USE_BARYCENTRIC : inFragPos is a placeholder (not used)
// #if !USE_BARYCENTRIC && !WIREFRAME : baryCoord is a placeholder (not used)
// #if !STOCHASTIC_SPLAT : fragCoord and splatId are placeholders (not used)
// #if !NEED_SURFACE_INFO : inNormal is a placeholder (not used)
FragmentOutput processFragment(float2 inFragPos, float4 inSplatCol, float3 baryCoord, float4 fragCoord, uint splatId, uint primitiveID, float3 inNormal)
{
  FragmentOutput output;

#if USE_BARYCENTRIC
  // Use barycentric extension to find the position of the fragment
  const float sqrt8 = sqrt(8.0);
  const float2 fragPos = (baryCoord.x * float2(-1, -1) + baryCoord.y * float2(1, 1) + baryCoord.z * float2(-1, 1)) * sqrt8;
#else
  const float2 fragPos = inFragPos;
#endif

  // Compute the positional squared distance from the center of the splat to the current fragment.
  const float A = dot(fragPos, fragPos);
  // Since the positional data in inFragPos has been scaled by sqrt(8), the squared result will be
  // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
  // defined by the rectangle formed by inFragPos. It also means it's farther
  // away than sqrt(8) standard deviations from the mean.
#if !WIREFRAME
  if(A > 8.0)
  {
    discard;
  }
#endif

#if DISABLE_OPACITY_GAUSSIAN
  const float opacity = 1.0;
#else
  // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
  // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
  // and since 'mean' is zero, we have X * X, which is the same as A:
  const float opacity = exp(-0.5 * A) * inSplatCol.a;
#endif

  // Threshold for early culling
  const float THRESHOLD = 1.0 / 255.0;

#if !WIREFRAME
  if(opacity <= THRESHOLD)
    discard;
#endif

#if STOCHASTIC_SPLAT
  {
    // Stochastic splat is incompatible with FTB (no transmittance accumulation needed)
    // Use the actual splat ID for random generation
    // Stochastic rendering: binary accept/reject based on opacity
    uint seed       = xxhash32(uint3(uint(fragCoord.x), uint(fragCoord.y), frameInfo.frameSampleId));
    seed            = xxhash32(uint3(seed, splatId, primitiveID));
    float randomVal = rand(seed);

    // Stochastic accept/reject decision based on opacity
    if(randomVal < opacity)
    {
      output.outColor = float4(inSplatCol.rgb, 1.0);
#if NEED_SURFACE_INFO
      // For stochastic mode, write surface outputs with full opacity (no blending needed)
      output.outDepth   = float4(fragCoord.z, 0, 0, 1.0);
      output.outNormal  = float4(inNormal, 1.0);
      output.outSplatId = splatId;
#endif
      return output; // no wireframe support in stochastic mode.
    }
    else
    {
      discard;
    }
  }
#else  // Alpha blending mode
  {
#if WIREFRAME
    if(A > 8.0)
      output.outColor = float4(0.0, 0.0, 0.0, 0.0);
    else
#endif
    {
#if FRONT_TO_BACK
      // Front-to-back "under" compositing requires premultiplied alpha
      // Output: float4(color.rgb * opacity, opacity)
      // Blend equation: SrcPremultiplied * (1 - DstAlpha) + Dst * 1
      output.outColor = float4(inSplatCol.rgb * opacity, opacity);
#else
      // Back-to-front "over" compositing (standard alpha blending)
      // Output: float4(color.rgb, opacity)
      // Blend equation: Src * SrcAlpha + Dst * (1 - SrcAlpha)
      output.outColor = float4(inSplatCol.rgb, opacity);
#endif
    }
  }
#endif
  // ===================================================

#if WIREFRAME
  const float wireframeAlpha = processWireframe(wireframeDefaultSettings(), baryCoord, fwidth(baryCoord));
  output.outColor            = lerp(output.outColor, float4(1.0, 0.0, 0.0, 1.0), wireframeAlpha);
#endif

#if NEED_SURFACE_INFO
  // Output normal weighted by opacity for alpha-blending integration (FTB mode)
  // Note: BTF with surface reconstruction only happens in stochastic mode which has early return
  output.outNormal = float4(inNormal * opacity, opacity);

#if FRONT_TO_BACK
  // FTB: Output depth for depth picking (first significant splat) using storage image
  // fragCoord.z is the normalized device depth [0,1]
  int2 pixelCoord = int2(fragCoord.xy);

#if FTB_SYNC_MODE == FTB_SYNC_INTERLOCK
  beginInvocationInterlock();
#endif
  float2 current           = depthTransmittanceBuffer[pixelCoord];
  float  depth             = current.r;
  float  prevTransmittance = current.g;
  float  transmittance     = prevTransmittance * (1.0 - opacity);
  if(depth == 0 && transmittance < frameInfo.depthIsoThreshold)
  {
    depth = fragCoord.z;
  }
  // Write back: depth in R, transmittance in G
  depthTransmittanceBuffer[pixelCoord] = float2(depth, transmittance);
#if FTB_SYNC_MODE == FTB_SYNC_INTERLOCK
  endInvocationInterlock();
#endif
#endif // FRONT_TO_BACK

  // Output global splat ID for material lookup in hybrid ray tracing mode
  output.outSplatId = splatId;
#endif

  return output;
}
