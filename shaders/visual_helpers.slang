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

#include "visual_helpers_shaderio.h.slang"

//--------------------------------------------------------------------------------------------------
// Vertex Stage
//--------------------------------------------------------------------------------------------------

// Vertex inputs (generic to support both grid and transform helper)
struct VertexInput
{
  float3 position : POSITION;   // Vertex position (world space for grid, model space for transform)
  float3 attrib0  : ATTRIB0;    // Grid: color (RGB), Transform: normal (XYZ)
};

// Vertex outputs
struct VertexOutput
{
  float4 position : SV_Position;
  float3 attrib0  : ATTRIB0;    // Grid: color, Transform: normal (unused)
  float4 color    : COLOR;      // Final color (from attrib or push constant)
};

[[vk::push_constant]] ConstantBuffer<PushConstantVisualHelpers> pc;

[shader("vertex")]
VertexOutput vertmain(VertexInput input)
{
  VertexOutput output;
  
  // Transform to clip space
  output.position = mul(float4(input.position, 1.0), pc.mvp);
  
  // Pass through attribute (color for grid, normal for transform)
  output.attrib0 = input.attrib0;
  
  // Determine final color based on mode
  if (pc.mode == HelperMode::eGrid)
  {
    // Grid: use per-vertex color from attribute
    output.color = float4(input.attrib0, 1.0);
  }
  else // HelperMode::eTransform
  {
    // Transform: use push constant color
    output.color = pc.color;
  }
  
  return output;
}

//--------------------------------------------------------------------------------------------------
// Fragment Stage
//--------------------------------------------------------------------------------------------------

// Scene depth texture (for occlusion detection)
[[vk::binding(0, 1)]] Texture2D<float> sceneDepth;
[[vk::binding(1, 1)]] SamplerState depthSampler;

// Fragment inputs
struct FragmentInput
{
  float4 position : SV_Position;  // Screen-space position (for depth and UV)
  float3 attrib0  : ATTRIB0;      // Grid: color, Transform: normal (unused)
  float4 color    : COLOR;        // Interpolated color
};

// Fragment output
struct FragmentOutput
{
  float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput fragmain(FragmentInput input)
{
  FragmentOutput output;
  
  // Sample scene depth for occlusion test
  // DLSS-aware UV scaling: map viewport position to depth buffer position
  float2 screenUV = input.position.xy / pc.viewportSize;
  float2 depthScale = pc.depthBufferSize / pc.viewportSize;
  float2 depthUV = screenUV * depthScale;
  float sceneDepthValue = sceneDepth.Sample(depthSampler, depthUV);
  
  // Get fragment depth
  float fragmentDepth = input.position.z;
  
  // Check if fragment is occluded (behind scene geometry)
  bool isOccluded = fragmentDepth > sceneDepthValue;
  
  // Apply mode-specific occlusion handling
  if (isOccluded)
  {
    if (pc.mode == HelperMode::eGrid)
    {
      // Grid: fully discard occluded fragments (no dithering)
      discard;
    }
    else // HelperMode::eTransform
    {
      // Transform: checkerboard pattern for occluded fragments (x-ray effect)
      int2 pixelCoord = int2(input.position.xy);
      int checker = (pixelCoord.x + pixelCoord.y) % 2;
      if (checker == 0)
      {
        discard;
      }
    }
  }
  
  // Output final color
  if (pc.mode == HelperMode::eGrid)
  {
    // Grid: apply alpha from push constant for LOD blending
    output.color = float4(input.color.rgb, pc.color.w);
  }
  else // HelperMode::eTransform
  {
    // Transform: use color as-is (alpha already in color)
    output.color = input.color;
  }
  
  return output;
}

