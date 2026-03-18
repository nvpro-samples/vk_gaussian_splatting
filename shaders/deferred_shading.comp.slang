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

// Deferred shading compute shader for pure rasterization pipelines
// Reads GBuffers (depth, normal, splat ID, color) and performs lighting

#include "shaderio.h"
#include "wavefront.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]      ConstantBuffer<FrameInfo>   frameInfo;
[[vk::binding(BINDING_ASSETS, 0)]]              ConstantBuffer<SceneAssets> assets;
[[vk::binding(BINDING_RASTER_NORMAL, 0)]]       RWTexture2D<float4>         normalBuffer;
[[vk::binding(BINDING_RASTER_DEPTH, 0)]]        RWTexture2D<float2>         depthBuffer;       // R=depth, G=transmittance
[[vk::binding(BINDING_RASTER_SPLATID, 0)]]      RWTexture2D<uint>           splatIdBuffer;
[[vk::binding(BINDING_RASTER_COLOR, 0)]]        RWTexture2D<float4>         colorBufferMain;   // Input: rasterized splat colors (COLOR_MAIN)
[[vk::binding(BINDING_DEFERRED_OUTPUT, 0)]]     RWTexture2D<float4>         outputImageMain;   // Output: final shaded image (COLOR_MAIN)
[[vk::binding(BINDING_RASTER_COLOR_AUX, 0)]]    RWTexture2D<float4>         colorBufferAux;    // Input: rasterized splat colors (COLOR_AUX1)
[[vk::binding(BINDING_DEFERRED_OUTPUT_AUX, 0)]] RWTexture2D<float4>         outputImageAux;    // Output: final shaded image (COLOR_AUX1)
// clang-format on

// Reconstruct world position from NDC depth and pixel coordinates
float3 reconstructWorldPos(float2 pixelF, float ndcDepth, float2 viewport)
{
  float2 ndc_xy  = pixelF / viewport * 2.0 - 1.0;
  float4 clipPos = float4(ndc_xy, ndcDepth, 1.0);

  // Unproject to view space
  float4 viewPos = mul(clipPos, frameInfo.projInverse);
  viewPos /= viewPos.w;

  // Transform to world space
  return mul(viewPos, frameInfo.viewInverse).xyz;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const int2   pixel    = int2(dispatchThreadID.xy);
  const float2 pixelF   = float2(pixel) + 0.5;
  const int2   viewport = int2(frameInfo.viewport);

  // Bounds check
  if(pixel.x >= viewport.x || pixel.y >= viewport.y)
    return;

  // Select buffer based on temporal accumulation state
  // Frame 0: use MAIN buffer, Frame 1+: use AUX buffer (ping-pong accumulation)
  const bool useAuxBuffer = (frameInfo.frameSampleId > 0);

  // Read integrated normal from surface reconstruction buffer
  float4 normalData = normalBuffer[pixel];

  // If no valid normal (alpha = 0), output background/color passthrough
  if(normalData.w < 0.001)
  {
    // No splat coverage - output the rasterized color directly (background or transparent)
    float4 colorIn = useAuxBuffer ? colorBufferAux[pixel] : colorBufferMain[pixel];
    if(useAuxBuffer)
      outputImageAux[pixel] = colorIn;
    else
      outputImageMain[pixel] = colorIn;
    return;
  }

  float3 normal = normalize(normalData.xyz);

  // Read picked depth from surface reconstruction buffer - depth is in R channel
  float ndcDepth = depthBuffer[pixel].r;

  // Reconstruct world position
  float3 worldPos = reconstructWorldPos(pixelF, ndcDepth, float2(viewport));

  // Read base color from appropriate buffer based on temporal accumulation state
  float4 baseColor = useAuxBuffer ? colorBufferAux[pixel] : colorBufferMain[pixel];
  float3 baseRadiance = baseColor.rgb;

  // Read splat ID and get material from splat set
  uint globalSplatId = splatIdBuffer[pixel];
  
  // Default material
  ObjMaterial mat = {};
  mat.diffuse     = float3(0.8, 0.8, 0.8);
  mat.ambient     = float3(0.1, 0.1, 0.1);
  mat.specular    = float3(0.0, 0.0, 0.0);
  mat.shininess   = 32.0;
  mat.needShading = 1;  // Default material needs lighting

  // Look up material from splat set if valid
  if(globalSplatId != 0xFFFFFFFF && assets.globalIndexTableAddress != 0)
  {
    // 64-bit byte offset to prevent overflow when globalSplatId > ~537M (buffer > 4 GB)
    uint64_t byteOffset = uint64_t(globalSplatId) * uint64_t(sizeof(GlobalSplatIndexEntry));
    GlobalSplatIndexEntry entry = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + byteOffset);
    if(entry.splatSetIndex < assets.splatSetCount)
    {
      SplatSetDesc splatSetDesc = assets.splatSetDescriptors[entry.splatSetIndex];
      
      // Set up material for lighting computation (same as raygen shader)
      mat.diffuse     = baseRadiance * splatSetDesc.material.diffuse;
      mat.ambient     = baseRadiance * splatSetDesc.material.ambient;
      mat.specular    = baseRadiance * splatSetDesc.material.specular;
      mat.shininess   = splatSetDesc.material.shininess;
      mat.emission    = baseRadiance * splatSetDesc.material.emission;
      mat.needShading = splatSetDesc.material.needShading;
    }
  }
  else
  {
    // No valid splat ID - use base color as diffuse
    mat.diffuse = baseRadiance;
  }

  // Compute view direction
  float3 viewDir = normalize(worldPos - frameInfo.cameraPosition);

#if (LIGHTING_MODE != LIGHTING_DISABLED)
  // Initialize output color with emission
  float3 color = mat.emission;

  // Only apply lighting if material needs it (has diffuse, ambient, or specular)
  if((mat.needShading != 0))
  {
    // Apply lighting from each light source (no shadows in pure rasterization)
    for(uint i = 0; i < assets.lightCount; ++i)
    {
      bool inShadow = false;  // No shadow rays in rasterization
      wavefrontComputeShadingDirectOnly(assets.lights[i], worldPos, normal, mat, viewDir, inShadow, double3(1.0), color);
    }

    if(assets.lightCount == 0)
    {
      // No lights in scene - use headlight (camera-attached point light)
      const LightSource headlight = createHeadlight(frameInfo.cameraPosition);
      bool              inShadow  = false;
      wavefrontComputeShadingDirectOnly(headlight, worldPos, normal, mat, viewDir, inShadow, double3(1.0), color);
    }
  }
#else
  // Shading disabled - keep raw splat color (baseRadiance contains the actual splat color)
  // Only emissive-only materials (light proxies) show emission instead
  float3 color = (mat.needShading != 0) ? baseRadiance : mat.emission;
#endif

  // Output final shaded color to appropriate buffer
  if(useAuxBuffer)
    outputImageAux[pixel] = float4(color, 1.0);
  else
    outputImageMain[pixel] = float4(color, 1.0);
}
