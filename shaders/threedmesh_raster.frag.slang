/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"
#include "shading.h.slang"
#if WIREFRAME
#include "nvshaders/wireframe.h.slang"
#include "nvshaders/fwidth.h.slang"
#endif

// clang-format off
[[vk::push_constant]]                      ConstantBuffer<PushConstant>  pcRaster;
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo>     frameInfo;
[[vk::binding(BINDING_ASSETS, 0)]]         ConstantBuffer<SceneAssets>   assets;
[[vk::binding(BINDING_MESH_TEXTURES, 0)]]  Sampler2D                    meshTextures[];

struct FragmentInput
{
  [[vk::location(1)]] float3 worldPos : POSITION_WS;
  [[vk::location(2)]] float3 worldNrm : NORMAL_WS;
  [[vk::location(3)]] float3 viewDir : VIEW_DIR;
  [[vk::location(4)]] float2 texCoord : TEXCOORD0;
  [[vk::location(5)]] float2 texCoord1 : TEXCOORD1;
  [[vk::location(6)]] float4 worldTangent : TANGENT_WS;
  uint primitiveID : SV_PrimitiveID; 
};

struct FragmentOutput
{
  [[vk::location(0)]] float4 color : SV_Target0;
};
// clang-format on

[shader("fragment")]
FragmentOutput main(FragmentInput input
#if WIREFRAME
                    , float3 baryCoord : SV_Barycentrics
#endif
)
{
  FragmentOutput output;

#if HYBRID_ENABLED
  output.color = float4(0.0, 0.0, 0.0, 0.0);
#else

#if FRONT_TO_BACK && NEED_SURFACE_INFO
  // FTB mode: Three-pass rendering for correct mesh/splat compositing
  // Pass 1 (ftbColorPass=0): Depth pre-pass - output black, write depth only
  // Pass 3 (ftbColorPass=1): Color pass - read transmittance, output mesh_color * transmittance

  if(pcRaster.ftbColorPass == 0)
  {
    // Depth pre-pass: output black (color doesn't matter, we just need depth)
    output.color = float4(0.0, 0.0, 0.0, 0.0);
    return output;
  }

  // Color pass: compute shading and modulate by transmittance
#endif

  // Get per-instance mesh descriptor using bindless access
  MeshDesc meshDesc = assets.meshes[pcRaster.objIndex];

  // Retrieve material for this triangle
  Material material = meshDesc.materialAddress[meshDesc.materialIndexAddress[input.primitiveID]];

  // Apply material textures (modifies material properties and normal in-place)
  float3 shadingNrm = input.worldNrm;
  applyMaterialTextures(material, meshTextures, input.texCoord, input.texCoord1, shadingNrm, input.worldTangent);

#if (LIGHTING_MODE != LIGHTING_DISABLED)
  float3 color = float3(0.0, 0.0, 0.0);  // Start with black

  // Add emissive contribution first (for light proxies and other emissive materials)
  color += material.emissive * material.emissiveStrength;

  // Only apply lighting if material needs it
  if(material.needShading == 1)
  {
    // Apply lighting from each light source (using bindless lights)
    for(int i = 0; i < assets.lightCount; ++i)
    {
      if(assets.lights[i].enabled == 0 || assets.lights[i].shadowOnly != 0)
        continue;
      bool inShadow = false;  // Rasterization doesn't have shadow rays (yet)
      wavefrontComputeShadingDirectOnly(assets.lights[i], input.worldPos, shadingNrm, material, input.viewDir, inShadow, float3(1.0), color);
    }

    if(assets.lightCount == 0)
    {
      // No lights in scene - use headlight (camera-attached point light with no attenuation)
      const LightSource headlight = createHeadlight(frameInfo.cameraPosition);
      bool              inShadow  = false;
      wavefrontComputeShadingDirectOnly(headlight, input.worldPos, shadingNrm, material, input.viewDir, inShadow, float3(1.0), color);
    }
  }
  // else: Pure emissive material (light proxy) - no lighting interaction
#else
  // Shading disabled - display unlit material color
  float3 color = material.emissive * material.emissiveStrength + material.baseColor;
#endif

  output.color = float4(color, 1);

  // Wireframe overlay: blend mesh triangle edges in red
#if WIREFRAME
  const float wireframeAlpha = processWireframe(wireframeDefaultSettings(), baryCoord, fwidth(baryCoord));
  output.color               = lerp(output.color, float4(1.0, 0.0, 0.0, 1.0), wireframeAlpha);
#endif

#endif

  return output;
}
