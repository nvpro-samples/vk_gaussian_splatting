/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"
#include "wavefront.h.slang"

// clang-format off
[[vk::push_constant]]                        ConstantBuffer<PushConstant>  pcRaster;
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]   ConstantBuffer<FrameInfo>     frameInfo;
[[vk::binding(BINDING_MESH_DESCRIPTORS, 0)]] StructuredBuffer<ObjDesc>     objDesc;
[[vk::binding(BINDING_LIGHT_SET, 0)]]        StructuredBuffer<LightSource> lights;

struct FragmentInput
{
  [[vk::location(1)]] float3 worldPos : TEXCOORD1;
  [[vk::location(2)]] float3 worldNrm : TEXCOORD2;
  [[vk::location(3)]] float3 viewDir : TEXCOORD3;
  // This was missing in GLSL version (there was a bug), added to get the source triangle ID.
  uint primitiveID : SV_PrimitiveID; 
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

#if HYBRID_ENABLED
  output.color = float4(0.0, 0.0, 0.0, 0.0);
#else

  ObjDesc objResource = objDesc[pcRaster.objIndex];
  // In GLSL version materialIndexAddress was erroneously dereferenced using pcRaster.objIndex. Now corrected using input.primitiveID.
  ObjMaterial material = objResource.materialAddress[objResource.materialIndexAddress[input.primitiveID]];

  float3 color = float3(0.0);

  wavefrontComputeShadingDirectOnly(lights, frameInfo.lightCount, input.worldPos, input.worldNrm, material, input.viewDir, color);

  output.color = float4(color, 1);

#endif

  return output;
}
