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

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

struct VertexInput
{
  [[vk::location(0)]] float3 position : POSITION;
  [[vk::location(1)]] float3 normal : NORMAL;
  //[[vk::location(2)]] float3 color : COLOR;
  //[[vk::location(3)]] float2 texCoord : TEXCOORD0;
};

struct VertexOutput
{
  [[vk::location(1)]] float3 worldPos : TEXCOORD1;
  [[vk::location(2)]] float3 worldNrm : TEXCOORD2;
  [[vk::location(3)]] float3 viewDir : TEXCOORD3;
  //[[vk::location(4)]] float2 texCoord : TEXCOORD4;
  float4 position : SV_Position;
};
// clang-format on

[shader("vertex")]
VertexOutput main(VertexInput input)
{
  VertexOutput output;

  float3 origin = mul(float4(0, 0, 0, 1), frameInfo.viewInverse).xyz;

  output.worldPos = mul(float4(input.position, 1.0), pcRaster.modelMatrix).xyz;
  output.viewDir  = output.worldPos - origin;
  // output.texCoord = input.texCoord;
  output.worldNrm = normalize(mul(input.normal, float3x3(transpose(pcRaster.modelMatrixInverse))));

  output.position = mul(mul(float4(output.worldPos, 1.0), frameInfo.viewMatrix), frameInfo.projectionMatrix);

  return output;
}
