/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 460

#include "common/shaders/functions.hlsli"
#include "device_host.h"

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  [[vk::location(0)]] float3 position : POSITION;
  [[vk::location(1)]] float3 normal : NORMAl;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  float3 position : POSIITON;
  float3 normal : NORMAL;
};

// Output of the vertex shader
struct VSout
{
  PSin stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};


[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(0)]] ConstantBuffer<FrameInfo> frameInfo;


// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  float4 pos = mul(pushConst.transfo, float4(input.position.xyz, 1.0));

  VSout output;
  output.sv_position = mul(frameInfo.proj, mul(frameInfo.view, pos));
  output.stage.normal = input.normal;
  output.stage.position = pos.xyz;

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage)
{
  float3 V = normalize(frameInfo.camPos - stage.position); // vector that goes from the hit position towards the origin of the ray
  float3 color = simpleShading(V, V, stage.normal, pushConst.color.xyz);

  PSout output;
  output.color = float4(color, 1.0);
  
  return output;
}
