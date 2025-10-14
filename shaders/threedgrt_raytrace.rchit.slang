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
#include "threedgrt_payload.h.slang"

// clang-format off
#if RTX_USE_MESHES
[[vk::binding(BINDING_MESH_DESCRIPTORS, 0)]] StructuredBuffer<ObjDesc> objDesc;
#endif
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// clang-format on

[shader("closesthit")]
void main(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
#if RTX_USE_MESHES
  //
  payload.dist[0] = RayTCurrent();

  //
  const ObjDesc objResource = objDesc[InstanceID()];

  payload.id[0] = int(InstanceID());                                        // to retrieve objResource in ray gen
  payload.id[1] = int(objResource.materialIndexAddress[PrimitiveIndex()]);  // to retrieve material in raygen

  // Indices of the triangle
  int3 tri = ((int3*)objResource.indexAddress)[PrimitiveIndex()];

  // Vertex of the triangle
  ObjVertex v0 = objResource.vertexAddress[tri.x];
  ObjVertex v1 = objResource.vertexAddress[tri.y];
  ObjVertex v2 = objResource.vertexAddress[tri.z];

  const float3 barycentrics =
      float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

  // Computing the coordinates of the hit position
  const float3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  const float3 worldPos = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));  // Transforming the position to world space

  // Computing the normal at hit position
  const float3 nrm      = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  const float3 worldNrm = normalize(mul(WorldToObject4x3(), nrm).xyz);  // Transforming the normal to world space

  payload.dist[1] = worldPos.x;
  payload.dist[2] = worldPos.y;
  payload.dist[3] = worldPos.z;
  payload.dist[4] = worldNrm.x;
  payload.dist[5] = worldNrm.y;
  payload.dist[6] = worldNrm.z;
#endif
}
