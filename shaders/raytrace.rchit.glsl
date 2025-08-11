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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require  // for [[unroll]]
#extension GL_GOOGLE_include_directive : require

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "shaderio.h"
#include "common.glsl"
#include "raycommon.glsl"


// Hit attributes
//#if WIREFRAME
hitAttributeEXT vec2 attrib;
//#endif

#if RTX_USE_MESHES
layout(set = 0, binding = BINDING_MESH_DESCRIPTORS, scalar) buffer ObjDesc_
{
  ObjDesc i[];
}
objDesc;
#endif

layout(buffer_reference, scalar) buffer Vertices
{
  Vertex v[];
};

layout(buffer_reference, scalar) buffer Indices
{
  ivec3 i[];
};

layout(buffer_reference, scalar) buffer MaterialIndices  // Material indices per face
{
  uint i[];
};

layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform _frameInfo
{
  FrameInfo frameInfo;
};

layout(push_constant) uniform _PushConstantRay
{
  PushConstantRay pcRay;
};


// clang-format on

void main()
{
#if RTX_USE_MESHES
  //
  writeDist(0, gl_HitTEXT);

  //
  ObjDesc         objResource = objDesc.i[gl_InstanceCustomIndexEXT];
  Indices         indices     = Indices(objResource.indexAddress);
  Vertices        vertices    = Vertices(objResource.vertexAddress);
  MaterialIndices matIndices  = MaterialIndices(objResource.materialIndexAddress);

  writeId(0, int(gl_InstanceCustomIndexEXT));     // to retrieve objResource in ray gen
  writeId(1, int(matIndices.i[gl_PrimitiveID]));  // to retrieve material in raygen - TODO force to -1 to display wireframe ?

  // Indices of the triangle
  ivec3 tri = indices.i[gl_PrimitiveID];

  // Vertex of the triangle
  Vertex v0 = vertices.v[tri.x];
  Vertex v1 = vertices.v[tri.y];
  Vertex v2 = vertices.v[tri.z];

  const vec3 barycentrics = vec3(1.0 - attrib.x - attrib.y, attrib.x, attrib.y);

  // Computing the coordinates of the hit position
  const vec3 pos      = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0));  // Transforming the position to world space

  // Computing the normal at hit position
  const vec3 nrm      = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  const vec3 worldNrm = normalize(vec3(nrm * gl_WorldToObjectEXT));  // Transforming the normal to world space

  writeDist(1, worldPos.x);
  writeDist(2, worldPos.y);
  writeDist(3, worldPos.z);
  writeDist(4, worldNrm.x);
  writeDist(5, worldNrm.y);
  writeDist(6, worldNrm.z);
#endif
}
