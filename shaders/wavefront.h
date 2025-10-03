/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _WAVEFRONT_H_
#define _WAVEFRONT_H_

//
#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_DIRECTIONAL 1

#ifdef __cplusplus
#include <glm/glm.hpp>
// used to assign fields defaults
#define DEFAULT(val) = val
namespace shaderio {
using namespace glm;
#else
// we are in glsl here
// used to skip fields init
// when included in glsl
#define DEFAULT(val)
// common extensions
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#endif

// Information of a obj model when referenced in a shader
struct ObjDesc
{
  //int      txtOffset;             // Texture index offset in the array of textures
  uint64_t vertexAddress;         // Address of the Vertex buffer
  uint64_t indexAddress;          // Address of the index buffer
  uint64_t materialAddress;       // Address of the material buffer
  uint64_t materialIndexAddress;  // Address of the triangle material index buffer
};

struct Vertex  // See ObjLoader, copy of VertexObj
{
  vec3 pos;
  vec3 nrm;
  //vec3 color;
  //vec2 texCoord;
};

// Structure holding the material for mesh objects
struct ObjMaterial
{
  vec3  ambient;
  vec3  diffuse;
  vec3  specular;
  vec3  transmittance;
  vec3  emission;
  float shininess;
  float ior;
  float dissolve;
  int   illum;
  int   textureID;
};

// Structure holding light source attributes
struct LightSource
{
  vec3 position   DEFAULT(vec3(0.0, 4.0, 0.0));
  float intensity DEFAULT(1.0);
  int type        DEFAULT(LIGHT_TYPE_DIRECTIONAL);
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

#undef DEFAULT
#endif