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
#include "nvshaders/slang_types.h"
// used to assign fields defaults
#define DEFAULT(val) = val
namespace shaderio {
#else
// we are in Slang here
// used to skip fields init
// when included in shaders
#define DEFAULT(val)
#endif

struct ObjVertex  // See ObjLoader, copy of VertexObj
{
  float3 pos;
  float3 nrm;
  //vec3 color;
  //vec2 texCoord;
};

// Structure holding the material for mesh objects
struct ObjMaterial
{
  float3 ambient;
  float3 diffuse;
  float3 specular;
  float3 transmittance;
  float3 emission;
  float  shininess;
  float  ior;
  float  dissolve;
  int    illum;
  int    textureID;
};

// Information of a obj model when referenced in a shader
struct ObjDesc
{
  //int      txtOffset;             // Texture index offset in the array of textures
  ObjVertex*   vertexAddress;         // Address of the Vertex buffer
  uint32_t*    indexAddress;          // Address of the index buffer
  ObjMaterial* materialAddress;       // Address of the material buffer
  uint32_t*    materialIndexAddress;  // Address of the triangle material index buffer
};

// Structure holding light source attributes
struct LightSource
{
  float3 position DEFAULT(float3(0.0, 4.0, 0.0));
  float intensity DEFAULT(1.0);
  int type        DEFAULT(LIGHT_TYPE_DIRECTIONAL);
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

#undef DEFAULT
#endif