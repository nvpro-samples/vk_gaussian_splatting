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

#ifndef _WAVEFRONT_H_
#define _WAVEFRONT_H_

#ifdef __cplusplus
#include "nvshaders/slang_types.h"
namespace shaderio {
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
  float3 ambient       = float3(0.0);
  float3 diffuse       = float3(0.0);
  float3 specular      = float3(0.0);
  float3 transmittance = float3(0.0);
  float3 emission      = float3(0.0);
  float  shininess     = 0.0;
  float  ior           = 0.0;
  float  dissolve      = 0.0;
  int    illum         = 0;
  int    textureID     = 0;
  int    needShading   = 0;  // Computed flag: 1 if diffuse/ambient/specular non-zero (set by CPU)
};

#ifdef __cplusplus
// Helper to update needShading flag based on material properties
// Call this before uploading material to GPU
inline void updateMaterialNeedsShading(ObjMaterial& mat)
{
  mat.needShading =
      (glm::length(mat.diffuse) > 0.001f || glm::length(mat.ambient) > 0.001f || glm::length(mat.specular) > 0.001f) ? 1 : 0;
}
#endif

// Information of a obj model when referenced in a shader
struct ObjDesc
{
  //int      txtOffset;             // Texture index offset in the array of textures
  ObjVertex*   vertexAddress;         // Address of the Vertex buffer
  uint32_t*    indexAddress;          // Address of the index buffer
  ObjMaterial* materialAddress;       // Address of the material buffer
  uint32_t*    materialIndexAddress;  // Address of the triangle material index buffer
};

// Light type enumeration (available in both C++ and Slang)
enum LightType : int32_t
{
  eDirectionalLight = 0,  // Backward compatible
  ePointLight       = 1,  // Backward compatible
  eSpotLight        = 2   // New
};

// Structure holding light source attributes
struct LightSource
{
  LightType type            = LightType::ePointLight;  // Light type (Point/Directional/Spot)
  float3    color           = float3(1.0, 1.0, 1.0);   // RGB color of the light
  float     intensity       = 1.0;
  float3    position        = float3(0.0, 0.0, 0.0);   // World position (from instance)
  float     range           = 10.0f;                   // Effective light range for point/spot lights
  float3    direction       = float3(0.0, 0.0, -1.0);  // Light direction (computed from rotation)
  float     innerConeAngle  = 30.0f;                   // Spot light: inner cone angle (degrees)
  float     outerConeAngle  = 45.0f;                   // Spot light: outer cone angle (degrees)
  int       attenuationMode = 2;                       // 0=None, 1=Linear, 2=Quadratic, 3=Physical
  float     radius          = 1.0f;                    // Light source radius for soft shadows (from proxyScale)
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

#endif