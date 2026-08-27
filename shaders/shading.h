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

#ifndef _SHADING_H_
#define _SHADING_H_

#ifdef __cplusplus
#include "nvshaders/slang_types.h"
namespace shaderio {
#endif

// PBR metallic-roughness material (GGX-based)
struct Material
{
  float3 baseColor           = float3(0.7, 0.7, 0.7);
  float  metallic            = 0.0;
  float  roughness           = 0.5;
  float  ior                 = 1.5;
  float  transmission        = 0.0;  // 0 = opaque, 1 = fully transmissive
  float  opacity             = 1.0;
  float  specularFactor      = 1.0;                    // KHR_materials_specular: dielectric specular weight
  float3 specularColorFactor = float3(1.0, 1.0, 1.0);  // KHR_materials_specular: dielectric specular color
  float  clearcoatFactor     = 0.0;                    // KHR_materials_clearcoat: strength of clearcoat layer
  float  clearcoatRoughness  = 0.0;                    // KHR_materials_clearcoat: roughness of clearcoat layer
  float3 emissive            = float3(0.0);
  float  emissiveStrength    = 1.0;  // KHR_materials_emissive_strength: multiplier on emissive
  int    maxBounces          = 3;    // Per-material max bounce count for path tracing (0 = direct only)

  // Texture indices into BINDING_MESH_TEXTURES array (-1 = no texture)
  int baseColorTexture          = -1;  // RGBA: modulates baseColor (.rgb) and opacity (.a)
  int metallicRoughnessTexture  = -1;  // glTF packed: roughness (G), metallic (B)
  int normalTexture             = -1;  // Tangent-space normal map
  int emissiveTexture           = -1;  // RGB: modulates emissive
  int occlusionTexture          = -1;  // R channel: ambient occlusion
  int specularTexture           = -1;  // KHR_materials_specular: A channel modulates specularFactor
  int specularColorTexture      = -1;  // KHR_materials_specular: RGB modulates specularColorFactor
  int clearcoatTexture          = -1;  // KHR_materials_clearcoat: R channel modulates clearcoatFactor
  int clearcoatRoughnessTexture = -1;  // KHR_materials_clearcoat: G channel modulates clearcoatRoughness

  // KHR_texture_transform: per-slot UV transform (identity by default)
  // Applied as: transformedUV = mul(float3(uv, 1.0), uvTransform).xy
  float3x2 baseColorUvTransform          = float3x2(1);
  float3x2 metallicRoughnessUvTransform  = float3x2(1);
  float3x2 normalUvTransform             = float3x2(1);
  float3x2 emissiveUvTransform           = float3x2(1);
  float3x2 occlusionUvTransform          = float3x2(1);
  float3x2 specularUvTransform           = float3x2(1);
  float3x2 specularColorUvTransform      = float3x2(1);
  float3x2 clearcoatUvTransform          = float3x2(1);
  float3x2 clearcoatRoughnessUvTransform = float3x2(1);

  // KHR_texture_transform: per-slot UV set selector (0 = TEXCOORD_0, 1 = TEXCOORD_1)
  int baseColorTexCoord          = 0;
  int metallicRoughnessTexCoord  = 0;
  int normalTexCoord             = 0;
  int emissiveTexCoord           = 0;
  int occlusionTexCoord          = 0;
  int specularTexCoord           = 0;
  int specularColorTexCoord      = 0;
  int clearcoatTexCoord          = 0;
  int clearcoatRoughnessTexCoord = 0;

  // KHR_materials_pbrSpecularGlossiness (legacy): replaces metallic-roughness model
  int    usePbrSpecularGlossiness       = 0;  // 1 = use SG workflow instead of MR
  float4 pbrSgDiffuseFactor             = float4(1.0, 1.0, 1.0, 1.0);
  float3 pbrSgSpecularFactor            = float3(1.0, 1.0, 1.0);
  float  pbrSgGlossinessFactor          = 1.0;
  int    pbrSgDiffuseTexture            = -1;
  int    pbrSgSpecularGlossinessTexture = -1;

  int needShading = 0;  // CPU-computed flag: 1 if material interacts with lighting
};

#ifdef __cplusplus
inline void updateMaterialNeedsShading(Material& mat)
{
  if(mat.usePbrSpecularGlossiness)
  {
    mat.needShading = 1;
    return;
  }
  bool interactsWithLight = glm::length(mat.baseColor) > 0.001f  //
                            || mat.metallic > 0.001f             //
                            || mat.specularFactor > 0.001f       //
                            || mat.clearcoatFactor > 0.001f      //
                            || mat.transmission > 0.001f;
  mat.needShading = interactsWithLight ? 1 : 0;
}
#endif

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
  int       enabled         = 1;                       // 0=disabled, 1=enabled
  int       shadowOnly      = 0;                       // 1 = gs-shadow light: only casts shadow-mask rays onto
                                                       // splat emissive, never contributes illumination (NEE skips it)
  int       castOnGs        = 0;                       // 1 = light illuminates AND also casts onto the GS shadow
                                                       // mask (a lit light whose shadow darkens splat emissive)
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

#endif