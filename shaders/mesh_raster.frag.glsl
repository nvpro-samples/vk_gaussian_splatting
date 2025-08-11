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

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "shaderio.h"
#include "wavefront.glsl"

layout(push_constant) uniform _PushConstantRaster
{
  PushConstant pcRaster;
};

// clang-format off
// Incoming 
layout(location = 1) in vec3 i_worldPos;
layout(location = 2) in vec3 i_worldNrm;
layout(location = 3) in vec3 i_viewDir;
// Outgoing
layout(location = 0) out vec4 o_color;

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform _frameInfo
{
  FrameInfo frameInfo;
};

layout(set = 0, binding = BINDING_MESH_DESCRIPTORS, scalar) buffer ObjDesc_
{
  ObjDesc i[];
}
objDesc;

layout(set = 0, binding = BINDING_LIGHT_SET, scalar) buffer LightSet_
{
  LightSource l[];
}
lights;

layout(buffer_reference, scalar) buffer Materials  // Materials
{
  ObjMaterial m[];
};

layout(buffer_reference, scalar) buffer MaterialIndices  // Material indices per face
{
  uint i[];
};
// clang-format on

void main()
{

#if HYBRID_ENABLED
  o_color = vec4(0.0, 0.0, 0.0, 0.0);
#else

  ObjDesc         objResource = objDesc.i[pcRaster.objIndex];
  Materials       materials   = Materials(objResource.materialAddress);
  MaterialIndices matIndices  = MaterialIndices(objResource.materialIndexAddress);
  ObjMaterial     mat         = materials.m[matIndices.i[pcRaster.objIndex]];

  // Material of the object
  vec3  matDiffuse    = mat.diffuse;
  vec3  matAmbient    = mat.ambient;
  vec3  matSpecular   = mat.specular;
  vec3  matRefractive = mat.transmittance;
  float ior           = mat.ior;  // 1.0 = pure transparency, 1.5111 window glass
  float matShininess  = mat.shininess;
  int   model         = mat.illum;

  vec3 color = vec3(0.0);

  for(int i = 0; i < frameInfo.lightCount; ++i)
  {
    const LightSource light = lights.l[i];

    // Light source
    const int  lightType      = light.type;
    const vec3 lightPosition  = light.position;
    float      lightIntensity = light.intensity;

    // Vector toward the light
    vec3 L;

    // Point light
    if(lightType == 0)
    {
      const vec3  lDir = lightPosition - i_worldPos;
      const float d    = length(lDir);
      lightIntensity   = lightIntensity / (d * d);
      L                = normalize(lDir);
    }
    else  // Directional light
    {
      L = normalize(lightPosition);
    }

    // Diffuse
    const vec3 fragDiffuse = computeDiffuse(matDiffuse, matAmbient, L, i_worldNrm);

    // Specular
    const vec3 fragSpecular = computeSpecular(matSpecular, matShininess, i_viewDir, L, i_worldNrm);

    // Result
    color += (fragDiffuse + fragSpecular) * lightIntensity;
  }

  o_color = vec4(color, 1);

#endif
}
