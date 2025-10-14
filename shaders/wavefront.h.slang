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

#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_GOOGLE_include_directive : require

#ifndef _WAVEFRONT_
#define _WAVEFRONT_

#include "wavefront.h"

vec3 wavefrontComputeDiffuse(vec3 diffuse, vec3 ambient, vec3 lightDir, vec3 normal)
{
  // Lambertian
  float dotNL = max(dot(normal, lightDir), 0.0);
  vec3  c     = diffuse * dotNL;
  //if(mat.illum >= 1)
  c += ambient;
  return c;
}

vec3 wavefrontComputeSpecular(vec3 ispecular, float shininess, vec3 viewDir, vec3 lightDir, vec3 normal)
{
  //if(mat.illum < 2)
  //  return vec3(0);

  const float kPi        = 3.14159265;
  const float kShininess = max(shininess, 4.0);

  // Specular
  const float kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
  vec3        V                   = normalize(-viewDir);
  vec3        R                   = reflect(-lightDir, normal);
  float       specular            = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

  return vec3(ispecular * specular);
}

void wavefrontComputeShadingDirectOnly(in LightSource lights[4],
                                       in int         lightCount,
                                       in vec3        worldPos,
                                       in vec3        worldNrm,
                                       in ObjMaterial mat,
                                       in vec3        viewDir,
                                       inout vec3     radiance)
{
  // Material of the object
  vec3  matDiffuse    = mat.diffuse;
  vec3  matAmbient    = mat.ambient;
  vec3  matSpecular   = mat.specular;
  vec3  matRefractive = mat.transmittance;
  float ior           = mat.ior;  // 1.0 = pure transparency, 1.5111 window glass
  float matShininess  = mat.shininess;
  int   model         = mat.illum;

  vec3 color = vec3(0.0);

  for(int i = 0; i < lightCount; ++i)
  {
    const LightSource light = lights[i];

    // Light source
    const int  lightType      = light.type;
    const vec3 lightPosition  = light.position;
    float      lightIntensity = light.intensity;

    // Vector toward the light
    vec3 L;

    // Point light
    if(lightType == 0)
    {
      const vec3  lDir = lightPosition - worldPos;
      const float d    = length(lDir);
      lightIntensity   = lightIntensity / (d * d);
      L                = normalize(lDir);
    }
    else  // Directional light
    {
      L = normalize(lightPosition);
    }

    // Diffuse
    const vec3 fragDiffuse = wavefrontComputeDiffuse(matDiffuse, matAmbient, L, worldNrm);

    // Specular
    const vec3 fragSpecular = wavefrontComputeSpecular(matSpecular, matShininess, viewDir, L, worldNrm);

    // Result
    radiance += (fragDiffuse + fragSpecular) * lightIntensity;
  }
}

void wavefrontComputeShading(in LightSource lights[4],
                             in int         lightCount,
                             in vec3        worldPos,
                             in vec3        worldNrm,
                             in ObjMaterial mat,
                             in vec3        worldRayDir,
                             inout dvec3    transmittance,
                             inout vec3     radiance,
                             inout int      done,
                             inout vec3     rayOrigin,
                             inout vec3     rayDir)
{

  vec3  matDiffuse    = mat.diffuse;
  vec3  matAmbient    = mat.ambient;
  vec3  matSpecular   = mat.specular;
  vec3  matRefractive = mat.transmittance;
  float ior           = mat.ior;  // 1 = pure transparency, 1.5111 window glass
  float matShininess  = mat.shininess;
  int   model         = mat.illum;

  for(int i = 0; i < lightCount; ++i)
  {
    const LightSource light = lights[i];

    // Vector toward the light
    vec3 L;

    // Light source
    const int  lightType      = light.type;
    const vec3 lightPosition  = light.position;
    float      lightIntensity = light.intensity;

    // Point light
    if(lightType == 0)
    {
      vec3  lDir     = lightPosition - worldPos;
      float d        = length(lDir);
      lightIntensity = lightIntensity / (d * d);
      L              = normalize(lDir);
    }
    else  // Directional light
    {
      L = normalize(lightPosition);
    }

    // Diffuse
    vec3 fragDiffuse = wavefrontComputeDiffuse(matDiffuse, matAmbient, L, worldNrm);

    // Specular
    vec3 fragSpecular = wavefrontComputeSpecular(matSpecular, matShininess, worldRayDir, L, worldNrm);

    // Result
    radiance += vec3(transmittance) * vec3(lightIntensity * (fragDiffuse + fragSpecular));
  }

  //
  if(model <= 0)  //
  {
    transmittance = dvec3(0.0);
  }
  else if(model == 1)  // reflection
  {
    transmittance *= matSpecular;
    done      = 0;  // set to 0 means continue to trace at next iteration
    rayOrigin = worldPos;
    rayDir    = reflect(worldRayDir, worldNrm);
  }
  else if(model >= 2)
  {
    transmittance *= matRefractive;
    done      = 0;  // set to 0 means continue to trace at next iteration
    rayOrigin = worldPos;

    // Determine if we're entering or exiting the surface
    float eta = 1.0 / ior;
    vec3  N   = worldNrm;

    if(dot(worldRayDir, worldNrm) > 0.0)
    {
      // Ray is inside the object: flip normal and eta
      N   = -worldNrm;
      eta = ior;
    }

    vec3 refractedDir = refract(worldRayDir, N, eta);
    if(length(refractedDir) > 0.0)
    {
      // Refraction succeeded
      rayDir = normalize(refractedDir);
    }
    else
    {
      // Total internal reflection fallback
      rayDir = reflect(worldRayDir, N);
    }
  }
}

#endif