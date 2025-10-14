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

#ifndef _WAVEFRONT_
#define _WAVEFRONT_

#include "wavefront.h"

float3 wavefrontComputeDiffuse(float3 diffuse, float3 ambient, float3 lightDir, float3 normal)
{
  // Lambertian
  float dotNL = max(dot(normal, lightDir), 0.0);
  float3  c     = diffuse * dotNL;
  //if(mat.illum >= 1)
  c += ambient;
  return c;
}

float3 wavefrontComputeSpecular(float3 ispecular, float shininess, float3 viewDir, float3 lightDir, float3 normal)
{
  //if(mat.illum < 2)
  //  return float3(0);

  const float kPi        = 3.14159265;
  const float kShininess = max(shininess, 4.0);

  // Specular
  const float kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
  float3        V                   = normalize(-viewDir);
  float3        R                   = reflect(-lightDir, normal);
  float       specular            = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

  return float3(ispecular * specular);
}

void wavefrontComputeShadingDirectOnly(StructuredBuffer<LightSource> lights,//in LightSource lights[4],
                                       in int         lightCount,
                                       in float3        worldPos,
                                       in float3        worldNrm,
                                       in ObjMaterial mat,
                                       in float3        viewDir,
                                       inout float3     radiance)
{
  // Material of the object
  float3  matDiffuse    = mat.diffuse;
  float3  matAmbient    = mat.ambient;
  float3  matSpecular   = mat.specular;
  float3  matRefractive = mat.transmittance;
  float ior           = mat.ior;  // 1.0 = pure transparency, 1.5111 window glass
  float matShininess  = mat.shininess;
  int   model         = mat.illum;

  float3 color = float3(0.0, 0.0, 0.0);

  for(int i = 0; i < lightCount; ++i)
  {
    const LightSource light = lights[i];

    // Light source
    const int  lightType      = light.type;
    const float3 lightPosition  = light.position;
    float      lightIntensity = light.intensity;

    // Vector toward the light
    float3 L;

    // Point light
    if(lightType == 0)
    {
      const float3  lDir = lightPosition - worldPos;
      const float d    = length(lDir);
      lightIntensity   = lightIntensity / (d * d);
      L                = normalize(lDir);
    }
    else  // Directional light
    {
      L = normalize(lightPosition);
    }

    // Diffuse
    const float3 fragDiffuse = wavefrontComputeDiffuse(matDiffuse, matAmbient, L, worldNrm);

    // Specular
    const float3 fragSpecular = wavefrontComputeSpecular(matSpecular, matShininess, viewDir, L, worldNrm);

    // Result
    radiance += (fragDiffuse + fragSpecular) * lightIntensity;
  }
}

void wavefrontComputeShading(StructuredBuffer<LightSource> lights, //in LightSource lights[4],
                             in int         lightCount,
                             in float3        worldPos,
                             in float3        worldNrm,
                             in ObjMaterial mat,
                             in float3        worldRayDir,
                             inout double3    transmittance,
                             inout float3     radiance,
                             inout int      done,
                             inout float3     rayOrigin,
                             inout float3     rayDir)
{

  float3  matDiffuse    = mat.diffuse;
  float3  matAmbient    = mat.ambient;
  float3  matSpecular   = mat.specular;
  float3  matRefractive = mat.transmittance;
  float ior           = mat.ior;  // 1 = pure transparency, 1.5111 window glass
  float matShininess  = mat.shininess;
  int   model         = mat.illum;

  for(int i = 0; i < lightCount; ++i)
  {
    const LightSource light = lights[i];

    // Vector toward the light
    float3 L;

    // Light source
    const int  lightType      = light.type;
    const float3 lightPosition  = light.position;
    float      lightIntensity = light.intensity;

    // Point light
    if(lightType == 0)
    {
      float3  lDir     = lightPosition - worldPos;
      float d        = length(lDir);
      lightIntensity = lightIntensity / (d * d);
      L              = normalize(lDir);
    }
    else  // Directional light
    {
      L = normalize(lightPosition);
    }

    // Diffuse
    float3 fragDiffuse = wavefrontComputeDiffuse(matDiffuse, matAmbient, L, worldNrm);

    // Specular
    float3 fragSpecular = wavefrontComputeSpecular(matSpecular, matShininess, worldRayDir, L, worldNrm);

    // Result
    radiance += float3(transmittance) * float3(lightIntensity * (fragDiffuse + fragSpecular));
  }

  //
  if(model <= 0)  //
  {
    transmittance = double3(0.0, 0.0, 0.0);
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
    float3  N   = worldNrm;

    if(dot(worldRayDir, worldNrm) > 0.0)
    {
      // Ray is inside the object: flip normal and eta
      N   = -worldNrm;
      eta = ior;
    }

    float3 refractedDir = refract(worldRayDir, N, eta);
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