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

#ifndef _WAVEFRONT_
#define _WAVEFRONT_

#include "wavefront.h"

// ============================================================================
// Lighting
// ============================================================================

// Compute light-to-surface vector (direction and distance), check if light is in range
// Returns false if light is culled (out of range), true if light should be processed
// Output: lightDir = direction TO light (for shadow rays), lightDist = distance to light
// For directional lights: lightDist is set to a large value (1e10)
bool computeLightToSurfaceVector(in LightSource light, in float3 worldPos, inout uint seed, out float3 lightDir, out float lightDist)
{
  if(light.type == LightType::eDirectionalLight)
  {
    // Directional: infinite range, direction is opposite of light.direction
    lightDir  = -normalize(light.direction);
    lightDist = 1e10;  // Large value for directional lights
    return true;       // Never culled
  }

  // Point / Spot lights
  float3 lightSamplePos = light.position;

#if SHADOWS_MODE == SHADOWS_SOFT
  // Sample a random point on a disk centered at the light position
  // The disk is perpendicular to the direction from world pos to light
  if(light.radius > 0.0f)
  {
    float3 toLight = normalize(light.position - worldPos);
    float3 tangent, bitangent;
    buildOrthonormalBasis(toLight, tangent, bitangent);
    lightSamplePos += sampleDisk(seed, light.radius * 0.5f, tangent, bitangent);
  }
#endif

  float3 toLight = lightSamplePos - worldPos;
  lightDist      = length(toLight);

  // Cull if beyond light range
  if(lightDist > light.range)
  {
    lightDir = float3(0.0);  // Initialize for safety
    return false;
  }

  lightDir = normalize(toLight);
  return true;
}

// Soft shadow disk sampling enabled when SHADOWS_MODE == SHADOWS_SOFT

#if SHADOWS_MODE == SHADOWS_SOFT
#include "nvshaders/random.h.slang"

// Build orthonormal basis from a single direction vector (Frisvad method)
void buildOrthonormalBasis(in float3 n, out float3 tangent, out float3 bitangent)
{
  if(n.z < -0.9999999f)
  {
    tangent   = float3(0.0f, -1.0f, 0.0f);
    bitangent = float3(-1.0f, 0.0f, 0.0f);
  }
  else
  {
    float a   = 1.0f / (1.0f + n.z);
    float b   = -n.x * n.y * a;
    tangent   = float3(1.0f - n.x * n.x * a, b, -n.x);
    bitangent = float3(b, 1.0f - n.y * n.y * a, -n.y);
  }
}

// Sample a point on a disk of given radius, centered at origin in the tangent plane
float3 sampleDisk(inout uint seed, float radius, float3 tangent, float3 bitangent)
{
  // Uniform disk sampling using rejection method or polar coordinates
  float r     = radius * sqrt(rand(seed));  // Square root for uniform area distribution
  float theta = 2.0f * 3.14159265f * rand(seed);
  return r * (cos(theta) * tangent + sin(theta) * bitangent);
}
#endif

// Create a default headlight at camera position (no attenuation, intensity 1.0)
// This provides basic illumination when no lights are present in the scene
LightSource createHeadlight(in float3 cameraPos)
{
  LightSource headlight;
  headlight.type            = LightType::ePointLight;
  headlight.color           = float3(1.0, 1.0, 1.0);
  headlight.intensity       = 1.0;
  headlight.position        = cameraPos;
  headlight.range           = 1e10;                    // Effectively infinite range
  headlight.attenuationMode = 0;                       // No attenuation
  headlight.direction       = float3(0.0, 0.0, -1.0);  // Not used for point light
  headlight.innerConeAngle  = 0.0;                     // Not used for point light
  headlight.outerConeAngle  = 0.0;                     // Not used for point light
  return headlight;
}

// Main light calculation dispatcher
float3 computeLight(in LightSource light, in float3 worldPos, in float3 worldNrm)
{
  switch(light.type)
  {
    case LightType::eDirectionalLight:
      return computeDirectionalLight(light, worldPos, worldNrm);
    case LightType::ePointLight:
      return computePointLight(light, worldPos, worldNrm);
    case LightType::eSpotLight:
      return computeSpotLight(light, worldPos, worldNrm);
    default:
      return float3(0.0);
  }
}

// Calculate directional light contribution
float3 computeDirectionalLight(in LightSource light, in float3 worldPos, in float3 worldNrm)
{
  // Directional lights use the direction field (computed from rotation)
  float3 L = -normalize(light.direction);  // Direction TO light (opposite of light direction)

  // Simple diffuse lighting (no attenuation for directional lights)
  float NdotL = max(dot(worldNrm, L), 0.0);

  return light.color * light.intensity * NdotL;
}

// Calculate point light contribution
float3 computePointLight(in LightSource light, in float3 worldPos, in float3 worldNrm)
{
  float3 toLight  = light.position - worldPos;
  float  distance = length(toLight);
  float3 L        = normalize(toLight);

  // Range check
  if(distance > light.range)
    return float3(0.0);

  // Attenuation based on mode
  float attenuation = 1.0;
  if(light.attenuationMode == 1)  // Linear
  {
    attenuation = 1.0 - (distance / light.range);
  }
  else if(light.attenuationMode == 2)  // Quadratic
  {
    attenuation = 1.0 / (1.0 + distance * distance);
  }
  else if(light.attenuationMode == 3)  // Physical
  {
    attenuation = 1.0 / (distance * distance + 0.01);
  }
  // Mode 0 (None) keeps attenuation = 1.0

  // Diffuse lighting
  float NdotL = max(dot(worldNrm, L), 0.0);

  return light.color * light.intensity * attenuation * NdotL;
}

// Calculate spot light contribution
float3 computeSpotLight(in LightSource light, in float3 worldPos, in float3 worldNrm)
{
  float3 toLight  = light.position - worldPos;
  float  distance = length(toLight);
  float3 L        = normalize(toLight);

  // Range check
  if(distance > light.range)
    return float3(0.0);

  // Distance attenuation (same as point light)
  float attenuation = 1.0;
  if(light.attenuationMode == 1)  // Linear
  {
    attenuation = 1.0 - (distance / light.range);
  }
  else if(light.attenuationMode == 2)  // Quadratic
  {
    attenuation = 1.0 / (1.0 + distance * distance);
  }
  else if(light.attenuationMode == 3)  // Physical
  {
    attenuation = 1.0 / (distance * distance + 0.01);
  }

  // Spot cone attenuation
  float3 spotDir = normalize(light.direction);
  float  theta   = dot(L, -spotDir);  // Angle between L and spot direction

  // Convert angles from degrees to cosines
  float innerCos = cos(radians(light.innerConeAngle));
  float outerCos = cos(radians(light.outerConeAngle));

  // Outside outer cone?
  if(theta < outerCos)
    return float3(0.0);

  // Smooth falloff between inner and outer cones
  float spotEffect = smoothstep(outerCos, innerCos, theta);

  // Diffuse lighting
  float NdotL = max(dot(worldNrm, L), 0.0);

  return light.color * light.intensity * attenuation * spotEffect * NdotL;
}

// ============================================================================
// Shading
// ============================================================================

void wavefrontComputeShadingDirectOnly(in LightSource light,
                                       in float3      worldPos,
                                       in float3      worldNrm,
                                       in ObjMaterial mat,
                                       in float3      viewDir,
                                       in bool        inShadow,
                                       in double3     transmittance,
                                       inout float3   radiance)
{
  // Material of the object
  float3 matDiffuse   = mat.diffuse;
  float3 matAmbient   = mat.ambient;
  float3 matSpecular  = mat.specular;
  float  matShininess = mat.shininess;

  // Always add ambient (not affected by shadows)
  radiance += float3(transmittance) * matAmbient;

  // Skip directional lighting if in shadow
  if(inShadow)
    return;

  // Use the unified light calculation for diffuse contribution
  float3 lightDiffuse = computeLight(light, worldPos, worldNrm);

  // Modulate by material diffuse
  float3 fragDiffuse = matDiffuse * lightDiffuse;

  // Compute L vector for specular calculation
  float3 L;
  if(light.type == LightType::eDirectionalLight)
  {
    L = -normalize(light.direction);
  }
  else  // Point or Spot
  {
    L = normalize(light.position - worldPos);
  }

  // Specular
  const float3 fragSpecular = wavefrontComputeSpecular(matSpecular, matShininess, viewDir, L, worldNrm);

  // Apply specular with light color and intensity
  float3 specularContribution = fragSpecular * light.color * light.intensity;

  // Result
  radiance += float3(transmittance) * (fragDiffuse + specularContribution);
}

void wavefrontComputeShading(in LightSource light,
                             in float3      worldPos,
                             in float3      worldNrm,
                             in ObjMaterial mat,
                             in float3      worldRayDir,
                             in bool        inShadow,
                             inout double3  transmittance,
                             inout float3   radiance,
                             inout int      done,
                             inout float3   rayOrigin,
                             inout float3   rayDir)
{
  float3 matDiffuse    = mat.diffuse;
  float3 matAmbient    = mat.ambient;
  float3 matSpecular   = mat.specular;
  float3 matRefractive = mat.transmittance;
  float  ior           = mat.ior;  // 1 = pure transparency, 1.5111 window glass
  float  matShininess  = mat.shininess;
  int    model         = mat.illum;

  // Always add ambient contribution (not affected by shadows)
  radiance += float3(transmittance) * matAmbient;

  // Skip directional lighting if in shadow
  if(!inShadow)
  {
    // Use the unified light calculation for diffuse contribution
    float3 lightDiffuse = computeLight(light, worldPos, worldNrm);

    // Modulate by material diffuse
    float3 fragDiffuse = matDiffuse * lightDiffuse;

    // Compute L vector for specular calculation
    float3 L;
    if(light.type == LightType::eDirectionalLight)
    {
      L = -normalize(light.direction);
    }
    else  // Point or Spot
    {
      L = normalize(light.position - worldPos);
    }

    // Specular
    float3 fragSpecular = wavefrontComputeSpecular(matSpecular, matShininess, worldRayDir, L, worldNrm);

    // Apply specular with light color and intensity
    fragSpecular *= light.color * light.intensity;

    // Result
    radiance += float3(transmittance) * (fragDiffuse + fragSpecular);
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
    float  eta = 1.0 / ior;
    float3 N   = worldNrm;

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

float3 wavefrontComputeDiffuse(float3 diffuse, float3 ambient, float3 lightDir, float3 normal)
{
  // Lambertian
  float  dotNL = max(dot(normal, lightDir), 0.0);
  float3 c     = diffuse * dotNL;
  // if(mat.illum >= 1)
  c += ambient;
  return c;
}

float3 wavefrontComputeSpecular(float3 ispecular, float shininess, float3 viewDir, float3 lightDir, float3 normal)
{
  // if(mat.illum < 2)
  //   return float3(0);

  const float kPi        = 3.14159265;
  const float kShininess = max(shininess, 4.0);

  // Specular
  const float kEnergyConservation = (2.0 + kShininess) / (2.0 * kPi);
  float3      V                   = normalize(-viewDir);
  float3      R                   = reflect(-lightDir, normal);
  float       specular            = kEnergyConservation * pow(max(dot(V, R), 0.0), kShininess);

  return float3(ispecular * specular);
}

#endif 