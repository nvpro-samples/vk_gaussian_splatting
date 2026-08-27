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

#ifndef _SHADING_
#define _SHADING_

#include "shading.h"
#include "nvshaders/functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"

// ============================================================================
// Environment rotation from Euler angles (degrees)
// ============================================================================

float3 applyEnvRotation(float3 dir, float3 eulerDeg)
{
  float3 r = eulerDeg * (M_PI / 180.0);
  float  cx = cos(r.x), sx = sin(r.x);
  float  cy = cos(r.y), sy = sin(r.y);
  float  cz = cos(r.z), sz = sin(r.z);

  // Rotation order: Y then X then Z (heading / pitch / roll)
  float3 d;
  d.x = dir.x * (cy * cz + sy * sx * sz) + dir.y * (-cy * sz + sy * sx * cz) + dir.z * (sy * cx);
  d.y = dir.x * (cx * sz) + dir.y * (cx * cz) + dir.z * (-sx);
  d.z = dir.x * (-sy * cz + cy * sx * sz) + dir.y * (sy * sz + cy * sx * cz) + dir.z * (cy * cx);
  return d;
}

// ============================================================================
// Material → PbrMaterial conversion
// ============================================================================

PbrMaterial toPbrMaterial(in Material mat, in float3 N, in float3 Ng)
{
  float clampedRoughness = max(mat.roughness, 0.001);
  PbrMaterial pbr = defaultPbrMaterial(mat.baseColor, mat.metallic, clampedRoughness, N, Ng);
  pbr.emissive     = mat.emissive * mat.emissiveStrength;
  pbr.opacity      = mat.opacity;
  pbr.ior2         = mat.ior;
  pbr.transmission = mat.transmission;
  // PbrMaterial no longer has an explicit isThinWalled flag; the BSDF derives it from
  // thickness (isThinWalled == thickness == 0). Preserve the original behavior: treat
  // non-transmissive materials as thin-walled (thickness 0) and transmissive ones as
  // volumetric (non-zero thickness).
  pbr.thickness = (mat.transmission == 0.0) ? 0.0 : 1.0;

  // KHR_materials_specular
  pbr.specular      = mat.specularFactor;
  pbr.specularColor = mat.specularColorFactor;

  // KHR_materials_clearcoat
  pbr.clearcoat          = mat.clearcoatFactor;
  pbr.clearcoatRoughness = max(mat.clearcoatRoughness, 0.001);
  pbr.Nc                 = N;

  return pbr;
}

// ============================================================================
// Material texture application
// ============================================================================

// Select UV set and apply KHR_texture_transform
float2 transformUV(float2 uv0, float2 uv1, int texCoord, float3x2 uvTransform)
{
  float2 uv = (texCoord == 1) ? uv1 : uv0;
  return mul(float3(uv, 1.0), uvTransform).xy;
}

// Apply material textures (base color, metallic-roughness, normal, emissive, occlusion).
// Modifies the Material in-place before lighting.
// uv0/uv1 are TEXCOORD_0 and TEXCOORD_1 from the vertex.
// worldTangent is float4(T.xyz, handedness) from the vertex attribute.
// worldNrm is modified in-place if a normal map is present.
void applyMaterialTextures(inout Material     mat,
                           in Sampler2D       meshTextures[],
                           in float2          uv0,
                           in float2          uv1,
                           inout float3       worldNrm,
                           in float4          worldTangent)
{
  // KHR_materials_pbrSpecularGlossiness: convert SG → MR before standard processing
  if(mat.usePbrSpecularGlossiness != 0)
  {
    float4 diffuse    = mat.pbrSgDiffuseFactor;
    float3 specular   = mat.pbrSgSpecularFactor;
    float  glossiness = mat.pbrSgGlossinessFactor;

    if(mat.pbrSgDiffuseTexture >= 0)
      diffuse *= meshTextures[mat.pbrSgDiffuseTexture].Sample(uv0);

    if(mat.pbrSgSpecularGlossinessTexture >= 0)
    {
      float4 sgTex = meshTextures[mat.pbrSgSpecularGlossinessTexture].Sample(uv0);
      specular  *= sgTex.rgb;
      glossiness *= sgTex.a;
    }

    // SG→MR conversion (matches nvpro_core2 convertSGToMR)
    const float dielectricF0 = 0.04;
    float specularIntensity  = max(specular.r, max(specular.g, specular.b));
    float metallic           = smoothstep(dielectricF0 + 0.01, dielectricF0 + 0.05, specularIntensity);

    float3 baseColor;
    if(metallic > 0.0)
      baseColor = specular;
    else
      baseColor = clamp(diffuse.rgb / max(1.0 - dielectricF0 * (1.0 - metallic), 0.001), 0.0, 1.0);

    float r = 1.0 - glossiness;
    mat.baseColor = baseColor;
    mat.metallic  = metallic;
    mat.roughness = r * r;
    mat.opacity   = diffuse.a;
  }

  // Base color texture — modulates baseColor and opacity
  if(mat.baseColorTexture >= 0)
  {
    float2 uv      = transformUV(uv0, uv1, mat.baseColorTexCoord, mat.baseColorUvTransform);
    float4 texColor = meshTextures[mat.baseColorTexture].Sample(uv);
    mat.baseColor *= texColor.rgb;
    mat.opacity   *= texColor.a;
  }

  // Metallic-roughness texture — glTF packing: G = roughness, B = metallic
  if(mat.metallicRoughnessTexture >= 0)
  {
    float2 uv = transformUV(uv0, uv1, mat.metallicRoughnessTexCoord, mat.metallicRoughnessUvTransform);
    float4 mr = meshTextures[mat.metallicRoughnessTexture].Sample(uv);
    mat.roughness *= mr.g;
    mat.metallic  *= mr.b;
  }

  // Emissive texture — modulates emissive factor
  if(mat.emissiveTexture >= 0)
  {
    float2 uv   = transformUV(uv0, uv1, mat.emissiveTexCoord, mat.emissiveUvTransform);
    float3 emTex = meshTextures[mat.emissiveTexture].Sample(uv).rgb;
    mat.emissive *= emTex;
  }

  // Occlusion texture — R channel modulates base color (diffuse occlusion)
  if(mat.occlusionTexture >= 0)
  {
    float2 uv = transformUV(uv0, uv1, mat.occlusionTexCoord, mat.occlusionUvTransform);
    float  ao = meshTextures[mat.occlusionTexture].Sample(uv).r;
    mat.baseColor *= ao;
  }

  // KHR_materials_specular textures
  if(mat.specularTexture >= 0)
  {
    float2 uv = transformUV(uv0, uv1, mat.specularTexCoord, mat.specularUvTransform);
    mat.specularFactor *= meshTextures[mat.specularTexture].Sample(uv).a;
  }
  if(mat.specularColorTexture >= 0)
  {
    float2 uv = transformUV(uv0, uv1, mat.specularColorTexCoord, mat.specularColorUvTransform);
    mat.specularColorFactor *= meshTextures[mat.specularColorTexture].Sample(uv).rgb;
  }

  // KHR_materials_clearcoat textures (R channel for factor, G channel for roughness)
  if(mat.clearcoatTexture >= 0)
  {
    float2 uv = transformUV(uv0, uv1, mat.clearcoatTexCoord, mat.clearcoatUvTransform);
    mat.clearcoatFactor *= meshTextures[mat.clearcoatTexture].Sample(uv).r;
  }
  if(mat.clearcoatRoughnessTexture >= 0)
  {
    float2 uv = transformUV(uv0, uv1, mat.clearcoatRoughnessTexCoord, mat.clearcoatRoughnessUvTransform);
    mat.clearcoatRoughness *= meshTextures[mat.clearcoatRoughnessTexture].Sample(uv).g;
  }

  // Normal map — uses explicit tangent frame from vertex attributes
  if(mat.normalTexture >= 0)
  {
    float2 uv            = transformUV(uv0, uv1, mat.normalTexCoord, mat.normalUvTransform);
    float3 tangentNormal = meshTextures[mat.normalTexture].Sample(uv).rgb * 2.0 - 1.0;

    float3 N = normalize(worldNrm);
    float3 T = normalize(worldTangent.xyz);
    T = normalize(T - dot(T, N) * N);
    float3 B = cross(N, T) * worldTangent.w;

    worldNrm = normalize(T * tangentNormal.x + B * tangentNormal.y + N * tangentNormal.z);
  }
}

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
// Light radiance at surface (no NdotL — for NEE MIS integration)
// ============================================================================

float3 computeLightRadiance(in LightSource light, in float3 worldPos)
{
  if(light.type == LightType::eDirectionalLight)
    return light.color * light.intensity;

  float3 toLight  = light.position - worldPos;
  float  distance = length(toLight);

  if(distance > light.range)
    return float3(0.0);

  float attenuation = 1.0;
  if(light.attenuationMode == 1)
    attenuation = 1.0 - (distance / light.range);
  else if(light.attenuationMode == 2)
    attenuation = 1.0 / (1.0 + distance * distance);
  else if(light.attenuationMode == 3)
    attenuation = 1.0 / (distance * distance + 0.01);

  float3 radiance = light.color * light.intensity * attenuation;

  if(light.type == LightType::eSpotLight)
  {
    float3 L       = normalize(toLight);
    float3 spotDir = normalize(light.direction);
    float  theta   = dot(L, -spotDir);
    float  innerCos = cos(radians(light.innerConeAngle));
    float  outerCos = cos(radians(light.outerConeAngle));
    if(theta < outerCos)
      return float3(0.0);
    radiance *= smoothstep(outerCos, innerCos, theta);
  }

  return radiance;
}

// ============================================================================
// Shading
// ============================================================================

// transmittance is taken as float3 (not double3): callers either pass
// float3(1.0) for raster paths or downcast their double3 transmittance from
// the path tracer. Keeping this fp32 prevents the Float64 SPIR-V capability
// from being emitted in raster fragment / deferred shading modules that
// would not otherwise need it (e.g. on devices without VkPhysicalDeviceFeatures::shaderFloat64).
void wavefrontComputeShadingDirectOnly(in LightSource light,
                                       in float3      worldPos,
                                       in float3      worldNrm,
                                       in Material    mat,
                                       in float3      viewDir,
                                       in bool        inShadow,
                                       in float3      transmittance,
                                       inout float3   radiance)
{
  if(inShadow)
    return;

  float3 L;
  if(light.type == LightType::eDirectionalLight)
    L = -normalize(light.direction);
  else
    L = normalize(light.position - worldPos);

  float NdotL = max(dot(worldNrm, L), 0.0);
  if(NdotL <= 0.0)
    return;

  PbrMaterial pbr = toPbrMaterial(mat, worldNrm, worldNrm);

  BsdfEvaluateData evalData;
  evalData.k1 = normalize(-viewDir);
  evalData.k2 = L;
  evalData.xi = float3(0.5, 0.5, 0.5);
  bsdfEvaluateSimple(evalData, pbr);

  float3 lightRad = computeLightRadiance(light, worldPos);
  radiance += transmittance * lightRad * (evalData.bsdf_diffuse + evalData.bsdf_glossy);
}

// ============================================================================
// Equirectangular mapping
// ============================================================================

float2 directionToEquirectUV(float3 dir)
{
  static const float kPi    = 3.14159265;
  static const float kInvPi = 1.0 / kPi;
  float theta = atan2(dir.x, dir.z);
  float phi   = acos(clamp(dir.y, -1.0, 1.0));
  float u     = theta * (0.5 * kInvPi) + 0.5;
  float v     = 1.0 - phi * kInvPi;
  return float2(u, v);
}

// HDR images loaded by stbi have row 0 at the top (v=0 = zenith), while
// directionToEquirectUV maps zenith to v=1 (matching the baked sky convention).
float2 directionToHdrEquirectUV(float3 dir)
{
  float2 uv = directionToEquirectUV(dir);
  uv.y = 1.0 - uv.y;
  return uv;
}

// ============================================================================
// Sky radiance sampling — analytical for sun disk, texture for sky
// ============================================================================

float3 sampleSkyRadiance(in float3 dirYUp, in SkyPhysicalParameters skyParams, in Sampler2D skyTex)
{
  float3 dirZUp                    = float3(dirYUp.x, dirYUp.z, dirYUp.y);
  float  sunSampleAngularRadius    = 1.5 * 0.00465 * skyParams.sunDiskScale;
  float  cosSunSampleAngularRadius = cos(sunSampleAngularRadius);

  if(dot(dirZUp, skyParams.sunDirection) >= cosSunSampleAngularRadius)
    return evalPhysicalSky(skyParams, dirZUp);

  return skyTex.SampleLevel(directionToEquirectUV(dirYUp), 0).rgb;
}

// ============================================================================
// One-sample NEE — randomly pick one punctual light OR sky environment.
// Returns pre-weighted radiance/pdf and combined PDF for downstream MIS.
// ============================================================================

struct NEESample
{
  float3 direction;
  float3 radianceOverPdf;
  float  pdf;
  float  distance;
};

NEESample sampleOneLightNEE(in float3                     worldPos,
                            in int                         lightCount,
                            in LightSource*                lights,
                            in bool                        envLighting,
                            in int                         envMode,
                            in SkyPhysicalParameters       skyParams,
                            in Sampler2D                   skyTex,
                            in StructuredBuffer<EnvAccel>  envAccel,
                            in float3                      envRotation,
                            in float                       envIntensity,
                            inout uint                     seed)
{
  NEESample result;
  result.direction       = float3(0.0);
  result.radianceOverPdf = float3(0.0);
  result.pdf             = 0.0;
  result.distance        = 1e10;

  float wLight = (lightCount > 0) ? 0.5 : 0.0;
  float wEnv   = envLighting ? 0.5 : 0.0;
  float total  = wLight + wEnv;
  if(total < 0.001)
    return result;

  float pLight = wLight / total;
  float pEnv   = wEnv / total;

  float r = rand(seed);

  if(r < pLight)
  {
    int         lightIdx = min(int(rand(seed) * float(lightCount)), lightCount - 1);
    LightSource light    = lights[lightIdx];

    // Disabled and shadow-only (gs-shadow) lights contribute no illumination.
    // Returning a zero sample keeps the estimator unbiased (same as enabled==0).
    if(light.enabled == 0 || light.shadowOnly != 0)
      return result;

    float3 lightDir;
    float  lightDist;
    if(!computeLightToSurfaceVector(light, worldPos, seed, lightDir, lightDist))
      return result;

    result.direction = lightDir;
    result.distance  = lightDist;

    float3 lightRad     = computeLightRadiance(light, worldPos);
    float  selectionPdf = pLight / float(lightCount);
    result.radianceOverPdf = lightRad / selectionPdf;
    result.pdf             = DIRAC;
  }
  else
  {
    float3 envDir;
    float3 envRad;
    float  envPdf;

    if(envMode == ENV_MODE_SKY)
    {
      float2            rnd       = float2(rand(seed), rand(seed));
      SkySamplingResult skySample = samplePhysicalSky(skyParams, rnd);
      skySample.direction         = float3(skySample.direction.x, skySample.direction.z, skySample.direction.y);

      if(skySample.pdf <= 0.0)
        return result;

      envDir = skySample.direction;
      envRad = sampleSkyRadiance(skySample.direction, skyParams, skyTex);
      envPdf = skySample.pdf;
    }
    else // ENV_MODE_HDR
    {
      float3 rand3  = float3(rand(seed), rand(seed), rand(seed));
      float3 toLight;
      float4 radPdf = environmentSample(skyTex, envAccel, rand3, toLight);

      if(radPdf.w <= 0.0)
        return result;

      envDir = applyEnvRotation(toLight, envRotation);
      envRad = radPdf.xyz * envIntensity;
      envPdf = radPdf.w;
    }

    result.direction = envDir;
    result.distance  = 1e6;

    float  combinedPdf = pEnv * envPdf;
    float3 radOverPdf  = envRad / combinedPdf;

    static const float kMaxRadOverPdf = 5000.0;
    float radLum = luminance(radOverPdf);
    if(radLum > kMaxRadOverPdf)
      radOverPdf *= kMaxRadOverPdf / radLum;

    result.radianceOverPdf = radOverPdf;
    result.pdf             = combinedPdf;
  }

  return result;
}

#endif 