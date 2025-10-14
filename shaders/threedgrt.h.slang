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

#ifndef _THREEDGRT_H_
#define _THREEDGRT_H_

#include "quaternions.h.slang"
#include "shaderio.h"
#include "threedgs_particles_storage.h.slang"

// fetch the position, scale and rotation
void fetchParticlePSR(in int splatIndex, out float3 particlePosition, out float3 particleScale, out float3x3 particleInvRotation)
{
  particlePosition = fetchCenter(splatIndex);
  particleScale    = exp(fetchScale(splatIndex));
  const float4 wxyzQuat  = normalize(fetchRotation(splatIndex));
  // vec4toQuat converts to scalar (w) last internal
  // transpose the pure rotation matrix to get its inverse
  particleInvRotation = quatToMat3Transpose(vec4toQuat(wxyzQuat));
}

void particleCannonicalRay(in float3  rayOrigin,
                           in float3  rayDirection,
                           in float3  particlePosition,
                           in float3  particleScale,
                           in float3x3  particleInvRotation,
                           out float3 particleRayOrigin,
                           out float3 particleRayDirection)
{
  const float3 giscl  = float3(1.0f) / particleScale;
  const float3 gposc  = (rayOrigin - particlePosition);
  // Row-major: vector * matrix
  const float3 gposcr = mul(gposc, particleInvRotation);
  particleRayOrigin = giscl * gposcr;

  // Row-major: vector * matrix
  const float3 rayDirR   = mul(rayDirection, particleInvRotation);
  const float3 grdu      = giscl * rayDirR;
  particleRayDirection = normalize(grdu);
}

float particleRayMinSquaredDistance(in float3 particleRayOrigin, in float3 particleRayDirection)
{
  const float3 gcrod = cross(particleRayDirection, particleRayOrigin);
  return dot(gcrod, gcrod);
}

float particleRayMaxKernelResponse(in float3 particleRayOrigin, in float3 particleRayDirection, in int32_t particleKernelDegree)
{
  const float grayDist = particleRayMinSquaredDistance(particleRayOrigin, particleRayDirection);

  /// generalized gaussian of degree n : scaling is s = -4.5/3^n
  switch(particleKernelDegree)
  {
    case 8:  // Zenzizenzizenzic
    {
      const float s          = -0.000685871056241;
      const float grayDistSq = grayDist * grayDist;
      return exp(s * grayDistSq * grayDistSq);
    }
    case 5:  // Quintic
    {
      const float s = -0.0185185185185;
      return exp(s * grayDist * grayDist * sqrt(grayDist));
    }
    case 4:  // Tesseractic
    {
      const float s = -0.0555555555556;
      return exp(s * grayDist * grayDist);
    }
    case 3:  // Cubic
    {
      const float s = -0.166666666667;
      return exp(s * grayDist * sqrt(grayDist));
    }
    case 1:  // Laplacian
    {
      const float s = -1.5f;
      return exp(s * sqrt(grayDist));
    }
    case 0:  // Linear
    {
      const float s = -0.329630334487;
      return max(1 + s * sqrt(grayDist), 0.f);
    }
    default:  // Quadratic
    {
      const float s = -0.5f;
      return exp(s * grayDist);
    }
  }
}

// distance to the gaussian center projection on the ray
float particleRayDistance(in float3 particleRayOrigin, in float3 particleRayDirection, in float3 particleScale)
{
  const float3 grds = particleScale * particleRayDirection * dot(particleRayDirection, particleRayOrigin * float3(-1.0));
  return sqrt(dot(grds, grds));
}

// Version with that fetches color used by 3DGRT
bool particleProcessHit(in FrameInfo frameInfo,
                        in float3      modelRayOrigin,
                        in float3      modelRayDirection,
                        in int       splatId,
                        inout double3  transmittance,
                        inout float3   radiance)
{
  //
  float3 particlePosition;
  float3 particleScale;
  float3x3 particleInvRotation;

  fetchParticlePSR(splatId, particlePosition, particleScale, particleInvRotation);

  float3 particleRayOrigin;
  float3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const float4  color           = fetchColor(splatId);
  const float particleDensity = color.w;

  const double  minParticleAlpha         = 1.0f / 255.0f;
  const int32_t particleKernelDegree     = KERNEL_DEGREE;
  const float   minParticleKernelDensity = KERNEL_MIN_RESPONSE;

  const float maxResponse = particleRayMaxKernelResponse(particleRayOrigin, particleRayDirection, particleKernelDegree);
  float       alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const bool acceptHit = (particleDensity > frameInfo.alphaCullThreshold) && (double(alpha) > minParticleAlpha)
                         && (maxResponse > minParticleKernelDensity);
  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    alpha = 1.0;
#endif
    // fetch radiance from SH coefs > degree 0
    const float3 vectorToParticleCenter = normalize(particlePosition - modelRayOrigin);
#if SHOW_SH_ONLY == 1
    const float3 grad = float3(0.5) + fetchViewDependentRadiance(splatId, vectorToParticleCenter);
#else
    const float3 grad = color.xyz + fetchViewDependentRadiance(splatId, vectorToParticleCenter);
#endif

    const float3 weight = float3(alpha * float3(transmittance));
    radiance += grad * weight;
    transmittance *= (1.0 - double(alpha));
  }

  return acceptHit;
}

// Version with pre fetched particle PSR and color. Used by 3DGUT.
bool particleProcessHitGut(in FrameInfo frameInfo,
                           in float3      modelRayOrigin,
                           in float3      modelRayDirection,
                           in int       splatId,
                           in float4      splatColor,
                           in float3      particlePosition,
                           in float3      particleScale,
                           in quat      particleRotation,  // normalized quaternion
                           out float    opacity)
{
  // attention, here the particleRotation is already a quaternion expressed with scalar last 
  const float3x3 particleInvRotation = quatToMat3Transpose(particleRotation);

  float3 particleRayOrigin;
  float3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const float   particleDensity          = splatColor.w;
  const double  minParticleAlpha         = 1.0f / 255.0f;
  const int32_t particleKernelDegree     = KERNEL_DEGREE;
  const float   minParticleKernelDensity = KERNEL_MIN_RESPONSE;

  const float maxResponse = particleRayMaxKernelResponse(particleRayOrigin, particleRayDirection, particleKernelDegree);
  const float alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const bool acceptHit = (particleDensity > frameInfo.alphaCullThreshold) && ((double)alpha > minParticleAlpha)
                         && (maxResponse > minParticleKernelDensity);
  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    opacity = 1.0;
#else
    opacity = alpha;
#endif
  }

  return acceptHit;
}

#endif