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

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require  // for [[unroll]]
#extension GL_ARB_shader_clock : require  // profiling timers
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_GOOGLE_include_directive : require

#ifndef _THREEDGRT_H_
#define _THREEDGRT_H_

#include "quaternions.glsl"
#include "shaderio.h"

#include "threedgs_particles_storage.glsl"

// fetch the position, scale and rotation
void fetchParticlePSR(in int splatIndex, out vec3 particlePosition, out vec3 particleScale, out mat3 particleInvRotation)
{
  particlePosition = fetchCenter(splatIndex);
  particleScale    = exp(fetchScale(splatIndex));
  const vec4 wxyzQuat  = normalize(fetchRotation(splatIndex));
  // vec4toQuat converts to scalar (w) last internal
  // transpose the pure rotation matrix to get its inverse
  particleInvRotation = quatToMat3Transpose(vec4toQuat(wxyzQuat));
}

void particleCannonicalRay(in vec3  rayOrigin,
                           in vec3  rayDirection,
                           in vec3  particlePosition,
                           in vec3  particleScale,
                           in mat3  particleInvRotation,
                           out vec3 particleRayOrigin,
                           out vec3 particleRayDirection)
{
  const vec3 giscl  = vec3(1.0f) / particleScale;
  const vec3 gposc  = (rayOrigin - particlePosition);
  const vec3 gposcr = particleInvRotation * gposc;
  particleRayOrigin = giscl * gposcr;

  const vec3 rayDirR   = particleInvRotation * rayDirection;
  const vec3 grdu      = giscl * rayDirR;
  particleRayDirection = normalize(grdu);
}

float particleRayMinSquaredDistance(in vec3 particleRayOrigin, in vec3 particleRayDirection)
{
  const vec3 gcrod = cross(particleRayDirection, particleRayOrigin);
  return dot(gcrod, gcrod);
}

float particleRayMaxKernelResponse(in vec3 particleRayOrigin, in vec3 particleRayDirection, in int32_t particleKernelDegree)
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
float particleRayDistance(in vec3 particleRayOrigin, in vec3 particleRayDirection, in vec3 particleScale)
{
  const vec3 grds = particleScale * particleRayDirection * dot(particleRayDirection, particleRayOrigin * vec3(-1.0));
  return sqrt(dot(grds, grds));
}

// Version with that fetches color used by 3DGRT
bool particleProcessHit(in FrameInfo frameInfo,
                        in vec3      modelRayOrigin,
                        in vec3      modelRayDirection,
                        in int       splatId,
                        inout dvec3  transmittance,
                        inout vec3   radiance)
{
  //
  vec3 particlePosition;
  vec3 particleScale;
  mat3 particleInvRotation;

  fetchParticlePSR(splatId, particlePosition, particleScale, particleInvRotation);

  vec3 particleRayOrigin;
  vec3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const vec4  color           = fetchColor(splatId);
  const float particleDensity = color.w;

  const double  minParticleAlpha         = 1.0f / 255.0f;
  const int32_t particleKernelDegree     = KERNEL_DEGREE;
  const float   minParticleKernelDensity = KERNEL_MIN_RESPONSE;

  const float maxResponse = particleRayMaxKernelResponse(particleRayOrigin, particleRayDirection, particleKernelDegree);
  float       alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const bool acceptHit = (particleDensity > frameInfo.alphaCullThreshold) && (alpha > minParticleAlpha)
                         && (maxResponse > minParticleKernelDensity);
  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    alpha = 1.0;
#endif
    // fetch radiance from SH coefs > degree 0
    const vec3 vectorToParticleCenter = normalize(particlePosition - modelRayOrigin);
#if SHOW_SH_ONLY == 1
    const vec3 grad = vec3(0.5) + fetchViewDependentRadiance(splatId, vectorToParticleCenter);
#else
    const vec3 grad = color.xyz + fetchViewDependentRadiance(splatId, vectorToParticleCenter);
#endif

    const vec3 weight = vec3(alpha * transmittance);
    radiance += grad * weight;
    transmittance *= (1.0 - alpha);
  }

  return acceptHit;
}

// Version with pre fetched particle PSR and color. Used by 3DGUT.
bool particleProcessHitGut(in FrameInfo frameInfo,
                           in vec3      modelRayOrigin,
                           in vec3      modelRayDirection,
                           in int       splatId,
                           in vec4      splatColor,
                           in vec3      particlePosition,
                           in vec3      particleScale,
                           in quat      particleRotation,  // normalized quaternion
                           out float    opacity)
{
  // attention, here the particleRotation is already a quaternion expressed with scalar last 
  const mat3 particleInvRotation = quatToMat3Transpose(particleRotation);

  vec3 particleRayOrigin;
  vec3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const float   particleDensity          = splatColor.w;
  const double  minParticleAlpha         = 1.0f / 255.0f;
  const int32_t particleKernelDegree     = KERNEL_DEGREE;
  const float   minParticleKernelDensity = KERNEL_MIN_RESPONSE;

  const float maxResponse = particleRayMaxKernelResponse(particleRayOrigin, particleRayDirection, particleKernelDegree);
  const float alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const bool acceptHit = (particleDensity > frameInfo.alphaCullThreshold) && (alpha > minParticleAlpha)
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