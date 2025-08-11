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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "shaderio.h"
#include "common.glsl"
#include "gs_particle.glsl"

layout(push_constant) uniform _PushConstantRay
{
  PushConstantRay pcRay;
};

bool particleDensityHitCustom(in vec3   rayOrigin,
                              in vec3   rayDirection,
                              in int    particleId,
                              in float  minHitDistance,
                              in float  maxHitDistance,
                              in float  maxParticleSquaredDistance,
                              out float hitDistance)
{
  vec3 particlePosition;
  vec3 particleScale;
  mat3 particleInvRotation;

  fetchParticlePSR(particleId, particlePosition, particleScale, particleInvRotation);

  vec3 canonicalRayOrigin;
  vec3 canonicalRayDirection;
  particleCannonicalRay(rayOrigin, rayDirection, particlePosition, particleScale, particleInvRotation,
                        canonicalRayOrigin, canonicalRayDirection);

  hitDistance = particleRayDistance(canonicalRayOrigin, canonicalRayDirection, particleScale);

  return (hitDistance > minHitDistance) && (hitDistance < maxHitDistance)
         && (particleRayMinSquaredDistance(canonicalRayOrigin, canonicalRayDirection) < maxParticleSquaredDistance);
}

bool particleDensityHitInstance(in vec3   canonicalRayOrigin,
                                in vec3   canonicalUnormalizedRayDirection,
                                in float  minHitDistance,
                                in float  maxHitDistance,
                                in float  maxParticleSquaredDistance,
                                out float hitDistance)
{
  const float numerator   = -dot(canonicalRayOrigin, canonicalUnormalizedRayDirection);
  const float denominator = dot(canonicalUnormalizedRayDirection, canonicalUnormalizedRayDirection);
  hitDistance             = numerator / denominator;
  return (hitDistance > minHitDistance) && (hitDistance < maxHitDistance)
         && (particleRayMinSquaredDistance(canonicalRayOrigin, normalize(canonicalUnormalizedRayDirection)) < maxParticleSquaredDistance);
}

void main()
{
  float hitT;
#if RTX_USE_INSTANCES
  // allways testing the same unit particle, no data fetch inside Hit test function
  if(particleDensityHitInstance(gl_ObjectRayOriginEXT, gl_ObjectRayDirectionEXT, gl_RayTminEXT, gl_RayTmaxEXT, 9.0F, hitT))
    reportIntersectionEXT(hitT, 0);
#else
  const int particleId = gl_PrimitiveID;

  const vec3 modelRayOrigin = vec3(pcRay.modelMatrixInverse * vec4(gl_WorldRayOriginEXT, 1.0));
  // modelMatrixTranspose is equivalent to inverse(transpose(modelMatrixInverse))
  const vec3 modelRayDirection = normalize(vec3(pcRay.modelMatrixTranspose * vec4(gl_WorldRayDirectionEXT, 1.0)));

  if(particleDensityHitCustom(modelRayOrigin, modelRayDirection, particleId, gl_RayTminEXT, gl_RayTmaxEXT, 9.0F, hitT))
    reportIntersectionEXT(hitT, 0);
#endif
}