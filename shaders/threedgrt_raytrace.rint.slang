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

#include "shaderio.h"
#include "threedgs_particles_storage.h.slang"
#include "threedgrt.h.slang"

// clang-format off
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// clang-format on

bool particleDensityHitCustom(in float3   rayOrigin,
                              in float3   rayDirection,
                              in int    particleId,
                              in float  minHitDistance,
                              in float  maxHitDistance,
                              in float  maxParticleSquaredDistance,
                              out float hitDistance)
{
  float3 particlePosition;
  float3 particleScale;
  float3x3 particleInvRotation;

  fetchParticlePSR(particleId, particlePosition, particleScale, particleInvRotation);

  float3 canonicalRayOrigin;
  float3 canonicalRayDirection;
  particleCannonicalRay(rayOrigin, rayDirection, particlePosition, particleScale, particleInvRotation,
                        canonicalRayOrigin, canonicalRayDirection);

  hitDistance = particleRayDistance(canonicalRayOrigin, canonicalRayDirection, particleScale);

  return (hitDistance > minHitDistance) && (hitDistance < maxHitDistance)
         && (particleRayMinSquaredDistance(canonicalRayOrigin, canonicalRayDirection) < maxParticleSquaredDistance);
}

bool particleDensityHitInstance(in float3   canonicalRayOrigin,
                                in float3   canonicalUnormalizedRayDirection,
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

[shader("intersection")]
void main()
{
  float hitT;
#if RTX_USE_INSTANCES
  // allways testing the same unit particle, no data fetch inside Hit test function
  if (particleDensityHitInstance(ObjectRayOrigin(), ObjectRayDirection(), RayTMin(), RayTCurrent(), 9.0F, hitT)) {
      BuiltInTriangleIntersectionAttributes attr;
      attr.barycentrics = float2(0.0, 0.0); // TODO: providing barycentrics will help rendering wireframe
      ReportHit(hitT, 0, attr);
  }
#else
  const int particleId = PrimitiveIndex();

  // The two following transformations are to compute processHit with transformed splat set model
  // Row-major: vector * matrix
  const float3 modelRayOrigin = mul(float4(WorldRayOrigin(), 1.0), pcRay.modelMatrixInverse).xyz;
  // Since the ray direction should not be affected by translation,
  // uses the inverse of the rotation - scale part of the model matrix.
  // Row-major: vector * matrix (extract 3x3 and multiply)
  const float3 modelRayDirection = normalize(mul(WorldRayDirection(), float3x3(pcRay.modelMatrixRotScaleInverse)));

  if (particleDensityHitCustom(modelRayOrigin, modelRayDirection, particleId, RayTMin(), RayTCurrent(), 9.0F, hitT)) {
      BuiltInTriangleIntersectionAttributes attr;
      attr.barycentrics = float2(0.0, 0.0); // TODO: providing barycentrics will help rendering wireframe
      ReportHit(hitT, 0, attr);
  }
#endif
}