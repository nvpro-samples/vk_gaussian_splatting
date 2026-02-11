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
#include "threedgrt_custom_attributes.h.slang"
#include "threedgrt_payload.h.slang"

#include "wireframe_extended.h.slang"

// clang-format off
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// clang-format on

[shader("intersection")]
void main()
{
#if RTX_USE_INSTANCES
  // allways testing the same unit particle, no data fetch inside Hit test function
  const float3 canonicalRayOrigin    = ObjectRayOrigin();
  const float3 canonicalRayDirection = ObjectRayDirection();

  CustomIntersectionAttributes attr;
  attr.aabbHitT     = PAYLOAD_INF_DISTANCE;
  attr.barycentrics = float2(0.5, 0.5);

#if WIREFRAME || DISABLE_OPACITY_GAUSSIAN
  // We perform an additional ray/aabb intersection for wireframe or if showing opaque particles
  // The built in Hardware aabb intersection is an approximation for fast evaluation and may not fit exactly
  // the AABB, it overestimates. This is not an issue when computing particleDensityHitInstance only, with
  // proper gaussian gradient (no visual artefacts). However when we need to display accurate sharp boundaries
  // we need to discard outliers from approximated built-in aabb hardware test.

  const float  aabbSize = 1.618033988749895;  // Golden ratio
  const float3 aabbMin  = float3(-aabbSize);
  const float3 aabbMax  = float3(aabbSize);

  float aabbHitT;
  if(!rayAABBIntersection(canonicalRayOrigin, canonicalRayDirection, aabbMin, aabbMax, RayTMin(), RayTCurrent(), aabbHitT))
    return;

  attr.aabbHitT = aabbHitT;
#endif

  float ellipsoidHitT = PAYLOAD_INF_DISTANCE;
  bool  hitEllipsoid =
      particleDensityHitInstance(canonicalRayOrigin, canonicalRayDirection, RayTMin(), RayTCurrent(), 9.0F, ellipsoidHitT);

#if !WIREFRAME
  if(hitEllipsoid)
    ReportHit(ellipsoidHitT, 0, attr);

#else
  // We will use hitKind as a bitfield.
  // Bit 4: particleHit flag (1=ellipsoid hit, 0=only AABB hit)
  uint hitKind = hitEllipsoid ? (1 << 4) : 0;

  // Compute barycentrics and update hitKind
  // This will set bits 0-3 (axis index, orientation, triangle)
  aabbFaceBarycentrics(canonicalRayOrigin, canonicalRayDirection, aabbHitT, aabbSize, hitKind, attr.barycentrics);

  const float reportT = hitEllipsoid ? ellipsoidHitT : aabbHitT;

  // we are sure we at least hit the bbox (otherwise early return)
  ReportHit(reportT, hitKind, attr);

#endif

#else

  // NO INSTANCES
  // Attention: this non instanced AABB branch code is forced disabled by UI,
  // activating can lead to GPU freeze due to massive AABBOX overlaps (see readme).

  const int particleId = PrimitiveIndex();

  // The two following transformations are to compute particleDensityHitCustom with transformed splat set model
  const float3 modelRayOrigin = mul(float4(WorldRayOrigin(), 1.0), pcRay.modelMatrixInverse).xyz;
  // Since the ray direction should not be affected by translation,
  // uses the inverse of the rotation - scale part of the model matrix.
  const float3 modelRayDirection = normalize(mul(WorldRayDirection(), float3x3(pcRay.modelMatrixRotScaleInverse)));

  float hitT;

  if(particleDensityHitCustom(modelRayOrigin, modelRayDirection, particleId, RayTMin(), RayTCurrent(), 9.0F, hitT))
  {
    CustomIntersectionAttributes attr;
    attr.aabbHitT = PAYLOAD_INF_DISTANCE;
    // no wireframe, never, we do not support for this branch of code that is disabled in UI.
    attr.barycentrics = float2(0.5, 0.5);
    ReportHit(hitT, 0, attr);
  }
#endif
}

// Ray-AABB intersection using slab method
// Returns true if ray intersects AABB, and outputs the near intersection distance
// Only reports front-facing intersections; rejects if ray origin is inside the box
bool rayAABBIntersection(in float3 rayOrigin,
                         in float3 rayDirection,
                         in float3 aabbMin,
                         in float3 aabbMax,
                         in float  minHitDistance,
                         in float  maxHitDistance,
                         out float hitDistance)
{
  const double3 invDir         = 1.0 / rayDirection;
  const double3 t0             = (aabbMin - rayOrigin) * invDir;
  const double3 t1             = (aabbMax - rayOrigin) * invDir;
  const double3 tmin           = min(t0, t1);
  const double3 tmax           = max(t0, t1);
  const double  tNearUnclamped = max(max(tmin.x, tmin.y), tmin.z);
  const double  tNear          = max(tNearUnclamped, double(minHitDistance));
  const double  tFar           = min(min(tmax.x, tmax.y), min(tmax.z, double(maxHitDistance)));

  // Reject if ray origin is inside the box (all entry points are behind us)
  if(tNearUnclamped < 0.0)
  {
    hitDistance = maxHitDistance;
    return false;
  }

  hitDistance = float(tNear);
  return (tNear <= tFar);
}

bool particleDensityHitInstance(in float3 canonicalRayOrigin,
                                in float3 canonicalUnormalizedRayDirection,
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

bool particleDensityHitCustom(in float3 rayOrigin,
                              in float3 rayDirection,
                              in int    particleId,
                              in float  minHitDistance,
                              in float  maxHitDistance,
                              in float  maxParticleSquaredDistance,
                              out float hitDistance)
{
  float3   particlePosition;
  float3   particleScale;
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
