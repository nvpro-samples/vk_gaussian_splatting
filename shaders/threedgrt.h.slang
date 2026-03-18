/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _THREEDGRT_H_
#define _THREEDGRT_H_

#include "quaternions.h.slang"
#include "shaderio.h"
#include "threedgs_particle_storage.h.slang"

static const float INF_DISTANCE = asfloat(0x7F7FFFFF);

struct ThreedgrtParticle
{
  int      id;           // particle Index
  float3   position;     // particle position (mean)
  float3   scale;        // particle scale
  quat     quaternion;   // particle rotation as a quaternion
  float3x3 invRotation;  // inverse of particle rotation matrix
  float4   color;        // particle color (Non view dependent RGB + density/opacity in alpha channel)

  // Indexed constructor
  // Fetch non view dependent data from storage via SplatSetDesc
  // If psrOnly is true, color is skipped, and only
  // Position Scale Rotations are loaded
  __init(SplatSetDesc splatSetDesc, int index, bool psrOnly = false)
  {
    id          = index;
    position    = fetchCenter(splatSetDesc, id);
    scale       = exp(fetchScale(splatSetDesc, id));
    quaternion  = vec4toQuat(normalize(fetchRotation(splatSetDesc, id)));
    invRotation = quatToMat3Transpose(quaternion);

    if(!psrOnly)
    {
      color = fetchColor(splatSetDesc, id);
    }
  }
}

void particleCannonicalRay(in float3   rayOrigin,
                           in float3   rayDirection,
                           in float3   particlePosition,
                           in float3   particleScale,
                           in float3x3 particleInvRotation,
                           out float3  particleRayOrigin,
                           out float3  particleRayDirection)
{
  const float3 giscl = float3(1.0f) / particleScale;
  const float3 gposc = (rayOrigin - particlePosition);
  // Row-major: vector * matrix
  const float3 gposcr = mul(gposc, particleInvRotation);
  particleRayOrigin   = giscl * gposcr;

  // Row-major: vector * matrix
  const float3 rayDirR = mul(rayDirection, particleInvRotation);
  const float3 grdu    = giscl * rayDirR;
  particleRayDirection = normalize(grdu);
}

float particleRayMinSquaredDistance(in float3 particleRayOrigin, in float3 particleRayDirection)
{
  const float3 gcrod = cross(particleRayDirection, particleRayOrigin);
  return dot(gcrod, gcrod);
}

float particleRayMaxKernelResponse<let particleKernelDegree : int32_t>(in float particleRayDist)
{
  const float grayDist = particleRayDist;

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

// Normal Computation Methods (controlled by NORMAL_METHOD macro from UI):
//   NORMAL_METHOD_MAX_DENSITY_PLANE (0): Uses computeEllipsoidNormalMaxDensityPlane()
//     Approximates the iso-density surface with a tangent plane at the Gaussian center
//     (StochasticSplats approach). Computes normal as n = Σ⁻¹ * μ where Σ is the covariance
//     matrix and μ is the position vector from camera to Gaussian center.
//     This method provides best results and is faster to compute.
//   NORMAL_METHOD_ISO_SURFACE (1): Uses computeEllipsoidNormal()
//     Ray-ellipsoid surface intersection approach in canonical space.
//     More geometrically accurate for individual particles.

#define USE_MAX_DENSITY_PLANE_NORMAL (NORMAL_METHOD == NORMAL_METHOD_MAX_DENSITY_PLANE)

// compute alpha + radiance, and optionally normal, for the particle
bool particleProcessHit<let kNormals : bool>(in FrameInfo    frameInfo,
                                             in SplatSetDesc splatSetDesc,
                                             in float3       modelRayOrigin,
                                             in float3       modelRayDirection,
                                             in float3x3     modelMatrixRotScaleInverse,
                                             in int          splatId,
                                             in float        hitT,
                                             out float       alpha,
                                             out float3      particleRad,
                                             out float3      normalWorld)
{
  alpha       = 0.0;
  particleRad = float3(0.0);
  normalWorld = float3(0.0);

  ThreedgrtParticle particle = { splatSetDesc, splatId };  // fetch the data, incl. color and opacity in color field

  // particle density culling
  const float particleDensity = particle.color.w;
  if(particleDensity <= frameInfo.alphaCullThreshold)
#pragma warning(disable: 41018)
    return false;
#pragma warning(default: 41018)

  // evaluate particle response
  float3 particleRayOrigin;
  float3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particle.position, particle.scale, particle.invRotation,
                        particleRayOrigin, particleRayDirection);

  const float rayDist = particleRayMinSquaredDistance(particleRayOrigin, particleRayDirection);

  const float maxResponse = particleRayMaxKernelResponse<KERNEL_DEGREE>(rayDist);
  alpha                   = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const float minParticleAlpha = 1.0f / 255.0f;
  const bool  acceptHit        = (alpha > frameInfo.alphaCullThreshold) && (maxResponse > KERNEL_MIN_RESPONSE);

  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    alpha = 1.0;
#endif
    // fetch radiance from SH coefs > degree 0
    const float3 vectorToParticleCenter = normalize(particle.position - modelRayOrigin);
#if SHOW_SH_ONLY == 1
    const float3 grad = float3(0.5) + fetchViewDependentRadiance(splatSetDesc, frameInfo.shDegree, splatId, vectorToParticleCenter);
#else
    const float3 grad =
        particle.color.xyz + fetchViewDependentRadiance(splatSetDesc, frameInfo.shDegree, splatId, vectorToParticleCenter);
#endif

    particleRad = grad;

    if(kNormals)
    {
      // Compute normal at the intersection point
      // Switch between two methods using USE_MAX_DENSITY_PLANE_NORMAL define
#if USE_MAX_DENSITY_PLANE_NORMAL
      const float3 normalModel = computeEllipsoidNormalMaxDensityPlane(frameInfo, modelRayOrigin, modelRayDirection, particle);
#else
      const float3 normalModel =
          computeEllipsoidNormal(frameInfo, modelRayOrigin, modelRayDirection, particleRayOrigin, particleRayDirection, particle);
#endif

      // Transform the accumulated normal from model space to world space
      // For normals, we need transpose(inverse(RS)) but for RS it is a noop
      // We store inverse(RS) in modelMatrixRotScaleInverse, then transpose() it here
      // Note: transpose() is fast (just index swapping) compared to inverse()
      normalWorld = normalize(mul(normalModel, transpose(modelMatrixRotScaleInverse)));
    }
  }

  return acceptHit;
}

// Integrates particle hit
void particleIntegrate(in float      alpha,
                       in float3     particleRad,
                       inout double3 transmittance,  // updated
                       inout float3  radiance,       // integrated
                       out float3    weight)
{
  weight = alpha * float3(transmittance);
  radiance += particleRad * weight;
  transmittance *= (1.0 - double(alpha));
}

// Version used by 3DGUT.
bool particleProcessHitGut(in FrameInfo frameInfo,
                           in float3    modelRayOrigin,
                           in float3    modelRayDirection,
                           in int       splatId,
                           in float4    splatColor,
                           in float3    particlePosition,
                           in float3    particleScale,
                           in float3x3  particleInvRotation,
                           out float    opacity)

{
  const float particleDensity = splatColor.a;
  if(particleDensity <= frameInfo.alphaCullThreshold)
#pragma warning(disable: 41018)
    return false;
#pragma warning(default: 41018)

  float3 particleRayOrigin;
  float3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const float rayDist = particleRayMinSquaredDistance(particleRayOrigin, particleRayDirection);

  const float maxResponse = particleRayMaxKernelResponse<KERNEL_DEGREE>(rayDist);
  const float alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const double minParticleAlpha = 1.0f / 255.0f;
  const bool   acceptHit        = ((double)alpha > minParticleAlpha) && (maxResponse > KERNEL_MIN_RESPONSE);

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

// Version used by 3DGUT with normal output for surface generation.
bool particleProcessHitGutWithNormal(in FrameInfo frameInfo,
                                     in float3    modelRayOrigin,
                                     in float3    modelRayDirection,
                                     in int       splatId,
                                     in float4    splatColor,
                                     in float3    particlePosition,
                                     in float3    particleScale,
                                     in float3x3  particleInvRotation,
                                     in float4x4  splatSetTransform,  // For world space normal transform
                                     out float    opacity,
                                     out float3   normalWorld)  // Output: normal in world space

{
  const float particleDensity = splatColor.a;
  if(particleDensity <= frameInfo.alphaCullThreshold)
#pragma warning(disable: 41018)
    return false;
#pragma warning(default: 41018)

  float3 particleRayOrigin;
  float3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation,
                        particleRayOrigin, particleRayDirection);

  const float rayDist = particleRayMinSquaredDistance(particleRayOrigin, particleRayDirection);

  const float maxResponse = particleRayMaxKernelResponse<KERNEL_DEGREE>(rayDist);
  const float alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const double minParticleAlpha = 1.0f / 255.0f;
  const bool   acceptHit        = ((double)alpha > minParticleAlpha) && (maxResponse > KERNEL_MIN_RESPONSE);

  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    opacity = 1.0;
#else
    opacity = alpha;
#endif

    // Compute normal using max density plane approximation
    // Create a ThreedgrtParticle struct for the normal computation
    ThreedgrtParticle particle;
    particle.id          = splatId;
    particle.position    = particlePosition;
    particle.scale       = particleScale;
    particle.invRotation = particleInvRotation;
    // color not needed for normal computation

#if USE_MAX_DENSITY_PLANE_NORMAL
    float3 normalModel = computeEllipsoidNormalMaxDensityPlane(frameInfo, modelRayOrigin, modelRayDirection, particle);
#else
    float3 normalModel =
        computeEllipsoidNormal(frameInfo, modelRayOrigin, modelRayDirection, particleRayOrigin, particleRayDirection, particle);
#endif

    // Transform normal from model space to world space
    // For normals: use transpose(inverse(RS)) but for rotation-scale matrix this is simplified
    // Extract upper-left 3x3 of splatSetTransform for the rotation-scale part
    float3x3 modelToWorldRS = float3x3(splatSetTransform[0].xyz, splatSetTransform[1].xyz, splatSetTransform[2].xyz);
    // For normals, we need inverse-transpose: transpose(inverse(RS))
    // Since modelToWorldRS is RS, inverse(RS) = S^-1 * R^T, and transpose of that = R * S^-1
    // However, for uniform scale or rotation-only, we can just use the matrix directly
    // For correctness with non-uniform scale, we'd need proper inverse-transpose
    // But the normal is already computed in model space accounting for scale
    normalWorld = normalize(mul(normalModel, modelToWorldRS));
  }

  return acceptHit;
}

// Compute the ellipsoid normal using the maximum density plane approximation
// Based on StochasticSplats approach: approximates the iso-density surface with a tangent plane
// at the Gaussian center, then derives the normal from the plane equation.
// Returns the normal in model space


float3 computeEllipsoidNormalMaxDensityPlane(in FrameInfo frameInfo, in float3 modelRayOrigin, in float3 modelRayDirection, in ThreedgrtParticle particle)
{
  float3 normalModel;

  // Check for degenerate scale components (very flat or line particles)
  bool3 isSmall    = particle.scale < float3(frameInfo.thinParticleThreshold);
  int         smallCount = (isSmall.x ? 1 : 0) + (isSmall.y ? 1 : 0) + (isSmall.z ? 1 : 0);

  float3x3 rotMat      = transpose(particle.invRotation);
  float3   localVector = modelRayOrigin - particle.position;

  if(smallCount == 0)
  {
    // Maximum density plane approximation
    // Theory: At the Gaussian center, the iso-density surface can be approximated by a tangent plane.
    // See StochasticSplats https://github.com/ubc-vision/stochasticsplats

    // Normal case: compute gradient-based normal
    float3 invScaleSq = float3(1.0) / (particle.scale * particle.scale);

    // Transform to canonical space, apply inverse scale squared, transform back
    float3 canonicalVector = mul(localVector, particle.invRotation);
    float3 scaledVector    = canonicalVector * invScaleSq;
    // Transform back to model space using R
    float3 gradient = mul(scaledVector, rotMat);

    // The gradient points in the direction of maximum density increase
    // For the normal, we want it pointing towards the ray origin (outward facing)
    float gradientLengthSq = dot(gradient, gradient);

    normalModel = gradient * rsqrt(gradientLengthSq);

    // Ensure normal points towards ray origin (same side as view direction)
    if(dot(normalModel, localVector) < 0.0)
    {
      normalModel = -normalModel;
    }
  }
  else if(smallCount == 1)
  {
    // Exactly one scale is small: flat particle (disk/ellipse)
    // Normal is perpendicular to the plane, along the small scale axis
    float3 axisLocal = (isSmall.x) ? float3(1.0, 0.0, 0.0) : ((isSmall.y) ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0));

    // Transform from canonical space to model space using rotation matrix
    normalModel = mul(axisLocal, rotMat);

    // Ensure normal points towards ray origin (outward facing)
    if(dot(normalModel, localVector) < 0.0)
    {
      normalModel = -normalModel;
    }
  }
  else
  {
    // Two or more scales are degenerate (line or point): fall back to view direction
    normalModel = -modelRayDirection;
  }

  return normalModel;
}

// Compute the ellipsoid normal using ray-sphere intersection in canonical space.
// Uses the same smallCount decomposition as computeEllipsoidNormalMaxDensityPlane.
// Returns the normal in model space
float3 computeEllipsoidNormal(in FrameInfo         frameInfo,
                              in float3            modelRayOrigin,
                              in float3            modelRayDirection,
                              in float3            particleRayOrigin,
                              in float3            particleRayDirection,
                              in ThreedgrtParticle particle)
{
  float3 normalModel;

  // Check for degenerate scale components using relative 1:50 ratio,
  // clamped to at least the UI-controlled thinParticleThreshold
  const float maxScale          = max(max(particle.scale.x, particle.scale.y), particle.scale.z);
  const float flatnessThreshold = max(0.02 * maxScale, frameInfo.thinParticleThreshold);
  bool3       isSmall           = particle.scale < float3(flatnessThreshold);
  int         smallCount        = (isSmall.x ? 1 : 0) + (isSmall.y ? 1 : 0) + (isSmall.z ? 1 : 0);

  float3x3 rotMat      = transpose(particle.invRotation);
  float3   localVector = modelRayOrigin - particle.position;

  if(smallCount == 0)
  {
    // Regular ellipsoid: ray-sphere intersection in canonical space
    // The sphere radius must encompass all accepted hits.
    // In the intersection shader, maxParticleSquaredDistance = 9.0, so the max
    // canonical distance at which hits are accepted is sqrt(9) = 3.0.
    // (This also matches KERNEL_MIN_RESPONSE = 0.0113 for the quadratic kernel:
    //  exp(-0.5 * 9) ≈ 0.0111)
    const float convergenceRadius = 3.0;
    float       surfaceHitT;
    const bool  hitSurface =
        raySphereIntersection(particleRayOrigin, particleRayDirection, convergenceRadius, 0.0, INF_DISTANCE, surfaceHitT);

    if(hitSurface)
    {
      // Compute hit point in canonical space
      const float3 hitPointCanonical = particleRayOrigin + surfaceHitT * particleRayDirection;

      // Normal in canonical space: gradient of ||p||^2 = r^2 is 2*p, so normal = normalize(p)
      const float3 normalCanonical = normalize(hitPointCanonical);

      // Transform normal back to model space (inverse-transpose for normals)
      const float3 normalScaled = normalCanonical / particle.scale;  // Apply inverse scale
      normalModel               = mul(normalScaled, rotMat);         // Apply rotation
      normalModel               = normalize(normalModel);
    }
    else
    {
      // Fallback: use negative ray direction as normal (shouldn't happen if hit is accepted)
      normalModel = -modelRayDirection;
    }
  }
  else if(smallCount == 1)
  {
    // Exactly one scale is small: flat particle (disk/ellipse)
    // Normal is perpendicular to the plane, along the small scale axis
    float3 axisLocal = (isSmall.x) ? float3(1.0, 0.0, 0.0) : ((isSmall.y) ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0));

    // Transform from canonical space to model space using rotation matrix
    normalModel = mul(axisLocal, rotMat);

    // Ensure normal points towards ray origin (outward facing)
    if(dot(normalModel, localVector) < 0.0)
    {
      normalModel = -normalModel;
    }
  }
  else
  {
    // Two or more scales are degenerate (line or point): fall back to view direction
    normalModel = -modelRayDirection;
  }

  return normalModel;
}


// Computes the actual parametric surface intersection of a ray with a sphere
// of radius sphereRadius in canonical space. Returns the nearest intersection point (t > 0).
// This is used for computing accurate geometric normals on the ellipsoid surface.
bool raySphereIntersection(in float3 rayOrigin, in float3 rayDirection, in float sphereRadius, in float minHitDistance, in float maxHitDistance, out float hitT)
{
  hitT = maxHitDistance;

  // Ray-sphere intersection: ||O + t*D||^2 = r^2


  // a*t^2 + b*t + c = 0
  const float a = dot(rayDirection, rayDirection);
  const float b = 2.0 * dot(rayOrigin, rayDirection);
  const float c = dot(rayOrigin, rayOrigin) - sphereRadius * sphereRadius;

  const float discriminant = b * b - 4.0 * a * c;

  if(discriminant < 0.0)
    return false;  // No intersection with sphere

  const float sqrtDisc = sqrt(discriminant);
  const float invA     = 1.0 / (2.0 * a);
  const float t1       = (-b - sqrtDisc) * invA;  // Near intersection
  const float t2       = (-b + sqrtDisc) * invA;  // Far intersection

  // Choose the nearest valid intersection in front of the ray
  if(t1 >= minHitDistance && t1 <= maxHitDistance)
  {
    hitT = t1;
    return true;
  }
  else if(t2 >= minHitDistance && t2 <= maxHitDistance)
  {
    hitT = t2;
    return true;
  }

  return false;
}

#endif 