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

#pragma once

/// This header extends #include "nvshaders/wireframe.h.slang"
/// It provides additional functions to handle fisheye cameras
/// And rendering of AABBs edges

#include "cameras.h.slang"

/// Computes ray differentials for fisheye camera in raytracing pipelines
/// Fisheye cameras use non-linear spherical projection, so we must compute actual neighboring rays
/// to get correct differentials for wireframe rendering
///
/// Unlike pinhole cameras where projection is linear, fisheye cameras map pixel positions to
/// ray directions using spherical coordinates. This function computes the true ray directions
/// for neighboring pixels using generateFisheyeRay() and derives the differentials from those,
/// ensuring wireframe edges are rendered correctly with proper thickness and anti-aliasing in
/// fisheye mode.
///
/// @param pixelPos Current pixel coordinates (x, y)
/// @param imageSize Image/framebuffer dimensions (width, height)
/// @param fov Field of view in radians
/// @param principalPoint Principal point offset (typically float2(0.0))
/// @param viewInverse Inverse view matrix (camera to world transform)
/// @param rayDirection Current ray direction (for reference)
/// @param differentialX [out] Ray differential in X direction (world space)
/// @param differentialY [out] Ray differential in Y direction (world space)
void computeDifferentialsFisheye(float2     pixelPos,
                                 float2     imageSize,
                                 float      fov,
                                 float2     principalPoint,
                                 float4x4   viewInverse,
                                 float3     rayDirection,
                                 out float3 differentialX,
                                 out float3 differentialY)
{
  // Compute the center pixel for consistent offset calculation (matching pinhole behavior)
  const float2 pixelCenter = pixelPos + float2(0.5);

  // Compute ray directions for neighboring pixels offset by 1 pixel in X and Y
  // This matches the pinhole approach where offsets are computed from (pixelPos + 1.5)
  float3 rayOriginDummy;  // Not used, but required by the function signature
  float3 rayDirOffsetX, rayDirOffsetY;

  // For X differential: use offset X, center Y (matching pinhole: dOffset.x, d.y)
  generateFisheyeRay(pixelCenter + float2(1.0, 0.0), imageSize, fov, principalPoint, viewInverse, rayOriginDummy, rayDirOffsetX);

  // For Y differential: use center X, offset Y (matching pinhole: d.x, dOffset.y)
  generateFisheyeRay(pixelCenter + float2(0.0, 1.0), imageSize, fov, principalPoint, viewInverse, rayOriginDummy, rayDirOffsetY);

  // Store the offset ray directions directly (not the difference!)
  // This matches the pinhole version which stores the normalized offset directions
  differentialX = rayDirOffsetX;
  differentialY = rayDirOffsetY;
}

/// Computes barycentric coordinates for an AABB face intersection in raytracing pipelines
/// AABBs have 6 quad faces, each subdivided into 2 triangles for wireframe rendering.
/// This function determines which face was hit, which triangle within that face, and
/// computes proper barycentric coordinates for wireframe edge detection.
///
/// The function uses a hybrid approach to determine the hit face:
/// - Position-based method: Reliable for normal, uniform transforms (checks which face the hit point is closest to)
/// - Ray-direction method: Reliable for anisotropic transforms like flat particles (uses largest ray component)
///
/// The hitKind parameter is encoded with face information for use by computeAABBFaceDerivativeHighp():
/// - Bits 0-1: Axis index (0=X, 1=Y, 2=Z) - direct 2-bit encoding
/// - Bit 2: Face orientation (0=positive side, 1=negative side)
/// - Bit 3: Triangle selection (0=lower triangle, 1=upper triangle)
///
/// Usage example in an intersection shader:
///   float2 barycentrics;
///   uint hitKind = 0;
///   aabbFaceBarycentrics(canonicalRayOrigin, canonicalRayDir, hitT, aabbHalfSize, hitKind, barycentrics);
///   // Use barycentrics with computeAABBFaceDerivativeHighp() for wireframe rendering
///
/// @param canonicalRayOrigin Ray origin in canonical/object space (after AABB transform applied)
/// @param canonicalRayDirection Ray direction in canonical/object space (does not need to be normalized)
/// @param hitT Ray parameter at AABB intersection point (distance from origin)
/// @param aabbHalfSize Half-size of the AABB (AABB extends from -aabbHalfSize to +aabbHalfSize on each axis)
/// @param hitKind [in/out] Hit kind flags; function sets bits encoding face/triangle information
///                         Bits 0-1: axis index (0-2), Bit 2: orientation, Bit 3: triangle
/// @param barycentrics [out] Barycentric coordinates (u, v) on the hit triangle in [0..1] range
///                           The third barycentric w can be computed as: w = 1-u-v (for lower triangle)
///                           or as: w = 1-v (for upper triangle, see implementation for details)
void aabbFaceBarycentrics(in float3 canonicalRayOrigin, in float3 canonicalRayDirection, in float hitT, in float aabbHalfSize, inout uint hitKind, out float2 barycentrics)
{
  const float aabbSize    = 2.0 * aabbHalfSize;
  const float invAabbSize = 1.0 / aabbSize;

  // Compute hit point at AABB entry in canonical space
  const float3 hitPoint = canonicalRayOrigin + hitT * canonicalRayDirection;

  // Determine which face was hit using TWO methods:
  // Method 1 (position-based): Reliable for normal transforms
  // Checks which face the hit point is closest to
  const float3 absPos     = abs(hitPoint);
  const float3 distToFace = abs(absPos - aabbHalfSize);

  float minDistPos  = distToFace.x;
  int   faceAxisPos = 0;
  if(distToFace.y < minDistPos)
  {
    minDistPos  = distToFace.y;
    faceAxisPos = 1;
  }
  if(distToFace.z < minDistPos)
  {
    minDistPos  = distToFace.z;
    faceAxisPos = 2;
  }

  // Method 2 (ray-direction-based): Reliable for anisotropic transforms (flat particles)
  // Assumes the largest ray component indicates the hit face
  const float3 absRayDir = abs(canonicalRayDirection);

  float maxComponent = absRayDir.x;
  int   faceAxisDir  = 0;
  if(absRayDir.y > maxComponent)
  {
    maxComponent = absRayDir.y;
    faceAxisDir  = 1;
  }
  if(absRayDir.z > maxComponent)
  {
    maxComponent = absRayDir.z;
    faceAxisDir  = 2;
  }

  // Use position-based method if it gives a clear result (minDistPos < threshold)
  // Otherwise use ray-direction method (for anisotropic/flat transforms)
  const int faceAxis = (minDistPos < 0.00001) ? faceAxisPos : faceAxisDir;

  // Encode face info in hitKind for derivative computation
  hitKind |= faceAxis;  // Bits 0-1: axis index (0=X, 1=Y, 2=Z) - direct encoding

  // Determine side based on hit point position for that axis
  if(hitPoint[faceAxis] < 0.0)
    hitKind |= (1 << 2);  // Bit 2: orientation (0=positive, 1=negative)

  // Compute barycentric coordinates on the quad face
  // Project hit point onto the face's 2D coordinate system
  // Vectorized: compute all coordinates at once, then select based on axis
  const float3 uvw = (hitPoint + aabbHalfSize) * invAabbSize;  // Map [-aabbHalfSize, aabbHalfSize] to [0, 1]
  
  float u, v;
  switch(faceAxis)
  {
    case 0:  u = uvw.y; v = uvw.z; break;  // X face: use Y,Z
    case 1:  u = uvw.x; v = uvw.z; break;  // Y face: use X,Z
    default: u = uvw.x; v = uvw.y; break;  // Z face: use X,Y
  }

  // Each quad face is split into two triangles
  // Triangle 1: (0,0)-(1,0)-(0,1) for u+v <= 1
  // Triangle 2: (1,0)-(1,1)-(0,1) for u+v > 1
  // Return proper triangle barycentric coordinates

  // Store which triangle in hitKind (bit 3)
  const bool useTriangle1 = (u + v <= 1.0);
  if(!useTriangle1)
    hitKind |= (1 << 3);  // Bit 3 = Triangle 2 (upper triangle)

  if(useTriangle1)
  {
    // Lower triangle: vertices at (0,0), (1,0), (0,1)
    // Barycentric coordinates: (1-u-v, u, v)
    barycentrics = float2(u, v);
  }
  else
  {
    // Upper triangle: vertices at (1,0), (1,1), (0,1)
    // Map (u,v) to barycentric coordinates for this triangle
    // Point (u,v) in quad space
    // Vertices: V0=(1,0), V1=(1,1), V2=(0,1)
    // Barycentric: w0 * V0 + w1 * V1 + w2 * V2 = (u,v)
    // Solve: w0*(1,0) + w1*(1,1) + w2*(0,1) = (u,v)
    // u = w0 + w1, v = w1 + w2, w0+w1+w2=1
    // Solution: w0 = 1-v, w1 = u+v-1, w2 = 1-u
    barycentrics = float2(u + v - 1.0, 1.0 - u);
  }
}

/// Computes barycentric coordinate derivatives for AABB face wireframe rendering in raytracing
/// This is the AABB equivalent of computeDerivativeHighp() for triangle meshes, specifically
/// designed for rendering wireframe edges on axis-aligned bounding box faces using double precision.
///
/// AABB faces are quads subdivided into two triangles. This function computes how barycentric
/// coordinates change across neighboring pixels, which is essential for proper wireframe edge
/// detection and anti-aliasing. It uses the hitKind information from aabbFaceBarycentrics() to
/// decode which face and triangle were hit, then computes derivatives along that specific face plane.
///
/// The function automatically suppresses the diagonal edge that splits each quad face into two
/// triangles, so only the actual AABB edges (the quad boundaries) are rendered as wireframe.
///
/// Usage example in a closest-hit shader for AABB:
///   float2 barycentrics;
///   uint hitKind = 0;
///   aabbFaceBarycentrics(canonicalRayOrigin, canonicalRayDir, hitT, aabbHalfSize, hitKind, barycentrics);
///   float3 bary3D = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
///   float3 deltas = computeAABBFaceDerivativeHighp(WorldToObject4x3(), WorldRayOrigin(),
///                                                    ObjectRayOrigin(), aabbHalfSize, differentialX,
///                                                    differentialY, bary3D, hitKind);
///   float wireframe = processWireframe(settings, bary3D, deltas);
///
/// @param worldToObject World-to-object space transformation matrix (4x3)
/// @param worldRayOrigin Ray origin in world space
/// @param objectRayOrigin Ray origin in object space
/// @param aabbHalfSize Half-size of the AABB (AABB extends from -aabbHalfSize to +aabbHalfSize on each axis)
/// @param differentialX Ray differential in X direction (from computeDifferentials or computeDifferentialsFisheye)
/// @param differentialY Ray differential in Y direction (from computeDifferentials or computeDifferentialsFisheye)
/// @param bary Current hit point barycentric coordinates (w, u, v) where w+u+v=1
/// @param kind Hit kind flags encoding face/triangle info (from aabbFaceBarycentrics)
///             Bits 0-1: axis index (0-2), Bit 2: orientation, Bit 3: triangle
/// @return Barycentric derivatives (absolute difference) analogous to fwidth(bary) for AABB faces
///         Diagonal edge component is set to 0 to suppress rendering of the quad subdivision edge
float3 computeAABBFaceDerivativeHighp(in float4x3 worldToObject,
                                      in float3   worldRayOrigin,
                                      in float3   objectRayOrigin,
                                      in double   aabbHalfSize,
                                      in float3   differentialX,
                                      in float3   differentialY,
                                      in float3   bary,
                                      in uint     kind)
{
  const double aabbSize    = 2.0 * aabbHalfSize;
  const double invAabbSize = 1.0 / aabbSize;

  // Determine which face was hit and build its tangent basis
  // Decode axis from bits 0-1 (direct encoding) and orientation from bit 2
  const int axisIdx = kind & 0x03;  // Bits 0-1: axis index (0=X, 1=Y, 2=Z) 
  const bool isNegativeSide = (kind & (1 << 2)) != 0;  // Bit 2: orientation
  const double faceOrientation = isNegativeSide ? -1.0 : 1.0;

  // Construct tangent basis using switch
  // Build orthonormal basis vectors programmatically for each face
  
  double3 planeNormal = double3(0, 0, 0);
  double3 tangentU    = double3(0, 0, 0);
  double3 tangentV    = double3(0, 0, 0);
  
  planeNormal[axisIdx] = faceOrientation;  // Normal points along the hit axis
  
  // Select tangent axes based on face axis (matches barycentric coordinate order)
  switch(axisIdx)
  {
    case 0:  tangentU[1] = 1.0; tangentV[2] = 1.0; break;  // X face: U=Y, V=Z
    case 1:  tangentU[0] = 1.0; tangentV[2] = 1.0; break;  // Y face: U=X, V=Z
    default: tangentU[0] = 1.0; tangentV[1] = 1.0; break;  // Z face: U=X, V=Y
  }

  const double3 planePoint = planeNormal * aabbHalfSize;
  const double  planeD     = dot(planeNormal, planePoint);

  // Differential ray X
  const float3  worldDiffX    = WorldRayOrigin() + differentialX;
  const double3 objDiffX      = mul(float4(worldDiffX, 1), WorldToObject4x3());
  const double3 dirDiffX      = normalize(objDiffX - ObjectRayOrigin());
  const double  denomX        = dot(planeNormal, dirDiffX);
  const double  invDenomX     = 1.0 / denomX;
  const double  tDiffX        = (planeD - dot(planeNormal, ObjectRayOrigin())) * invDenomX;
  const double3 hitDiffX      = ObjectRayOrigin() + tDiffX * dirDiffX;
  const double  uDiffX        = (dot(hitDiffX, tangentU) + aabbHalfSize) * invAabbSize;
  const double  vDiffX        = (dot(hitDiffX, tangentV) + aabbHalfSize) * invAabbSize;

  // Differential ray Y
  const float3  worldDiffY    = WorldRayOrigin() + differentialY;
  const double3 objDiffY      = mul(float4(worldDiffY, 1), WorldToObject4x3());
  const double3 dirDiffY      = normalize(objDiffY - ObjectRayOrigin());
  const double  denomY        = dot(planeNormal, dirDiffY);
  const double  invDenomY     = 1.0 / denomY;
  const double  tDiffY        = (planeD - dot(planeNormal, ObjectRayOrigin())) * invDenomY;
  const double3 hitDiffY      = ObjectRayOrigin() + tDiffY * dirDiffY;
  const double  uDiffY        = (dot(hitDiffY, tangentU) + aabbHalfSize) * invAabbSize;
  const double  vDiffY        = (dot(hitDiffY, tangentV) + aabbHalfSize) * invAabbSize;

  // Decode triangle from hitKind (bit 3)
  const bool isTriangle1 = ((kind & (1 << 3)) == 0);

  double3 baryDiffX, baryDiffY;
  if(isTriangle1)
  {
    baryDiffX = double3(1.0 - uDiffX - vDiffX, uDiffX, vDiffX);
    baryDiffY = double3(1.0 - uDiffY - vDiffY, uDiffY, vDiffY);
  }
  else
  {
    baryDiffX = double3(1.0 - vDiffX, uDiffX + vDiffX - 1.0, 1.0 - uDiffX);
    baryDiffY = double3(1.0 - vDiffY, uDiffY + vDiffY - 1.0, 1.0 - uDiffY);
  }

  double3 derivativeD = abs(baryDiffX - bary) + abs(baryDiffY - bary);

  // Suppress diagonal edges (the edges that split the quad face into two triangles)
  // Triangle 1: diagonal is component 0 (edge where w=1-u-v ~ 0)
  // Triangle 2: diagonal is component 1 (edge where u+v-1 ~ 0)
  // Set derivative to 0 to indicate no edge (small change in barycentrics)
  if(isTriangle1)
  {
    derivativeD.x = 0.0;  // Suppress first component (diagonal edge)
  }
  else
  {
    derivativeD.y = 0.0;  // Suppress second component (diagonal edge)
  }

  const float3 derivative = float3(derivativeD);

  return derivative;
}
