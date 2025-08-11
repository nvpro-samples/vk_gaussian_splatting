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
#extension GL_GOOGLE_include_directive : require
#extension GL_ARB_shader_clock : require  // profiling timers
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

//
void rotationMatrixTranspose(vec4 q, out mat3 ret)
{
  const float r = q.x;
  const float x = q.y;
  const float y = q.z;
  const float z = q.w;

  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float rx = r * x;
  const float ry = r * y;
  const float rz = r * z;

  // Compute rotation matrix from quaternion
  ret[0] = vec3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry));
  ret[1] = vec3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx));
  ret[2] = vec3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy)));

  ret = transpose(ret);  // JEM addition, todo, just generate in proper locations instead of transposing afterward
}

// fetch the position, scale and rotation
void fetchParticlePSR(in int splatIndex, out vec3 particlePosition, out vec3 particleScale, out mat3 particleInvRotation)
{
  particlePosition = fetchCenter(splatIndex);
  particleScale    = exp(fetchScale(splatIndex));
  vec4 quaternion  = normalize(fetchRotation(splatIndex));
  rotationMatrixTranspose(quaternion, particleInvRotation);
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

