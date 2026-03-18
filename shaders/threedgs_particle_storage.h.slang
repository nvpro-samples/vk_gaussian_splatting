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

#ifndef _THREEDGS_PARTICLE_STORAGE_H_
#define _THREEDGS_PARTICLE_STORAGE_H_

#include <shaderio.h>

#include "threedgs_particle_textures.h.slang"
#include "threedgs_particle_buffers.h.slang"

//==============================================================================
// Multi-Instance Support: Global Index Resolution
//==============================================================================

// Resolve a global splat ID to (descriptor, local splat index)
// Used by raster and compute shaders where indices[] contains global IDs
void resolveGlobalSplatID(in SceneAssets assets, in uint globalSplatID, out SplatSetDesc desc, out uint localSplatID)
{
  // Explicit 64-bit byte offset to avoid 32-bit overflow when globalSplatID * sizeof(GlobalSplatIndexEntry)
  // exceeds 4 GB (happens at ~537M splats, i.e. 6 instances of a 106M model).
  uint64_t byteOffset = uint64_t(globalSplatID) * uint64_t(sizeof(GlobalSplatIndexEntry));
  GlobalSplatIndexEntry entry = *(GlobalSplatIndexEntry*)(assets.globalIndexTableAddress + byteOffset);
  desc                        = assets.splatSetDescriptors[entry.splatSetIndex];
  localSplatID                = entry.splatIndex;
}

//==============================================================================
// Constants
//==============================================================================

static const float sqrt8    = sqrt(8.0);
static const float SH_C1    = 0.4886025119029199f;
static const float SH_C2[5] = { 1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742 };
static const float SH_C3[7] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f };

//==============================================================================
// Getter Functions
//==============================================================================

float3 fetchCenter(in SplatSetDesc desc, in uint splatIndex)
{
  if(desc.storage == SplatStorage::eBuffers)
    return fetchCenterFromBuffer(desc, splatIndex);
  else
    return fetchCenterFromTexture(desc, splatIndex);
}
float3 fetchScale(in SplatSetDesc desc, in uint splatIndex)
{
  if(desc.storage == SplatStorage::eBuffers)
    return fetchScaleFromBuffer(desc, splatIndex);
  else
    return fetchScaleFromTexture(desc, splatIndex);
}
float4 fetchRotation(in SplatSetDesc desc, in uint splatIndex)
{
  if(desc.storage == SplatStorage::eBuffers)
    return fetchRotationFromBuffer(desc, splatIndex);
  else
    return fetchRotationFromTexture(desc, splatIndex);
}
float4 fetchColor(in SplatSetDesc desc, in uint splatIndex)
{
  if(desc.storage == SplatStorage::eBuffers)
    return fetchColorFromBuffer(desc, splatIndex);
  else
    return fetchColorFromTexture(desc, splatIndex);
}
float3x3 fetchCovariance(in SplatSetDesc desc, in uint splatIndex)
{
  if(desc.storage == SplatStorage::eBuffers)
    return fetchCovarianceFromBuffer(desc, splatIndex);
  else
    return fetchCovarianceFromTexture(desc, splatIndex);
}

// caller must ensure that requested degree is <= desc.shDegree
void fetchSh(in SplatSetDesc desc, in uint splatIndex, in uint degree, out float3 shd1[3], out float3 shd2[5], out float3 shd3[7])
{
  if(desc.storage == SplatStorage::eBuffers)
    return fetchShFromBuffer(desc, splatIndex, degree, shd1, shd2, shd3);
  else
    return fetchShFromTexture(desc, splatIndex, degree, shd1, shd2, shd3);
}

float3 fetchViewDependentRadiance(in SplatSetDesc desc, in uint requestedShDegree, in uint splatIndex, in float3 worldViewDir)
{
  // contribution is null if MAX_SH_DEGREE < 1
  float3 rgb = float3(0.0, 0.0, 0.0);

  uint degree = min(desc.shDegree, requestedShDegree); // requestedShDegree is the one requested by renderer

  if(degree >= 1)
  {
    // SH coefficients for degree 1 (1,2,3)
    float3 shd1[3];
    // SH coefficients for degree 2 (4 5 6 7 8)
    float3 shd2[5];
    // SH coefficients for degree 3 (9,10,11,12,13,14,15)
    float3 shd3[7];

    // fetch the data (only what is needed according to degree)
    fetchSh(desc, splatIndex, degree, shd1, shd2, shd3);

    const float x = worldViewDir.x;
    const float y = worldViewDir.y;
    const float z = worldViewDir.z;

    // Degree 1 contributions
    rgb += SH_C1 * (-shd1[0] * y + shd1[1] * z - shd1[2] * x);

    if(degree >= 2)
    {
      const float xx = x * x;
      const float yy = y * y;
      const float zz = z * z;
      const float xy = x * y;
      const float yz = y * z;
      const float xz = x * z;

      // Degree 2 contributions
      rgb += (SH_C2[0] * xy) * shd2[0] + (SH_C2[1] * yz) * shd2[1] + (SH_C2[2] * (2.0 * zz - xx - yy)) * shd2[2]
             + (SH_C2[3] * xz) * shd2[3] + (SH_C2[4] * (xx - yy)) * shd2[4];

      if(degree >= 3)
      {
        const float xyy = x * yy;
        const float yzz = y * zz;
        const float zxx = z * xx;
        const float xyz = x * y * z;

        // Degree 3 contributions
        rgb += SH_C3[0] * shd3[0] * (3.0 * x * x - y * y) * y + SH_C3[1] * shd3[1] * x * y * z
               + SH_C3[2] * shd3[2] * (4.0 * z * z - x * x - y * y) * y
               + SH_C3[3] * shd3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
               + SH_C3[4] * shd3[4] * x * (4.0 * z * z - x * x - y * y) + SH_C3[5] * shd3[5] * (x * x - y * y) * z
               + SH_C3[6] * shd3[6] * x * (x * x - 3.0 * y * y);
      }
    }
  }
  return rgb;
}

#endif
