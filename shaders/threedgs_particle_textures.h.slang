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

#ifndef _THREEDGS_PARTICLE_TEXTURES_H_
#define _THREEDGS_PARTICLE_TEXTURES_H_

#include <shaderio.h>

// This array contains all splat textures indexed by values in SplatSetDesc
[[vk::binding(BINDING_SPLAT_TEXTURES, 0)]]
Sampler2D allSplatTextures[];

// data texture accessors
// Uses uint64_t to prevent overflow when splatIndex * stride > 2^32 (e.g. SH stride=12 at >358M splats)
int2 getDataPos(in uint splatIndex, in uint stride, in uint offset, in int2 dimensions)
{
  const uint64_t fullOffset = uint64_t(splatIndex) * uint64_t(stride) + uint64_t(offset);

  return int2(uint(fullOffset % uint64_t(dimensions.x)), uint(fullOffset / uint64_t(dimensions.x)));
}

// data texture accessors (float stride variant for covariance packing)
int2 getDataPosF(in uint splatIndex, in float stride, in uint offset, in int2 dimensions)
{
  const uint64_t fullOffset = uint64_t(float(splatIndex) * stride) + uint64_t(offset);

  return int2(uint(fullOffset % uint64_t(dimensions.x)), uint(fullOffset / uint64_t(dimensions.x)));
}


float3 fetchCenterFromTexture(in SplatSetDesc desc, in uint splatIndex)
{
  int2 texSize;
  allSplatTextures[desc.centersTexture].GetDimensions(texSize.x, texSize.y);
  return allSplatTextures[desc.centersTexture].Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).xyz;
}

float3 fetchScaleFromTexture(in SplatSetDesc desc, in uint splatIndex)
{
  int2 texSize;
  allSplatTextures[desc.scalesTexture].GetDimensions(texSize.x, texSize.y);
  return allSplatTextures[desc.scalesTexture].Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).xyz;
}

float4 fetchRotationFromTexture(in SplatSetDesc desc, in uint splatIndex)
{
  int2 texSize;
  allSplatTextures[desc.rotationsTexture].GetDimensions(texSize.x, texSize.y);
  return allSplatTextures[desc.rotationsTexture].Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0));
}

// Fetch RGBA color from texture.
// Unlike SH (which needs range remapping for UINT8 since SH values live in [-1,1]),
// RGBA colors are natively in [0,1]. All three VkFormats (R8G8B8A8_UNORM,
// R16G16B16A16_SFLOAT, R32G32B32A32_SFLOAT) return correct [0,1] float4 from Load(),
// so no RGBA_FORMAT-dependent conversion is needed here.
float4 fetchColorFromTexture(in SplatSetDesc desc, in uint splatIndex)
{
  int2 texSize;
  allSplatTextures[desc.colorsTexture].GetDimensions(texSize.x, texSize.y);
  return allSplatTextures[desc.colorsTexture].Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0));
}

float3x3 fetchCovarianceFromTexture(in SplatSetDesc desc, in uint splatIndex)
{
  // Use RGBA texture map to store sets of 3 elements requires some offset shifting depending on splatIndex

  const uint  oddOffset        = uint(splatIndex) & uint(0x00000001);
  const uint  doubleOddOffset  = oddOffset * uint(2);
  const bool  isEven           = oddOffset == uint(0);
  const uint  nearestEvenIndex = uint(splatIndex) - oddOffset;
  const float fOddOffset       = float(oddOffset);

  int2 texSize;
  allSplatTextures[desc.covariancesTexture].GetDimensions(texSize.x, texSize.y);

  const float4 sampledCovarianceA =
      allSplatTextures[desc.covariancesTexture].Load(int3(getDataPosF(nearestEvenIndex, 1.5, oddOffset, texSize), 0));

  const float4 sampledCovarianceB =
      allSplatTextures[desc.covariancesTexture].Load(int3(getDataPosF(nearestEvenIndex, 1.5, oddOffset + uint(1), texSize), 0));

  const float3 cov3D_M11_M12_M13 =
      float3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) + float3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;

  const float3 cov3D_M22_M23_M33 =
      float3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) + float3(sampledCovarianceB.gba) * fOddOffset;

  return float3x3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
                  cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
}

// fetch SH coefficients from data textures
// caller must ensure that requested degree is <= desc.shDegree
void fetchShFromTexture(in SplatSetDesc desc, in uint splatIndex, in uint degree, out float3 shd1[3], out float3 shd2[5], out float3 shd3[7])
{
  const float SphericalHarmonics8BitCompressionRange = 2.0;
  const float3 vec8BitSHShift = float3(SphericalHarmonics8BitCompressionRange / 2.0, SphericalHarmonics8BitCompressionRange / 2.0,
                                       SphericalHarmonics8BitCompressionRange / 2.0);

  int2 texSize;
  allSplatTextures[desc.shTexture].GetDimensions(texSize.x, texSize.y);

  const uint stride = 12;  // 12 for degree 3, 6 for degree 2
  // fetching degree 1
  const float4 sampledSH0123 = allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 0, texSize), 0));
  const float4 sampledSH4567 = allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 1, texSize), 0));
  const float4 sampledSH891011 = allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 2, texSize), 0));

  const float3 sh1 = sampledSH0123.rgb;
  const float3 sh2 = float3(sampledSH0123.a, sampledSH4567.rg);
  const float3 sh3 = float3(sampledSH4567.ba, sampledSH891011.r);

#if SH_FORMAT != FORMAT_UINT8
  shd1[0] = sh1;
  shd1[1] = sh2;
  shd1[2] = sh3;
#else
  shd1[0] = sh1 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd1[1] = sh2 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd1[2] = sh3 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
#endif

  // fetching degree 2
  if(degree >= 2)
  {
    const float4 sampledSH12131415 = allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 3, texSize), 0));
    const float4 sampledSH16171819 = allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 4, texSize), 0));
    const float4 sampledSH20212223 = allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 5, texSize), 0));

    const float3 sh4 = sampledSH891011.gba;
    const float3 sh5 = sampledSH12131415.rgb;
    const float3 sh6 = float3(sampledSH12131415.a, sampledSH16171819.rg);
    const float3 sh7 = float3(sampledSH16171819.ba, sampledSH20212223.r);
    const float3 sh8 = sampledSH20212223.gba;

#if SH_FORMAT != FORMAT_UINT8
    shd2[0] = sh4;
    shd2[1] = sh5;
    shd2[2] = sh6;
    shd2[3] = sh7;
    shd2[4] = sh8;
#else
    shd2[0] = sh4 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
    shd2[1] = sh5 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
    shd2[2] = sh6 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
    shd2[3] = sh7 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
    shd2[4] = sh8 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
#endif

    // Fetching degree 3
    if(degree >= 3)
    {
      const float4 sampledSH24252627 =
          allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 6, texSize), 0));
      const float4 sampledSH28293031 =
          allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 7, texSize), 0));
      const float4 sampledSH32333435 =
          allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 8, texSize), 0));
      const float4 sampledSH36373839 =
          allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 9, texSize), 0));
      const float4 sampledSH404142 =
          allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 10, texSize), 0));
      const float4 sampledSH434445 =
          allSplatTextures[desc.shTexture].Load(int3(getDataPos(splatIndex, stride, 11, texSize), 0));

      const float3 sh9  = sampledSH24252627.rgb;
      const float3 sh10 = float3(sampledSH24252627.a, sampledSH28293031.rg);
      const float3 sh11 = float3(sampledSH28293031.ba, sampledSH32333435.r);
      const float3 sh12 = sampledSH32333435.gba;
      const float3 sh13 = sampledSH36373839.rgb;
      const float3 sh14 = float3(sampledSH36373839.a, sampledSH404142.rg);
      const float3 sh15 = float3(sampledSH404142.ba, sampledSH434445.r);

#if SH_FORMAT != FORMAT_UINT8
      shd3[0] = sh9;
      shd3[1] = sh10;
      shd3[2] = sh11;
      shd3[3] = sh12;
      shd3[4] = sh13;
      shd3[5] = sh14;
      shd3[6] = sh15;
#else
      shd3[0] = sh9 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      shd3[1] = sh10 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      shd3[2] = sh11 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      shd3[3] = sh12 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      shd3[4] = sh13 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      shd3[5] = sh14 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      shd3[6] = sh15 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
#endif
    }
  }
}

#endif 