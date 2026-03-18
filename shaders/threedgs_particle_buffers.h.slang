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

#ifndef _THREEDGS_PARTICLE_BUFFERS_H_
#define _THREEDGS_PARTICLE_BUFFERS_H_

#include <shaderio.h>

// ===================================================================================
// Buffer fetch functions for per-particle splat data (centers, scales, rotations, etc.)
//
// All functions use explicit 64-bit byte offset arithmetic to compute the element
// address, rather than pointer subscript notation (e.g. ptr[uint64_index]).
//
// Reason: Slang may generate SPIR-V that truncates the byte offset in ptr[index]
// to 32-bit, causing address wrap-around when the buffer exceeds 4 GB.
// By computing the byte offset explicitly in uint64 and adding it to the base
// device address, the full 64-bit range is preserved. The final element access
// uses only small constant indices (0..44) that are safe in 32-bit.
//
// 4 GB overflow thresholds per buffer (single splat set, float32):
//   Centers     (3 floats/splat):  >358M splats
//   Scales      (3 floats/splat):  >358M splats
//   Rotations   (4 floats/splat):  >268M splats
//   Colors f32  (4 floats/splat):  >268M splats
//   Covariances (6 floats/splat):  >179M splats  (first to overflow in practice)
//   SH degree 3 (45 elems/splat):  >95M (f32), >191M (f16), >382M (u8)
// ===================================================================================

// 3 floats per splat (x, y, z) — buffer exceeds 4 GB above ~358M splats
float3 fetchCenterFromBuffer(in SplatSetDesc desc, in uint splatIndex)
{
  uint64_t byteOffset = uint64_t(splatIndex) * uint64_t(3) * uint64_t(sizeof(float));
  float*   ptr        = (float*)(uint64_t(desc.centersAddress) + byteOffset);
  return float3(ptr[0], ptr[1], ptr[2]);
}

// 3 floats per splat (sx, sy, sz) — buffer exceeds 4 GB above ~358M splats
float3 fetchScaleFromBuffer(in SplatSetDesc desc, in uint splatIndex)
{
  uint64_t byteOffset = uint64_t(splatIndex) * uint64_t(3) * uint64_t(sizeof(float));
  float*   ptr        = (float*)(uint64_t(desc.scalesAddress) + byteOffset);
  return float3(ptr[0], ptr[1], ptr[2]);
}

// 4 floats per splat (quaternion) — buffer exceeds 4 GB above ~268M splats
float4 fetchRotationFromBuffer(in SplatSetDesc desc, in uint splatIndex)
{
  uint64_t byteOffset = uint64_t(splatIndex) * uint64_t(4) * uint64_t(sizeof(float));
  float*   ptr        = (float*)(uint64_t(desc.rotationsAddress) + byteOffset);
  return float4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

// 4 elements per splat (RGBA) — overflow threshold depends on format:
//   float32: >268M, float16: >537M, uint8: >1073M
float4 fetchColorFromBuffer(in SplatSetDesc desc, in uint splatIndex)
{
#if RGBA_FORMAT == FORMAT_FLOAT32
  uint64_t byteOffset = uint64_t(splatIndex) * uint64_t(4) * uint64_t(sizeof(float));
  float*   ptr        = (float*)(desc.colorsAddress + byteOffset);
  return float4(ptr[0], ptr[1], ptr[2], ptr[3]);
#elif RGBA_FORMAT == FORMAT_FLOAT16
  uint64_t   byteOffset = uint64_t(splatIndex) * uint64_t(4) * uint64_t(sizeof(float16_t));
  float16_t* ptr        = (float16_t*)(desc.colorsAddress + byteOffset);
  return float4(float(ptr[0]), float(ptr[1]), float(ptr[2]), float(ptr[3]));
#elif RGBA_FORMAT == FORMAT_UINT8
  uint64_t byteOffset = uint64_t(splatIndex) * uint64_t(4) * uint64_t(sizeof(uint8_t));
  uint8_t* ptr        = (uint8_t*)(desc.colorsAddress + byteOffset);
  return float4(float(ptr[0]) / 255.0f, float(ptr[1]) / 255.0f,
                float(ptr[2]) / 255.0f, float(ptr[3]) / 255.0f);
#else
#error "Unsupported RGBA format"
#endif
}

// 6 floats per splat (symmetric 3x3 matrix) — buffer exceeds 4 GB above ~179M splats.
// This is the first buffer to overflow in practice for large single-file models.
float3x3 fetchCovarianceFromBuffer(in SplatSetDesc desc, in uint splatIndex)
{
  uint64_t byteOffset = uint64_t(splatIndex) * uint64_t(6) * uint64_t(sizeof(float));
  float*   ptr        = (float*)(uint64_t(desc.covariancesAddress) + byteOffset);

  const float3 cov3D_M11_M12_M13 = float3(ptr[0], ptr[1], ptr[2]);
  const float3 cov3D_M22_M23_M33 = float3(ptr[3], ptr[4], ptr[5]);

  return float3x3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
                  cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
}

// 45 elements per splat (SH degree 3) — buffer exceeds 4 GB above ~95M (f32), ~191M (f16), ~382M (u8).
// Caller must ensure that requested degree is <= desc.shDegree.
void fetchShFromBuffer(in SplatSetDesc desc, in uint splatIndex, in uint degree, out float3 shd1[3], out float3 shd2[5], out float3 shd3[7])
{
  const uint splatStride = 45;  // Number of elements per splat

#if SH_FORMAT == FORMAT_FLOAT32
  uint64_t byteOffset = uint64_t(splatStride) * uint64_t(splatIndex) * uint64_t(sizeof(float));
  float*   shPtr      = (float*)(desc.shAddress + byteOffset);
#elif SH_FORMAT == FORMAT_FLOAT16
  uint64_t   byteOffset = uint64_t(splatStride) * uint64_t(splatIndex) * uint64_t(sizeof(float16_t));
  float16_t* shPtr      = (float16_t*)(desc.shAddress + byteOffset);
#elif SH_FORMAT == FORMAT_UINT8
  uint64_t byteOffset                                    = uint64_t(splatStride) * uint64_t(splatIndex) * uint64_t(sizeof(uint8_t));
  uint8_t* shPtr                                         = (uint8_t*)(desc.shAddress + byteOffset);
  const float SphericalHarmonics8BitCompressionRange     = 2.0;
  const float SphericalHarmonics8BitCompressionHalfRange = SphericalHarmonics8BitCompressionRange / 2.0f;
#else
#error "Unsupported SH format"
#endif

  // Degree 1 (3 coefficients)
#if SH_FORMAT == FORMAT_UINT8
  shd1[0] = float3(float(shPtr[0]), float(shPtr[1]), float(shPtr[2]))
                / 255.0f * SphericalHarmonics8BitCompressionRange
            - SphericalHarmonics8BitCompressionHalfRange;
  shd1[1] = float3(float(shPtr[3]), float(shPtr[4]), float(shPtr[5]))
                / 255.0f * SphericalHarmonics8BitCompressionRange
            - SphericalHarmonics8BitCompressionHalfRange;
  shd1[2] = float3(float(shPtr[6]), float(shPtr[7]), float(shPtr[8]))
                / 255.0f * SphericalHarmonics8BitCompressionRange
            - SphericalHarmonics8BitCompressionHalfRange;
#else
  shd1[0] = float3(float(shPtr[0]), float(shPtr[1]), float(shPtr[2]));
  shd1[1] = float3(float(shPtr[3]), float(shPtr[4]), float(shPtr[5]));
  shd1[2] = float3(float(shPtr[6]), float(shPtr[7]), float(shPtr[8]));
#endif

  if (degree >= 2)
  {
    // Degree 2 (5 coefficients)
#if SH_FORMAT == FORMAT_UINT8
    shd2[0] = float3(float(shPtr[9]), float(shPtr[10]), float(shPtr[11]))
                  / 255.0f * SphericalHarmonics8BitCompressionRange
              - SphericalHarmonics8BitCompressionHalfRange;
    shd2[1] = float3(float(shPtr[12]), float(shPtr[13]), float(shPtr[14]))
                  / 255.0f * SphericalHarmonics8BitCompressionRange
              - SphericalHarmonics8BitCompressionHalfRange;
    shd2[2] = float3(float(shPtr[15]), float(shPtr[16]), float(shPtr[17]))
                  / 255.0f * SphericalHarmonics8BitCompressionRange
              - SphericalHarmonics8BitCompressionHalfRange;
    shd2[3] = float3(float(shPtr[18]), float(shPtr[19]), float(shPtr[20]))
                  / 255.0f * SphericalHarmonics8BitCompressionRange
              - SphericalHarmonics8BitCompressionHalfRange;
    shd2[4] = float3(float(shPtr[21]), float(shPtr[22]), float(shPtr[23]))
                  / 255.0f * SphericalHarmonics8BitCompressionRange
              - SphericalHarmonics8BitCompressionHalfRange;
#else
    shd2[0] = float3(float(shPtr[9]), float(shPtr[10]), float(shPtr[11]));
    shd2[1] = float3(float(shPtr[12]), float(shPtr[13]), float(shPtr[14]));
    shd2[2] = float3(float(shPtr[15]), float(shPtr[16]), float(shPtr[17]));
    shd2[3] = float3(float(shPtr[18]), float(shPtr[19]), float(shPtr[20]));
    shd2[4] = float3(float(shPtr[21]), float(shPtr[22]), float(shPtr[23]));
#endif

    if (degree >= 3)
    {
      // Degree 3 (7 coefficients)
#if SH_FORMAT == FORMAT_UINT8
      shd3[0] = float3(float(shPtr[24]), float(shPtr[25]), float(shPtr[26]))
                    / 255.0f * SphericalHarmonics8BitCompressionRange
                - SphericalHarmonics8BitCompressionHalfRange;
      shd3[1] = float3(float(shPtr[27]), float(shPtr[28]), float(shPtr[29]))
                    / 255.0f * SphericalHarmonics8BitCompressionRange
                - SphericalHarmonics8BitCompressionHalfRange;
      shd3[2] = float3(float(shPtr[30]), float(shPtr[31]), float(shPtr[32]))
                    / 255.0f * SphericalHarmonics8BitCompressionRange
                - SphericalHarmonics8BitCompressionHalfRange;
      shd3[3] = float3(float(shPtr[33]), float(shPtr[34]), float(shPtr[35]))
                    / 255.0f * SphericalHarmonics8BitCompressionRange
                - SphericalHarmonics8BitCompressionHalfRange;
      shd3[4] = float3(float(shPtr[36]), float(shPtr[37]), float(shPtr[38]))
                    / 255.0f * SphericalHarmonics8BitCompressionRange
                - SphericalHarmonics8BitCompressionHalfRange;
      shd3[5] = float3(float(shPtr[39]), float(shPtr[40]), float(shPtr[41]))
                    / 255.0f * SphericalHarmonics8BitCompressionRange
                - SphericalHarmonics8BitCompressionHalfRange;
      shd3[6] = float3(float(shPtr[42]), float(shPtr[43]), float(shPtr[44]))
                    / 255.0f * SphericalHarmonics8BitCompressionRange
                - SphericalHarmonics8BitCompressionHalfRange;
#else
      shd3[0] = float3(float(shPtr[24]), float(shPtr[25]), float(shPtr[26]));
      shd3[1] = float3(float(shPtr[27]), float(shPtr[28]), float(shPtr[29]));
      shd3[2] = float3(float(shPtr[30]), float(shPtr[31]), float(shPtr[32]));
      shd3[3] = float3(float(shPtr[33]), float(shPtr[34]), float(shPtr[35]));
      shd3[4] = float3(float(shPtr[36]), float(shPtr[37]), float(shPtr[38]));
      shd3[5] = float3(float(shPtr[39]), float(shPtr[40]), float(shPtr[41]));
      shd3[6] = float3(float(shPtr[42]), float(shPtr[43]), float(shPtr[44]));
#endif
    }
  }
}

#endif
