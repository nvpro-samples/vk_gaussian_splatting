/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#extension GL_EXT_shader_explicit_arithmetic_types : require

#if DATA_STORAGE == STORAGE_TEXTURES
// textures map describing the 3DGS model
layout(set = 0, binding = BINDING_CENTERS_TEXTURE) uniform sampler2D centersTexture;
layout(set = 0, binding = BINDING_COLORS_TEXTURE) uniform sampler2D colorsTexture;
layout(set = 0, binding = BINDING_COVARIANCES_TEXTURE) uniform sampler2D covariancesTexture;
layout(set = 0, binding = BINDING_SH_TEXTURE) uniform sampler2D sphericalHarmonicsTexture;
#else
// buffers describing the 3DGS model (alternative to textures)
layout(set = 0, binding = BINDING_CENTERS_BUFFER) buffer _centersBuffer
{
  float centersBuffer[];
};
layout(set = 0, binding = BINDING_COLORS_BUFFER) buffer _colorsBuffer
{
  float colorsBuffer[];
};
layout(set = 0, binding = BINDING_COVARIANCES_BUFFER) buffer _covariancesBuffer
{
  float covariancesBuffer[];
};
layout(set = 0, binding = BINDING_SH_BUFFER) buffer _sphericalHarmonicsBuffer
{
#if SH_FORMAT == FORMAT_FLOAT32
  float sphericalHarmonicsBuffer[];
#else
#if SH_FORMAT == FORMAT_FLOAT16
  float16_t sphericalHarmonicsBuffer[];
#else
#if SH_FORMAT == FORMAT_UINT8
  uint8_t sphericalHarmonicsBuffer[];
#else
#error "Unsupported SH format"
#endif
#endif
#endif
};
#endif

////////////
// constants

const float sqrt8   = sqrt(8.0);
const float SH_C1   = 0.4886025119029199f;
const float SH_C2[] = {1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742};
const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                       -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

// data texture accessors
ivec2 getDataPos(in uint splatIndex, in uint stride, in uint offset, in ivec2 dimensions)
{
  const uint fullOffset = splatIndex * stride + offset;

  return ivec2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

// data texture accessors
ivec2 getDataPosF(in uint splatIndex, in float stride, in uint offset, in ivec2 dimensions)
{
  const uint fullOffset = uint(float(splatIndex) * stride) + offset;

  return ivec2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

#if DATA_STORAGE == STORAGE_TEXTURES
// fetch center value from texture map
vec3 fetchCenter(in uint splatIndex)
{
  return vec3(texelFetch(centersTexture, getDataPos(splatIndex, 1, 0, textureSize(centersTexture, 0)), 0));
}
#else
// fetch center value from data buffer
vec3 fetchCenter(in uint splatIndex)
{
  return vec3(centersBuffer[splatIndex * 3 + 0], centersBuffer[splatIndex * 3 + 1], centersBuffer[splatIndex * 3 + 2]);
}
#endif

#if DATA_STORAGE == STORAGE_TEXTURES
// fetchColor replaces fetchSH0 since non view dependent color is precomputed on CPU
vec4 fetchColor(in uint splatIndex)
{
  return texelFetch(colorsTexture, getDataPos(splatIndex, 1, 0, textureSize(colorsTexture, 0)), 0);
}
#else
// fetch center value from data buffer
vec4 fetchColor(in uint splatIndex)
{
  return vec4(colorsBuffer[splatIndex * 4 + 0], colorsBuffer[splatIndex * 4 + 1], colorsBuffer[splatIndex * 4 + 2],
              colorsBuffer[splatIndex * 4 + 3]);
}
#endif

#if DATA_STORAGE == STORAGE_TEXTURES
// fetch from data textures
void fetchSh(in uint  splatIndex,
             out vec3 shd1[3]
#if MAX_SH_DEGREE >= 2
             , out vec3 shd2[5]
#endif
#if MAX_SH_DEGREE >= 3
             , out vec3 shd3[7]
#endif
)
{
  const float SphericalHarmonics8BitCompressionRange = 2.0;
  const vec3  vec8BitSHShift                         = vec3(SphericalHarmonics8BitCompressionRange / 2.0);
  
  const uint stride = 12;  // 12 for degree 3, 6 for degree 2
  // fetching degree 1
  const vec4 sampledSH0123 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 0, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH4567 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 1, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH891011 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 2, textureSize(sphericalHarmonicsTexture, 0)), 0);

  const vec3 sh1 = sampledSH0123.rgb;
  const vec3 sh2 = vec3(sampledSH0123.a, sampledSH4567.rg);
  const vec3 sh3 = vec3(sampledSH4567.ba, sampledSH891011.r);

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
#if MAX_SH_DEGREE >= 2
  const vec4 sampledSH12131415 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 3, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH16171819 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 4, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH20212223 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 5, textureSize(sphericalHarmonicsTexture, 0)), 0);

  const vec3 sh4 = sampledSH891011.gba;
  const vec3 sh5 = sampledSH12131415.rgb;
  const vec3 sh6 = vec3(sampledSH12131415.a, sampledSH16171819.rg);
  const vec3 sh7 = vec3(sampledSH16171819.ba, sampledSH20212223.r);
  const vec3 sh8 = sampledSH20212223.gba;

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
#endif

  // Fetching degree 3
#if MAX_SH_DEGREE >= 3
  const vec4 sampledSH24252627 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 6, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH28293031 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 7, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH32333435 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 8, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH36373839 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 9, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH404142 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 10, textureSize(sphericalHarmonicsTexture, 0)), 0);
  const vec4 sampledSH434445 =
      texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, stride, 11, textureSize(sphericalHarmonicsTexture, 0)), 0);

  const vec3 sh9  = sampledSH24252627.rgb;
  const vec3 sh10 = vec3(sampledSH24252627.a, sampledSH28293031.rg);
  const vec3 sh11 = vec3(sampledSH28293031.ba, sampledSH32333435.r );
  const vec3 sh12 = sampledSH32333435.gba;
  const vec3 sh13 = sampledSH36373839.rgb;
  const vec3 sh14 = vec3(sampledSH36373839.a, sampledSH404142.rg);
  const vec3 sh15 = vec3(sampledSH404142.ba, sampledSH434445.r);

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
#endif

}
#else
// fetch from data buffers
void fetchSh(
  in uint splatIndex ,out vec3 shd1[3] 
#if MAX_SH_DEGREE >= 2
  ,out vec3 shd2[5]
#endif
#if MAX_SH_DEGREE >= 3
  ,out vec3 shd3[7]
#endif
)
{
  const uint splatStride = 45;

  const float SphericalHarmonics8BitCompressionRange     = 2.0;
  const float SphericalHarmonics8BitCompressionHalfRange = SphericalHarmonics8BitCompressionRange / 2.0;
  const vec3  vec8BitSHShift                             = vec3(SphericalHarmonics8BitCompressionHalfRange);
  const float SphericalHarmonics8BitScale                = SphericalHarmonics8BitCompressionRange / 255.0f;

  // fetching degree 1
  const vec3 sh1 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 2]);

  const vec3 sh2 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 2]);

  const vec3 sh3 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd1[0] = sh1;
  shd1[1] = sh2;
  shd1[2] = sh3;
#else
  shd1[0] = sh1 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd1[1] = sh2 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd1[2] = sh3 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif

  // fetching degree 2
#if MAX_SH_DEGREE >= 2
  const vec3 sh4 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 2]);

  const vec3 sh5 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 2]);

  const vec3 sh6 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 2]);

  const vec3 sh7 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 2]);

  const vec3 sh8 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd2[0] = sh4;
  shd2[1] = sh5;
  shd2[2] = sh6;
  shd2[3] = sh7;
  shd2[4] = sh8;
#else
  shd2[0] = sh4 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[1] = sh5 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[2] = sh6 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[3] = sh7 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[4] = sh8 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif
#endif

  // fetching degree 3
#if MAX_SH_DEGREE >= 3
  const vec3 sh9 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 0],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 1],
                        sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 2]);

  const vec3 sh10 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 0],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 1],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 2]);

  const vec3 sh11 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 0],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 1],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 2]);

  const vec3 sh12 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 0],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 1],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 2]);

  const vec3 sh13 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 0],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 1],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 2]);

  const vec3 sh14 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 0],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 1],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 2]);

  const vec3 sh15 = vec3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 0],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 1],
                         sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd3[0] = sh9;
  shd3[1] = sh10;
  shd3[2] = sh11;
  shd3[3] = sh12;
  shd3[4] = sh13;
  shd3[5] = sh14;
  shd3[6] = sh15;
#else
  shd3[0] = sh9 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[1] = sh10 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[2] = sh11 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[3] = sh12 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[4] = sh13 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[5] = sh14 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[6] = sh15 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif
#endif
}
#endif

#if DATA_STORAGE == STORAGE_TEXTURES
mat3 fetchCovariance(in uint splatIndex)
{
  // Use RGBA texture map to store sets of 3 elements requires some offset shifting depending on splatIndex

  const uint  oddOffset        = uint(splatIndex) & uint(0x00000001);
  const uint  doubleOddOffset  = oddOffset * uint(2);
  const bool  isEven           = oddOffset == uint(0);
  const uint  nearestEvenIndex = uint(splatIndex) - oddOffset;
  const float fOddOffset       = float(oddOffset);

  const vec4 sampledCovarianceA =
      texelFetch(covariancesTexture, getDataPosF(nearestEvenIndex, 1.5, oddOffset, textureSize(covariancesTexture, 0)), 0);

  const vec4 sampledCovarianceB =
      texelFetch(covariancesTexture,
                 getDataPosF(nearestEvenIndex, 1.5, oddOffset + uint(1), textureSize(covariancesTexture, 0)), 0);

  const vec3 cov3D_M11_M12_M13 =
      vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) + vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;

  const vec3 cov3D_M22_M23_M33 =
      vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) + vec3(sampledCovarianceB.gba) * fOddOffset;

  return mat3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
              cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
}
#else
mat3 fetchCovariance(in uint splatIndex)
{
  // Use RGBA texture map to store sets of 3 elements requires some offset shifting depending on splatIndex
  const vec3 cov3D_M11_M12_M13 = vec3(covariancesBuffer[splatIndex * 6 + 0], covariancesBuffer[splatIndex * 6 + 1],
                                      covariancesBuffer[splatIndex * 6 + 2]);
  const vec3 cov3D_M22_M23_M33 = vec3(covariancesBuffer[splatIndex * 6 + 3], covariancesBuffer[splatIndex * 6 + 4],
                                      covariancesBuffer[splatIndex * 6 + 5]);

  return mat3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
              cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
}
#endif