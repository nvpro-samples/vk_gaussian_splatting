/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#extension GL_EXT_shader_explicit_arithmetic_types : require

////////////
// constants

const float sqrt8    = sqrt(8.0);
const float minAlpha = 1.0 / 255.0;

const float SH_C1    = 0.4886025119029199f;
const float[5] SH_C2 = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);

const float SphericalHarmonics8BitCompressionRange     = 3.0;
const float SphericalHarmonics8BitCompressionHalfRange = SphericalHarmonics8BitCompressionRange / 2.0;
const vec3  vec8BitSHShift                             = vec3(SphericalHarmonics8BitCompressionHalfRange);


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

