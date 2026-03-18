/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Octahedral Normal Encoding/Decoding
// Reference: Meyer et al. 2010 "On Floating-Point Normal Vectors"
// Projects unit vector to octahedron, then to unit square for 2D encoding

#ifndef OCTAHEDRAL_NORMAL_H_SLANG
#define OCTAHEDRAL_NORMAL_H_SLANG

// Helper: wrap octahedral coordinates for the bottom hemisphere
float2 octWrap(float2 v)
{
  return (1.0 - abs(v.yx)) * select(v.xy >= 0.0, float2(1.0), float2(-1.0));
}

// Encode unit normal to octahedral representation in [-1, 1]^2
float2 octEncode(float3 n)
{
  // Project to octahedron (L1 norm = 1)
  float2 p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));

  // Wrap bottom hemisphere
  return (n.z < 0.0) ? octWrap(p) : p;
}

// Decode octahedral representation back to unit normal
float3 octDecode(float2 f)
{
  float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));

  // Unwrap bottom hemisphere
  if(n.z < 0.0)
  {
    n.xy = octWrap(n.xy);
  }

  return normalize(n);
}

// Quantize float2 in [-1,1] to packed uint (16 bits per component)
// Uses signed normalization: [-1,1] -> [0, 65535]
uint octPack(float2 f)
{
  // Map [-1, 1] to [0, 65535]
  uint x = uint(clamp((f.x * 0.5 + 0.5) * 65535.0, 0.0, 65535.0));
  uint y = uint(clamp((f.y * 0.5 + 0.5) * 65535.0, 0.0, 65535.0));
  return (y << 16) | x;
}

// Unpack uint back to float2 in [-1, 1]
float2 octUnpack(uint packed)
{
  float x = float(packed & 0xFFFF) / 65535.0 * 2.0 - 1.0;
  float y = float(packed >> 16) / 65535.0 * 2.0 - 1.0;
  return float2(x, y);
}

// Encode unit normal to packed uint (full pipeline)
uint encodeNormalOctahedral(float3 n)
{
  return octPack(octEncode(n));
}

// Decode packed uint back to unit normal (full pipeline)
float3 decodeNormalOctahedral(uint packed)
{
  return octDecode(octUnpack(packed));
}

#endif  // OCTAHEDRAL_NORMAL_H_SLANG

