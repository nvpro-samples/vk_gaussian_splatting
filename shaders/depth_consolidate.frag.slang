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

// Depth consolidation fragment shader
// Reads picked depth from the raster depth buffer (storage image) and writes to hardware depth buffer
// This consolidates mesh depth (already in hw depth) with splat picked depth (in storage image)

#include "shaderio.h"

// Storage image containing picked depth (R) and transmittance (G)
[[vk::binding(BINDING_RASTER_DEPTH, 0)]] RWTexture2D<float2> depthTransmittanceBuffer;

struct FragmentOutput
{
  float depth : SV_Depth;
};

[shader("fragment")]
FragmentOutput main(float4 fragCoord : SV_Position)
{
  FragmentOutput output;
  
  int2 pixelCoord = int2(fragCoord.xy);
  float2 depthTrans = depthTransmittanceBuffer[pixelCoord];
  float pickedDepth = depthTrans.r;
  
  // If picked depth is valid (> 0), write it to hw depth buffer
  // Otherwise discard to keep existing mesh depth
  if(pickedDepth > 0.0001)
  {
    output.depth = pickedDepth;
  }
  else
  {
    discard;
  }
  
  return output;
}

