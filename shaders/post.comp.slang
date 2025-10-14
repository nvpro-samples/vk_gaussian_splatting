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

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]  ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(POST_BINDING_MAIN_IMAGE, 0)]] RWTexture2D<float4>       mainColorImage;
[[vk::binding(POST_BINDING_AUX1_IMAGE, 0)]] RWTexture2D<float4>       aux1ColorImage;
// clang-format on

[numthreads(32, 32, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const int2 pixel = int2(dispatchThreadID.xy);

  if(pixel.x >= int(frameInfo.viewport.x) || pixel.y >= int(frameInfo.viewport.y))
    return;

  // Do accumulation over time
  const float  a          = 1.0F / float(frameInfo.frameSampleId + 1);
  const float4 mainColor  = mainColorImage[pixel];
  const float4 aux1Color  = aux1ColorImage[pixel];
  float4       finalColor = lerp(mainColor, aux1Color, a);

  mainColorImage[pixel] = float4(finalColor.rgb, 1.0);
}
