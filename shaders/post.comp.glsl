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

#version 460

#extension GL_GOOGLE_include_directive : enable
#include "shaderio.h"

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform _frameInfo
{
  FrameInfo frameInfo;
};

layout(set = 0, binding = POST_BINDING_MAIN_IMAGE, rgba8) uniform image2D mainColorImage;
layout(set = 0, binding = POST_BINDING_AUX1_IMAGE, rgba8) uniform image2D aux1ColorImage;

layout(local_size_x = 32) in;
layout(local_size_Y = 32) in;

void main()
{
  const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

  if(pixel.x >= int(frameInfo.viewport.x) || pixel.y >= int(frameInfo.viewport.y) )
    return;

  // Do accumulation over time
  const float a          = 1.0F / float(frameInfo.frameSampleId + 1);
  const vec4  mainColor = imageLoad(mainColorImage, pixel);
  const vec4  aux1Color = imageLoad(aux1ColorImage, pixel);
  vec4        finalColor = mix(mainColor, aux1Color, a);

  imageStore(mainColorImage, pixel, vec4(finalColor.rgb, 1.0));
}
