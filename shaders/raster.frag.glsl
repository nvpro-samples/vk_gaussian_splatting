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
#version 450


#extension GL_EXT_mesh_shader : require
#extension GL_KHR_vulkan_glsl : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#include "device_host.h"

precision highp float;

layout(location = 0) in vec2 inFragPos;
layout(location = 1) perprimitiveEXT in flat vec4 inSplatCol;

layout(location = 0) out vec4 outColor;

// we could write to manage alignment automatically
// layout(set = 0, binding = 0, scalar) uniform FrameInfo_
// but it may be less performant than aligning 
// attribute in the struct (see device_host.h comment)
layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

void main () {

    // Compute the positional squared distance from the center of the splat to the current fragment.
    const float A = dot(inFragPos, inFragPos);
    // Since the positional data in vPosition has been scaled by sqrt(8), the squared result will be
    // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
    // defined by the rectangle formed by vPosition. It also means it's farther
    // away than sqrt(8) standard deviations from the mean.
    if (A > 8.0) discard;
    const vec3 color = inSplatCol.rgb;

    // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
    // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
    // and since 'mean' is zero, we have X * X, which is the same as A:
    // disableAlphaGaussian flag is used to enable/disable the opacity and force to 1
    const float opacity = float(frameInfo.opacityGaussianDisabled)
                    + float(1 - frameInfo.opacityGaussianDisabled) * exp(-0.5 * A) * inSplatCol.a;

    outColor = vec4(color.rgb, opacity);

}
