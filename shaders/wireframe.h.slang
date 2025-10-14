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

#include "nvshaders/slang_types.h"
#include "nvshaders/constants.h.slang"

/* thickness;thicknessVar;smoothing;screenSpace;backFaceColor;enableDash;dashRepeats;dashLength;;
    {1.0F, {1.0F, 1.0F}, 1.0F, 1, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F},    // default
    {1.0F, {1.0F, 1.0F}, 0.5F, 1, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F},    // Wire dot
    {0.1F, {0.0F, 1.0F}, 0.1F, 0, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F},    // Star
    {0.1F, {0.7F, 0.0F}, 0.1F, 0, {0.07F, 0.0F, 0.0F}, 1, 10, 0.8F},  // Flake
    {0.3F, {1.F, 1.F}, 2.F, 1, {0.06F, 0.06F, 0.06F}, 0, 8, 0.4F},    // Thin
    {0.5F, {1.F, 1.F}, 1.F, 1, {0.1F, 0.1F, 0.1F}, 1, 1, 1.F},        // Wire line
    {0.5F, {1.F, 1.F}, 1.F, 1, {0.1F, 0.1F, 0.1F}, 1, 20, 0.5F},      // Stipple
*/

struct WireframeSettings
{
  float thickness;
  vec2  thicknessVar;
  float smoothing;
  bool  stipple;
  float stippleLength;
  float stippleRepeats;
  bool  screenSpace;
};

WireframeSettings wireframeDefaultSettings()
{
  WireframeSettings settings;
  settings.thickness      = 1.0;
  settings.thicknessVar   = vec2(1.0, 1.0);
  settings.smoothing      = 1.0;
  settings.stipple        = false;
  settings.stippleLength  = 0.5;
  settings.stippleRepeats = 20.0;
  settings.screenSpace    = true;
  return settings;
}

// Return the width [0..1] for which the line should be displayed or not
float getLineWidth(in vec3 deltas, in float thickness, in float smoothing, in vec3 barys)
{
  barys         = smoothstep(deltas * (thickness), deltas * (thickness + smoothing), barys);
  float minBary = min(barys.x, min(barys.y, barys.z));
  return 1.0 - minBary;
}

// Position along the edge [0..1]
float edgePosition(vec3 baryWeights)
{
  return max(baryWeights.z, max(baryWeights.y, baryWeights.x));
}

// Return 0 or 1 if edgePos should be diplayed or not
float stipple(in float stippleRepeats, in float stippleLength, in float edgePos)
{
  float offset = 1.0 / stippleRepeats;
  offset *= 0.5 * stippleLength;
  float pattern = fract((edgePos + offset) * stippleRepeats);
  return 1.0 - step(stippleLength, pattern);
}

// Vary the thickness along the edge
float edgeThickness(in vec2 thicknessVar, in float edgePos)
{
  return mix(thicknessVar.x, thicknessVar.y, (1.0 - sin(edgePos * M_PI)));
}

// return a value in [0..1] that express the proportion of wireframe line contribution
// The returned value can be used to mix with object color for instance
float processWireframe(WireframeSettings settings, in float3 barys)
{

  // Wireframe Settings
  float thickness = settings.thickness * 0.5;  // Thickness for both side of the edge, must be divided by 2
  float smoothing = settings.thickness * settings.smoothing;  // Could be thickness

  // Uniform position on the edge [0, 1]
  float edgePos = edgePosition(barys);

  // Vary the thickness along the edge
  thickness *= edgeThickness(settings.thicknessVar, edgePos);

  // fwidth  return the sum of the absolute value of derivatives in x and y
  // which makes the width in screen space
  float3 deltas = settings.screenSpace ? fwidth(barys) : float3(1, 1, 1);

  // Get the wireframe line width
  float lineWidth = getLineWidth(deltas, thickness, smoothing, barys);

  if(settings.stipple)
  {
    float stippleFact = stipple(settings.stippleRepeats, settings.stippleLength, edgePos);
    lineWidth *= stippleFact;  // 0 or 1
  }

  return lineWidth;
}