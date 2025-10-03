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

#ifndef _CAMERAS_
#define _CAMERAS_

#include "nvshaders/slang_types.h"
#include "nvshaders/random.h.slang"
#include "nvshaders/constants.h.slang"

void generatePinholeRay(in vec2 pixel, in vec2 resolution, in mat4 viewInverse, in mat4 projInverse, out vec3 rayOrigin, out vec3 rayDirection)
{
  // Computes the ray for this fragment (pinhole)
  const vec2 pixelCenter = pixel + vec2(0.5);
  const vec2 inUV        = pixelCenter / resolution;
  vec2       d           = inUV * 2.0 - 1.0;

  rayOrigin = (viewInverse * vec4(0, 0, 0, 1)).xyz;

  vec4 target  = projInverse * vec4(d.x, d.y, 1, 1);
  rayDirection = normalize(viewInverse * vec4(target.xyz, 0)).xyz;
}

// return false if pixel is out of fov
bool generateFisheyeRay(in vec2 pixel, in vec2 resolution, in float fov, in vec2 principalPoint, in mat4 viewInverse, out vec3 rayOrigin, out vec3 rayDirection)
{
  // Account for principal point (offsets from the center)
  pixel.x -= principalPoint.x;
  pixel.y += principalPoint.y;

  // pixel values are now in normalized device coordinates [-1.0,1.0]
  const vec2  ndc = (pixel / (resolution - 1.0)) * 2.0 - vec2(1.0);
  const float u   = ndc.x;
  const float v   = ndc.y;

  const float r        = sqrt(u * u + v * v);
  const bool  outOfFov = r > 1.0;

  const float epsilon = 1e-9;
  float       phiCos  = abs(r) > epsilon ? u / r : 0.0;
  phiCos              = clamp(phiCos, -1.0, 1.0);
  float phi           = acos(phiCos);
  phi                 = v < 0.0 ? -phi : phi;
  const float theta   = r * fov * 0.5;

  const vec3 direction = vec3(cos(phi) * sin(theta), -sin(phi) * sin(theta), -cos(theta));

  // Transform ray direction from camera space to world space
  rayDirection = normalize(viewInverse * vec4(direction, 0)).xyz;
  // Generate ray origin in world coordinates
  rayOrigin = (viewInverse * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

  return !outOfFov;
}

// Applies thin-lens depth-of-field perturbation to the ray origin and direction
void depthOfField(inout uint seed, in float focusDist, in float aperture, in mat4 viewInverse, inout vec3 rayOrigin, inout vec3 rayDirection)
{
  // This correctly finds where the original ray would hit the focal plane.
  const vec3 focalPoint = rayDirection * focusDist;

  // Random Sampling on Lens
  const float r1 = rand(seed) * M_TWO_PI;
  const float r2 = rand(seed) * aperture;

  // Aperture Position
  const vec4 camRight          = viewInverse * vec4(1, 0, 0, 0);
  const vec4 camUp             = viewInverse * vec4(0, 1, 0, 0);
  const vec3 randomAperturePos = (cos(r1) * camRight.xyz + sin(r1) * camUp.xyz) * sqrt(r2);

  // New Ray Direction
  const vec3 finalRayDir = normalize(focalPoint - randomAperturePos);

  // Set the new ray origin and direction with depth-of-field perturbation
  rayOrigin += randomAperturePos;  // Random point on lens
  rayDirection = finalRayDir;      // Still goes through focal point
}

#endif