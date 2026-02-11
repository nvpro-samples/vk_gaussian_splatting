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

#pragma once

/// Converts HSB (Hue, Saturation, Brightness) color space to RGB color space
/// This function performs a smooth color space transformation using polynomial interpolation
/// for high-quality color conversion. The conversion uses a smoothstep-like curve for the
/// RGB component interpolation, producing perceptually smooth color gradients.
///
/// @param hsbColor Input color in HSB format where:
///   - x (Hue): Color angle in normalized range [0..1], where 0=red, 1/6=yellow,
///              1/3=green, 1/2=cyan, 2/3=blue, 5/6=magenta, 1=red (wraps around)
///   - y (Saturation): Color intensity/purity in range [0..1], where 0=grayscale, 1=pure color
///   - z (Brightness): Color brightness/value in range [0..1], where 0=black, 1=full brightness
/// @return RGB color in range [0..1] for each component (red, green, blue)
///
/// Reference: https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB

float3 hsbToRgb(float3 hsbColor)
{
  float3 rgb = clamp(abs(fmod((hsbColor.x * 6.0) + float3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  rgb        = (rgb * rgb * (3.0 - (2.0 * rgb)));
  return (hsbColor.z * lerp(float3(1.0), rgb, hsbColor.y));
}
