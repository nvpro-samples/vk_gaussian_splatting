/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

// Convert sRGB to linear RGB (accurate version following sRGB spec)
float3 srgbToLinear(float3 srgb)
{
  float3 result;
  result.x = (srgb.x <= 0.04045) ? (srgb.x / 12.92) : pow((srgb.x + 0.055) / 1.055, 2.4);
  result.y = (srgb.y <= 0.04045) ? (srgb.y / 12.92) : pow((srgb.y + 0.055) / 1.055, 2.4);
  result.z = (srgb.z <= 0.04045) ? (srgb.z / 12.92) : pow((srgb.z + 0.055) / 1.055, 2.4);
  return result;
}

// Convert linear RGB to CIE XYZ color space (D65 illuminant)
float3 rgbToXYZ(float3 linearRgb)
{
  // sRGB/Rec.709 to XYZ transformation matrix
  float3 xyz;
  xyz.x = dot(linearRgb, float3(0.4124564, 0.3575761, 0.1804375));
  xyz.y = dot(linearRgb, float3(0.2126729, 0.7151522, 0.0721750));
  xyz.z = dot(linearRgb, float3(0.0193339, 0.1191920, 0.9503041));
  return xyz;
}

// Convert XYZ to perceptual opponent color space (simplified Hunt-adjusted)
float3 xyzToOpponentColor(float3 xyz)
{
  float Y     = xyz.y;
  float RG    = (xyz.x - xyz.y) * 0.5;           // Red-Green opponent
  float BY    = (xyz.y - xyz.z) * 0.3;           // Blue-Yellow opponent
  float Ystar = (Y > 0.008856) ? pow(Y, 1.0 / 3.0) : (7.787 * Y + 16.0 / 116.0);
  return float3(Ystar, RG, BY);
}

// Combined RGB to opponent color conversion (for perceptual metrics like FLIP)
float3 rgbToOpponentColor(float3 linearRgb)
{
  return xyzToOpponentColor(rgbToXYZ(linearRgb));
}

//==============================================================================
// FLIP Color Space Transformations (Reference Implementation)
//==============================================================================

// Convert linear RGB to LMS (Long, Medium, Short cone response)
// Uses Hunt-Pointer-Estevez matrix optimized for D65 illuminant
// Reference: FLIP paper, Andersson et al.
float3 linearRgbToLMS(float3 linearRgb)
{
  // Hunt-Pointer-Estevez transformation matrix
  const float3x3 RGB_TO_LMS = float3x3(
     0.31670331,  0.70299344, -0.01969366,  // L (Long wavelength - reddish)
     0.10938715,  0.87060437,  0.01990658,  // M (Medium wavelength - greenish)
     0.01840087,  0.10476914,  0.87470614   // S (Short wavelength - bluish)
  );
  return mul(RGB_TO_LMS, linearRgb);
}

// Apply Hunt luminance adaptation to LMS values
// Models how the visual system adapts to different luminance levels
// Reference: Hunt effect in color appearance models
float3 huntAdjustment(float3 lms, float adaptationLuminance)
{
  // Hunt adaptation factor based on adapting luminance
  // Default adaptation luminance = 1.0 for display-referred imagery
  float La = adaptationLuminance;
  
  // Hunt's formula: F_L = 0.2 * k^(1/3) * (1 - exp(-0.42 * k^(1/3)))
  // where k = 5 * La
  float k = 5.0 * La;
  float k_cbrt = pow(k, 1.0 / 3.0);
  float F_L = 0.2 * k_cbrt * (1.0 - exp(-0.42 * k_cbrt));
  
  // Apply adaptation factor to LMS values
  return lms * F_L;
}

// Convert Hunt-adjusted LMS to YCxCz perceptually uniform opponent space
// This is FLIP's custom color space optimized for perceptual uniformity
// Y = luminance, Cx = red-green, Cz = yellow-blue
float3 lmsToYCxCz(float3 huntLMS)
{
  // Transformation matrix from Hunt-LMS to YCxCz
  // Y is derived from M channel (most sensitive to luminance)
  // Cx and Cz are opponent channels
  const float3x3 LMS_TO_YCXCZ = float3x3(
    0.0,      1.0,     0.0,      // Y  = M (luminance from medium cone)
    1.0,      -1.0,    0.0,      // Cx = L - M (red-green opponent)
    0.0,      1.0,    -1.0       // Cz = M - S (yellow-blue opponent)
  );
  
  return mul(LMS_TO_YCXCZ, huntLMS);
}

// Complete color transformation pipeline for FLIP: sRGB → Linear → LMS → Hunt → YCxCz
float3 srgbToFLIPColorSpace(float3 srgb, float adaptationLuminance)
{
  float3 linear = srgbToLinear(srgb);
  float3 lms = linearRgbToLMS(linear);
  float3 huntLMS = huntAdjustment(lms, adaptationLuminance);
  float3 ycxcz = lmsToYCxCz(huntLMS);
  return ycxcz;
}

// Turbo colormap - perceptually uniform, great for depth
// With x in [0,1]
// https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization/
// https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
float3 turboColormap(in float x)
{
  const float4 kRedVec4   = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const float4 kGreenVec4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const float4 kBlueVec4  = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const float2 kRedVec2   = float2(-152.94239396, 59.28637943);
  const float2 kGreenVec2 = float2(4.27729857, 2.82956604);
  const float2 kBlueVec2  = float2(-89.90310912, 27.34824973);

  x         = saturate(x);
  float4 v4 = float4(1.0, x, x * x, x * x * x);
  float2 v2 = v4.zw * v4.z;
  return float3(dot(v4, kRedVec4) + dot(v2, kRedVec2), dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
}
