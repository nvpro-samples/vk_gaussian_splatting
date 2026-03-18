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

/*
 * Comparison Composite Shader
 * 
 * This compute shader composites two images (capture and current) with a 
 * vertical split divider that can be moved left/right. Used for before/after
 * comparison of rendering results. Supports showing reference, current render,
 * or difference images on each side.
 */

#include "image_compare_shaderio.h"
#include "color.h.slang"

// clang-format off
[[vk::binding(ComparisonBinding::eCaptureImage)]] Texture2D<float4> captureImage;
[[vk::binding(ComparisonBinding::eCurrentImage)]] Texture2D<float4> currentImage;
[[vk::binding(ComparisonBinding::eOutputImage)]] RWTexture2D<float4> outputImage;
[[vk::binding(ComparisonBinding::eSampler)]] SamplerState linearSampler;

[[vk::push_constant]] ConstantBuffer<PushConstantComparison> pushC;
// clang-format on

// Workgroup size
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
  int2 outputSize = pushC.outputSize;
  int2 pixelCoord = int2(threadID.xy);

  // Early exit if outside output bounds
  if(pixelCoord.x >= outputSize.x || pixelCoord.y >= outputSize.y)
    return;

  // Calculate split position in pixels
  float splitPosFloat = pushC.splitPosition * float(outputSize.x);
  int   splitPos      = int(splitPosFloat);

  // Divider width (total 5 pixels: 1 white center + 2 black on each side)
  const int dividerHalfWidth   = 2;
  const int dividerCenterWidth = 1;

  // Determine which side of the split we're on
  bool isOnLeftSide = pixelCoord.x < splitPos;

  // Check if we're in the divider region
  int  distFromSplit          = abs(pixelCoord.x - splitPos);
  bool isInDividerBlackBorder = distFromSplit <= dividerHalfWidth;
  bool isInDividerWhiteCenter = distFromSplit <= (dividerCenterWidth / 2);

  float4 color;

  if(isInDividerBlackBorder)
  {
    // Draw divider
    if(isInDividerWhiteCenter)
    {
      // White center line
      color = float4(1.0, 1.0, 1.0, 1.0);
    }
    else
    {
      // Black border
      color = float4(0.0, 0.0, 0.0, 1.0);
    }
  }
  else
  {
    // Calculate normalized UV coordinates in output space
    float2 outputUV = (float2(pixelCoord) + 0.5) / float2(outputSize);

    // Select which display mode to use based on which side of the divider we're on
    DisplayMode displayMode = isOnLeftSide ? pushC.leftSide : pushC.rightSide;
    color                   = getDisplayColor(displayMode, pixelCoord, outputUV, outputSize);
  }

  // Write output
  outputImage[pixelCoord] = color;
}

// Helper function to get the color for a specific display mode
float4 getDisplayColor(DisplayMode displayMode, int2 pixelCoord, float2 uv, int2 outputSize)
{
  if(displayMode == DisplayMode::eCapture)
  {
    return sampleImage(captureImage, pixelCoord, uv, pushC.captureImgSize, outputSize);
  }
  else if(displayMode == DisplayMode::eCurrent)
  {
    return sampleImage(currentImage, pixelCoord, uv, pushC.currentImgSize, outputSize);
  }
  else if(displayMode == DisplayMode::eDifferenceRaw)
  {
    // Grayscale difference mode
    float4 refColor = sampleImage(captureImage, pixelCoord, uv, pushC.captureImgSize, outputSize);
    float4 curColor = sampleImage(currentImage, pixelCoord, uv, pushC.currentImgSize, outputSize);

    // Compute absolute difference and amplify for visibility
    float4 diff = abs(refColor - curColor);
    diff        = min(diff * pushC.differenceAmplify, float4(1.0, 1.0, 1.0, 1.0));

    return float4(diff.rgb, 1.0);
  }
  else if(displayMode == DisplayMode::eDifferenceRedGray)
  {
    // Red on grayscale difference mode (like ImageMagick)
    float4 refColor = sampleImage(captureImage, pixelCoord, uv, pushC.captureImgSize, outputSize);
    float4 curColor = sampleImage(currentImage, pixelCoord, uv, pushC.currentImgSize, outputSize);

    // Compute absolute difference
    float4 diff = abs(refColor - curColor);

    // Calculate luminance of the difference to determine intensity
    float diffIntensity = dot(diff.rgb, float3(0.299, 0.587, 0.114));

    // Amplify and clamp
    diffIntensity = min(diffIntensity * pushC.differenceAmplify, 1.0);

    // Convert current image to grayscale for base
    float grayValue = dot(curColor.rgb, float3(0.299, 0.587, 0.114));

    // Blend red over grayscale base based on difference intensity
    float3 resultColor = lerp(float3(grayValue, grayValue, grayValue), float3(1.0, 0.0, 0.0), diffIntensity);

    return float4(resultColor, 1.0);
  }
  else if(displayMode == DisplayMode::eDifferenceRedOnly)
  {
    // Red only difference mode (no background)
    float4 refColor = sampleImage(captureImage, pixelCoord, uv, pushC.captureImgSize, outputSize);
    float4 curColor = sampleImage(currentImage, pixelCoord, uv, pushC.currentImgSize, outputSize);

    // Compute absolute difference
    float4 diff = abs(refColor - curColor);

    // Calculate luminance of the difference to determine intensity
    float diffIntensity = dot(diff.rgb, float3(0.299, 0.587, 0.114));

    // Amplify and clamp
    diffIntensity = min(diffIntensity * pushC.differenceAmplify, 1.0);

    // Show only red where there are differences, black elsewhere
    float3 resultColor = float3(diffIntensity, 0.0, 0.0);

    return float4(resultColor, 1.0);
  }
  else  // DisplayMode::eFLIPError
  {
    // FLIP perceptual error heatmap
    float4 refColor = sampleImage(captureImage, pixelCoord, uv, pushC.captureImgSize, outputSize);
    float4 curColor = sampleImage(currentImage, pixelCoord, uv, pushC.currentImgSize, outputSize);

    // Compute multi-scale spatial contrast for both images
    float refContrast = computeMultiScaleContrast(captureImage, pixelCoord, pushC.captureImgSize);
    float curContrast = computeMultiScaleContrast(currentImage, pixelCoord, pushC.currentImgSize);

    // Compute FLIP perceptual error
    float flipError = computeFLIP(refColor.rgb, curColor.rgb, refContrast, curContrast);

    // Convert to heatmap color using Turbo colormap (perceptually uniform)
    float3 heatmapColor = turboColormap(flipError);

    return float4(heatmapColor, 1.0);
  }
}

// Multi-scale spatial filtering for improved edge/texture detection
float computeMultiScaleContrast(Texture2D<float4> img, int2 coord, int2 imgSize)
{
  if(coord.x < 2 || coord.y < 2 || coord.x >= imgSize.x - 2 || coord.y >= imgSize.y - 2)
    return 0.0;

  float totalContrast = 0.0;

  // Scale 1: Fine details (3x3 kernel, weight 0.5)
  {
    float c  = dot(img.Load(int3(coord, 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float n  = dot(img.Load(int3(coord + int2(0, -1), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float s  = dot(img.Load(int3(coord + int2(0, 1), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float e  = dot(img.Load(int3(coord + int2(1, 0), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float w  = dot(img.Load(int3(coord + int2(-1, 0), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float gx = abs(e - w);
    float gy = abs(n - s);
    totalContrast += sqrt(gx * gx + gy * gy) * 0.5;
  }

  // Scale 2: Medium details (5x5 kernel, weight 0.3)
  {
    float c   = dot(img.Load(int3(coord, 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float n2  = dot(img.Load(int3(coord + int2(0, -2), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float s2  = dot(img.Load(int3(coord + int2(0, 2), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float e2  = dot(img.Load(int3(coord + int2(2, 0), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float w2  = dot(img.Load(int3(coord + int2(-2, 0), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float gx2 = abs(e2 - w2) * 0.5;
    float gy2 = abs(n2 - s2) * 0.5;
    totalContrast += sqrt(gx2 * gx2 + gy2 * gy2) * 0.3;
  }

  // Scale 3: Coarse details (diagonal, weight 0.2)
  {
    float ne  = dot(img.Load(int3(coord + int2(1, -1), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float sw  = dot(img.Load(int3(coord + int2(-1, 1), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float nw  = dot(img.Load(int3(coord + int2(-1, -1), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float se  = dot(img.Load(int3(coord + int2(1, 1), 0)).rgb, float3(0.2126, 0.7152, 0.0722));
    float gd1 = abs(ne - sw);
    float gd2 = abs(nw - se);
    totalContrast += sqrt(gd1 * gd1 + gd2 * gd2) * 0.2;
  }

  return totalContrast;
}

// Compute FLIP perceptual error
// See https://github.com/NVlabs/flip
float computeFLIP(float3 refColor, float3 curColor, float refContrast, float curContrast)
{
  float3 refLinear = srgbToLinear(refColor);
  float3 curLinear = srgbToLinear(curColor);

  float3 refOpp = rgbToOpponentColor(refLinear);
  float3 curOpp = rgbToOpponentColor(curLinear);

  float lumDiff    = abs(refOpp.x - curOpp.x);
  float chromaDiff = length(refOpp.yz - curOpp.yz);
  float colorError = lumDiff * 0.75 + chromaDiff * 0.25;

  float avgContrast         = (refContrast + curContrast) * 0.5;
  float contrastSensitivity = 1.0 / (1.0 + avgContrast * 8.0);
  float maxContrast         = max(refContrast, curContrast);
  float edgeMasking         = 1.0 - saturate(maxContrast * 4.0);

  float perceptualError = colorError * (0.4 + 0.6 * contrastSensitivity) * (0.3 + 0.7 * edgeMasking);
  perceptualError       = pow(saturate(perceptualError), 0.75);

  return perceptualError;
}

// Helper function to sample an image (handles both Load and SampleLevel)
float4 sampleImage(Texture2D<float4> image, int2 pixelCoord, float2 uv, int2 imageSize, int2 outputSize)
{
  if(all(imageSize == outputSize))
  {
    return image.Load(int3(pixelCoord, 0));
  }
  else
  {
    return image.SampleLevel(linearSampler, uv, 0);
  }
}
