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

// Shader interface definitions for ImageCompare module
// Shared between C++ and Slang shaders

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <glm/vec2.hpp>
#include "nvshaders/slang_types.h"
namespace shaderio::imcmp {
#endif

//--------------------------------------------------------------------------------------------------
// Comparison Display Modes
//--------------------------------------------------------------------------------------------------

// What to display on each side of the comparison view
enum class DisplayMode
{
  eCapture           = 0,  // Show reference image
  eCurrent           = 1,  // Show real-time render
  eDifferenceRaw     = 2,  // Show difference between reference and real-time
  eDifferenceRedGray = 3,  // Show difference in red on grayscale (like ImageMagick)
  eDifferenceRedOnly = 4,  // Show difference in red only (no background)
  eFLIPError         = 5   // Show FLIP perceptual error map (heatmap visualization)
};

//--------------------------------------------------------------------------------------------------
// FLIP Quality Modes
//--------------------------------------------------------------------------------------------------

// FLIP computation quality/performance trade-offs
enum class FLIPMode
{
  eDisabled  = 0,  // Only compute MSE/PSNR (fastest, ~0.01ms)
  eApprox    = 1,  // Fast approximation: Sobel + YCxCz + Minkowski (~0.1ms, 85% accurate)
  eReference = 2   // Full reference: Gaussian pyramid + all features (~1ms, 100% accurate)
};

#ifndef __cplusplus
// Slang doesn't support enum class, use int constants
static const int FLIPMode_Disabled  = 0;
static const int FLIPMode_Approx    = 1;
static const int FLIPMode_Reference = 2;
#endif

//--------------------------------------------------------------------------------------------------
// Descriptor Bindings
//--------------------------------------------------------------------------------------------------

// Bindings for set 0 of Comparison Composite shader
enum class ComparisonBinding
{
  eCaptureImage = 0,  // saved reference image
  eCurrentImage = 1,  // current frame image
  eOutputImage  = 2,  // output composite image
  eSampler      = 3   // linear sampler for image scaling
};

// Bindings for set 0 of Metrics Compute shader
enum class MetricsBinding
{
  eCaptureImage = 0,  // saved reference image
  eCurrentImage = 1,  // current frame image
  eResultBuffer = 2,  // output buffer for MSE result
  eSampler      = 3   // linear sampler for image scaling
};

//--------------------------------------------------------------------------------------------------
// Push Constants
//--------------------------------------------------------------------------------------------------

// Push constant for comparison composite shader
struct PushConstantComparison
{
  float       splitPosition     = 0.5f;                   // Split position [0.0, 1.0]
  int2        outputSize        = int2(0, 0);             // Output image size (viewport size)
  int2        currentImgSize    = int2(0, 0);             // Current image size (may differ for DLSS buffers)
  int2        captureImgSize    = int2(0, 0);             // Reference image size
  DisplayMode leftSide          = DisplayMode::eCapture;  // What to display on left side
  DisplayMode rightSide         = DisplayMode::eCurrent;  // What to display on right side
  float       differenceAmplify = 5.0f;                   // Amplification factor for difference modes
};

// Push constant for metrics compute shader
struct PushConstantMetrics
{
  int2  captureImgSize  = int2(0, 0);  // Reference image size
  int2  currentImgSize  = int2(0, 0);  // Current image size
  float sampleDivider   = 1.0f;        // Pre-calculated divider (width * height * 3) for normalization
  float pixelsPerDegree = 67.0f;  // Pixels per degree of visual angle (default: 0.7m viewing distance on 24" monitor)
  FLIPMode flipMode     = FLIPMode::eReference;  // FLIP quality mode (disabled/approx/reference)
};

#ifdef __cplusplus
}  // namespace shaderio::imcmp
#endif
