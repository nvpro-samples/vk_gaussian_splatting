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

// Shader interface definitions for Visual Helpers (Grid + Transform Helper)
// Shared between C++ and Slang shaders

#pragma once

#ifdef __cplusplus
#include <glm/glm.hpp>
#include "nvshaders/slang_types.h"
namespace shaderio::visual_helpers {
#endif

//--------------------------------------------------------------------------------------------------
// Render Modes
//--------------------------------------------------------------------------------------------------

// Visual helper rendering modes
enum class HelperMode
{
  eGrid      = 0,  // Infinite grid rendering
  eTransform = 1   // Transform manipulation helper (gizmo)
};

//--------------------------------------------------------------------------------------------------
// Push Constants
//--------------------------------------------------------------------------------------------------

// Unified push constant for visual helpers rendering
struct PushConstantVisualHelpers
{
  float4x4   mvp;              // Model-View-Projection matrix
  float4     color;            // Base color (RGBA) - for transform, or grid alpha in .w
  float2     viewportSize;     // Viewport dimensions (native resolution)
  float2     depthBufferSize;  // Depth buffer valid region (DLSS render size or viewport size)
  HelperMode mode;             // Rendering mode: eGrid or eTransform
  uint       componentID;      // Transform component ID for picking (unused for now, reserved)
  float2     padding;          // Padding for alignment
};

#ifdef __cplusplus
}  // namespace shaderio::visual_helpers
#endif

