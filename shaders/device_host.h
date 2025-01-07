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

// Warning, struct members must 
// be aligned to 128 bits
// 
struct FrameInfo
{
  mat4 projectionMatrix;
  mat4 viewMatrix;

  vec3 cameraPosition;
  float orthoZoom;

  vec2  focal;
  vec2  viewport;

  vec2  basisViewport;
  int   orthographicMode;
  int   pointCloudModeEnabled;
  
  float inverseFocalAdjustment;
  int   sphericalHarmonicsDegree;
  int   sphericalHarmonics8BitMode;
  float splatScale;

  int   showShOnly;
  int   opacityGaussianDisabled;
  int   splatCount;
  int   gpuSorting;
};

struct PushConstant
{
  mat4 transfo;
  vec4 color;
};
