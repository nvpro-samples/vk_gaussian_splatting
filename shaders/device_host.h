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

// type of method used for sorting
#define SORTING_GPU_SYNC_RADIX 0
#define SORTING_CPU_ASYNC_MONO 1
#define SORTING_CPU_ASYNC_MULTI 2

// type of pipeline used
#define PIPELINE_MESH 0
#define PIPELINE_VERT 1
#define PIPELINE_RTX 2

// type of frustum culling
#define FRUSTUM_CULLING_NONE 0
#define FRUSTUM_CULLING_DIST 1
#define FRUSTUM_CULLING_VERT 2
#define FRUSTUM_CULLING_MESH 3

// bindings for set 0
#define BINDING_FRAME_INFO_UBO 0
#define BINDING_CENTERS_TEXTURE 1
#define BINDING_COLORS_TEXTURE 2
#define BINDING_COVARIANCES_TEXTURE 3
#define BINDING_SH_TEXTURE 4
#define BINDING_DISTANCES_BUFFER 5
#define BINDING_INDICES_BUFFER 6
#define BINDING_INDIRECT_BUFFER 7
#define BINDING_CENTERS_BUFFER 8
#define BINDING_COLORS_BUFFER 9
#define BINDING_COVARIANCES_BUFFER 10
#define BINDING_SH_BUFFER 11

// Warning, struct members must
// be aligned to 128 bits
//
struct FrameInfo
{
  mat4 projectionMatrix;
  mat4 viewMatrix;

  vec3  cameraPosition;
  float orthoZoom;

  vec2 focal;
  vec2 viewport;

  vec2 basisViewport;
  int  orthographicMode;
  int  pointCloudModeEnabled;

  float inverseFocalAdjustment;
  int   sphericalHarmonicsDegree;
  int   sphericalHarmonics8BitMode;  // unused
  float splatScale;

  int   showShOnly;
  int   splatCount;
  int   sortingMethod;
  float frustumDilation;  // for frustum culling
  float alphaCullThreshold; // for alpha culling

};

// not used
struct PushConstant
{
  mat4 transfo;
  vec4 color;
};

// indirect parameters for
// - vkCmdDrawIndexedIndirect (first 6 attr)
// - vkCmdDrawMeshTasksIndirectEXT (last 3 attr)
struct IndirectParams
{
  // for vkCmdDrawIndexedIndirect
  uint32_t indexCount;     //= 6;  // allways = 6 indices for the quad (2 triangles)
  uint32_t instanceCount;  //= 0;  // will be incremented by the distance compute shader
  uint32_t firstIndex;     //= 0;  // allways zero
  uint32_t vertexOffset;   //= 0;  // allways zero
  uint32_t firstInstance;  //= 0;  // allways zero
  // for vkCmdDrawMeshTasksIndirectEXT
  uint32_t groupCountX;  //=0;  // Will be incremented by distance the compute shader
  uint32_t groupCountY;  //=1;  // Allways one workgroup on Y
  uint32_t groupCountZ;  //=1;  // Allways one workgroup on Z
};