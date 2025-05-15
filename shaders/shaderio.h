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

#ifndef _DEVICE_HOST_H_
#define _DEVICE_HOST_H_

// type of method used for sorting
#define SORTING_GPU_SYNC_RADIX 0
#define SORTING_CPU_ASYNC_MONO 1
#define SORTING_CPU_ASYNC_MULTI 2

// type of model storage
#define STORAGE_BUFFERS 0
#define STORAGE_TEXTURES 1

// format for SH storage
#define FORMAT_FLOAT32 0
#define FORMAT_FLOAT16 1
#define FORMAT_UINT8 2

// type of pipeline used
#define PIPELINE_MESH 0
#define PIPELINE_VERT 1
#define PIPELINE_RTX 2

// type of frustum culling
#define FRUSTUM_CULLING_NONE 0
#define FRUSTUM_CULLING_AT_DIST 1
#define FRUSTUM_CULLING_AT_RASTER 2

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

// location for vertex attributes
// (only for vertex shader mode)
#define ATTRIBUTE_LOC_POSITION 0
#define ATTRIBUTE_LOC_SPLAT_INDEX 1

// Distance shader workgroup size
#define DISTANCE_COMPUTE_WORKGROUP_SIZE 256

// Mesh shader workgroup size
// This configuration is optimized for NVIDIA hardware
#define RASTER_MESH_WORKGROUP_SIZE 32

#ifdef __cplusplus
#include <glm/glm.hpp>
// used to assign fields defaults
#define DEFAULT(val) = val
namespace shaderio {
using namespace glm;
#else
// used to skip fields init
// when included in glsl
#define DEFAULT(val)
// common extensions
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#endif

// Warning, struct members must be aligned
// we group by packs of 128 bits
struct FrameInfo
{
  mat4 projectionMatrix;
  mat4 viewMatrix;

  vec3                         cameraPosition;
  float inverseFocalAdjustment DEFAULT(1.0f);

  vec2 focal;
  // vec2 viewport;
  vec2 basisViewport;

  float orthoZoom      DEFAULT(1.0f);  //
  int orthographicMode DEFAULT(0);     // disabled, in [0,1]
  int splatCount       DEFAULT(0);     //
  float splatScale     DEFAULT(1.0f);  // in {0.1, 2.0}

  int sortingMethod        DEFAULT(SORTING_GPU_SYNC_RADIX);
  float frustumDilation    DEFAULT(0.2f);           // for frustum culling, 2% scale
  float alphaCullThreshold DEFAULT(1.0f / 255.0f);  // for alpha culling
};

// TODO will be used for model transformation
struct PushConstant
{
  mat4 transfo;
};

// indirect parameters for
// - vkCmdDrawIndexedIndirect (first 6 attr)
// - vkCmdDrawMeshTasksIndirectEXT (last 3 attr)
struct IndirectParams
{
  // for vkCmdDrawIndexedIndirect
  uint32_t indexCount    DEFAULT(6);  // allways = 6 indices for the quad (2 triangles)
  uint32_t instanceCount DEFAULT(0);  // will be incremented by the distance compute shader
  uint32_t firstIndex    DEFAULT(0);  // allways zero
  uint32_t vertexOffset  DEFAULT(0);  // allways zero
  uint32_t firstInstance DEFAULT(0);  // allways zero

  // for vkCmdDrawMeshTasksIndirectEXT
  uint32_t groupCountX DEFAULT(0);  // Will be incremented by the distance compute shader
  uint32_t groupCountY DEFAULT(1);  // Allways one workgroup on Y
  uint32_t groupCountZ DEFAULT(1);  // Allways one workgroup on Z
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

#undef DEFAULT
#endif