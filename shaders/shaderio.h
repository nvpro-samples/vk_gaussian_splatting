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
#define PIPELINE_VERT 0
#define PIPELINE_MESH 1
#define PIPELINE_RTX 2
#define PIPELINE_HYBRID 3  // Hybrid rendering: raster primary rays, raytrace secondary rays

// visualization mode
#define VISUALIZE_FINAL 0
#define VISUALIZE_CLOCK 1
#define VISUALIZE_DEPTH 2
#define VISUALIZE_RAYHITS 3

// type of frustum culling
#define FRUSTUM_CULLING_NONE 0
#define FRUSTUM_CULLING_AT_DIST 1
#define FRUSTUM_CULLING_AT_RASTER 2

// particle format (PF), RTX
// not used in shaders (using RTX_USE_AABBS compiler defined instead)
// used only by UI but here to be easier to find
#define PARTICLE_FORMAT_ICOSAHEDRON 0
#define PARTICLE_FORMAT_PARAMETRIC 1

// degree of the splat kernel, RTX
#define KERNEL_DEGREE_QUINTIC 5
#define KERNEL_DEGREE_TESSERACTIC 4
#define KERNEL_DEGREE_CUBIC 3
#define KERNEL_DEGREE_QUADRATIC 2
#define KERNEL_DEGREE_LAPLACIAN 1
#define KERNEL_DEGREE_LINEAR 0

// bindings for set 0 (common to Raster and RTX)
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
#define BINDING_SCALES_TEXTURE 12
#define BINDING_ROTATIONS_TEXTURE 13
#define BINDING_SCALES_BUFFER 14
#define BINDING_ROTATIONS_BUFFER 15
#define BINDING_OPACITY_TEXTURE 16
#define BINDING_OPACITY_BUFFER 17
#define BINDING_RTX_PAYLOAD_BUFFER 18
#define BINDING_MESH_DESCRIPTORS 19
#define BINDING_LIGHT_SET 20

// bindings for set 1 of RTX
#define RTX_BINDING_OUTIMAGE 0        // Ray tracer output image
#define RTX_BINDING_TLAS_SPLATS 1     // Top-level acceleration structure for splats
#define RTX_BINDING_TLAS_MESH 2       // Top-level acceleration structure for meshes
#define RTX_BINDING_PAYLOAD_BUFFER 3  // the alternative to payload stack (less efficient)

// location for vertex attributes
// (only for vertex shader mode)
#define ATTRIBUTE_LOC_POSITION 0
#define ATTRIBUTE_LOC_SPLAT_INDEX 1
// used for mesh rasterization
#define ATTRIBUTE_LOC_MESH_POSITION 0
#define ATTRIBUTE_LOC_MESH_NORMAL 1

#ifdef __cplusplus
#include <glm/glm.hpp>
// used to assign fields defaults
#define DEFAULT(val) = val
namespace shaderio {
using namespace glm;
#else
// we are in glsl here
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
  mat4 viewInverse;  // Camera inverse view matrix
  mat4 projInverse;  // Camera inverse projection matrix

  vec3                         cameraPosition;
  float inverseFocalAdjustment DEFAULT(1.0f);

  vec2 focal;
  vec2 viewport;

  vec2                     basisViewport;
  float orthoZoom          DEFAULT(1.0f);  //
  int32_t orthographicMode DEFAULT(0);     // disabled, in [0,1]

  int32_t splatCount DEFAULT(0);     //
  float splatScale   DEFAULT(1.0f);  // in {0.1, 2.0}

  float frustumDilation    DEFAULT(0.2f);           // for frustum culling, 2% scale
  float alphaCullThreshold DEFAULT(1.0f / 255.0f);  // for alpha culling

  int32_t lightCount DEFAULT(0);
  ivec2              cursor;        // position of the mouse cursor for debug
  int32_t maxPasses  DEFAULT(200);  // RTX maximum hits during marching

  float alphaClamp       DEFAULT(0.99f);  // 0.99 in original paper
  float minTransmittance DEFAULT(0.01f);  // 0.1  in original paper ? TO check
  int32_t rtxMaxBounces  DEFAULT(3);

  float multiplier DEFAULT(1.0f);  // for alternative visualization modes

  int32_t frameSampleId  DEFAULT(0);       // the frame sample index since last frame reset
  int32_t frameSampleMax DEFAULT(200);     // maximum number of frame after which we stop accumulating frames raytracing
  float focalDist        DEFAULT(1.5f);    // focal distance to compute depth of field
  float aperture         DEFAULT(0.009f);  // aperture distance to compute depth of field, 0 does no DOF effect
};

// Push constant for raster
struct PushConstant
{
  // model transformation
  mat4 modelMatrix;
  mat4 modelMatrixInverse;
  // index of the mesh beeing rendered if any
  uint32_t objIndex;
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

  // for debug info readback, TODO shall be in some other buffer
  int32_t particleID DEFAULT(-1);   // Will be set by the ID of the nearest splat on the ray path
  float particleDist DEFAULT(0.0);  // Will be set by the dist to the nearest splat on the ray path
};

// Information of a obj model when referenced in a shader
struct ObjDesc
{
  //int      txtOffset;             // Texture index offset in the array of textures
  uint64_t vertexAddress;         // Address of the Vertex buffer
  uint64_t indexAddress;          // Address of the index buffer
  uint64_t materialAddress;       // Address of the material buffer
  uint64_t materialIndexAddress;  // Address of the triangle material index buffer
};

// Structure holding the material for mesh objects
struct ObjMaterial
{
  vec3  ambient;
  vec3  diffuse;
  vec3  specular;
  vec3  transmittance;
  vec3  emission;
  float shininess;
  float ior;
  float dissolve;
  int   illum;
  int   textureID;
};

//
#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_DIRECTIONAL 1

// Structure holding light source attributes
struct LightSource
{
  vec3 position   DEFAULT(vec3(0.0, 4.0, 0.0));
  float intensity DEFAULT(1.0);
  int type        DEFAULT(LIGHT_TYPE_DIRECTIONAL);
};

// Push constant specific to raytracing
struct PushConstantRay
{
  // vec4 clearColor; // TODO move that in frameInfo ?
  // model transformation
  mat4 modelMatrix;
  mat4 modelMatrixInverse;
  mat4 modelMatrixTranspose;
  // stores the splat indices and vertices
  // provisionaly placed here, shall not be in push constant,
  // rather in some objects description buffer
  uint64_t vertexAddress;  // Address of the Vertex buffer
  uint64_t indexAddress;   // Address of the index buffer
};

struct Vertex  // See ObjLoader, copy of VertexObj
{
  vec3 pos;
  vec3 nrm;
  //vec3 color;
  //vec2 texCoord;
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

struct HitBufferPayload
{
  int   id;
  float dist;
};

#ifndef __cplusplus
struct hitPayload
{
#if USE_RTX_PAYLOAD_BUFFER == 0
  int   id[PAYLOAD_ARRAY_SIZE];
  float dist[PAYLOAD_ARRAY_SIZE];
#if WIREFRAME
  vec2 bary[PAYLOAD_ARRAY_SIZE];  // hit barycentrics
#endif
#endif
};
#endif

#undef DEFAULT
#undef START_BINDING
#undef END_BINDING
#endif
