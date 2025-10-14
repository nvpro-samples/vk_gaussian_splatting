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

/*
* Some mathematical formulations and comments have been directly retained from
* https://github.com/mkkellogg/GaussianSplats3D. Original source code  
* licence hereafter.
* ----------------------------------
* The MIT License (MIT)
* 
* Copyright (c) 2023 Mark Kellogg
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#version 450

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : require

#include "shaderio.h"
#include "threedgs.glsl"
#include "threedgs_particles_storage.glsl"

// Parallel Processing : Each global invocation (thread) processes one splat.
// Batch Processing : The workgroup can process up to RASTER_MESH_WORKGROUP_SIZE splats(outputQuadCount)

layout(local_size_x = RASTER_MESH_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 4 * RASTER_MESH_WORKGROUP_SIZE, max_primitives = 2 * RASTER_MESH_WORKGROUP_SIZE) out;

// Per primitive output
layout(location = 0) perprimitiveEXT out vec4 outSplatCol[];
// Per vertex output
#if !USE_BARYCENTRIC
layout(location = 1) out vec2 outFragPos[];
#endif

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform _frameInfo
{
  FrameInfo frameInfo;
};

layout(push_constant) uniform _PushConstantRaster
{
  PushConstant pcRaster;
};

// sorted indices
layout(set = 0, binding = BINDING_INDICES_BUFFER) buffer _indices
{
  uint32_t indices[];
};
// to get the actual number of splats (after culling if any)
layout(set = 0, binding = BINDING_INDIRECT_BUFFER, scalar) buffer _indirect
{
  IndirectParams indirect;
};

// used when quad need to be discard
void emitDegeneratedQuad(void)
{
  gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 0].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 1].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 2].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 3].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
}

void main()
{
  const uint32_t baseIndex = gl_GlobalInvocationID.x;
#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
  // if culling is already performed we use the subset of splats
  const uint splatCount = indirect.instanceCount;
#else
  // otherwise we use all the splats
  const uint splatCount = frameInfo.splatCount;
#endif
  const uint outputQuadCount = min(RASTER_MESH_WORKGROUP_SIZE, splatCount - gl_WorkGroupID.x * RASTER_MESH_WORKGROUP_SIZE);

  if(gl_LocalInvocationIndex == 0)
  {
    // set the number of vertices and primitives to put out just once for the complete workgroup
    SetMeshOutputsEXT(outputQuadCount * 4, outputQuadCount * 2);
  }

  if(baseIndex < splatCount)
  {
    const uint splatIndex = indices[baseIndex];

    // emit primitives (triangles) as soon as possible
    gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex * 2 + 0] = uvec3(0, 2, 1) + gl_LocalInvocationIndex * 4;
    gl_PrimitiveTriangleIndicesEXT[gl_LocalInvocationIndex * 2 + 1] = uvec3(2, 0, 3) + gl_LocalInvocationIndex * 4;

    const mat4 modelViewMatrix = frameInfo.viewMatrix * pcRaster.modelMatrix;

    // Fetches are done as early as possible.
    vec4       splatColor  = fetchColor(splatIndex);
    const mat3 cov3Dm      = fetchCovariance(splatIndex);
    const vec3 splatCenter = fetchCenter(splatIndex);

    const vec4 viewCenter = modelViewMatrix * vec4(splatCenter, 1.0);
    const vec4 clipCenter  = frameInfo.projectionMatrix * viewCenter;

#if SHOW_SH_ONLY == 1
    splatColor.rgb = vec3(0.5);
#endif

    // fetch radiance from SH coefs > degree 0
    // const vec3 worldViewDir = normalize(splatCenter - frameInfo.cameraPosition);
    const vec3 worldViewDir = normalize(splatCenter - vec3(pcRaster.modelMatrixInverse * vec4(frameInfo.cameraPosition, 1.0)));

    splatColor.rgb += fetchViewDependentRadiance(splatIndex, worldViewDir);

    // alpha based culling
    if(splatColor.a < frameInfo.alphaCullThreshold)
    {
      // Early return to discard splat
      emitDegeneratedQuad();
      return;
    }

    // emit per primitive color as early as possible for perf reasons, 
    // only for original 3DGS, see later on for MipSplatting
#if MS_ANTIALIASING == 0
    outSplatCol[gl_LocalInvocationIndex * 2 + 0] = splatColor;
    outSplatCol[gl_LocalInvocationIndex * 2 + 1] = splatColor;
#endif

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_RASTER
    const float clip = (1.0 + frameInfo.frustumDilation) * clipCenter.w;
    if(abs(clipCenter.x) > clip || abs(clipCenter.y) > clip
       || clipCenter.z < (0.f - frameInfo.frustumDilation) * clipCenter.w || clipCenter.z > clipCenter.w)
    {
      // Early return to discard splat
      emitDegeneratedQuad();
      return;
    }
#endif

    // the vertices of the quad
    const vec2 positions[4] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};

#if !USE_BARYCENTRIC
    // emit per vertex attributes as early as possible
    [[unroll]] for(uint i = 0; i < 4; ++i)
    {
      // Scale the fragment position data we send to the fragment shader
      outFragPos[gl_LocalInvocationIndex * 4 + i] = positions[i].xy * sqrt8;
    }
#endif

    // Computes the projected covariance
    const vec3 cov2Dv = threedgsCovarianceProjection(cov3Dm, viewCenter, frameInfo.focal, modelViewMatrix);

    // computes the basis vectors of the extent of the projected covariance
    // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
    vec2 basisVector1, basisVector2;
    if(!threedgsProjectedExtentBasis(cov2Dv, sqrt8, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
    {
      // Early return to discard splat
      emitDegeneratedQuad();
      return;
    }

#if MS_ANTIALIASING == 1
    // emit the color with alpha compensation
    outSplatCol[gl_LocalInvocationIndex * 2 + 0] = splatColor;
    outSplatCol[gl_LocalInvocationIndex * 2 + 1] = splatColor;
#endif

    /////////////////////////////
    // emiting quad vertices

    const vec3 ndcCenter = clipCenter.xyz / clipCenter.w;

    [[unroll]] for(uint i = 0; i < 4; ++i)
    {
      const vec2 fragPos = positions[i].xy;

      const vec2 ndcOffset = vec2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                             * frameInfo.inverseFocalAdjustment;

      const vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + i].gl_Position = quadPos;
    }
  }
}
