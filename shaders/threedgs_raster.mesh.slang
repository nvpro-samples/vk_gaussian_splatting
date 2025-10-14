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

#include "shaderio.h"
#include "threedgs.h.slang"
#include "threedgs_particles_storage.h.slang"

// Parallel Processing : Each global invocation (thread) processes one splat.
// Batch Processing : The workgroup can process up to RASTER_MESH_WORKGROUP_SIZE splats(outputQuadCount)

static const int MAX_VERTICES   = 4 * RASTER_MESH_WORKGROUP_SIZE;
static const int MAX_PRIMITIVES = 2 * RASTER_MESH_WORKGROUP_SIZE;

// clang-format off

// Per primitive output
struct PrimitiveOutput
{
  [[vk::location(0)]] nointerpolation float4 outSplatCol : COLOR0;
};

// Per vertex output
struct VertexOutput
{
#if !USE_BARYCENTRIC
  [[vk::location(1)]] float2 outFragPos : TEXCOORD0;
#endif
  [[vk::location(2)]] float4 position : SV_Position;
};

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]  ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;
// sorted indices
[[vk::binding(BINDING_INDICES_BUFFER, 0)]]  RWStructuredBuffer<uint32_t> indices;
// to get the actual number of splats (after culling if any)
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]] RWStructuredBuffer<IndirectParams> indirectBuffer;

// clang-format on

// used when quad need to be discard
void emitDegeneratedQuad(uint localIndex, out OutputVertices<VertexOutput, MAX_VERTICES> verts)
{
  [unroll]
  for(uint i = 0; i < 4; ++i)
  {
    verts[localIndex * 4 + i].position = float4(0.0, 0.0, 2.0, 1.0);
  }
}

[numthreads(RASTER_MESH_WORKGROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
[shader("mesh")]
void main(
  uint3 groupThreadID : SV_GroupThreadID,
  uint3 groupID : SV_GroupID,
  uint3 dispatchThreadID : SV_DispatchThreadID,
  OutputIndices<uint3, MAX_PRIMITIVES> tris,
  OutputVertices<VertexOutput, MAX_VERTICES> verts,
  OutputPrimitives<PrimitiveOutput, MAX_PRIMITIVES> prims
)
{
  const uint32_t baseIndex = dispatchThreadID.x;
  const uint localIndex = groupThreadID.x;
  
#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
  // if culling is already performed we use the subset of splats
  const uint splatCount = indirectBuffer[0].instanceCount;
#else
  // otherwise we use all the splats
  const uint splatCount = frameInfo.splatCount;
#endif
  const uint outputQuadCount = min(RASTER_MESH_WORKGROUP_SIZE, splatCount - groupID.x * RASTER_MESH_WORKGROUP_SIZE);

  if(localIndex == 0)
  {
    // set the number of vertices and primitives to put out just once for the complete workgroup
    SetMeshOutputCounts(outputQuadCount * 4, outputQuadCount * 2);
  }

  if(baseIndex < splatCount)
  {
    const uint splatIndex = indices[baseIndex];

    // emit primitives (triangles) as soon as possible
    tris[localIndex * 2 + 0] = uint3(0, 2, 1) + localIndex * 4;
    tris[localIndex * 2 + 1] = uint3(2, 0, 3) + localIndex * 4;

    const float4x4 modelViewMatrix = mul(pcRaster.modelMatrix, frameInfo.viewMatrix);

    // Fetches are done as early as possible.
    float4       splatColor  = fetchColor(splatIndex);
    const float3x3 cov3Dm      = fetchCovariance(splatIndex);
    const float3 splatCenter = fetchCenter(splatIndex);

    // Row-major: vector * matrix
    const float4 viewCenter = mul(float4(splatCenter, 1.0), modelViewMatrix);
    const float4 clipCenter  = mul(viewCenter, frameInfo.projectionMatrix);

#if SHOW_SH_ONLY == 1
    splatColor.rgb = float3(0.5, 0.5, 0.5);
#endif

    // fetch radiance from SH coefs > degree 0
    // const float3 worldViewDir = normalize(splatCenter - frameInfo.cameraPosition);
    // Row-major: vector * matrix (camera position transformed by inverse model matrix)
    const float3 worldViewDir = normalize(splatCenter - mul(float4(frameInfo.cameraPosition, 1.0), pcRaster.modelMatrixInverse).xyz);

    splatColor.rgb += fetchViewDependentRadiance(splatIndex, worldViewDir);

    // alpha based culling
    if(splatColor.a < frameInfo.alphaCullThreshold)
    {
      // Early return to discard splat
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

    // emit per primitive color as early as possible for perf reasons, 
    // only for original 3DGS, see later on for MipSplatting
#if MS_ANTIALIASING == 0
    prims[localIndex * 2 + 0].outSplatCol = splatColor;
    prims[localIndex * 2 + 1].outSplatCol = splatColor;
#endif

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_RASTER
    const float clip = (1.0 + frameInfo.frustumDilation) * clipCenter.w;
    if(abs(clipCenter.x) > clip || abs(clipCenter.y) > clip
       || clipCenter.z < (0.0 - frameInfo.frustumDilation) * clipCenter.w || clipCenter.z > clipCenter.w)
    {
      // Early return to discard splat
      emitDegeneratedQuad(localIndex, verts);
      return;
    }
#endif

    // the vertices of the quad
    const float2 positions[4] = {float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)};

#if !USE_BARYCENTRIC
    // emit per vertex attributes as early as possible
    [unroll]
    for(uint i = 0; i < 4; ++i)
    {
      // Scale the fragment position data we send to the fragment shader
      verts[localIndex * 4 + i].outFragPos = positions[i].xy * sqrt8;
    }
#endif

    // Computes the projected covariance
    const float3 cov2Dv = threedgsCovarianceProjection(cov3Dm, viewCenter, frameInfo.focal, modelViewMatrix);

    // computes the basis vectors of the extent of the projected covariance
    // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
    float2 basisVector1, basisVector2;
    if(!threedgsProjectedExtentBasis(cov2Dv, sqrt8, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
    {
      // Early return to discard splat
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MS_ANTIALIASING == 1
    // emit the color with alpha compensation
    prims[localIndex * 2 + 0].outSplatCol = splatColor;
    prims[localIndex * 2 + 1].outSplatCol = splatColor;
#endif

    /////////////////////////////
    // emiting quad vertices

    const float3 ndcCenter = clipCenter.xyz / clipCenter.w;

    [unroll]
    for(uint i = 0; i < 4; ++i)
    {
      const float2 fragPos = positions[i].xy;

      const float2 ndcOffset = float2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                             * frameInfo.inverseFocalAdjustment;

      const float4 quadPos = float4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      verts[localIndex * 4 + i].position = quadPos;
    }
  }
}
