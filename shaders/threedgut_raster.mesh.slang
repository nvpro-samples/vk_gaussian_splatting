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

#include "shaderio.h"
#include "threedgs.h.slang"
#include "threedgs_particles_storage.h.slang"
#include "threedgut.h.slang"

// Parallel Processing : Each global invocation (thread) processes one splat.
// Batch Processing : The workgroup can process up to RASTER_MESH_WORKGROUP_SIZE splats(outputQuadCount)

static const int MAX_VERTICES   = 4 * RASTER_MESH_WORKGROUP_SIZE;
static const int MAX_PRIMITIVES = 2 * RASTER_MESH_WORKGROUP_SIZE;

// clang-format off

// Per primitive output
struct PrimitiveOutput
{
  [[vk::location(0)]] nointerpolation uint splatId : SPLATID;
  [[vk::location(1)]] nointerpolation float4 splatCol : COLOR0;
  [[vk::location(2)]] nointerpolation float3 splatPosition : POSITION0;
  [[vk::location(3)]] nointerpolation float3 splatScale : SCALE0;
  [[vk::location(4)]] nointerpolation float4 splatRotation : ROTATION0;  // a quaternion
};

// Per vertex output
struct VertexOutput
{
  float4 position : SV_Position;
};

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;
// sorted indices
[[vk::binding(BINDING_INDICES_BUFFER, 0)]] RWStructuredBuffer<uint32_t> indices;
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
  out OutputVertices<VertexOutput, MAX_VERTICES> verts,
  out OutputPrimitives<PrimitiveOutput, MAX_PRIMITIVES> prims,
  out OutputIndices<uint3, MAX_PRIMITIVES> triangles)
{
  const uint localIndex = groupThreadID.x;
  const uint32_t baseIndex = dispatchThreadID.x;
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
    triangles[localIndex * 2 + 0] = uint3(0, 2, 1) + localIndex * 4;
    triangles[localIndex * 2 + 1] = uint3(2, 0, 3) + localIndex * 4;
    
    // Fetch particle minimal data
    // Fetches are done as early as possible.
    // Moving those after culling statments will reduce the performance
    float4       splatColor  = fetchColor(splatIndex);  // contains opacity as .a
    const float3 splatCenter = fetchCenter(splatIndex);
    const float3 splatScale  = exp(fetchScale(splatIndex));
    // INRIA quaternions are scalar first: w in splatRotation.x and xyz in splatRotation.yzw)
    // our internal representation (hence splatRotation), uses xyz, w. vec4ToQuat does convertion
    const float4 splatRotation = normalize(vec4toQuat(fetchRotation(splatIndex)));

    // provides the splatIndex to the fragment shader
    prims[localIndex * 2 + 0].splatId = splatIndex;
    prims[localIndex * 2 + 1].splatId = splatIndex;
    
    // emit per primitive data ASAP
    prims[localIndex * 2 + 0].splatPosition = splatCenter;
    prims[localIndex * 2 + 1].splatPosition = splatCenter;
    prims[localIndex * 2 + 0].splatScale    = splatScale;
    prims[localIndex * 2 + 1].splatScale    = splatScale;
    prims[localIndex * 2 + 0].splatRotation = splatRotation;
    prims[localIndex * 2 + 1].splatRotation = splatRotation;

    // fetch radiance from SH coefs > degree 0, we work in model coordinates
    const float3 worldViewDir = normalize(splatCenter - mul(float4(frameInfo.cameraPosition, 1.0), pcRaster.modelMatrixInverse).xyz);

#if SHOW_SH_ONLY == 1
    splatColor.rgb = float3(0.5);
#endif
    splatColor.rgb += fetchViewDependentRadiance(splatIndex, worldViewDir);

    // alpha based culling
    if(splatColor.a < frameInfo.alphaCullThreshold)
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MS_ANTIALIASING == 0
    // emit per primitive color as early as possible for performance reasons
    prims[localIndex * 2 + 0].splatCol = splatColor;
    prims[localIndex * 2 + 1].splatCol = splatColor;
#endif

#if CAMERA_TYPE == CAMERA_PINHOLE
    CameraModelParameters sensorModel = initPerfectPinholeCamera(frameInfo.nearFar, frameInfo.viewport, frameInfo.focal);
#else
    CameraModelParameters sensorModel = initPerfectFisheyeCamera(frameInfo.viewport, frameInfo.focal);
#endif

    // camera pose in world space, used by projector to compute the ray.
    SensorState sensorState = initGlobalShutterSensorState(frameInfo.viewTrans, frameInfo.viewQuat);

    int2 resolution = int2(int(frameInfo.viewport.x), int(frameInfo.viewport.y));

    float3 particleSensorRay;
    float2 particleProjCenter;
    float3 particleProjCovariance;

    if(!threedgutParticleProjection(resolution, sensorModel, pcRaster.modelMatrix, sensorState, splatCenter, splatScale,
                                    quatToMat3(splatRotation), particleProjCenter, particleProjCovariance))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

// Method from paper used for cuda raster
#if EXTENT_METHOD == EXTENT_CONIC
    float2  extent;
    float4  conicOpacity;
    float maxConicOpacityPower;

    if(!threedgutProjectedExtentConicOpacity(particleProjCovariance, splatColor.a, extent, conicOpacity, maxConicOpacityPower))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }

#if MS_ANTIALIASING == 1
    splatColor.a = conicOpacity.a;
#endif

// Method optimized for rasterisation pipelines
#elif EXTENT_METHOD == EXTENT_EIGEN

    float2 basisVector1, basisVector2;

    if(!threedgsProjectedExtentBasis(particleProjCovariance, 3.33, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
    {
      emitDegeneratedQuad(localIndex, verts);
      return;
    }
#endif

#if MS_ANTIALIASING == 1
    // emit the fragment color with compensation
    prims[localIndex * 2 + 0].splatCol = splatColor;
    prims[localIndex * 2 + 1].splatCol = splatColor;
#endif

    // Convert projected particle center from pixel coordinates to NDC [-1, 1]
    // particleProjCenter is in pixel coordinates, need to convert to NDC
    const float2 ndcXY = (particleProjCenter / frameInfo.viewport) * 2.0 - 1.0;

    const float4 worldSplatCenter = mul(float4(splatCenter, 1.0), pcRaster.modelMatrix);
    const float4 viewSplatCenter  = mul(worldSplatCenter, frameInfo.viewMatrix);
    // TODO: projection matrix is not correct with fisheye, this is a coarse aprox to compute depth
    const float4 clipSplatCenter = mul(viewSplatCenter, frameInfo.projectionMatrix);

    const float3 ndcCenter = float3(ndcXY, clipSplatCenter.z / clipSplatCenter.w);
#if EXTENT_METHOD == EXTENT_CONIC
    const float2 ndcExtent = extent * frameInfo.basisViewport * 2.0;
#endif

    /////////////////////////////
    // emiting quad vertices

    const float2 vertexPositions[4] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};

    [unroll]
    for(uint i = 0; i < 4; ++i)
    {
      const float2 vertPos = vertexPositions[i].xy;

#if EXTENT_METHOD == EXTENT_CONIC

      const float2 ndcOffset = vertPos * ndcExtent;

#elif EXTENT_METHOD == EXTENT_EIGEN

      const float2 ndcOffset = float2(vertPos.x * basisVector1 + vertPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                             * frameInfo.inverseFocalAdjustment;

#endif
      const float4 quadPos = float4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      verts[localIndex * 4 + i].position = quadPos;
    }
  }
}