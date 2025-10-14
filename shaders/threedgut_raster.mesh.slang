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

#version 450

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : require

#include "shaderio.h"
#include "threedgs.glsl"
#include "threedgs_particles_storage.glsl"
#include "threedgut.glsl"

// Parallel Processing : Each global invocation (thread) processes one splat.
// Batch Processing : The workgroup can process up to RASTER_MESH_WORKGROUP_SIZE splats(outputQuadCount)

layout(local_size_x = RASTER_MESH_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 4 * RASTER_MESH_WORKGROUP_SIZE, max_primitives = 2 * RASTER_MESH_WORKGROUP_SIZE) out;

// Per primitive output
layout(location = 0) perprimitiveEXT out uint outSplatId[];
layout(location = 1) perprimitiveEXT out vec4 outSplatCol[];
layout(location = 2) perprimitiveEXT out vec3 outSplatPosition[];
layout(location = 3) perprimitiveEXT out vec3 outSplatScale[];
layout(location = 4) perprimitiveEXT out vec4 outSplatRotation[];  // a quaternion

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
    // provides the splatIndex to the fragment shader
    outSplatId[gl_LocalInvocationIndex * 2 + 0] = splatIndex;
    outSplatId[gl_LocalInvocationIndex * 2 + 1] = splatIndex;

    // Fetch particle minimal data
    // Fetches are done as early as possible.
    // Moving those after culling statments will reduce the performance
    vec4       splatColor  = fetchColor(splatIndex);  // contains opacity as .a
    const vec3 splatCenter = fetchCenter(splatIndex);
    const vec3 splatScale  = exp(fetchScale(splatIndex));
    // INRIA quaternions are scalar first: w in splatRotation.x and xyz in splatRotation.yzw)
    // our internal representation (hence splatRotation), uses xyz, w. vec4ToQuat does convertion
    const vec4 splatRotation = normalize(vec4toQuat(fetchRotation(splatIndex)));

    // emit per primitive data ASAP
    outSplatPosition[gl_LocalInvocationIndex * 2 + 0] = splatCenter;
    outSplatPosition[gl_LocalInvocationIndex * 2 + 1] = splatCenter;
    outSplatScale[gl_LocalInvocationIndex * 2 + 0]    = splatScale;
    outSplatScale[gl_LocalInvocationIndex * 2 + 1]    = splatScale;
    outSplatRotation[gl_LocalInvocationIndex * 2 + 0] = splatRotation;
    outSplatRotation[gl_LocalInvocationIndex * 2 + 1] = splatRotation;

    // fetch radiance from SH coefs > degree 0, we work in model coordinates
    const vec3 worldViewDir = normalize(splatCenter - vec3(pcRaster.modelMatrixInverse * vec4(frameInfo.cameraPosition, 1.0)));

#if SHOW_SH_ONLY == 1
    splatColor.rgb = vec3(0.5);
#endif
    splatColor.rgb += fetchViewDependentRadiance(splatIndex, worldViewDir);

    // alpha based culling
    if(splatColor.a < frameInfo.alphaCullThreshold)
    {
      emitDegeneratedQuad();
      return;
    }

#if MS_ANTIALIASING == 0
    // emit per primitive color as early as possible for performance reasons
    outSplatCol[gl_LocalInvocationIndex * 2 + 0] = splatColor;
    outSplatCol[gl_LocalInvocationIndex * 2 + 1] = splatColor;
#endif

#if CAMERA_TYPE == CAMERA_PINHOLE
    CameraModelParameters sensorModel = initPerfectPinholeCamera(frameInfo.nearFar, frameInfo.viewport, frameInfo.focal);
#else
    CameraModelParameters sensorModel = initPerfectFisheyeCamera(frameInfo.viewport, frameInfo.focal);
#endif

    // camera pose in world space, used by projector to compute the ray.
    SensorState sensorState = initGlobalShutterSensorState(frameInfo.viewTrans, frameInfo.viewQuat);

    ivec2 resolution = ivec2(frameInfo.viewport.x, frameInfo.viewport.y);

    vec3 particleSensorRay;
    vec2 particleProjCenter;
    vec3 particleProjCovariance;

    if(!threedgutParticleProjection(resolution, sensorModel, pcRaster.modelMatrix, sensorState, splatCenter, splatScale,
                                    quatToMat3(splatRotation), particleProjCenter, particleProjCovariance))
    {
      emitDegeneratedQuad();
      return;
    }

// Method from paper used for cuda raster
#if EXTENT_METHOD == EXTENT_CONIC
    vec2  extent;
    vec4  conicOpacity;
    float maxConicOpacityPower;

    if(!threedgutProjectedExtentConicOpacity(particleProjCovariance, splatColor.a, extent, conicOpacity, maxConicOpacityPower))
    {
      emitDegeneratedQuad();
      return;
    }

#if MS_ANTIALIASING == 1
    splatColor.a = conicOpacity.a;
#endif

// Method optimized for rasterisation pipelines
#elif EXTENT_METHOD == EXTENT_EIGEN

    vec2 basisVector1, basisVector2;

    if(!threedgsProjectedExtentBasis(particleProjCovariance, 3.33, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
    {
      emitDegeneratedQuad();
      return;
    }
#endif

#if MS_ANTIALIASING == 1
    // emit the fragment color with compensation
    outSplatCol[gl_LocalInvocationIndex * 2 + 0] = splatColor;
    outSplatCol[gl_LocalInvocationIndex * 2 + 1] = splatColor;
#endif

    // Convert projected particle center from pixel coordinates to NDC [-1, 1]
    // particleProjCenter is in pixel coordinates, need to convert to NDC
    const vec2 ndcXY = (particleProjCenter / frameInfo.viewport) * 2.0 - 1.0;

    const vec4 worldSplatCenter = pcRaster.modelMatrix * vec4(splatCenter, 1.0);
    const vec4 viewSplatCenter  = frameInfo.viewMatrix * worldSplatCenter;
    // TODO: projection matrix is not correct with fisheye, this is a coarse aprox to compute depth
    const vec4 clipSplatCenter = frameInfo.projectionMatrix * viewSplatCenter;

    const vec3 ndcCenter = vec3(ndcXY, clipSplatCenter.z / clipSplatCenter.w);
#if EXTENT_METHOD == EXTENT_CONIC
    const vec2 ndcExtent = extent * frameInfo.basisViewport * 2.0;
#endif

    /////////////////////////////
    // emiting quad vertices

    const vec2 vertexPositions[4] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};

    [[unroll]] for(uint i = 0; i < 4; ++i)
    {
      const vec2 vertPos = vertexPositions[i].xy;

#if EXTENT_METHOD == EXTENT_CONIC

      const vec2 ndcOffset = vertPos * ndcExtent;

#elif EXTENT_METHOD == EXTENT_EIGEN

      const vec2 ndcOffset = vec2(vertPos.x * basisVector1 + vertPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                             * frameInfo.inverseFocalAdjustment;

#endif
      const vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + i].gl_Position = quadPos;
    }
  }
}