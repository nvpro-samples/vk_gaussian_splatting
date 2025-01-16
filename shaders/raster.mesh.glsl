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

#version 450

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require
#define UNROLL_LOOP [[UNROLL]]
//#extension GL_KHR_shader_subgroup : enable
//#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_GOOGLE_include_directive : require
#include "device_host.h"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 128, max_primitives = 64) out;

// Per vertex output
layout(location = 0) out vec2 outFragPos[];
// Per primitive output
layout(location = 1) perprimitiveEXT out vec4 outSplatCol[];

// in order to manage alignment automatically we could write:
// layout(set = 0, binding = 0, scalar) uniform FrameInfo_
// but it may be less performant than aligning
// attribute in the struct (see device_host.h comment)
layout(set = 0, binding = 0) uniform _frameInfo
{
  FrameInfo frameInfo;
};

// sorted indices
layout(set = 0, binding = 6) buffer _indices
{
  uint32_t indices[];
};

layout(set = 0, binding = 1) uniform sampler2D centersTexture;
layout(set = 0, binding = 2) uniform sampler2D colorsTexture;
layout(set = 0, binding = 3) uniform sampler2D covariancesTexture;
layout(set = 0, binding = 4) uniform sampler2D sphericalHarmonicsTexture;

const float sqrt8    = sqrt(8.0);
const float minAlpha = 1.0 / 255.0;

// texture accessors
ivec2 getDataPos(in uint splatIndex, in uint stride, in uint offset, in ivec2 dimensions)
{
  const uint fullOffset = splatIndex * stride + offset;

  return ivec2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

ivec2 getDataPosF(in uint splatIndex, in float stride, in uint offset, in ivec2 dimensions)
{
  const uint fullOffset = uint(float(splatIndex) * stride) + offset;

  return ivec2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

const float SH_C1    = 0.4886025119029199f;
const float[5] SH_C2 = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);

const float SphericalHarmonics8BitCompressionRange     = 3.0;
const float SphericalHarmonics8BitCompressionHalfRange = SphericalHarmonics8BitCompressionRange / 2.0;
const vec3  vec8BitSHShift                             = vec3(SphericalHarmonics8BitCompressionHalfRange);

void main()
{
  const uint32_t baseIndex       = gl_GlobalInvocationID.x;
  const int      splatCount      = frameInfo.splatCount;
  const uint     outputQuadCount = min(32, splatCount - gl_WorkGroupID.x * 32);

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

    ///////
    //
    const uint  oddOffset        = uint(splatIndex) & uint(0x00000001);
    const uint  doubleOddOffset  = oddOffset * uint(2);
    const bool  isEven           = oddOffset == uint(0);
    const uint  nearestEvenIndex = uint(splatIndex) - oddOffset;
    const float fOddOffset       = float(oddOffset);

    const vec3 splatCenter = vec3(texelFetch(centersTexture, getDataPos(splatIndex, 1, 0, textureSize(centersTexture, 0)), 0));

    const mat4 transformModelViewMatrix = frameInfo.viewMatrix;
    const vec4 viewCenter               = transformModelViewMatrix * vec4(splatCenter, 1.0);

    // culling
    const vec4  clipCenter = frameInfo.projectionMatrix * viewCenter;
    const float clip       = 1.2 * clipCenter.w;
    if(frameInfo.frustumCulling == FRUSTUM_CULLING_MESH
       && (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip || clipCenter.y < -clip || clipCenter.y > clip))
    {
      // Early return to discard splat
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 0].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 1].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 2].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 3].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }

    // the vertices of the quad
    const vec2 positions[4] = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};

    // emit per vertex attributes as early as possible
    UNROLL_LOOP
    for(uint i = 0; i < 4; ++i)
    {
      // Scale the fragment position data we send to the fragment shader
      outFragPos[gl_LocalInvocationIndex * 4 + i] = positions[i].xy * sqrt8;
    }

    // work on color
    vec4 splatColor = texelFetch(colorsTexture, getDataPos(splatIndex, 1, 0, textureSize(colorsTexture, 0)), 0);
    if(frameInfo.showShOnly == 1)
    {
      splatColor.r = 0.5;
      splatColor.g = 0.5;
      splatColor.b = 0.5;
    }

    if(frameInfo.sphericalHarmonicsDegree >= 1)
    {

      const vec3 worldViewDir = normalize(splatCenter - frameInfo.cameraPosition);

      const vec4 sampledSH0123 =
          texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, 6, 0, textureSize(sphericalHarmonicsTexture, 0)), 0);
      const vec4 sampledSH4567 =
          texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, 6, 1, textureSize(sphericalHarmonicsTexture, 0)), 0);
      const vec4 sampledSH891011 =
          texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, 6, 2, textureSize(sphericalHarmonicsTexture, 0)), 0);
      const vec3 sh1 = sampledSH0123.rgb;
      const vec3 sh2 = vec3(sampledSH0123.a, sampledSH4567.rg);
      const vec3 sh3 = vec3(sampledSH4567.ba, sampledSH891011.r);


      //if (sphericalHarmonics8BitMode == 1) {
      //    sh1 = sh1 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      //    sh2 = sh2 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      //    sh3 = sh3 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
      //}

      const float x = worldViewDir.x;
      const float y = worldViewDir.y;
      const float z = worldViewDir.z;
      splatColor.rgb += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);

      if(frameInfo.sphericalHarmonicsDegree >= 2)
      {

        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float yz = y * z;
        const float xz = x * z;

        const vec4 sampledSH12131415 =
            texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, 6, 3, textureSize(sphericalHarmonicsTexture, 0)), 0);
        const vec4 sampledSH16171819 =
            texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, 6, 4, textureSize(sphericalHarmonicsTexture, 0)), 0);
        const vec4 sampledSH20212223 =
            texelFetch(sphericalHarmonicsTexture, getDataPos(splatIndex, 6, 5, textureSize(sphericalHarmonicsTexture, 0)), 0);

        const vec3 sh4 = sampledSH891011.gba;
        const vec3 sh5 = sampledSH12131415.rgb;
        const vec3 sh6 = vec3(sampledSH12131415.a, sampledSH16171819.rg);
        const vec3 sh7 = vec3(sampledSH16171819.ba, sampledSH20212223.r);
        const vec3 sh8 = sampledSH20212223.gba;

        //if (sphericalHarmonics8BitMode == 1) {
        //    sh4 = sh4 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //    sh5 = sh5 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //    sh6 = sh6 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //    sh7 = sh7 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //    sh8 = sh8 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //}

        splatColor.rgb += (SH_C2[0] * xy) * sh4 + (SH_C2[1] * yz) * sh5 + (SH_C2[2] * (2.0 * zz - xx - yy)) * sh6
                          + (SH_C2[3] * xz) * sh7 + (SH_C2[4] * (xx - yy)) * sh8;
      }
    }

    // emit per primitive color as early as possible for perf reasons
    outSplatCol[gl_LocalInvocationIndex * 2 + 0] = splatColor;
    outSplatCol[gl_LocalInvocationIndex * 2 + 1] = splatColor;

    // Use RGBA texture map to store sets of 3 elements requires some offset shifting depending on splatIndex
    const vec4 sampledCovarianceA =
        texelFetch(covariancesTexture, getDataPosF(nearestEvenIndex, 1.5, oddOffset, textureSize(covariancesTexture, 0)), 0);
    const vec4 sampledCovarianceB =
        texelFetch(covariancesTexture,
                   getDataPosF(nearestEvenIndex, 1.5, oddOffset + uint(1), textureSize(covariancesTexture, 0)), 0);

    const vec3 cov3D_M11_M12_M13 =
        vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) + vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
    const vec3 cov3D_M22_M23_M33 =
        vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) + vec3(sampledCovarianceB.gba) * fOddOffset;

    // Construct the 3D covariance matrix
    const mat3 Vrk =
        mat3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
             cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);

    mat3 J;
    if(frameInfo.orthographicMode == 1)
    {
      // Since the projection is linear, we don't need an approximation
      J = transpose(mat3(frameInfo.orthoZoom, 0.0, 0.0, 0.0, frameInfo.orthoZoom, 0.0, 0.0, 0.0, 0.0));
    }
    else
    {
      // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
      // 3D covariance matrix instead of using the actual projection matrix because that transformation would
      // require a non-linear component (perspective division) which would yield a non-gaussian result.
      const float s = 1.0 / (viewCenter.z * viewCenter.z);
      J             = mat3(frameInfo.focal.x / viewCenter.z, 0., -(frameInfo.focal.x * viewCenter.x) * s, 0.,
                           frameInfo.focal.y / viewCenter.z, -(frameInfo.focal.y * viewCenter.y) * s, 0., 0., 0.);
    }

    // Concatenate the projection approximation with the model-view transformation
    const mat3 W = transpose(mat3(transformModelViewMatrix));
    const mat3 T = W * J;

    // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
    mat3 cov2Dm = transpose(T) * Vrk * T;

    const bool antialiased = false;
    if(antialiased)
    {
      const float detOrig = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
      cov2Dm[0][0] += 0.3;
      cov2Dm[1][1] += 0.3;
      const float detBlur      = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
      const float compensation = sqrt(max(detOrig / detBlur, 0.0));

      splatColor.a *= compensation;

      // overwrite
      outSplatCol[gl_LocalInvocationIndex * 2 + 0] = splatColor;
      outSplatCol[gl_LocalInvocationIndex * 2 + 1] = splatColor;
    }
    else
    {
      cov2Dm[0][0] += 0.3;
      cov2Dm[1][1] += 0.3;
    }

    // alpha based culling
    if(splatColor.a < minAlpha)
    {
      // Early return to discard splat
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 0].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 1].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 2].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 3].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }

    // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
    // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
    // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
    // need cov2Dm[1][0] because it is a symetric matrix.
    const vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

    const vec3 ndcCenter = clipCenter.xyz / clipCenter.w;

    // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
    // so that we can determine the 2D basis for the splat. This is done using the method described
    // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
    // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * eigen-value), which is
    // equal to scaling them by sqrt(8) standard deviations.
    //
    // This is a different approach than in the original work at INRIA. In that work they compute the
    // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
    // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
    // times the maximum eigen-value, or 3 standard deviations. They then use the inverse 2D covariance
    // matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by calculating the
    // full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
    const float a           = cov2Dv.x;
    const float d           = cov2Dv.z;
    const float b           = cov2Dv.y;
    const float D           = a * d - b * b;
    const float trace       = a + d;
    const float traceOver2  = 0.5 * trace;
    const float term2       = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
    float       eigenValue1 = traceOver2 + term2;
    float       eigenValue2 = traceOver2 - term2;

    if(frameInfo.pointCloudModeEnabled == 1)
    {
      eigenValue1 = eigenValue2 = 0.2;
    }

    // from original code
    if(eigenValue2 <= 0.0)
    {
      // Early return to discard splat
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 0].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 1].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 2].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + 3].gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
      return;
    }

    const vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
    // since the eigen vectors are orthogonal, we derive the second one from the first
    const vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

    // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
    const vec2 basisVector1 = eigenVector1 * frameInfo.splatScale * min(sqrt8 * sqrt(eigenValue1), 2048.0);
    const vec2 basisVector2 = eigenVector2 * frameInfo.splatScale * min(sqrt8 * sqrt(eigenValue2), 2048.0);

    /////////////////////////////
    // emiting quad vertices

    UNROLL_LOOP
    for(uint i = 0; i < 4; ++i)
    {
      const vec2 fragPos = positions[i].xy;

      const vec2 ndcOffset = vec2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                             * frameInfo.inverseFocalAdjustment;

      const vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

      gl_MeshVerticesEXT[gl_LocalInvocationIndex * 4 + i].gl_Position = quadPos;
    }
  }
}