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

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#include "device_host.h"
#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in uint32_t inSplatIndex;

layout(location = 0) out vec2 outFragPos;
layout(location = 1) out vec4 outFragCol;

// unused for the time beeing
layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

// in order to manage alignment automatically we could write:
// layout(set = 0, binding = ..., scalar) uniform FrameInfo_
// but it may be less performant than aligning
// attribute in the struct (see device_host.h comment)
layout(set = 0, binding = BINDING_FRAME_INFO_UBO) uniform _frameInfo
{
  FrameInfo frameInfo;
};

// textures map describing the 3DGS model
layout(set = 0, binding = BINDING_CENTERS_TEXTURE) uniform sampler2D centersTexture;
layout(set = 0, binding = BINDING_COLORS_TEXTURE) uniform sampler2D colorsTexture;
layout(set = 0, binding = BINDING_COVARIANCES_TEXTURE) uniform sampler2D covariancesTexture;
layout(set = 0, binding = BINDING_SH_TEXTURE) uniform sampler2D sphericalHarmonicsTexture;

void main()
{
  const uint splatIndex = inSplatIndex;

  //
#ifdef USE_DATA_TEXTURES
  const vec3 splatCenter = fetchCenter(centersTexture, splatIndex);
#else
  const vec3 splatCenter = fetchCenter(splatIndex);
#endif

  const mat4 transformModelViewMatrix = frameInfo.viewMatrix;
  const vec4 viewCenter               = transformModelViewMatrix * vec4(splatCenter, 1.0);

  const vec4 clipCenter = frameInfo.projectionMatrix * viewCenter;

  const float clip = 1.2 * clipCenter.w;
  if(frameInfo.frustumCulling == FRUSTUM_CULLING_VERT
     && (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip || clipCenter.y < -clip || clipCenter.y > clip))
  {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  const vec2 fragPos = inPosition.xy;
  // emit as early as possible
  // Scale the position data we send to the fragment shader
  outFragPos = fragPos * sqrt8;

#ifdef USE_DATA_TEXTURES
  vec4 splatColor = fetchColor(colorsTexture, splatIndex);
#else
  vec4 splatColor = fetchColor(splatIndex);
#endif
  if(frameInfo.showShOnly == 1)
  {
    splatColor.r = 0.5;
    splatColor.g = 0.5;
    splatColor.b = 0.5;
  }

  if(frameInfo.sphericalHarmonicsDegree >= 1)
  {
    // SH coefficients for degree 1 (1,2,3)
    vec3 shd1[3];
    // SH coefficients for degree 2 (4 5 6 7 8)
    vec3 shd2[5];
    // fetch the data (only what is needed according to degree)
#ifdef USE_DATA_TEXTURES
    fetchSh(sphericalHarmonicsTexture, splatIndex, frameInfo.sphericalHarmonicsDegree,
            frameInfo.sphericalHarmonics8BitMode, shd1, shd2);
#else
    fetchSh(splatIndex, frameInfo.sphericalHarmonicsDegree, frameInfo.sphericalHarmonics8BitMode, shd1, shd2);
#endif

    const vec3  worldViewDir = normalize(splatCenter - frameInfo.cameraPosition);
    const float x            = worldViewDir.x;
    const float y            = worldViewDir.y;
    const float z            = worldViewDir.z;
    splatColor.rgb += SH_C1 * (-shd1[0] * y + shd1[1] * z - shd1[2] * x);

    if(frameInfo.sphericalHarmonicsDegree >= 2)
    {
      const float xx = x * x;
      const float yy = y * y;
      const float zz = z * z;
      const float xy = x * y;
      const float yz = y * z;
      const float xz = x * z;

      splatColor.rgb += (SH_C2[0] * xy) * shd2[0] + (SH_C2[1] * yz) * shd2[1] + (SH_C2[2] * (2.0 * zz - xx - yy)) * shd2[2]
                        + (SH_C2[3] * xz) * shd2[3] + (SH_C2[4] * (xx - yy)) * shd2[4];
    }
  }

  // emit as early as possible for perf reasons
  outFragCol = splatColor;

  // Fetch and construct the 3D covariance matrix
#ifdef USE_DATA_TEXTURES
  const mat3 Vrk = fetchCovariance(covariancesTexture, splatIndex);
#else
  const mat3 Vrk = fetchCovariance(splatIndex);
#endif

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
    float s = 1.0 / (viewCenter.z * viewCenter.z);
    J       = mat3(frameInfo.focal.x / viewCenter.z, 0., -(frameInfo.focal.x * viewCenter.x) * s, 0.,
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
    outFragCol = splatColor;
  }
  else
  {
    cov2Dm[0][0] += 0.3;
    cov2Dm[1][1] += 0.3;
  }

  // alpha based culling
  if(splatColor.a < minAlpha)
  {
    // JEM added gl_position set for proper discard
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
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
    // JEM added gl_position set for proper discard
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  const vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
  // since the eigen vectors are orthogonal, we derive the second one from the first
  const vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

  // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
  const vec2 basisVector1 = eigenVector1 * frameInfo.splatScale * min(sqrt8 * sqrt(eigenValue1), 2048.0);
  const vec2 basisVector2 = eigenVector2 * frameInfo.splatScale * min(sqrt8 * sqrt(eigenValue2), 2048.0);

  const vec2 ndcOffset = vec2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                         * frameInfo.inverseFocalAdjustment;

  const vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
  gl_Position        = quadPos;
}
