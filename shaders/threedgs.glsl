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

#extension GL_EXT_shader_explicit_arithmetic_types : require

#ifndef _3DGS_CORE_H_
#define _3DGS_CORE_H_

// cov3Dm: 3D covariance matrix
// splatCenterView: splat center in view coordinates
// returns: the upper-left 2x2 portion of the projected 3D covariance matrix (see comments in function body). 
vec3 threedgsCovarianceProjection(mat3 cov3Dm, vec4 splatCenterView, vec2 focal, mat4 modelViewTransform)
{
  // TODO: the ortho mode is not supported yet
#if ORTHOGRAPHIC_MODE == 1
  // Since the projection is linear, we don't need an approximation
  const mat3 J = transpose(mat3(orthoZoom, 0.0, 0.0, 0.0, orthoZoom, 0.0, 0.0, 0.0, 0.0));
#else
  // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
  // 3D covariance matrix instead of using the actual projection matrix because that transformation would
  // require a non-linear component (perspective division) which would yield a non-gaussian result.
  const float s = 1.0 / (splatCenterView.z * splatCenterView.z);
  const mat3  J = mat3(focal.x / splatCenterView.z, 0., -(focal.x * splatCenterView.x) * s, 0.,
                       focal.y / splatCenterView.z, -(focal.y * splatCenterView.y) * s, 0., 0., 0.);
#endif

  // Concatenate the projection approximation with the model-view transformation
  const mat3 W = transpose(mat3(modelViewTransform));
  const mat3 T = W * J;

  // Transform the 3D covariance matrix (cov3Dm) to compute the 2D covariance matrix
  const mat3 cov2Dm = transpose(T) * cov3Dm * T;

  // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
  // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
  // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
  // need cov2Dm[1][0] because it is a symetric matrix.
  return vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);
}

// This function ingests the projected 2D covariance and outputs the basis vectors of its 2D extent
// opacity is updated if MipSplatting antialiasing is applied.
bool threedgsProjectedExtentBasis(vec3 cov2Dv, float stdDev, float splatScale, inout float opacity, out vec2 basisVector1, out vec2 basisVector2)
{

#if MS_ANTIALIASING == 1
  // This mode is used when model is reconstructed using MipSplatting
  // https://niujinshuchong.github.io/mip-splatting/
  const float detOrig = cov2Dv[0] * cov2Dv[2] - cov2Dv[1] * cov2Dv[1];
#endif

  cov2Dv[0] += 0.3;
  cov2Dv[2] += 0.3;

#if MS_ANTIALIASING == 1
  const float detBlur = cov2Dv[0] * cov2Dv[2] - cov2Dv[1] * cov2Dv[1];
  // apply the alpha compensation
  opacity *= sqrt(max(detOrig / detBlur, 0.0));
#endif

  // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
  // so that we can determine the 2D basis for the splat. This is done using the method described
  // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
  // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
  // by normalizing the eigen-vectors and then multiplying them by (stdDev * eigen-value), which is
  // equal to scaling them by stdDev standard deviations.
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

  if(eigenValue2 <= 0.0)
  {
    return false;
  }

#if POINT_CLOUD_MODE
  eigenValue1 = eigenValue2 = 0.2;
#endif

  const vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
  // since the eigen vectors are orthogonal, we derive the second one from the first
  const vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

  basisVector1 = eigenVector1 * splatScale * min(stdDev * sqrt(eigenValue1), 2048.0);
  basisVector2 = eigenVector2 * splatScale * min(stdDev * sqrt(eigenValue2), 2048.0);

  return true;
}

#endif