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

#ifndef _GUT_PROJECTIOR_H_
#define _GUT_PROJECTIOR_H_

// File translated from github-grut\threedgut_tracer\include\3dgut\kernels\cuda\renderers\gutProjector.cuh

#include "threedgut_definitions.h.slang"
#include "threedgut_camera_projections.h.slang"

bool threedgutParticleProjection(float2      resolution,
                                 SensorModel sensorModel,
                                 float4x4    toWorldMatrix,
                                 SensorState sensorShutterState,
                                 float3      particlePosition,
                                 float3      particleScale,
                                 float3x3    particleRotation,
                                 out float2  particleProjCenter,
                                 out float3  particleProjCovariance)
{
  float3 particleMean = particlePosition;

  /* no Znear culling for now
  if((particleMean.x * sensorMatrix[0][2] + particleMean.y * sensorMatrix[1][2] + particleMean.z * sensorMatrix[2][2]
      + sensorMatrix[3][2])
     < 0.2) // Params.ParticleMinSensorZ)
  {
    return false;
  }
  */

  int    numValidPoints = 0;
  float2 projectedSigmaPoints[2 * GUT_D + 1];

  const float Lambda = GUT_LAMBDA;

  if(projectPointWithShutter(mul(float4(particleMean, 1.0), toWorldMatrix).xyz, resolution, sensorModel,
                             sensorShutterState, GUT_IN_IMAGE_MARGIN_FACTOR, projectedSigmaPoints[0]))
  {
    numValidPoints++;
  }
  particleProjCenter = projectedSigmaPoints[0] * (GUT_LAMBDA / (GUT_D + GUT_LAMBDA));

  const float weightI = 1.0 / (2.0 * (GUT_D + GUT_LAMBDA));
  [[unroll]]
  for(int i = 0; i < GUT_D; ++i)
  {
    float3 delta = GUT_DELTA * particleScale[i] * particleRotation[i];

    if(projectPointWithShutter(mul(float4(particleMean + delta, 1.0), toWorldMatrix).xyz, resolution, sensorModel,
                               sensorShutterState, GUT_IN_IMAGE_MARGIN_FACTOR, projectedSigmaPoints[i + 1]))
    {
      numValidPoints++;
    }
    particleProjCenter += weightI * projectedSigmaPoints[i + 1];

    if(projectPointWithShutter(mul(float4(particleMean - delta, 1.0), toWorldMatrix).xyz, resolution, sensorModel,
                               sensorShutterState, GUT_IN_IMAGE_MARGIN_FACTOR, projectedSigmaPoints[i + 1 + GUT_D]))
    {
      numValidPoints++;
    }
    particleProjCenter += weightI * projectedSigmaPoints[i + 1 + GUT_D];
  }

  if(GUT_REQUIRE_ALL_SIGMA_POINTS_VALID)
  {
    if(numValidPoints < (2 * GUT_D + 1))
    {
#pragma warning(disable: 41018)  // Disable warning 41018 : returning without initializing some variables/parameters
      return false;
#pragma warning(default: 41018)
    }
  }
  else if(numValidPoints == 0)
  {
#pragma warning(disable: 41018)  // Disable warning 41018 : returning without initializing some variables/parameters
    return false;
#pragma warning(default: 41018)
  }

  {
    float2      centeredPoint = projectedSigmaPoints[0] - particleProjCenter;
    const float weight0       = GUT_LAMBDA / (GUT_D + GUT_LAMBDA) + (1.0 - GUT_ALPHA * GUT_ALPHA + GUT_BETA);
    particleProjCovariance =
        weight0
        * float3(centeredPoint.x * centeredPoint.x, centeredPoint.x * centeredPoint.y, centeredPoint.y * centeredPoint.y);
  }
  [[unroll]]
  for(int i = 0; i < 2 * GUT_D; ++i)
  {
    float2 centeredPoint = projectedSigmaPoints[i + 1] - particleProjCenter;
    particleProjCovariance +=
        weightI
        * float3(centeredPoint.x * centeredPoint.x, centeredPoint.x * centeredPoint.y, centeredPoint.y * centeredPoint.y);
  }

  return true;
}


bool threedgutProjectedExtentConicOpacity(float3 covariance, float opacity, out float2 extent, out float4 conicOpacity, out float maxConicOpacityPower)
{
  float3 dilatedCovariance = float3(covariance.x + GUT_COVARIANCE_DILATION, covariance.y, covariance.z + GUT_COVARIANCE_DILATION);
  float dilatedCovDet = dilatedCovariance.x * dilatedCovariance.z - dilatedCovariance.y * dilatedCovariance.y;
  if(dilatedCovDet == 0.0)
  {
#pragma warning(disable: 41018)  // Disable warning 41018 : returning without initializing some variables/parameters
    return false;
#pragma warning(default: 41018)
  }
  conicOpacity.xyz = float3(dilatedCovariance.z, -dilatedCovariance.y, dilatedCovariance.x) / dilatedCovDet;

  // See Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting" https://github.com/autonomousvision/mip-splatting
#if MS_ANTIALIASING == 1
  float covDet            = covariance.x * covariance.z - covariance.y * covariance.y;
  float convolutionFactor = sqrt(max(0.000025, covDet / dilatedCovDet));
  conicOpacity.w          = opacity * convolutionFactor;
#else
  conicOpacity.w = opacity;
#endif

  if(conicOpacity.w < GUT_ALPHA_THRESHOLD)
  {
#pragma warning(disable: 41018)  // Disable warning 41018 : returning without initializing some variables/parameters
    return false;
#pragma warning(default: 41018)
  }

  maxConicOpacityPower = log(conicOpacity.w / GUT_ALPHA_THRESHOLD);
  float extentFactor   = GUT_TIGHT_OPACITY_BOUNDING ? min(3.33, sqrt(2.0 * maxConicOpacityPower)) : 3.33;
  float minLambda      = 0.01;
  float mid            = 0.5 * (dilatedCovariance.x + dilatedCovariance.z);
  float lambda         = mid + sqrt(max(minLambda, mid * mid - dilatedCovDet));
  float radius         = extentFactor * sqrt(lambda);
  if(GUT_RECT_BOUNDING)
  {
    extent = min(extentFactor * sqrt(float2(dilatedCovariance.x, dilatedCovariance.z)), float2(radius));
  }
  else
  {
    extent = float2(radius);
  }

  return radius > 0.0;
}

#endif 