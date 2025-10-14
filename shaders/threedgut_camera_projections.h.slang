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

#ifndef _GUT_CAMERA_PROJECTIONS_H_
#define _GUT_CAMERA_PROJECTIONS_H_

#include "quaternions.glsl"
#include "threedgut_definitions.glsl"
#include "threedgut_sensors.glsl"

#define SensorModel CameraModelParameters

// Following functions
// translated from github-grut\threedgut_tracer\include\3dgut\kernels\cuda\sensors\cameraProjections.cuh

// Computes 2-norm of a [x,y] vector in a numerically stable way
float stableNorm2(vec2 vec)
{
  float absX   = abs(vec.x);
  float absY   = abs(vec.y);
  float minVal = min(absX, absY);
  float maxVal = max(absX, absY);
  if(maxVal <= 0.0)
  {
    return 0.0;
  }
  float minMaxRatio = minVal / maxVal;
  return maxVal * sqrt(1.0 + minMaxRatio * minMaxRatio);
}

// Evaluates an N-1 degree polynomial y=f(x) using numerically stable Horner scheme.
// With:
// f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
// Here N=4, replace 4 by N for a generalization
float evalPolyHorner4(float[4] coeffs, float x)
{
  float y = coeffs[4 - 1];
  [[unroll]] for(int i = 4 - 2; i >= 0; --i)
  {
    y = x * y + coeffs[i];
  }
  return y;
}

float relativeShutterTime(SensorModel sensorModel, vec2 resolution, vec2 position)
{
  switch(sensorModel.shutterType)
  {
    case RollingTopToBottomShutter:
      return floor(position.y) / (resolution.y - 1.0);
    case RollingLeftToRightShutter:
      return floor(position.x) / (resolution.x - 1.0);
    case RollingBottomToTopShutter:
      return (resolution.y - ceil(position.y)) / (resolution.y - 1.0);
    case RollingRightToLeftShutter:
      return (resolution.x - ceil(position.x)) / (resolution.x - 1.0);
    default:
      return 0.5;
  }
}

bool withinResolution(vec2 resolution, float tolerance, vec2 p)
{
  const vec2 tolMargin = resolution * tolerance;
  return (p.x > -tolMargin.x) && (p.y > -tolMargin.y) && (p.x < resolution.x + tolMargin.x)
         && (p.y < resolution.y + tolMargin.y);
}

bool projectPointPinhole(OpenCVPinholeProjectionParameters sensorParams, vec2 resolution, vec3 position, float tolerance, out vec2 projected)
{

  if(position.z <= 0.0)
  {
    projected = vec2(0.0);
    return false;
  }

  vec2 uvNormalized = position.xy / position.z;

  // computeDistortion
  const vec2  uvSquared = uvNormalized * uvNormalized;
  const float r2        = uvSquared.x + uvSquared.y;
  const float a1        = 2.0 * uvNormalized.x * uvNormalized.y;
  const float a2        = r2 + 2.0 * uvSquared.x;
  const float a3        = r2 + 2.0 * uvSquared.y;

  const float icD_numerator =
      1.0 + r2 * (sensorParams.radialCoeffs[0] + r2 * (sensorParams.radialCoeffs[1] + r2 * sensorParams.radialCoeffs[2]));
  const float icD_denominator =
      1.0 + r2 * (sensorParams.radialCoeffs[3] + r2 * (sensorParams.radialCoeffs[4] + r2 * sensorParams.radialCoeffs[5]));
  const float icD = icD_numerator / icD_denominator;

  const vec2 delta = vec2(sensorParams.tangentialCoeffs[0] * a1 + sensorParams.tangentialCoeffs[1] * a2
                              + r2 * (sensorParams.thinPrismCoeffs[0] + r2 * sensorParams.thinPrismCoeffs[1]),
                          sensorParams.tangentialCoeffs[0] * a3 + sensorParams.tangentialCoeffs[1] * a1
                              + r2 * (sensorParams.thinPrismCoeffs[2] + r2 * sensorParams.thinPrismCoeffs[3]));

  // Project using ideal pinhole model (apply radial / tangential / thin-prism distortions)
  // in case radial distortion is within limits
  const vec2 uvND = icD * uvNormalized + delta;

  const float kMinRadialDist = 0.8;
  const float kMaxRadialDist = 1.2;
  const bool  validRadial    = (icD > kMinRadialDist) && (icD < kMaxRadialDist);
  if(validRadial)
  {
    projected = uvND * sensorParams.focalLength + sensorParams.principalPoint;
  }
  else
  {
    // If the radial distortion is out-of-limits, the computed coordinates will be unreasonable
    // (might even flip signs) - check on which side of the image we overshoot, and set the coordinates
    // out of the image bounds accordingly. The coordinates will be clipped to
    // viable range and direction but the exact values cannot be trusted / are still invalid
    const float roiClippingRadius = length(resolution);
    projected                     = (roiClippingRadius / sqrt(r2)) * uvNormalized + sensorParams.principalPoint;
  }

  return validRadial && withinResolution(resolution, tolerance, projected);
}

void applyRenderingResolution(vec2 nominalResolution, vec2 renderingResolution, inout vec2 projected)
{
  if(renderingResolution != nominalResolution)
  {
    const float rescalingFactor = renderingResolution.x / nominalResolution.x;
    // apply vertical offset for different aspect ratios
    projected.y += 0.5 * (renderingResolution.y / rescalingFactor - nominalResolution.y);
    projected *= rescalingFactor;
  }
}

bool projectPointFisheye(OpenCVFisheyeProjectionParameters sensorParams, vec2 resolution, vec3 position, float tolerance, out vec2 projected)
{
  const float eps       = 1e-7;
  const float rho       = max(stableNorm2(position.xy), eps);
  const float thetaFull = atan(rho, position.z);
  // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
  // In particular for omnidirectional cameras, this prevents points outside the FOV to be
  // wrongly projected to in-image-domain points because of badly constrained polynomials outside
  // the effective FOV (which is different to the image boundaries).
  //
  // These FOV-clamped projections will be marked as *invalid*
  const float theta = min(thetaFull, sensorParams.maxAngle);
  // Evaluate forward polynomial
  // (radial distances to the principal point in the normalized image domain (up to focal length scales))
  const float theta2 = theta * theta;
  const float delta  = (theta * (evalPolyHorner4(sensorParams.radialCoeffs, theta2) * theta2 + 1.0)) / rho;
  projected          = sensorParams.focalLength * position.xy * delta + sensorParams.principalPoint;

  // We do not need this for our use case
  // applyRenderingResolution(sensorParams.nominalResolution, vec2(resolution), projected);

  return (theta < sensorParams.maxAngle) && withinResolution(resolution, tolerance, projected);
}


bool projectPoint(SensorModel sensorModel, vec2 resolution, vec3 position, float tolerance, out vec2 projected)
{
  switch(sensorModel.modelType)
  {
    case OpenCVPinholeModel:
      return projectPointPinhole(sensorModel.ocvPinholeParams, resolution, position, tolerance, projected);
    case OpenCVFisheyeModel:
      return projectPointFisheye(sensorModel.ocvFisheyeParams, resolution, position, tolerance, projected);
    default:
      projected = vec2(0.0);
      return false;
  }
}


bool projectPointWithShutter(vec3 position, vec2 resolution, SensorModel sensorModel, SensorState sensorState, float tolerance, out vec2 projectedPosition)
{
  // Coordinate system conversion from Right Up Front to Right Up Back
  vec3 rubToRufFlipPos  = vec3(1.0, 1.0, -1.0);
  vec4 rubToRufFlipQuat = vec4(-1.0, -1.0, 1.0, 1.0);  // the scalar term w is always left unchanged

  const vec3 tStart = sensorState.startPose.translation * rubToRufFlipPos;
  const quat qStart = sensorState.startPose.quaternion * rubToRufFlipQuat;

  // Position of the particle in world coordinates
  const vec3 wc_startPosition = rotateByQuaternion(position * rubToRufFlipPos, qStart) + tStart;

  bool validProjection = projectPoint(sensorModel, resolution, wc_startPosition, tolerance, projectedPosition);
  if(sensorModel.shutterType == GlobalShutter)
  {
    return validProjection;
  }

  ////////////////////////////////////////////
  // Attention,
  // the rest of the code has not been tested, we only support
  // sensorModel.shutterType == GlobalShutter for now

  const vec3 tEnd = sensorState.endPose.translation * rubToRufFlipPos;
  const quat qEnd = sensorState.endPose.quaternion * rubToRufFlipQuat;

  const vec3 wc_endPosition = rotateByQuaternion(position * rubToRufFlipPos, qEnd) + tEnd;

  if(!validProjection)
  {
    validProjection = projectPoint(sensorModel, resolution, wc_endPosition, tolerance, projectedPosition);
    if(!validProjection)
    {
      return false;
    }
  }

  // Compute the new timestamp and project again
  [[unroll]] for(int i = 0; i < GUT_N_ROLLING_SHUTTER_ITERATIONS; ++i)
  {
    const float alpha = relativeShutterTime(sensorModel, resolution, projectedPosition);

    const vec3 wc_interPosition =
        rotateByQuaternion(position * rubToRufFlipPos, quatSlerp(qStart, qEnd, alpha)) + mix(tStart, tEnd, alpha);

    validProjection = projectPoint(sensorModel, resolution, wc_interPosition, tolerance, projectedPosition);
  }

  return validProjection;
}

#endif
