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

#ifndef _GUT_CAMERA_MODELS_H_
#define _GUT_CAMERA_MODELS_H_

// File translated from github-grut\threedgut_tracer\include\3dgut\sensors\cameraModels.h

struct OpenCVPinholeProjectionParameters
{
  float2  nearFar;
  float2  principalPoint;
  float2  focalLength;
  float radialCoeffs[6];
  float2  tangentialCoeffs;
  float4  thinPrismCoeffs;
};

struct OpenCVFisheyeProjectionParameters
{
  float2  principalPoint;
  float2  focalLength;
  float radialCoeffs[4];
  float maxAngle;
  float2  nominalResolution;
};

struct CameraModelParameters
{
  int                               shutterType;
  int                               modelType;
  OpenCVPinholeProjectionParameters ocvPinholeParams;
  OpenCVFisheyeProjectionParameters ocvFisheyeParams;
};

// Enum values for ShutterType
static const int RollingTopToBottomShutter = 0;
static const int RollingLeftToRightShutter = 1;
static const int RollingBottomToTopShutter = 2;
static const int RollingRightToLeftShutter = 3;
static const int GlobalShutter             = 4;

// Enum values for ModelType
static const int OpenCVPinholeModel = 0;
static const int OpenCVFisheyeModel = 1;
static const int EmptyModel         = 2;
static const int Unsupported        = 3;


CameraModelParameters initPerfectPinholeCamera(float2 nearFar, float2 viewport, float2 focalLength)
{
  CameraModelParameters cameraModel;
  cameraModel.shutterType                       = GlobalShutter;
  cameraModel.modelType                         = OpenCVPinholeModel;
  cameraModel.ocvPinholeParams.nearFar          = nearFar;
  cameraModel.ocvPinholeParams.principalPoint   = viewport / 2.0;  // center of viewport
  cameraModel.ocvPinholeParams.focalLength      = focalLength;
  cameraModel.ocvPinholeParams.tangentialCoeffs = float2(0, 0);
  cameraModel.ocvPinholeParams.thinPrismCoeffs  = float4(0, 0, 0, 0);
  cameraModel.ocvPinholeParams.radialCoeffs[0]  = 0;
  cameraModel.ocvPinholeParams.radialCoeffs[1]  = 0;
  cameraModel.ocvPinholeParams.radialCoeffs[2]  = 0;
  cameraModel.ocvPinholeParams.radialCoeffs[3]  = 0;
  cameraModel.ocvPinholeParams.radialCoeffs[4]  = 0;
  cameraModel.ocvPinholeParams.radialCoeffs[5]  = 0;
  return cameraModel;
}

// Following functions to compute maxAngle parameter for fisheye

// Given an image size component (x or y) and corresponding principal point component (x or y),
// returns the maximum distance (in image domain units) from the principal point to either image boundary.
float computeMaxDistanceToBorder(float imageSizeComponent, float principalPointComponent)
{
  float center = 0.5 * imageSizeComponent;
  if(principalPointComponent > center)
  {
    return principalPointComponent;
  }
  else
  {
    return imageSizeComponent - principalPointComponent;
  }
}

// Compute the maximum radius from the principal point to the image boundaries.
float computeMaxRadius(float2 imageSize, float2 principalPoint)
{

  float2 maxDiag = float2(computeMaxDistanceToBorder(imageSize.x, principalPoint.x),
                      computeMaxDistanceToBorder(imageSize.y, principalPoint.y));
  return length(maxDiag);
}

// Estimate max angle for fisheye
float computeMaxAngle(float2 resolution, float2 principalPoint, float2 focalLength)
{
  float maxRadiusPixels = computeMaxRadius(resolution, principalPoint);
  float fovAngleX       = 2.0 * maxRadiusPixels / focalLength.x;
  float fovAngleY       = 2.0 * maxRadiusPixels / focalLength.y;
  float maxAngle        = max(fovAngleX, fovAngleY) / 2.0;
  return maxAngle;
}

CameraModelParameters initPerfectFisheyeCamera(float2 viewport, float2 focalLength)
{
  const float2 principalPoint = viewport / 2.0;  // center of viewport

  CameraModelParameters cameraModel;
  cameraModel.shutterType                        = GlobalShutter;
  cameraModel.modelType                          = OpenCVFisheyeModel;
  cameraModel.ocvFisheyeParams.principalPoint    = principalPoint;
  cameraModel.ocvFisheyeParams.focalLength       = focalLength;
  cameraModel.ocvFisheyeParams.nominalResolution = viewport;
  cameraModel.ocvFisheyeParams.maxAngle          = computeMaxAngle(viewport, principalPoint, focalLength);
  cameraModel.ocvFisheyeParams.radialCoeffs[0]   = 0;
  cameraModel.ocvFisheyeParams.radialCoeffs[1]   = 0;
  cameraModel.ocvFisheyeParams.radialCoeffs[2]   = 0;
  cameraModel.ocvFisheyeParams.radialCoeffs[3]   = 0;

  return cameraModel;
}

#endif
