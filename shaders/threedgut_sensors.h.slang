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

// File translated from github-grut\threedgut_tracer\include\3dgut\sensors\sensors.h

#include "quaternions.h.slang"
#include "threedgut_camera_models.h.slang"

struct SensorPose
{
  float3 translation;
  quat quaternion;  // (xyz, w), scalar last
};

struct SensorState
{
  int64_t    startTimestamp;
  SensorPose startPose;
  int64_t    endTimestamp;
  SensorPose endPose;
};

SensorState initGlobalShutterSensorState(float3 translation, quat quaternion)
{
  // camera pose in world space, used by projector to compute the ray.
  SensorState sensorState;
  sensorState.startPose.translation = translation;
  sensorState.startPose.quaternion  = quaternion;
  // For global shutter, start and end poses are the same (we do not use endPose)
  sensorState.endPose.translation = translation;
  sensorState.endPose.quaternion  = quaternion;
  sensorState.startTimestamp      = 0;
  sensorState.endTimestamp        = 1;
  return sensorState;
}