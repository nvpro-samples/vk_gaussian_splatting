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

#ifndef _QUATERNIONS_H_
#define _QUATERNIONS_H_

// internal representation of the quaternion is xyz the vector and w the scalar
// so default constructor quat() takes w as last parameter
#define quat vec4

// create a quaternion, w scalar is first parameter
quat scalarsToQuat(float w, float x, float y, float z) {
  return vec4(x, y, z, w);
}

// convert a quaternion stored on a vec4 with scalar first storage (w on x)
quat vec4toQuat(vec4 scalarFirst)
{
  return vec4(scalarFirst.yzw, scalarFirst.x);
}

// convert a rotation quaternion to a 3x3 rotation matrix
mat3 quatToMat3(quat q)
{
  const float x = q.x, y = q.y, z = q.z, w = q.w;

  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float wx = w * x;
  const float wy = w * y;
  const float wz = w * z;

  return mat3(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 
              2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 
              2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy));
}

// convert a rotation quaternion to a 3x3 rotation matrix and 
// transpose the result which provides its inverse matrix (since we 
// deal with pure rotation matrix)
mat3 quatToMat3Transpose(quat q)
{
  const float x = q.x, y = q.y, z = q.z, w = q.w;

  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float wx = w * x;
  const float wy = w * y;
  const float wz = w * z;

  return mat3(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 
    2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 
    2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy));
}

// converts a pure 3x3 rotation matrix into to a quaternion
quat mat3toQuat(mat3 m)
{
  float trace = m[0][0] + m[1][1] + m[2][2];
  quat q;

  if(trace > 0.0)
  {
    float s    = sqrt(trace + 1.0) * 2.0;
    float invS = 1.0 / s;
    q.w         = 0.25 * s;
    q.x        = (m[2][1] - m[1][2]) * invS;
    q.y         = (m[0][2] - m[2][0]) * invS;
    q.z         = (m[1][0] - m[0][1]) * invS;
  }
  else if((m[0][0] > m[1][1]) && (m[0][0] > m[2][2]))
  {
    float s    = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
    float invS = 1.0 / s;
    q.w        = (m[2][1] - m[1][2]) * invS;
    q.x        = 0.25 * s;
    q.y        = (m[0][1] + m[1][0]) * invS;
    q.z        = (m[0][2] + m[2][0]) * invS;
  }
  else if(m[1][1] > m[2][2])
  {
    float s    = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
    float invS = 1.0 / s;
    q.w        = (m[0][2] - m[2][0]) * invS;
    q.x        = (m[0][1] + m[1][0]) * invS;
    q.y        = 0.25 * s;
    q.z        = (m[1][2] + m[2][1]) * invS;
  }
  else
  {
    float s    = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
    float invS = 1.0 / s;
    q.w        = (m[1][0] - m[0][1]) * invS;
    q.x        = (m[0][2] + m[2][0]) * invS;
    q.y        = (m[1][2] + m[2][1]) * invS;
    q.z        = 0.25 * s;
  }

  return q;
}

// rotates the position using the rotation quaternion q 
vec3 rotateByQuaternion(vec3 position, vec4 q)
{
  // Apply rotation using the formula:
  // p' = p + 2 * q_w * (q_xyz x p) + 2 * q_xyz x (q_xyz x p)
  vec3 t = 2.0 * cross(q.xyz, position);
  return position + q.w * t + cross(q.xyz, t);
}

quat quatMultiply(quat q1, quat q2)
{
  quat result;
  result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
  result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  return result;
}

quat quatSlerp(quat q1, quat q2, float t)
{
  // Compute the cosine of the angle between the two vectors.
  float dotProduct = dot(q1, q2);

  // If the dot product is negative, slerp won't take the shorter path.
  // Note that q and -q are equivalent when the negation is applied to all four components of the quaternion.
  // Fix by reversing one quaternion.
  if(dotProduct < 0.0)
  {
    q2         = -q2;
    dotProduct = -dotProduct;
  }

  const float DOT_THRESHOLD = 0.995;
  if(dotProduct > DOT_THRESHOLD)
  {
    // If the inputs are too close for comfort, linearly interpolate
    // and normalize the result.
    quat result = q1 + t * (q2 - q1);
    return normalize(result);
  }

  // Since dot is in range [0, DOT_THRESHOLD], acos is safe
  float theta_0     = acos(dotProduct);  // theta_0 = angle between input vectors
  float theta       = theta_0 * t;       // theta = angle between q1 and result
  float sin_theta   = sin(theta);        // compute this value only once
  float sin_theta_0 = sin(theta_0);      // compute this value only once

  float s0 = cos(theta) - dotProduct * sin_theta / sin_theta_0;  // == sin(theta_0 - theta) / sin(theta_0)
  float s1 = sin_theta / sin_theta_0;

  return (s0 * q1) + (s1 * q2);
}

#endif