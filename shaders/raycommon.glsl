/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"

const int PAYLOAD_INVALID_ID   = -1;
float     PAYLOAD_INF_DISTANCE = uintBitsToFloat(0x7F7FFFFF);

// access payload buffer
#if USE_RTX_PAYLOAD_BUFFER
layout(set = 1, binding = ePayloadBuffer, scalar) buffer _payload
{
  HitBufferPayload payload[];
};
#endif

// access payload stack
// prd (Per Ray Data)
#ifdef USED_FROM_RAY_GEN
layout(location = 0) rayPayloadEXT hitPayload prd;
#else
layout(location = 0) rayPayloadInEXT hitPayload prd;
#endif

int readId(in uint i) 
{
#if USE_RTX_PAYLOAD_BUFFER
  const uint index = (gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x) * PAYLOAD_ARRAY_SIZE + i;
  return payload[index].id;
#else
  return prd.id[i];
 #endif
}

void writeId(in uint i, in int id)
{
#if USE_RTX_PAYLOAD_BUFFER
  const uint index  = (gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x) * PAYLOAD_ARRAY_SIZE + i;
  payload[index].id = id;
#else
  prd.id[i] = id;
#endif
}

float readDist(in uint i)
{
#if USE_RTX_PAYLOAD_BUFFER
  const uint index = (gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x) * PAYLOAD_ARRAY_SIZE + i;
  return payload[index].dist;
#else
  return prd.dist[i];
#endif
}

void writeDist(in uint i, in float dist)
{
#if USE_RTX_PAYLOAD_BUFFER
  const uint index    = (gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x) * PAYLOAD_ARRAY_SIZE + i;
  payload[index].dist = dist;
#else
  prd.dist[i] = dist;
#endif
}

vec2 readBary(in uint i)
{
#if USE_RTX_PAYLOAD_BUFFER 
  return vec2(0.0);
#else
#if WIREFRAME
  return prd.bary[i];
#else
  return vec2(0.0);
#endif
#endif
}

void writeBary(in uint i, in vec2 bary)
{
#if USE_RTX_PAYLOAD_BUFFER
  return;
#else
#if WIREFRAME
  prd.bary[i] = bary;
#endif
#endif
}