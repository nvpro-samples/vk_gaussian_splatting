/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"

static const int   PAYLOAD_INVALID_ID   = -1;
static const float PAYLOAD_INF_DISTANCE = asfloat(0x7F7FFFFF);

static const bool kStochasticSort = (RTX_TRACE_STRATEGY == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT);
static const bool kWireframe      = WIREFRAME;
static const bool kPayloadSize    = PAYLOAD_ARRAY_SIZE;

typedef HitPayloadGeneric<kStochasticSort, kWireframe, kPayloadSize> HitPayload;

// the payload structure used for both Particle and Mesh tracing
struct HitPayloadGeneric<let kStochasticSplat : bool = false, let kWireframe : bool = false, let kSize : int>
{
  int   id[kSize];
  float dist[kSize];
  uint  tlasInstanceOffset;  // Base offset for current TLAS (multi-TLAS support)

  // Raygen-provided seed for stochastic any-hit
  Conditional<uint, kStochasticSplat>   rngSeed;
  Conditional<float4, kStochasticSplat> color[kSize];
  Conditional<float3, kStochasticSplat> normal[kSize];

  Conditional<float3, kWireframe> differentialX;
  Conditional<float3, kWireframe> differentialY;
  Conditional<float, kWireframe>  wireframeDist;
  Conditional<float, kWireframe>  wireframeAlpha;
};
