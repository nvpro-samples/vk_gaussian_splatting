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
static const bool kWireframe     = (WIREFRAME != 0);

// Each payload array is independently sized to minimize register pressure:
//   kDistSize: max(SPP, meshPayloadSlots) — dist[] holds mesh packed hit data (11 or 15 slots) or particle distances
//   kIdSize:   max(SPP, 2 if meshes)      — id[] holds SPP particle IDs + mesh needs id[0]=objId, id[1]=matId
//   kSppSize:  PARTICLES_SPP              — splatSetIdx[] is particle-only; stochastic color/normal are per-sample
static const int kDistSize = PAYLOAD_ARRAY_SIZE;
static const int kSppSize  = PARTICLES_SPP;
#if RTX_HAS_MESHES && PARTICLES_SPP < 2
static const int kIdSize   = 2;    // Mesh closest-hit needs id[0]=objId, id[1]=matId
#else
static const int kIdSize   = PARTICLES_SPP;
#endif

typedef HitPayloadGeneric<kStochasticSort, kWireframe, kDistSize, kIdSize, kSppSize> HitPayload;

// Payload structure shared by Particle and Mesh tracing.
// Array sizes are decoupled so that mesh hit packing (which needs many dist[] slots)
// does not inflate particle-only arrays (splatSetIdx[], stochastic color/normal).
//
// RTX_HAS_PARTICLES encodes the RTX splat descriptor count (clamped): 0 = no particle
// code, 1 = single splat set (descriptor index 0 implied, splatSetIdx[] and
// currentSplatSetInstance removed from the payload), 2 = multi-set code path.
// Access the descriptor index only through the accessors below so the single-set
// specialization stays confined to this struct.
struct HitPayloadGeneric<let kStochasticSplat : bool = false, let kWireframe : bool = false,
                         let kDistSize : int, let kIdSize : int = kDistSize, let kSppSize : int = kDistSize>
{
  // Particle tracing: indexed by [0..SPP-1] in all three arrays.
  // Mesh closest-hit: packs hit data into dist[0..meshSlots-1], id[0]=objId, id[1]=matId.
  int   id[kIdSize];            // Particle: local splat ID | Mesh: objId + matId (2 entries)
#if RTX_HAS_PARTICLES >= 2
  int   splatSetIdx[kSppSize];  // Particle-only: splat set descriptor index (meshes never use this)
#endif
  float dist[kDistSize];        // Particle: hit distances | Mesh: packed hit data (11 or 15 slots)
#if RTX_HAS_PARTICLES >= 2
  uint  currentSplatSetInstance;  // Current splat-set-instance descriptor index (RTX_USE_INSTANCES mode)
#endif
  uint  rayBounce;                // Bounce index: 0 = primary camera ray, >0 = secondary (reflection/refraction/shadow)

  // Per-sample splat set descriptor index. Single-set mode: constant 0, no storage.
  int getSplatSetIdx(int i)
  {
#if RTX_HAS_PARTICLES >= 2
    return splatSetIdx[i];
#else
    return 0;
#endif
  }

  [mutating]
  void setSplatSetIdx(int i, int value)
  {
#if RTX_HAS_PARTICLES >= 2
    splatSetIdx[i] = value;
#endif
  }

  // Splat-set-instance descriptor index for RTX_USE_INSTANCES mode.
  // Single-set mode: constant 0, no storage.
  uint getCurrentSplatSetInstance()
  {
#if RTX_HAS_PARTICLES >= 2
    return currentSplatSetInstance;
#else
    return 0;
#endif
  }

  [mutating]
  void setCurrentSplatSetInstance(uint value)
  {
#if RTX_HAS_PARTICLES >= 2
    currentSplatSetInstance = value;
#endif
  }

  // Stochastic any-hit: color and normal are computed in the any-hit shader and stored per sample.
  // Sized by kSppSize (not kDistSize) since mesh hit packing does not use these.
  Conditional<uint, kStochasticSplat>   rngSeed;
  Conditional<float4, kStochasticSplat> color[kSppSize];
  Conditional<float3, kStochasticSplat> normal[kSppSize];

  Conditional<float3, kWireframe> differentialX;
  Conditional<float3, kWireframe> differentialY;
  Conditional<float, kWireframe>  wireframeDist;
  Conditional<float, kWireframe>  wireframeAlpha;
};
