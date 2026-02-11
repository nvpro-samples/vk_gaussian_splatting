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

static const int PAYLOAD_INVALID_ID   = -1;
static const float PAYLOAD_INF_DISTANCE = asfloat(0x7F7FFFFF);

// the payload structure used for both Particle and Mesh tracing

struct HitPayload
{
    int id[PAYLOAD_ARRAY_SIZE];
    float dist[PAYLOAD_ARRAY_SIZE];
#if WIREFRAME
    float3 differentialX;
    float3 differentialY;
    float wireframeDist;
#endif
};
