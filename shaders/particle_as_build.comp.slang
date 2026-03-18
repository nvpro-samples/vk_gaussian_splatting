/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"
#include "quaternions.h.slang"
#include "threedgs_particle_storage.h.slang"

[[vk::push_constant]] ParticleAsBuildPushConstants pc;

struct AccelInstance
{
  float4 row0;
  float4 row1;
  float4 row2;
  uint   instanceCustomIndexAndMask;
  uint   instanceShaderBindingTableRecordOffsetAndFlags;
  uint2  accelerationStructureReference;  // uint64_t split
};

struct AabbPositions
{
  float minX;
  float minY;
  float minZ;
  float maxX;
  float maxY;
  float maxZ;
};

static const float kIcosasVrtScale = 1.0704662767622;  // Compensated: original 0.6615845 * phi, since verts scaled by 1/phi to fit [-1,1]^3
static const float kAabbVrtScale   = 1.0; // AABB geometry is already [-1,1]^3, keep original scale
static const uint  kGeometryTriangles = 0;
static const uint  kGeometryAabbs     = 1;

// Unit icosahedron scaled by 1/phi so max extent fits in [-1,1]^3 (coherent with AABB mode)
static const float3 kIcosaVerts[12] = {
    float3(-0.618034f,  1.0f,       0.0f),
    float3( 0.618034f,  1.0f,       0.0f),
    float3( 0.0f,       0.618034f, -1.0f),
    float3(-1.0f,       0.0f,      -0.618034f),
    float3(-1.0f,       0.0f,       0.618034f),
    float3( 0.0f,       0.618034f,  1.0f),
    float3( 1.0f,       0.0f,       0.618034f),
    float3( 0.0f,      -0.618034f,  1.0f),
    float3(-0.618034f, -1.0f,       0.0f),
    float3( 0.0f,      -0.618034f, -1.0f),
    float3( 1.0f,       0.0f,      -0.618034f),
    float3( 0.618034f, -1.0f,       0.0f),
};

static const uint kIcosaInds[60] = {
    0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 1,
    6, 1, 5, 6, 5, 7, 6, 7, 11, 6, 11, 10, 6, 10, 1,
    8, 4, 3, 8, 3, 9, 8, 9, 11, 8, 11, 7, 8, 7, 4,
    9, 3, 2, 9, 2, 10, 9, 10, 11, 5, 4, 7, 1, 10, 2
};

float kernelScale(float density, float modulatedMinResponse, float kernelDegree, bool adaptiveClamping)
{
  float responseModulation = adaptiveClamping ? density : 1.0f;
  float minResponse        = min(modulatedMinResponse / responseModulation, 0.97f);

  if(kernelDegree == 0.0f)
  {
    return ((1.0f - minResponse) / 3.0f) / -0.329630334487f;
  }

  float b = kernelDegree;
  float a = -4.5f / pow(3.0f, b);
  return pow(log(minResponse) / a, 1.0f / b);
}


AccelInstance makeInstance(float4x4 world, uint customIndex, uint mask, uint sbtOffset, uint flags, uint64_t blasAddress)
{
  AccelInstance inst;

  float4 r0 = mul(float4(1.0f, 0.0f, 0.0f, 0.0f), world);
  float4 r1 = mul(float4(0.0f, 1.0f, 0.0f, 0.0f), world);
  float4 r2 = mul(float4(0.0f, 0.0f, 1.0f, 0.0f), world);

  inst.row0 = r0;
  inst.row1 = r1;
  inst.row2 = r2;

  inst.instanceCustomIndexAndMask = (customIndex & 0x00FFFFFFu) | ((mask & 0xFFu) << 24);
  inst.instanceShaderBindingTableRecordOffsetAndFlags = (sbtOffset & 0x00FFFFFFu) | ((flags & 0xFFu) << 24);

  inst.accelerationStructureReference = uint2((uint)(blasAddress & 0xFFFFFFFFu), (uint)((blasAddress >> 32) & 0xFFFFFFFFu));
  return inst;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint globalId = dispatchThreadId.x;
  if(pc.writeGeometry != 0)
  {
    if(pc.geometryMode == 0)
    {
      if(pc.geometryType == kGeometryAabbs)
      {
        if(globalId == 0 && pc.aabbBufferAddress != 0)
        {
          AabbPositions* aabb = (AabbPositions*)pc.aabbBufferAddress;
          aabb[0].minX = -1.0f;
          aabb[0].minY = -1.0f;
          aabb[0].minZ = -1.0f;
          aabb[0].maxX = 1.0f;
          aabb[0].maxY = 1.0f;
          aabb[0].maxZ = 1.0f;
        }
      }
      else if(pc.geometryType == kGeometryTriangles)
      {
        if(pc.vertexBufferAddress != 0 && globalId < 12)
        {
          float3* outVerts = (float3*)pc.vertexBufferAddress;
          outVerts[globalId] = kIcosaVerts[globalId];
        }
        if(pc.indexBufferAddress != 0 && globalId < 60)
        {
          uint* outInds = (uint*)pc.indexBufferAddress;
          outInds[globalId] = kIcosaInds[globalId];
        }
      }
    }
    else if(pc.geometryMode == 2)
    {
      // Per-splat-set geometry generation (one BLAS per splat set)
      if(pc.splatSetDescriptorAddress == 0)
        return;

      SplatSetDesc* descs = (SplatSetDesc*)pc.splatSetDescriptorAddress;
      uint splatSetIndex = pc.instanceBaseIndex;
      SplatSetDesc desc = descs[splatSetIndex];

      const uint splatCount = pc.instanceCount;
      const uint totalVerts = splatCount * 12;
      const uint totalInds  = splatCount * 60;

      if(pc.geometryType == kGeometryAabbs)
      {
        if(pc.aabbBufferAddress != 0 && globalId < splatCount)
        {
          uint splatId = pc.splatBaseIndex + globalId;
          float3 position = fetchCenter(desc, splatId);
          float3 scale    = exp(fetchScale(desc, splatId));
          quat   rotation = normalize(vec4toQuat(fetchRotation(desc, splatId)));
          float  density  = fetchColor(desc, splatId).w;
          float kerScale = kernelScale(density, pc.kernelMinResponse, float(pc.kernelDegree), pc.kernelAdaptiveClamping != 0);
          float3 totalScale = scale * kIcosasVrtScale * kerScale;
          float3x3 rot = quatToMat3(rotation);
          float3x3 scaledRot = float3x3(rot[0] * totalScale.x, rot[1] * totalScale.y, rot[2] * totalScale.z);
          float4x4 local = float4x4(
              scaledRot[0].x, scaledRot[1].x, scaledRot[2].x, position.x,
              scaledRot[0].y, scaledRot[1].y, scaledRot[2].y, position.y,
              scaledRot[0].z, scaledRot[1].z, scaledRot[2].z, position.z,
              0.0f,           0.0f,           0.0f,           1.0f);
          float4x4 world = mul(transpose(desc.transform), local);

          float3 minV = float3(1e30f, 1e30f, 1e30f);
          float3 maxV = float3(-1e30f, -1e30f, -1e30f);
          for(uint v = 0; v < 12; ++v)
          {
            float3 p = mul(world, float4(kIcosaVerts[v], 1.0f)).xyz;
            minV = min(minV, p);
            maxV = max(maxV, p);
          }
          AabbPositions* aabb = (AabbPositions*)pc.aabbBufferAddress;
          aabb[globalId].minX = minV.x;
          aabb[globalId].minY = minV.y;
          aabb[globalId].minZ = minV.z;
          aabb[globalId].maxX = maxV.x;
          aabb[globalId].maxY = maxV.y;
          aabb[globalId].maxZ = maxV.z;
        }
      }
      else if(pc.geometryType == kGeometryTriangles)
      {
        if(pc.vertexBufferAddress != 0 && globalId < totalVerts)
        {
          uint localSplatId = globalId / 12;
          uint splatId = pc.splatBaseIndex + localSplatId;
          uint vId = globalId - localSplatId * 12;
          float3 position = fetchCenter(desc, splatId);
          float3 scale    = exp(fetchScale(desc, splatId));
          quat   rotation = normalize(vec4toQuat(fetchRotation(desc, splatId)));
          float  density  = fetchColor(desc, splatId).w;
          float kerScale = kernelScale(density, pc.kernelMinResponse, float(pc.kernelDegree), pc.kernelAdaptiveClamping != 0);
          float3 totalScale = scale * kIcosasVrtScale * kerScale;
          float3x3 rot = quatToMat3(rotation);
          float3x3 scaledRot = float3x3(rot[0] * totalScale.x, rot[1] * totalScale.y, rot[2] * totalScale.z);
          float4x4 local = float4x4(
              scaledRot[0].x, scaledRot[1].x, scaledRot[2].x, position.x,
              scaledRot[0].y, scaledRot[1].y, scaledRot[2].y, position.y,
              scaledRot[0].z, scaledRot[1].z, scaledRot[2].z, position.z,
              0.0f,           0.0f,           0.0f,           1.0f);
          float4x4 world = mul(transpose(desc.transform), local);

          float3* outVerts = (float3*)pc.vertexBufferAddress;
          outVerts[globalId] = mul(world, float4(kIcosaVerts[vId], 1.0f)).xyz;
        }
        if(pc.indexBufferAddress != 0 && globalId < totalInds)
        {
          uint localSplatId = globalId / 60;
          uint splatId = pc.splatBaseIndex + localSplatId;
          uint iId = globalId - localSplatId * 60;
          uint* outInds = (uint*)pc.indexBufferAddress;
          outInds[globalId] = kIcosaInds[iId] + localSplatId * 12;
        }
      }
    }
  }

  uint instanceLocalId = globalId;
  if(instanceLocalId >= pc.instanceCount)
    return;
  if(pc.tlasInstanceBufferAddress == 0 || pc.splatSetDescriptorAddress == 0)
    return;

  SplatSetDesc*          descs      = (SplatSetDesc*)pc.splatSetDescriptorAddress;
  AccelInstance*         instances  = (AccelInstance*)pc.tlasInstanceBufferAddress;

  if(pc.instanceMode == 1)
  {
    uint instanceIndex = pc.instanceBaseIndex + instanceLocalId;
    SplatSetDesc desc = descs[instanceIndex];
    float4x4 world = transpose(desc.transform);
    uint64_t blasAddress = desc.blasAddress;
    const uint kMask = (blasAddress != 0) ? 0xFFu : 0u;
    const uint kSbtOffset = 0u;
    const uint kFlags = 0x8u;  // VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR
    instances[instanceLocalId] = makeInstance(world, instanceIndex, kMask, kSbtOffset, kFlags, blasAddress);
    return;
  }

  if(pc.globalIndexTableAddress == 0)
    return;

  uint globalSplatId = pc.instanceBaseIndex + instanceLocalId;
  // 64-bit byte offset to prevent overflow when globalSplatId > ~537M (buffer > 4 GB)
  uint64_t gitByteOffset = uint64_t(globalSplatId) * uint64_t(sizeof(GlobalSplatIndexEntry));
  GlobalSplatIndexEntry entry = *(GlobalSplatIndexEntry*)(pc.globalIndexTableAddress + gitByteOffset);
  SplatSetDesc          desc  = descs[entry.splatSetIndex];
  uint splatLocalId = entry.splatIndex;

  float3 position = fetchCenter(desc, splatLocalId);
  float3 scale    = exp(fetchScale(desc, splatLocalId));
  quat   rotation = normalize(vec4toQuat(fetchRotation(desc, splatLocalId)));
  float  density  = fetchColor(desc, splatLocalId).w;

  float kerScale = kernelScale(density, pc.kernelMinResponse, float(pc.kernelDegree), pc.kernelAdaptiveClamping != 0);
  float vrtScale = (pc.geometryType == kGeometryTriangles) ? kIcosasVrtScale : kAabbVrtScale;
  float3 totalScale = scale * vrtScale * kerScale;

  float3x3 rot = quatToMat3(rotation);
  float3x3 scaledRot = float3x3(rot[0] * totalScale.x, rot[1] * totalScale.y, rot[2] * totalScale.z);

  float4x4 local = float4x4(
      scaledRot[0].x, scaledRot[1].x, scaledRot[2].x, position.x,
      scaledRot[0].y, scaledRot[1].y, scaledRot[2].y, position.y,
      scaledRot[0].z, scaledRot[1].z, scaledRot[2].z, position.z,
      0.0f,           0.0f,           0.0f,           1.0f);

  // Match CPU path: world = instanceTransform * localTransform
  float4x4 world = mul(transpose(desc.transform), local);

  const uint kMask = 0xFFu;
  const uint kSbtOffset = 0u;
  const uint kFlags = 0x8u;  // VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR

  instances[instanceLocalId] = makeInstance(world, globalSplatId, kMask, kSbtOffset, kFlags, pc.blasAddress);

}

