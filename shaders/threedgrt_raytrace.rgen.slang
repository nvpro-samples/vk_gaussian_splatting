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

#include "nvshaders/random.h.slang"
#include "nvshaders/wireframe.h.slang"

#include "shaderio.h"
#include "cameras.h.slang"
#include "color.h.slang"
#include "wavefront.h.slang"

#include "threedgs_particles_storage.h.slang"
#include "threedgrt.h.slang"
#include "threedgrt_payload.h.slang"

// clang-format off
[[vk::binding(RTX_BINDING_TLAS_SPLATS, 1)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(RTX_BINDING_OUTIMAGE, 1)]] RWTexture2D<float4> image;
[[vk::binding(RTX_BINDING_AUX1, 1)]] RWTexture2D<float4> imageAux;
[[vk::binding(RTX_BINDING_OUTDEPTH, 1)]] RWTexture2D<float> depthBuffer;
#if RTX_USE_MESHES
[[vk::binding(RTX_BINDING_TLAS_MESH, 1)]] RaytracingAccelerationStructure topLevelASMesh;
#endif
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstantRay> pcRay;
// TODO to be removed, just for debug readback
[[vk::binding(BINDING_INDIRECT_BUFFER, 0)]] RWStructuredBuffer<IndirectParams> indirectBuffer;
#if RTX_USE_MESHES
[[vk::binding(BINDING_MESH_DESCRIPTORS, 0)]] StructuredBuffer<ObjDesc> objDesc;
[[vk::binding(BINDING_LIGHT_SET, 0)]] StructuredBuffer<LightSource> lights;
#endif
// clang-format on

double maxComponent(in double3 v)
{
  return max(v.x, max(v.y, v.z));
}

[shader("raygeneration")]
void main()
{
  float3 rayOrigin;
  float3 rayDirection;

  const uint2  launchID        = DispatchRaysIndex().xy;
  const uint2  launchSize      = DispatchRaysDimensions().xy;
  const float2 launchIdFloat   = float2(launchID.x, launchID.y);      // notation required to prevent warning
  const float2 launchSizeFloat = float2(launchSize.x, launchSize.y);  // notation required to prevent warning

  // generate the ray for this pixel
#if CAMERA_TYPE == CAMERA_PINHOLE
  generatePinholeRay(launchIdFloat, launchSizeFloat, frameInfo.viewInverse, frameInfo.projInverse, rayOrigin, rayDirection);
#else
  if(!generateFisheyeRay(launchIdFloat, launchSizeFloat, frameInfo.fovRad, float2(0.0), frameInfo.viewInverse, rayOrigin, rayDirection))
  {
    image[launchID] = float4(0.0, 0.0, 0.0, 1.0);
    return;
  }
#endif

  // add Depth-of-Field perturbation to the ray
#if((PIPELINE != PIPELINE_HYBRID) && RTX_DOF_ENABLED)
  // Initialize the random number
  uint seed = xxhash32(uint3(launchID, frameInfo.frameSampleId));

  depthOfField(seed, frameInfo.focusDist, frameInfo.aperture, frameInfo.viewInverse, rayOrigin, rayDirection);
#endif

  // will use Any hit only
  uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
  // will use Closest hit only
  uint rayFlagsMesh = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

  float       tMax = 10000.0;
  const float epsT = 1e-9;

  int rayHitsCount = 0;

  float3  radiance      = float3(0.0f);
  double3 transmittance = float3(1.0f);

  bool meshHit = false;  // did we have a mesh hit by primary ray
  int  bounce  = 0;
  int  done    = 1;

  uint64_t clockStart = clockARB();

  // for debug feedback and alternative visu modes
  bool  closestParticleFound = false;
  float closestParticleDist  = PAYLOAD_INF_DISTANCE;
  int   closestParticleId    = PAYLOAD_INVALID_ID;

#if HYBRID_ENABLED
  float4 pixel;
  if(frameInfo.frameSampleId <= 0)  // no temporal or first frame
    pixel = image[launchID];
  else
    pixel = imageAux[launchID];
  radiance      = pixel.xyz;
  transmittance = float3(1.0 - pixel.w);
#endif

  // Create the payload
  HitPayload payload = {};

#if WIREFRAME
  computeDifferentials(launchIdFloat, launchSizeFloat, frameInfo.viewInverse, frameInfo.projInverse,
                       payload.differentialX, payload.differentialY);
#endif

  // bounce loop
  while(true)
  {
    // each new bounce reinits Tmin
    float tMin               = 0.001;
    tMax                     = 10000.0;
    float rayLastHitDistance = max(0.0f, tMin - epsT);

    // trace the mesh with one closest hit if meshes are activated
    float meshHitDist = PAYLOAD_INF_DISTANCE;

#if RTX_USE_MESHES

    float3 meshHitWorldPos;
    float3 meshHitWorldNrm;
    int    meshHitObjId;
    int    meshHitMatId;

    payload.dist[0] = PAYLOAD_INF_DISTANCE;
    payload.id[0]   = PAYLOAD_INVALID_ID;
    payload.id[1]   = PAYLOAD_INVALID_ID;

// #define USE_SER
#ifdef USE_SER
    hitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj);
    hitObjectTraceRayNV(hObj,                       // the hitObject instance
                        topLevelASMesh,             // acceleration structure
                        rayFlagsMesh,               // rayFlags
                        0xFF,                       // cullMask
                        0,                          // sbtRecordOffset
                        0,                          // sbtRecordStride
                        0,                          // missIndex
                        rayOrigin,                  // ray origin
                        rayLastHitDistance + epsT,  // ray min range
                        rayDirection,               // ray direction
                        tMax + epsT,                // ray max range
                        0                           // payload (location = 0)
    );
    reorderThreadNV(hObj);
    hitObjectExecuteShaderNV(hObj, 0);
#else
    RayDesc rayMesh;
    rayMesh.Origin    = rayOrigin;
    rayMesh.Direction = rayDirection;
    rayMesh.TMin      = rayLastHitDistance + epsT;
    rayMesh.TMax      = tMax + epsT;

    TraceRay(topLevelASMesh,  // acceleration structure
             rayFlagsMesh,    // rayFlags
             0xFF,            // cullMask
             0,               // sbtRecordOffset
             0,               // sbtRecordStride
             0,               // missIndex
             rayMesh,         // ray
             payload          // payload (location = 0)
    );
#endif
    meshHitDist = payload.dist[0];
    //
    if(meshHitDist < PAYLOAD_INF_DISTANCE)
    {
      tMax = meshHitDist;
      // collect info to compute shading and potential new ray later on
      meshHitObjId    = payload.id[0];
      meshHitMatId    = payload.id[1];
      meshHitWorldPos = float3(payload.dist[1], payload.dist[2], payload.dist[3]);
      meshHitWorldNrm = float3(payload.dist[4], payload.dist[5], payload.dist[6]);
      // Early return if meshDepthOnly pre-pass
      if(pcRay.meshDepthOnly)
      {
        const float4 clip = mul(mul(float4(meshHitWorldPos, 1.0), frameInfo.viewMatrix), frameInfo.projectionMatrix);
        const float3 ndc  = clip.xyz / clip.w;
        depthBuffer[launchID] = ndc.z;
        // clear image buffer
        if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
          image[launchID] = float4(0.0, 0.0, 0.0, 1.0);
        else
          imageAux[launchID] = float4(0.0, 0.0, 0.0, 1.0);
        return;
      }
      //
      if(bounce == 0)
        meshHit = true;
    }
    else if(pcRay.meshDepthOnly)
    {
      // Early return if meshDepthOnly pre-pass
      depthBuffer[launchID] = 1.0;
      // clear image buffer
      if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
        image[launchID] = float4(0.0, 0.0, 0.0, 1.0);
      else
        imageAux[launchID] = float4(0.0, 0.0, 0.0, 1.0);
      return;
    }
#endif

    // when hybrid is enabled, we trace the particles only for secondary rays
#if HYBRID_ENABLED
    if(bounce > 0)
    {
#endif
      // trace the gaussians with multi-pass any hit
      int outerIdx = 0;

      // The two following are to compute particleProcessHit with transformed splat set model
      const float3 splatSetModelRayOrigin = mul(float4(rayOrigin, 1.0), pcRay.modelMatrixInverse).xyz;
      // Since the ray direction should not be affected by translation,
      // uses the inverse of the rotation-scale part of the model matrix.
      const float3 splatSetModelRayDirection = normalize(mul(rayDirection, float3x3(pcRay.modelMatrixRotScaleInverse)));

      while(outerIdx < frameInfo.maxPasses && (rayLastHitDistance <= tMax)
            && (maxComponent(transmittance) > double(frameInfo.minTransmittance)))
      {
        // prepare the payload
        [[unroll]]
        for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
        {
          payload.id[i]   = PAYLOAD_INVALID_ID;
          payload.dist[i] = PAYLOAD_INF_DISTANCE;
        }
#if WIREFRAME
        // Infinite means no wireframe hit by the ray
        payload.wireframeDist = PAYLOAD_INF_DISTANCE;
#endif

        // trace the PAYLOAD_ARRAY_SIZE any hits
        RayDesc raySplat;
        raySplat.Origin    = rayOrigin;
        raySplat.Direction = rayDirection;
        raySplat.TMin      = rayLastHitDistance + epsT;
        raySplat.TMax      = tMax + epsT;

        TraceRay(topLevelAS,  // acceleration structure
                 rayFlags,    // rayFlags
                 0xFF,        // cullMask
                 0,           // sbtRecordOffset
                 0,           // sbtRecordStride
                 0,           // missIndex
                 raySplat,    // ray
                 payload      // payload (location = 0)
        );

        const int firstId = payload.id[0];

        // no more hits found
        if(firstId == PAYLOAD_INVALID_ID)
        {
          break;
        }

        // evaluate the sorted hits
        [[unroll]]
        for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
        {
          const int   splatId = payload.id[i];
          const float dist    = payload.dist[i];

          if((splatId != PAYLOAD_INVALID_ID) && (maxComponent(transmittance) > double(frameInfo.minTransmittance)))
          {
            bool acceptedHit = particleProcessHit(frameInfo, splatSetModelRayOrigin, splatSetModelRayDirection, splatId,
                                                  transmittance, radiance);
            rayHitsCount += int(acceptedHit);

            // debug feedback and alternative visualizations
            if(!closestParticleFound && acceptedHit && outerIdx == 0)
            {
              closestParticleId    = splatId;
              closestParticleDist  = dist;
              closestParticleFound = true;
            }
#if WIREFRAME
            if(dist == payload.wireframeDist)
            {
              radiance      = float3(1.0, 0.0, 0.0);  // wireframe color
              transmittance = float3(0.0);            // opaque
            }
#endif
            // we move on in any case
            rayLastHitDistance = max(rayLastHitDistance, dist);
          }
        }

        //
        outerIdx++;
      }

#if HYBRID_ENABLED
    }
#endif


#if RTX_USE_MESHES
    // process mesh shading if needed
    if((meshHitDist < PAYLOAD_INF_DISTANCE) && (maxComponent(transmittance) > double(frameInfo.minTransmittance)))
    {
      // retrieve the material for this hit
      ObjMaterial meshHitMaterial = objDesc[meshHitObjId].materialAddress[meshHitMatId];
      wavefrontComputeShading(lights, frameInfo.lightCount, meshHitWorldPos, meshHitWorldNrm, meshHitMaterial,
                              rayDirection, transmittance, radiance, done, rayOrigin, rayDirection);
    }
#endif

    // continue with secondary rays if needed (decided by wavefrontComputeShading)
    bounce++;
    if(done == 1 || bounce > frameInfo.rtxMaxBounces)
      break;

    // prepare ray for next iteration
    // new rayOrigin and direction was set by wavefrontComputeShading
    // Next iteration will stop if a reflective material isn't hit
    done = 1;

  }  // while(true)

  // debug feedback
  if(closestParticleFound && all(launchID == frameInfo.cursor))
  {
    indirectBuffer[0].particleID   = closestParticleId;
    indirectBuffer[0].particleDist = closestParticleDist;
  }

  //
  float3 fragRadiance = radiance;

  uint64_t clock = clockARB() - clockStart;

#if VISUALIZE == VISUALIZE_CLOCK
  fragRadiance = hsbToRgb(
      float3(lerp(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * 0.1 * float(clock) / float(1 << 20))), 1.0, 1.));
#endif

#if VISUALIZE == VISUALIZE_DEPTH
  if(closestParticleFound)
  {
    fragRadiance = hsbToRgb(
        float3(lerp(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * float(closestParticleDist) / 256.0F)), 1.0, 1.));
  }
#endif

#if VISUALIZE == VISUALIZE_RAYHITS
  fragRadiance = float3(frameInfo.multiplier * 10 * float(rayHitsCount) / 255, 0.0, 0.0);
#endif

#if VISUALIZE == VISUALIZE_FINAL
#if HYBRID_ENABLED
  if(meshHit)
  {
    if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
      image[launchID] = float4(fragRadiance, 1.0);
    else
      imageAux[launchID] = float4(fragRadiance, 1.0);
  }
  return;
#else
  if(frameInfo.frameSampleId > 0)
  {
    // Do accumulation over time
    const float  a        = 1.0F / float(frameInfo.frameSampleId + 1);
    const float4 oldColor = image[launchID];
    fragRadiance          = lerp(oldColor, float4(fragRadiance, 1.0), a).xyz;
  }
#endif
#endif

// depending on previous build conditions this code might become unreachable (which is normal)
#pragma warning(disable: 41000)  // Disable warning warning 41000: unreachable code detected
  image[launchID] = float4(fragRadiance, 1.0);
}
