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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require  // for [[unroll]]
#extension GL_GOOGLE_include_directive : require
#extension GL_ARB_shader_clock : require  // profiling timers

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_NV_shader_invocation_reorder : enable

#include "nvshaders/slang_types.h"
#include "nvshaders/random.h.slang"

#include "shaderio.h"
#include "cameras.glsl"
#include "color.glsl"
#include "wavefront.glsl"

#include "threedgs_particles_storage.glsl"
#include "threedgrt.glsl"
#define USED_FROM_RAY_GEN
#include "threedgrt_payload.glsl"

layout(set = 1, binding = RTX_BINDING_TLAS_SPLATS) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = RTX_BINDING_OUTIMAGE, rgba8) uniform image2D image;
layout(set = 1, binding = RTX_BINDING_AUX1, rgba8) uniform image2D imageAux;
// for proper use of depth buffer as an output image2D we must specify writeonly and remove the format
// layout(set = 1, binding = RTX_BINDING_OUTDEPTH, r32f) uniform image2D depthBuffer;
layout(set = 1, binding = RTX_BINDING_OUTDEPTH) writeonly uniform image2D depthBuffer;
#if RTX_USE_MESHES
layout(set = 1, binding = RTX_BINDING_TLAS_MESH) uniform accelerationStructureEXT topLevelASMesh;
#endif

// scalar prevents alignment issues
layout(set = 0, binding = BINDING_FRAME_INFO_UBO, scalar) uniform _frameInfo
{
  FrameInfo frameInfo;
};

layout(push_constant) uniform _PushConstantRay
{
  PushConstantRay pcRay;
};

// TODO to be removed, just for debug readback
layout(set = 0, binding = BINDING_INDIRECT_BUFFER, scalar) writeonly buffer _indirect
{
  IndirectParams indirect;
};

#if RTX_USE_MESHES
layout(set = 0, binding = BINDING_MESH_DESCRIPTORS, scalar) buffer ObjDesc_
{
  ObjDesc i[];
}
objDesc;
#endif

layout(buffer_reference, scalar) buffer Vertices  // vertex positions of a icosa or an obj mesh
{
  vec3 v[];
};
layout(buffer_reference, scalar) buffer Indices  // Triangle indices
{
  ivec3 i[];
};
layout(buffer_reference, scalar) buffer Materials  // Materials
{
  ObjMaterial m[];
};
layout(set = 0, binding = BINDING_LIGHT_SET, scalar) buffer LightSet_
{
  LightSource l[4];
}
lights;

double maxComponent(in dvec3 v)
{
  return max(v.x, max(v.y, v.z));
}

void main()
{
  vec3 rayOrigin;
  vec3 rayDirection;

  // generate the ray for this pixel
#if CAMERA_TYPE == CAMERA_PINHOLE
  generatePinholeRay(gl_LaunchIDEXT.xy, vec2(gl_LaunchSizeEXT.xy), frameInfo.viewInverse, frameInfo.projInverse, rayOrigin, rayDirection);
#else
  if(!generateFisheyeRay(gl_LaunchIDEXT.xy, vec2(gl_LaunchSizeEXT.xy), frameInfo.fovRad, vec2(0.0),
                         frameInfo.viewInverse, rayOrigin, rayDirection))
  {
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(0.0, 0.0, 0.0, 1.0));
    return;
  }
#endif

  // add Depth-of-Field perturbation to the ray
#if((PIPELINE != PIPELINE_HYBRID) && RTX_DOF_ENABLED)
  // Initialize the random number
  uint seed = xxhash32(uvec3(gl_LaunchIDEXT.xy, frameInfo.frameSampleId));

  depthOfField(seed, frameInfo.focusDist, frameInfo.aperture, frameInfo.viewInverse, rayOrigin, rayDirection);
#endif

  // will use Any hit only
  uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT | gl_RayFlagsNoOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  // will use Closest hit only
  uint rayFlagsMesh = gl_RayFlagsCullBackFacingTrianglesEXT;

  float       tMax = 10000.0;
  const float epsT = 1e-9;

  int rayHitsCount = 0;

  vec3  radiance      = vec3(0.0f);
  dvec3 transmittance = vec3(1.0f);

  bool meshHit = false;  // did we have a mesh hit by primary ray
  int  bounce  = 0;
  int  done    = 1;

  uint64_t clockStart = clockARB();

  // for debug feedback and alternative visu modes
  bool  closestParticleFound = false;
  float closestParticleDist  = PAYLOAD_INF_DISTANCE;
  int   closestParticleId    = PAYLOAD_INVALID_ID;

#if HYBRID_ENABLED
  vec4 pixel;
  if(frameInfo.frameSampleId <= 0)  // no temporal or first frame
    pixel = imageLoad(image, ivec2(gl_LaunchIDEXT.xy));
  else
    pixel = imageLoad(imageAux, ivec2(gl_LaunchIDEXT.xy));
  radiance      = vec3(pixel);
  transmittance = vec3(1.0 - pixel.w);
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

    vec3 meshHitWorldPos;
    vec3 meshHitWorldNrm;
    int  meshHitObjId;
    int  meshHitMatId;

    writeDist(0, PAYLOAD_INF_DISTANCE);
    writeId(0, PAYLOAD_INVALID_ID);
    writeId(1, PAYLOAD_INVALID_ID);

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
    traceRayEXT(topLevelASMesh,             // acceleration structure
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
#endif
    meshHitDist = readDist(0);
    //
    if(meshHitDist < PAYLOAD_INF_DISTANCE)
    {
      tMax = meshHitDist;
      // collect info to compute shading and potential new ray later on
      meshHitObjId    = readId(0);
      meshHitMatId    = readId(1);
      meshHitWorldPos = vec3(readDist(1), readDist(2), readDist(3));
      meshHitWorldNrm = vec3(readDist(4), readDist(5), readDist(6));
      // Early return if meshDepthOnly pre-pass
      if(pcRay.meshDepthOnly)
      {
        const vec4 clip = frameInfo.projectionMatrix * frameInfo.viewMatrix * vec4(meshHitWorldPos, 1.0);
        const vec3 ndc  = clip.xyz / clip.w;
        imageStore(depthBuffer, ivec2(gl_LaunchIDEXT.xy), vec4(ndc.z, 0.0, 0.0, 0.0));
        // clear image buffer
        if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
          imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(0.0, 0.0, 0.0, 1.0));
        else
          imageStore(imageAux, ivec2(gl_LaunchIDEXT.xy), vec4(0.0, 0.0, 0.0, 1.0));
        return;
      }
      //
      if(bounce == 0)
        meshHit = true;
    }
    else if(pcRay.meshDepthOnly)
    {
      // Early return if meshDepthOnly pre-pass
      imageStore(depthBuffer, ivec2(gl_LaunchIDEXT.xy), vec4(1.0, 0.0, 0.0, 0.0));
      // clear image buffer
      if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
        imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(0.0, 0.0, 0.0, 1.0));
      else
        imageStore(imageAux, ivec2(gl_LaunchIDEXT.xy), vec4(0.0, 0.0, 0.0, 1.0));
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
      const vec3 splatSetModelRayOrigin = vec3(pcRay.modelMatrixInverse * vec4(rayOrigin, 1.0));
      // Since the ray direction should not be affected by translation,
      // uses the inverse of the rotation-scale part of the model matrix.
      const vec3 splatSetModelRayDirection = normalize(mat3(pcRay.modelMatrixRotScaleInverse) * rayDirection);

      while(outerIdx < frameInfo.maxPasses && (rayLastHitDistance <= tMax) && (maxComponent(transmittance) > frameInfo.minTransmittance))
      {
        // prepare the payload
        [[unroll]] for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
        {
          writeId(i, PAYLOAD_INVALID_ID);
          writeDist(i, PAYLOAD_INF_DISTANCE);
        }

        // trace the PAYLOAD_ARRAY_SIZE any hits
        traceRayEXT(topLevelAS,                 // acceleration structure
                    rayFlags,                   // rayFlags
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

        const int firstId = readId(0);

        // no more hits found
        if(firstId == PAYLOAD_INVALID_ID)
        {
          break;
        }

        // evaluate the sorted hits
        [[unroll]] for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
        {
          const int   splatId = readId(i);
          const float dist    = readDist(i);

          if((splatId != PAYLOAD_INVALID_ID) && (maxComponent(transmittance) > frameInfo.minTransmittance))
          {
#if WIREFRAME
            // we render wireframe only for primary rays.
            if(bounce == 0)
            {
              const vec3  bary      = readBary3(i);
              const float threshold = 0.02;
              if(bary.x < threshold || bary.y < threshold || bary.z < threshold)
              {
                radiance      = vec3(1.0, 0.0, 0.0);  // wireframe color
                transmittance = vec3(0.0);            // opaque
                break;
              }
            }
#endif  // end of display wireframe

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
    if((meshHitDist < PAYLOAD_INF_DISTANCE) && (maxComponent(transmittance) > frameInfo.minTransmittance))
    {
      // retrieve the material for this hit
      ObjDesc     objResource     = objDesc.i[meshHitObjId];
      Materials   materials       = Materials(objResource.materialAddress);
      ObjMaterial meshHitMaterial = materials.m[meshHitMatId];

      wavefrontComputeShading(lights.l, frameInfo.lightCount, meshHitWorldPos, meshHitWorldNrm, meshHitMaterial,
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
  if(closestParticleFound && gl_LaunchIDEXT.xy == frameInfo.cursor)
  {
    indirect.particleID   = closestParticleId;
    indirect.particleDist = closestParticleDist;
  }

  //
  vec3 fragRadiance = radiance;

  uint64_t clock = clockARB() - clockStart;

#if VISUALIZE == VISUALIZE_CLOCK
  fragRadiance =
      hsbToRgb(vec3(mix(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * 0.1 * float(clock) / float(1 << 20))), 1.0, 1.));
#endif

#if VISUALIZE == VISUALIZE_DEPTH
  if(closestParticleFound)
  {
    fragRadiance = hsbToRgb(
        vec3(mix(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * float(closestParticleDist) / 256.0F)), 1.0, 1.));
  }
#endif

#if VISUALIZE == VISUALIZE_RAYHITS
  fragRadiance = vec3(frameInfo.multiplier * 10 * float(rayHitsCount) / 255, 0.0, 0.0);
#endif

#if VISUALIZE == VISUALIZE_FINAL
#if HYBRID_ENABLED
  if(meshHit)
  {
    if(frameInfo.frameSampleId <= 0)  // < 0 means temporal sampling off, 0 means first frame
      imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(fragRadiance, 1.0));
    else
      imageStore(imageAux, ivec2(gl_LaunchIDEXT.xy), vec4(fragRadiance, 1.0));
  }
  return;
#else
  if(frameInfo.frameSampleId > 0)
  {
    // Do accumulation over time
    const float a        = 1.0F / float(frameInfo.frameSampleId + 1);
    const vec4  oldColor = imageLoad(image, ivec2(gl_LaunchIDEXT.xy));
    fragRadiance         = mix(oldColor, vec4(fragRadiance, 1.0), a).xyz;
  }
#endif
#endif

  imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(fragRadiance, 1.0));
}
