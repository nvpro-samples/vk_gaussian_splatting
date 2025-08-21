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

#include "shaderio.h"
#include "common.glsl"
#include "wavefront.glsl"
#include "gs_particle.glsl"

#define USED_FROM_RAY_GEN
#include "raycommon.glsl"

layout(set = 1, binding = RTX_BINDING_TLAS_SPLATS) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = RTX_BINDING_OUTIMAGE, rgba8) uniform image2D image;
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
  LightSource l[];
}
lights;

// For wireframe

// Function to transform a world-space point into screen space
vec2 worldToScreen(vec3 worldPos)
{
  vec4 clipPos = frameInfo.projectionMatrix * frameInfo.viewMatrix * vec4(worldPos, 1.0);
  vec2 ndcPos  = clipPos.xy / clipPos.w;             // Normalize by w to get NDC
  return (ndcPos * 0.5 + 0.5) * frameInfo.viewport;  // Convert to screen space
}

// Function to compute the shortest distance from a point to an edge in screen space
float edgeDistance(vec2 P, vec2 A, vec2 B)
{
  vec2  AB = B - A;
  vec2  AP = P - A;
  float t  = clamp(dot(AP, AB) / dot(AB, AB), 0.0, 1.0);
  return length(AP - t * AB);
}

// End for wireframe


bool processHit(in vec3 modelRayOrigin, in vec3 modelRayDirection, in int splatId, inout dvec3 transmittance, inout vec3 radiance, in float distance)
{
  //
  vec3  particlePosition;
  vec3  particleScale;
  mat3  particleInvRotation;
  
  fetchParticlePSR(splatId, particlePosition, particleScale, particleInvRotation);

  vec3 particleRayOrigin;
  vec3 particleRayDirection;
  particleCannonicalRay(modelRayOrigin, modelRayDirection, particlePosition, particleScale, particleInvRotation, 
                        particleRayOrigin, particleRayDirection);

  const vec4 color           = fetchColor(splatId);
  float     particleDensity = color.w;

  const double  minParticleAlpha         = 1.0f / 255.0f;
  const int32_t particleKernelDegree     = KERNEL_DEGREE;
  const float  minParticleKernelDensity = KERNEL_MIN_RESPONSE;

  const float maxResponse = particleRayMaxKernelResponse(particleRayOrigin, particleRayDirection, particleKernelDegree);
  float       alpha       = min(frameInfo.alphaClamp, maxResponse * particleDensity);

  const bool acceptHit = (particleDensity > frameInfo.alphaCullThreshold) && (alpha > minParticleAlpha)
                         && (maxResponse > minParticleKernelDensity);
  if(acceptHit)
  {
#if DISABLE_OPACITY_GAUSSIAN
    alpha = 1.0;
#endif
    // fetch radiance from SH coefs > degree 0
    const vec3 vectorToParticleCenter = normalize(particlePosition - modelRayOrigin);
#if SHOW_SH_ONLY == 1
    const vec3 grad = vec3(0.5) + fetchViewDependentRadiance(splatId, vectorToParticleCenter);
#else
    const vec3 grad = color.xyz + fetchViewDependentRadiance(splatId, vectorToParticleCenter);
#endif

    const vec3 weight = vec3(alpha * transmittance);
    radiance += grad *weight;
    transmittance *= (1.0 - alpha);
    return true;
  }
  // uncomment to debug
//#define WITH_ELSE
#ifdef WITH_ELSE
  else
  {
    radiance = vec3(0, 0, 1);
  }
#endif

  return false;
}

#if RTX_USE_MESHES
void processMeshHit(in vec3     worldPos,
                    in vec3     worldNrm,
                    in int      objectId,
                    in int      materialId,
                    in vec3     worldRayDir,
                    inout dvec3 transmittance,
                    inout vec3  radiance,
                    inout int   done,
                    inout vec3  rayOrigin,
                    inout vec3  rayDir)
{

  // retireve the material for this hit
  ObjDesc     objResource = objDesc.i[objectId];
  Materials   materials   = Materials(objResource.materialAddress);
  ObjMaterial mat         = materials.m[materialId];

  vec3  matDiffuse    = mat.diffuse;
  vec3  matAmbient    = mat.ambient;
  vec3  matSpecular   = mat.specular;
  vec3  matRefractive = mat.transmittance;
  float ior           = mat.ior;    // 1 = pure transparency, 1.5111 widow glass
  float matShininess  = mat.shininess;
  int   model         = mat.illum;

   for(int i = 0; i < frameInfo.lightCount; ++i)
  {
    const LightSource light = lights.l[i];

    // Vector toward the light
    vec3 L;

    // Light source
    const int  lightType      = light.type;
    const vec3 lightPosition  = light.position;
    float      lightIntensity = light.intensity;

    // Point light
    if(lightType == 0)
    {
      vec3  lDir     = lightPosition - worldPos;
      float d        = length(lDir);
      lightIntensity = lightIntensity / (d * d);
      L              = normalize(lDir);
    }
    else  // Directional light
    {
      L = normalize(lightPosition);
    }

    // Diffuse
    vec3 fragDiffuse = computeDiffuse(matDiffuse, matAmbient, L, worldNrm);

    // Specular
    vec3 fragSpecular = computeSpecular(matSpecular, matShininess, worldRayDir, L, worldNrm);

    // Result
    radiance += vec3(transmittance) * vec3(lightIntensity * (fragDiffuse + fragSpecular));
  }

  //
  if(model <= 0 ) // 
  {
    transmittance = dvec3(0.0);
  }
  else if(model == 1) // reflection
  {
    transmittance *= matSpecular;
    done      = 0;  // set to 0 means continue to trace at next iteration
    rayOrigin = worldPos;
    rayDir    = reflect(worldRayDir, worldNrm);
  }
  else if(model >= 2)
  {
    transmittance *= matRefractive;
    done      = 0;  // set to 0 means continue to trace at next iteration
    rayOrigin = worldPos;

    // Determine if we're entering or exiting the surface
    float eta = 1.0 / ior;
    vec3  N   = worldNrm;

    if(dot(worldRayDir, worldNrm) > 0.0)
    {
      // Ray is inside the object: flip normal and eta
      N   = -worldNrm;
      eta = ior;
    }

    vec3 refractedDir = refract(worldRayDir, N, eta);
    if(length(refractedDir) > 0.0)
    {
      // Refraction succeeded
      rayDir = normalize(refractedDir);
    }
    else
    {
      // Total internal reflection fallback
      rayDir = reflect(worldRayDir, N);
    }
  }
}
#endif

vec3 HsbToRgb(vec3 hsbColor)
{
  vec3 rgb = clamp(abs(mod((hsbColor.x * 6.0) + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  rgb      = (rgb * rgb * (3.0 - (2.0 * rgb)));
  return (hsbColor.z * mix(vec3(1.0), rgb, hsbColor.y));
}

double maxComponent(in dvec3 v)
{
  return max(v.x, max(v.y, v.z));
}

uint xxhash32(uvec3 p)
{
  const uvec4 primes = uvec4(2246822519U, 3266489917U, 668265263U, 374761393U);
  uint        h32;
  h32 = p.z + primes.w + p.x * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 += p.y * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 = primes.x * (h32 ^ (h32 >> 15));
  h32 = primes.y * (h32 ^ (h32 >> 13));
  return h32 ^ (h32 >> 16);
}

uint pcg(inout uint state)
{
  uint prev = state * 747796405u + 2891336453u;
  uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
  state     = prev;
  return (word >> 22u) ^ word;
}

float rand(inout uint seed)
{
  uint r = pcg(seed);
  return float(r) * (1.F / float(0xffffffffu));
}

void main()
{
  const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
  const vec2 inUV        = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
  vec2       d           = inUV * 2.0 - 1.0;

  vec4 origin    = frameInfo.viewInverse * vec4(0, 0, 0, 1);
  vec4 target    = frameInfo.projInverse * vec4(d.x, d.y, 1, 1);
  vec4 direction = normalize(frameInfo.viewInverse * vec4(target.xyz, 0));

  // will use Any hit only
  uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT | gl_RayFlagsNoOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  // will use Closest hit only
  uint rayFlagsMesh = gl_RayFlagsCullBackFacingTrianglesEXT;

  float tMax = 10000.0;
  const float epsT = 1e-9;

  vec3 rayOrigin    = origin.xyz;
  vec3 rayDirection = direction.xyz;

  // Depth-of-Field
#if !HYBRID_ENABLED && RTX_DOF_ENABLED
#define TWO_PI 6.28318530718
  
  // Initialize the random number
  uint seed = xxhash32(uvec3(gl_LaunchIDEXT.xy, frameInfo.frameSampleId));

  vec3  focalPoint        = rayDirection * frameInfo.focalDist;
  float cam_r1            = rand(seed) * TWO_PI;
  float cam_r2            = rand(seed) * frameInfo.aperture;
  vec4  cam_right         = frameInfo.viewInverse * vec4(1, 0, 0, 0);
  vec4  cam_up            = frameInfo.viewInverse * vec4(0, 1, 0, 0);
  vec3  randomAperturePos = (cos(cam_r1) * cam_right.xyz + sin(cam_r1) * cam_up.xyz) * sqrt(cam_r2);
  vec3  finalRayDir       = normalize(focalPoint - randomAperturePos);

  // Set the new ray origin and direction with depth-of-field
  rayOrigin += randomAperturePos;
  rayDirection = finalRayDir;
#endif

  int rayHitsCount = 0;

  vec3  radiance      = vec3(0.0f);
  dvec3 transmittance = vec3(1.0f);

  int bounce = 0;
  int done   = 1;

  uint64_t clockStart = clockARB();

  // for debug feedback and alternative visu modes
  bool  closestParticleFound = false;
  float closestParticleDist  = PAYLOAD_INF_DISTANCE;
  int   closestParticleId    = PAYLOAD_INVALID_ID;

#if HYBRID_ENABLED
  vec4 pixel = imageLoad(image, ivec2(gl_LaunchIDEXT.xy));
  radiance   = vec3(pixel);
  transmittance = vec3(1.0-pixel.w);
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
    int meshHitObjId;
    int meshHitMatId;

    writeDist(0, PAYLOAD_INF_DISTANCE);
    writeId(0, PAYLOAD_INVALID_ID);
    writeId(1, PAYLOAD_INVALID_ID);

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

    meshHitDist = readDist(0);
    if(meshHitDist < PAYLOAD_INF_DISTANCE)
    {
      tMax = meshHitDist;
      // collect info to compute shading and potential new ray later on
      meshHitObjId    = readId(0);
      meshHitMatId    = readId(1);
      meshHitWorldPos = vec3(readDist(1), readDist(2), readDist(3));
      meshHitWorldNrm = vec3(readDist(4), readDist(5), readDist(6));
    }
#endif

#if HYBRID_ENABLED
    if(bounce > 0)
    {
#endif
    // trace the gaussians with multi-pass any hit
    int outerIdx = 0;

    // The two following are to compute processHit with transformed splat set model
    const vec3 splatSetModelRayOrigin = vec3(pcRay.modelMatrixInverse * vec4(rayOrigin, 1.0));
    // modelMatrixTranspose is equivalent to inverse(transpose(modelMatrixInverse))
    const vec3 splatSetModelRayDirection = normalize(vec3(pcRay.modelMatrixTranspose * vec4(rayDirection, 1.0)));

    while(outerIdx < frameInfo.maxPasses && (rayLastHitDistance <= tMax) && (maxComponent(transmittance) > frameInfo.minTransmittance))
    {
      // prepare the payload
      [[unroll]] for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
      {
        writeId(i, PAYLOAD_INVALID_ID);
        writeDist(i, PAYLOAD_INF_DISTANCE);
      }

//#define USE_SER
#ifdef USE_SER
      hitObjectNV hObj;
      hitObjectRecordEmptyNV(hObj);
      hitObjectTraceRayNV(hObj, topLevelAS, rayFlags, 0xFF, 0, 0, 0, rayOrigin, rayLastHitDistance + epsT,
                          rayDirection, tMax + epsT, 0);
      reorderThreadNV(hObj);
      hitObjectExecuteShaderNV(hObj, 0);
#else
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
#endif
      const int firstId = readId(0);

      // no more hits found
      if(firstId == PAYLOAD_INVALID_ID)
      {
        break;
      }

      // evaluate the sorted hits
      [[unroll]] for(int i = 0; i < PAYLOAD_ARRAY_SIZE; ++i)
      {
        const int splatId = readId(i);
        const float dist = readDist(i);

        if((splatId != PAYLOAD_INVALID_ID) && (maxComponent(transmittance) > frameInfo.minTransmittance))
        {
#if WIREFRAME
          {
            // Compute the hit position in world space using barycentric coordinates
            const vec2  bary = readBary(i);
            const float u    = bary.x;
            const float v    = bary.y;
            const float w    = 1.0 - u - v;

            // Define wireframe thickness threshold
            float threshold = 0.02;
            if(u < threshold || v < threshold || w < threshold)
            {
              radiance      = vec3(1.0, 0.0, 0.0);  // wireframe color
              transmittance = vec3(0.0);            // opaque
              break;
            }
          }

#endif  // end of display wireframe

          bool acceptedHit = processHit(splatSetModelRayOrigin, splatSetModelRayDirection, splatId, transmittance, radiance, dist);
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
      processMeshHit(meshHitWorldPos, meshHitWorldNrm, meshHitObjId, meshHitMatId, rayDirection,
                     transmittance, radiance, done, rayOrigin, rayDirection);
    }
#endif

    // continue with secondary rays if needed (decided by processMeshHit)
    bounce++;
    if(done == 1 || bounce > frameInfo.rtxMaxBounces)
      break;

    // prepare ray for next iteration
    // new rayOrigin and direction was set by processMeshHit
    // Next iteration will stop if a reflective material isn't hit
    done = 1;

  }  // while(true)

  // debug feedback
  if(closestParticleFound && gl_LaunchIDEXT.xy == frameInfo.cursor)
  {
    indirect.particleID = closestParticleId;
    indirect.particleDist = closestParticleDist;
  }

  //
  vec3 fragRadiance = radiance;

  uint64_t clock = clockARB() - clockStart;

#if VISUALIZE == VISUALIZE_CLOCK
  fragRadiance = HsbToRgb(vec3(mix(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * 0.1 * float(clock) / float(1 << 20))), 1.0, 1.));
#endif

#if VISUALIZE == VISUALIZE_DEPTH
  if(closestParticleFound)
  {
    fragRadiance = HsbToRgb(
        vec3(mix(0.65, 0.02, smoothstep(0.0, 1.0, frameInfo.multiplier * float(closestParticleDist) / 256.0F)), 1.0, 1.));
  }
#endif

#if VISUALIZE == VISUALIZE_RAYHITS
  fragRadiance = vec3(frameInfo.multiplier * 10 * float(rayHitsCount) / 255, 0.0, 0.0);
#endif

#if VISUALIZE == VISUALIZE_FINAL && !HYBRID_ENABLED
  if(frameInfo.frameSampleId > 0)
  {
    // Do accumulation over time
    const float a        = 1.0F / float(frameInfo.frameSampleId + 1);
    const vec4  oldColor = imageLoad(image, ivec2(gl_LaunchIDEXT.xy));
    fragRadiance         = mix(oldColor, vec4(fragRadiance, 1.0), a).xyz;
  }
#endif

  imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(fragRadiance, 1.0));
}
