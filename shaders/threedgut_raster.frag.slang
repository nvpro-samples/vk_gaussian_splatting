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

#include "nvshaders/random.h.slang"
#include "nvshaders/fwidth.h.slang"
#include "nvshaders/wireframe.h.slang"

#include "shaderio.h"
#include "cameras.h.slang"
#include "threedgrt.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(BINDING_ASSETS, 0)]] ConstantBuffer<SceneAssets> assets;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

#if FRONT_TO_BACK && NEED_SURFACE_INFO
// For FTB: depth buffer accessed as storage image (R=depth, G=transmittance)
// globallycoherent ensures memory coherency between interlock-protected fragment invocations
[[vk::binding(BINDING_RASTER_DEPTH, 0)]] globallycoherent RWTexture2D<float2> depthTransmittanceBuffer;
#endif

struct FragmentInput
{
  [[vk::location(0)]] nointerpolation uint splatId : SPLATID;
  [[vk::location(1)]] nointerpolation float4 splatCol : COLOR0;
  [[vk::location(2)]] nointerpolation float3 splatPosition : POSITION0;
  [[vk::location(3)]] nointerpolation float3 splatScale : SCALE0;
  [[vk::location(4)]] nointerpolation float3 splatInvRotation0 : ROTATION0;  // matrix row 0
  [[vk::location(5)]] nointerpolation float3 splatInvRotation1 : ROTATION1;  // matrix row 1
  [[vk::location(6)]] nointerpolation float3 splatInvRotation2 : ROTATION2;  // matrix row 2
  float4 position : SV_Position;
#if WIREFRAME
  float3 barycentrics : SV_Barycentrics;
#endif
#if STOCHASTIC_SPLAT
  uint primitiveId : SV_PrimitiveID;
#endif
};

struct FragmentOutput
{
  [[vk::location(0)]] float4 color : SV_Target0;
#if NEED_SURFACE_INFO
  [[vk::location(1)]] float4 outNormal : SV_Target1;  // Normals (integrated mode) or zeros (reconstructed mode)
#if !FRONT_TO_BACK
  [[vk::location(2)]] float4 outDepth : SV_Target2;   // Integrated depth (R=depth, A=opacity for blending) - BTF only
  [[vk::location(3)]] uint   outSplatId : SV_Target3; // Global splat ID for material lookup
#else
  [[vk::location(2)]] uint   outSplatId : SV_Target2; // Global splat ID for material lookup (shifted for FTB)
  // Note: In FTB mode, picked depth is stored in storage image (depthTransmittanceBuffer.r)
#endif
#endif
};
// clang-format on

[shader("fragment")]
FragmentOutput main(FragmentInput input)
{
  FragmentOutput output;

  // use this to visualize full splat rectangular extent.
#if 0
  {
    float4 color = input.splatCol;
    output.color = float4(color.rgb, 1.0);
    return output;
  }
#endif

  // prepare a ray the evaluation of the particle response
  float3 rayOrigin;
  float3 rayDirection;

  // generate the ray for this pixel
#if CAMERA_TYPE == CAMERA_PINHOLE
  generatePinholeRay(input.position.xy, float2(0.5), frameInfo.viewport, frameInfo.viewInverse, frameInfo.projInverse,
                     rayOrigin, rayDirection);
#else
  if(!generateFisheyeRay(input.position.xy, frameInfo.viewport, frameInfo.fovRad, float2(0.0), frameInfo.viewInverse,
                         rayOrigin, rayDirection))
  {
    discard;
  }
#endif

  // add Depth-of-Field perturbation to the ray
#if DOF_MODE != DOF_DISABLED
  // Initialize the random number
  uint seed = xxhash32(uint3(int(input.position.x), int(input.position.y), frameInfo.frameSampleId));

  depthOfField(seed, frameInfo.focusDist, frameInfo.aperture, frameInfo.viewInverse, rayOrigin, rayDirection);
#endif

  // Resolve global splat ID to correct descriptor and local index
  SplatSetDesc splatSetDesc;
  uint         splatIndex;
  resolveGlobalSplatID(assets, input.splatId, splatSetDesc, splatIndex);

  // The two following transformations are to compute processHit with transformed splat set model
  const float3 splatSetModelRayOrigin = mul(float4(rayOrigin, 1.0), splatSetDesc.transformInverse).xyz;
  // Since the ray direction should not be affected by translation,
  // uses the inverse of the rotation-scale part of the model matrix (upper-left 3x3 of transformInverse)
  const float3 splatSetModelRayDirection = normalize(mul(rayDirection, float3x3(splatSetDesc.transformInverse)));

  // Reconstruct the inverse rotation matrix from the three row vectors
  const float3x3 splatInvRotation = float3x3(input.splatInvRotation0, input.splatInvRotation1, input.splatInvRotation2);

  float  opacity;
  float3 normalWorld = float3(0.0);
  bool   acceptedHit;

#if NEED_SURFACE_INFO
  // Use the version that computes normals for surface generation
  acceptedHit = particleProcessHitGutWithNormal(frameInfo, splatSetModelRayOrigin, splatSetModelRayDirection, int(splatIndex),
                                                input.splatCol, input.splatPosition, input.splatScale, splatInvRotation,
                                                splatSetDesc.transform, opacity, normalWorld);
#else
  // Use resolved local splatIndex for hit processing
  acceptedHit = particleProcessHitGut(frameInfo, splatSetModelRayOrigin, splatSetModelRayDirection, int(splatIndex),
                                      input.splatCol, input.splatPosition, input.splatScale, splatInvRotation, opacity);
#endif

  if(!acceptedHit)
  {
#if !WIREFRAME
    discard;
#else
    output.color = float4(0.0, 0.0, 0.0, 0.0);
#endif
  }
  else
  {
#if STOCHASTIC_SPLAT
    // Stochastic splat is incompatible with FTB (no transmittance accumulation needed)
    // Stochastic rendering: binary accept/reject based on opacity
    uint seed       = xxhash32(uint3(uint(input.position.x), uint(input.position.y), frameInfo.frameSampleId));
    seed            = xxhash32(uint3(seed, input.splatId, input.primitiveId));
    float randomVal = rand(seed);

    if(randomVal < opacity)
    {
      // ACCEPTED: Output full color (opaque)
      output.color = float4(input.splatCol.rgb, 1.0);
#if NEED_SURFACE_INFO
      // For stochastic mode, write surface outputs with full opacity (no blending needed)
      output.outDepth   = float4(input.position.z, 0, 0, 1.0);
      output.outNormal  = float4(normalWorld, 1.0);
      output.outSplatId = input.splatId;
#endif
      return output;  // Early return - no wireframe support in stochastic mode
    }
    else
    {
      // REJECTED: Discard fragment
      discard;
    }
#else
    // Standard alpha blending mode
#if FRONT_TO_BACK
    // Front-to-back "under" compositing requires premultiplied alpha
    // Output: float4(color.rgb * opacity, opacity)
    output.color = float4(input.splatCol.rgb * opacity, opacity);
#else
    // Back-to-front "over" compositing (standard alpha blending)
    output.color = float4(input.splatCol.rgb, opacity);
#endif
#endif
  }

#if WIREFRAME
  const float wireframeAlpha = processWireframe(wireframeDefaultSettings(), input.barycentrics, fwidth(input.barycentrics));
  output.color = lerp(output.color, float4(1.0, 0.0, 0.0, 1.0), wireframeAlpha);
#endif

  // Surface generation outputs (normal, depth, splat ID) - FTB mode
  // Note: BTF with surface reconstruction only happens in stochastic mode which has early return
#if NEED_SURFACE_INFO
  // Output normal weighted by opacity for alpha-blending integration
  output.outNormal = float4(normalWorld * opacity, opacity);

#if FRONT_TO_BACK
  // FTB: Output depth for depth picking (first significant splat) using storage image
  // input.position.z is the normalized device depth [0,1]
  int2 pixelCoord = int2(input.position.xy);
  
#if FTB_SYNC_MODE == FTB_SYNC_INTERLOCK
  beginInvocationInterlock();
#endif
  float2 current           = depthTransmittanceBuffer[pixelCoord];
  float  depth             = current.r;
  float  prevTransmittance = current.g;
  float  transmittance     = prevTransmittance * (1.0 - opacity);
  if(depth == 0 && transmittance < frameInfo.depthIsoThreshold)
  {
    depth = input.position.z;
  }
  // Write back: depth in R, transmittance in G
  depthTransmittanceBuffer[pixelCoord] = float2(depth, transmittance);
#if FTB_SYNC_MODE == FTB_SYNC_INTERLOCK
  endInvocationInterlock();
#endif
#endif // FRONT_TO_BACK

  // Output global splat ID for material lookup in hybrid ray tracing mode
  output.outSplatId = input.splatId;
#endif

  return output;
}
