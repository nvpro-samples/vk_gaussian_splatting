/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asset_manager_vk.h"
#include <nvvk/check_error.hpp>
#include <iostream>

namespace vk_gaussian_splatting {

void AssetManagerVk::init(nvapp::Application*                                 app,
                          nvvk::ResourceAllocator*                            alloc,
                          nvvk::StagingUploader*                              uploader,
                          nvutils::CameraManipulator*                         cameraManip,
                          VkSampler*                                          sampler,
                          nvvk::PhysicalDeviceInfo*                           deviceInfo,
                          VkPhysicalDeviceAccelerationStructurePropertiesKHR* accelStructProps,
                          nvutils::ProfilerTimeline*                          profilerTimeline)
{
  // Store references
  m_app      = app;
  m_alloc    = alloc;
  m_uploader = uploader;

  // Initialize all asset managers
  cameras.init(cameraManip);
  lights.init(app, alloc, uploader);
  meshes.init(app, alloc, uploader, accelStructProps);
  splatSets.init(app, alloc, uploader, sampler, deviceInfo, accelStructProps, profilerTimeline);

  // Set cross-references
  lights.setMeshSet(&meshes);

  // Create initial assets buffer (empty placeholder)
  sceneAssets = {};
  NVVK_CHECK(m_alloc->createBuffer(assetsBuffer, sizeof(shaderio::SceneAssets), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
  NVVK_DBG_NAME(assetsBuffer.buffer);

  LOGI("AssetManagerVk::init: Created assets buffer (handle=%p, size=%zu bytes)\n", (void*)assetsBuffer.buffer,
       sizeof(shaderio::SceneAssets));
}

void AssetManagerVk::deinit()
{
  // CRITICAL: Process any remaining VRAM updates (deletions) before deinitializing.
  // splatSets.deinit() internally calls reset() which only MARKS assets for deferred
  // deletion. Without calling processVramUpdates, splat set data buffers (centers, scales,
  // rotations, colors, SH) would leak since processVramUpdates won't run after deinit.
  splatSets.reset();
  processVramUpdates(false);

  // Destroy assets buffer
  if(assetsBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(assetsBuffer);
    assetsBuffer = {};
  }

  // Deinitialize in reverse order of dependencies
  splatSets.deinit();
  lights.deinit();  // before meshes, important
  meshes.deinit();
  cameras.deinit();
}

void AssetManagerVk::reset()
{
  // vkDeviceWaitIdle is called by the parent GaussianSplatting::reset() before calling this

  // Reset all asset managers
  cameras.reset();
  lights.reset();  // before meshes, important
  meshes.reset();
  splatSets.reset();

  // CRITICAL: Process VRAM updates immediately to ensure reset completes before any new load
  // This prevents race condition where a load happens before reset's cleanup is processed
  // This matches the DELETE flow: mark dirty → processVramUpdates → clean state
  // NOTE: We pass processRtx=false since RTX structures were already destroyed in reset()
  processVramUpdates(false);

  std::cout << "AssetManagerVk::reset: Complete (VRAM updates processed)" << std::endl;
}

void AssetManagerVk::processVramUpdates(bool processRtx)
{
  // Process each manager's deferred updates
  // Note: Order matters - lights may create/update mesh instances (proxies)

  if(static_cast<uint32_t>(lights.pendingRequests))
  {
    lights.processVramUpdates();
  }

  if(static_cast<uint32_t>(meshes.pendingRequests))
  {
    meshes.processVramUpdates(processRtx);
  }

  if(static_cast<uint32_t>(splatSets.pendingRequests))
  {
    splatSets.processVramUpdates(processRtx);
  }

  // Update assets buffer with current descriptor addresses
  // This must happen after all manager updates to ensure pointers are current
  updateAssetsBuffer();
}

bool AssetManagerVk::hasPendingRequests(bool excludeRtxRequests) const
{
  // Consolidate pending requests from all managers
  uint32_t lightsRequests    = static_cast<uint32_t>(lights.pendingRequests);
  uint32_t meshesRequests    = static_cast<uint32_t>(meshes.pendingRequests);
  uint32_t splatSetsRequests = static_cast<uint32_t>(splatSets.pendingRequests);

  if(excludeRtxRequests)
  {
    // In raster mode, exclude ONLY pure RTX requests (BLAS/TLAS)
    // UpdateTransformsOnly is NOT excluded because it triggers descriptor buffer updates
    // which are needed for raster shaders (MeshDesc contains transform matrices)
    constexpr uint32_t RTX_REQUESTS_MASK_MESHES = static_cast<uint32_t>(MeshManagerVk::Request::eRebuildTLAS)
                                                  | static_cast<uint32_t>(MeshManagerVk::Request::eRebuildBLAS);

    constexpr uint32_t RTX_REQUESTS_MASK_SPLATS = static_cast<uint32_t>(SplatSetManagerVk::Request::eRebuildTLAS)
                                                  | static_cast<uint32_t>(SplatSetManagerVk::Request::eRebuildBLAS);

    meshesRequests &= ~RTX_REQUESTS_MASK_MESHES;
    splatSetsRequests &= ~RTX_REQUESTS_MASK_SPLATS;
  }

  return lightsRequests || meshesRequests || splatSetsRequests;
}

void AssetManagerVk::updateAssetsBuffer()
{
  // Reset and populate the RAM structure with current asset data
  sceneAssets = {};

  // Mesh descriptors (includes both regular objects and light proxies, differentiated by MeshType)
  {
    sceneAssets.meshes          = reinterpret_cast<shaderio::MeshDesc*>(meshes.objectDescriptionsBuffer.address);
    sceneAssets.meshCount       = static_cast<uint32_t>(meshes.instances.size());
    sceneAssets.meshTlasAddress = meshes.rtAccelerationStructures.tlas.address;
  }

  // Light sources
  {
    sceneAssets.lights     = reinterpret_cast<shaderio::LightSource*>(lights.lightsBuffer.address);
    sceneAssets.lightCount = static_cast<uint32_t>(lights.size());
  }

  // Gaussian splat set instances (from manager)
  {
    // Per-instance descriptors are used by raster/hybrid and global index table.
    sceneAssets.splatSetDescriptors = reinterpret_cast<shaderio::SplatSetDesc*>(splatSets.getGPUDescriptorArrayAddress());
    // Per-TLAS-instance descriptors are used by RTX when BLAS is split (optional).
    sceneAssets.splatSetDescriptorsRtx = reinterpret_cast<shaderio::SplatSetDesc*>(splatSets.getRtxDescriptorArrayAddress());
    sceneAssets.splatSetCount = static_cast<uint32_t>(splatSets.getInstanceCount());

    // Multi-TLAS array (for large scenes exceeding maxInstanceCount)
    const auto& tlasArray                 = splatSets.getTlasArray();
    sceneAssets.splatSetTlasArrayAddress  = splatSets.getTlasAddress();          // Address of TLAS address buffer
    sceneAssets.splatSetTlasOffsetAddress = tlasArray.tlasOffsetBuffer.address;  // Address of TLAS offset buffer
    sceneAssets.splatSetTlasCount         = tlasArray.tlasCount;

    // Global indexing for unified sorting
    sceneAssets.globalIndexTableAddress         = splatSets.getGlobalIndexTableAddress();
    sceneAssets.splatSetGlobalIndexTableAddress = splatSets.getSplatSetGlobalIndexTableAddress();
    sceneAssets.totalGlobalSplatCount           = splatSets.getTotalGlobalSplatCount();

    // Sorting buffer addresses (bindless access, from SplatSetManagerVk)
    sceneAssets.splatSortingDistancesAddress = reinterpret_cast<uint32_t*>(splatSets.getSplatSortingDistancesAddress());
    sceneAssets.splatSortingIndicesAddress   = reinterpret_cast<uint32_t*>(splatSets.getSplatSortingIndicesAddress());
  }

  // Upload updated data to existing buffer (buffer was created in init())
  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_uploader->appendBuffer(assetsBuffer, 0, std::span(&sceneAssets, 1)));
  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();
}

}  // namespace vk_gaussian_splatting
