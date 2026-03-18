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

#pragma once

#include "light_manager_vk.h"
#include "mesh_manager_vk.h"
#include "splat_set_manager_vk.h"
#include "camera_set.h"

namespace vk_gaussian_splatting {

/**
 * AssetManagerVk - Centralized manager for all scene assets
 * 
 * Provides a single point of access to all asset types:
 * - Lights
 * - Meshes
 * - Splat Sets
 * - Cameras
 * - SceneAssets uniform buffer (bindless asset structure)
 * 
 * This improves encapsulation and makes the GaussianSplatting class cleaner.
 */
class AssetManagerVk
{
public:
  // Public asset managers
  LightManagerVk    lights;
  MeshManagerVk     meshes;
  SplatSetManagerVk splatSets;
  CameraSet         cameras;

  // Assets buffer (GPU uniform buffer containing SceneAssets structure)
  nvvk::Buffer assetsBuffer{};

  // Assets data (RAM copy of SceneAssets structure)
  shaderio::SceneAssets sceneAssets{};

  AssetManagerVk()  = default;
  ~AssetManagerVk() = default;

  // Initialize all asset managers and create assets buffer
  void init(nvapp::Application*                                 app,
            nvvk::ResourceAllocator*                            alloc,
            nvvk::StagingUploader*                              uploader,
            nvutils::CameraManipulator*                         cameraManip,
            VkSampler*                                          sampler,
            nvvk::PhysicalDeviceInfo*                           deviceInfo,
            VkPhysicalDeviceAccelerationStructurePropertiesKHR* accelStructProps,
            nvutils::ProfilerTimeline*                          profilerTimeline);

  // Deinitialize all asset managers and destroy assets buffer
  void deinit();

  // Reset all asset managers (for scene reset, not app exit)
  void reset();

  // Process all deferred VRAM updates (calls each manager's processVramUpdates)
  void processVramUpdates(bool processRtx = true);

  // Centralized pending request detection (consolidates all managers)
  // @param excludeRtxRequests If true, exclude RTX-only requests (RebuildBLAS, RebuildTLAS, UpdateTransformsOnly)
  //                           Used in raster mode to avoid processing deferred RTX requests
  bool hasPendingRequests(bool excludeRtxRequests = false) const;

  // Update the SceneAssets uniform buffer contents with current asset data (GPU upload)
  // Note: Descriptor set binding is handled by GaussianSplatting (renderer concern)
  void updateAssetsBuffer();

private:
  nvapp::Application*      m_app{nullptr};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::StagingUploader*   m_uploader{nullptr};
};

}  // namespace vk_gaussian_splatting
