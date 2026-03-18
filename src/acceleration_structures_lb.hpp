/*
 * Copyright (c) 2014-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <string>
#include <sstream>
#include <queue>
#include <vector>

#include <glm/glm.hpp>
#include <volk.h>

#include "nvvk/acceleration_structures.hpp"

namespace vk_gaussian_splatting {


// This is a verison of nvvk::AccelerationStructureHelper that supports large buffers
//
// Helper class for building both Bottom-Level Acceleration Structures (BLAS) and
// Top-Level Acceleration Structures (TLAS). This utility
// abstracts the complexity of acceleration structure generation while allowing
// compacting, updating, and managing buffers.
// For more advanced control one shall use nvvk::BlasBuilder and
// nvvk::AccelerationStructureBuildData.
class AccelerationStructureHelperLB
{
public:
  ~AccelerationStructureHelperLB() { assert(!m_transientPool && "deinit missing"); }

  void init(nvvk::ResourceAllocator* alloc,
            nvvk::StagingUploader*   uploader,
            nvvk::QueueInfo          queueInfo,
            VkDeviceSize             hintMaxAccelerationStructureSize = 512'000'000,
            VkDeviceSize             hintMaxScratchStructureSize      = 128'000'000);

  // Destroys all BLAS and TLAS resources, buffers, and clears internal state.
  // Must be called before rebuilding acceleration structures to avoid memory
  // leaks or double allocations.
  void deinit(void);

  // free both the TLAS and the BLAS set that were created using build methods,
  // some new builds and wait can be invoked afterward
  void deinitAccelerationStructures(void);

public:
  // BLAS related

  std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;
  std::vector<nvvk::AccelerationStructure>          blasSet{};  // Bottom-level AS set
  nvvk::AccelerationStructureBuilder::Stats         blasBuildStatistics;
  nvvk::LargeBuffer                                 blasScratchBuffer{};  // Can be >4GB for large BLAS builds

  // Builds a set of Bottom Level Acceleration Structures(BLAS) from a
  // Vector of geometry descriptors used for each BLAS
  // Same buildFlags will apply to each BLAS generated from asGeoInfoSet.
  // Returns VK_SUCCESS on success, or error code on failure (e.g., VK_ERROR_OUT_OF_DEVICE_MEMORY)
  VkResult blasSubmitBuildAndWait(const std::vector<nvvk::AccelerationStructureGeometryInfo>& asGeoInfoSet,
                                  VkBuildAccelerationStructureFlagsKHR                        buildFlags);

public:
  // TLAS related

  nvvk::AccelerationStructureBuildData tlasBuildData{};
  nvvk::AccelerationStructure          tlas{};                 // Top-level AS
  nvvk::LargeBuffer                    tlasInstancesBuffer{};  // Can be >4GB for large scenes
  nvvk::LargeBuffer                    tlasScratchBuffer{};    // Can be >4GB for large TLAS builds
  size_t                               tlasSize{0};

  // Builds the Top-Level Acceleration Structure (TLAS) from a list of instances
  // add VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR to buildFlags if you intend to use tlasUpdate
  // Returns VK_SUCCESS on success, or error code on failure (e.g., VK_ERROR_OUT_OF_DEVICE_MEMORY)
  VkResult tlasSubmitBuildAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances,
                                  VkBuildAccelerationStructureFlagsKHR                   buildFlags);

  // Updates an existing TLAS with an updated list of instances.
  // If instance count differs from original, a rebuild is performed instead of an update.
  // TLAS must have been built with the VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR flag.
  // Returns VK_SUCCESS on success, or error code on failure
  VkResult tlasSubmitUpdateAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances);

private:
  nvvk::QueueInfo                                    m_queueInfo;
  nvvk::ResourceAllocator*                           m_alloc{nullptr};
  nvvk::StagingUploader*                             m_uploader{};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelStructProps{};
  VkDeviceSize                                       m_blasAccelerationStructureBudget{};
  VkDeviceSize                                       m_blasScratchBudget{};
  VkCommandPool                                      m_transientPool{};
};

}  // namespace vk_gaussian_splatting
