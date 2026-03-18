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


#include <assert.h>

#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <fmt/format.h>

#include "nvvk/debug_util.hpp"
#include "nvvk/commands.hpp"

#include "acceleration_structures_lb.hpp"

using namespace vk_gaussian_splatting;
using namespace nvvk;

//////////////////////////////////////////////////////////////////////////

// Returns the maximum scratch buffer size needed for building all provided acceleration structures.
// This function iterates through a vector of AccelerationStructureBuildData, comparing the scratch
// size required for each structure and returns the largest value found.
//
// Returns:
//   The maximum scratch size needed as a VkDeviceSize.
VkDeviceSize getMaxScratchSize(const std::vector<AccelerationStructureBuildData>& asBuildData)
{
  VkDeviceSize maxScratchSize = 0;
  for(const auto& blas : asBuildData)
  {
    maxScratchSize = std::max(maxScratchSize, blas.sizeInfo.buildScratchSize);
  }
  return maxScratchSize;
}


void AccelerationStructureHelperLB::init(ResourceAllocator* alloc,
                                         StagingUploader*   uploader,
                                         QueueInfo          queueInfo,
                                         VkDeviceSize       hintMaxAccelerationStructureSize /*= 512'000'000*/,
                                         VkDeviceSize       hintMaxScratchStructureSize /*= 128'000'000*/)
{
  assert(!m_transientPool && "init() called multiple times");

  m_queueInfo                       = queueInfo;
  m_alloc                           = alloc;
  m_uploader                        = uploader;
  m_blasAccelerationStructureBudget = hintMaxAccelerationStructureSize;
  m_blasScratchBudget               = hintMaxScratchStructureSize;
  m_transientPool                   = createTransientCommandPool(alloc->getDevice(), queueInfo.familyIndex);

  m_accelStructProps                 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext                       = &m_accelStructProps;
  vkGetPhysicalDeviceProperties2(alloc->getPhysicalDevice(), &props2);
}

void AccelerationStructureHelperLB::deinit(void)
{
  if(m_transientPool)
    vkDestroyCommandPool(m_alloc->getDevice(), m_transientPool, nullptr);
  m_transientPool    = nullptr;
  m_alloc            = nullptr;
  m_queueInfo        = {};
  m_uploader         = nullptr;
  m_accelStructProps = {};
}

void AccelerationStructureHelperLB::deinitAccelerationStructures(void)
{
  // BLAS related
  for(auto& b : blasSet)
  {
    if(b.accel)
      m_alloc->destroyAcceleration(b);
  }
  blasSet.clear();
  blasBuildData.clear();
  blasBuildStatistics = {};
  if(blasScratchBuffer.buffer)
    m_alloc->destroyLargeBuffer(blasScratchBuffer);
  blasScratchBuffer = {};

  // TLAS related
  if(tlas.accel)
    m_alloc->destroyAcceleration(tlas);
  if(tlasInstancesBuffer.buffer)
    m_alloc->destroyLargeBuffer(tlasInstancesBuffer);
  if(tlasScratchBuffer.buffer)
    m_alloc->destroyLargeBuffer(tlasScratchBuffer);

  tlas                = {};
  tlasInstancesBuffer = {};
  tlasScratchBuffer   = {};
  tlasBuildData       = {};
  tlasSize            = {};
}

VkResult AccelerationStructureHelperLB::blasSubmitBuildAndWait(const std::vector<AccelerationStructureGeometryInfo>& asGeoInfoSet,
                                                               VkBuildAccelerationStructureFlagsKHR buildFlags)
{
  VkDevice device = m_alloc->getDevice();

  assert(blasSet.empty() && "we must not invoke build if already built. use deinit before.");

  // Prepare the BLAS build data
  blasBuildData.reserve(asGeoInfoSet.size());

  for(const auto& asGeoInfo : asGeoInfoSet)
  {
    AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};

    buildData.addGeometry(asGeoInfo);

    buildData.finalizeGeometry(m_alloc->getDevice(), buildFlags);

    blasBuildData.emplace_back(buildData);
  }

  // build the set of BLAS
  blasSet.resize(blasBuildData.size());

  // Find the most optimal size for our scratch buffer, and get the addresses of the scratch buffers
  // to allow a maximum of BLAS to be built in parallel, within the budget
  AccelerationStructureBuilder blasBuilder;
  blasBuilder.init(m_alloc);
  VkDeviceSize hintScratchBudget = m_blasScratchBudget;  // Limiting the size of the scratch buffer to 2MB
  VkDeviceSize scratchSize       = blasBuilder.getScratchSize(hintScratchBudget, blasBuildData);

  VkResult result = m_alloc->createLargeBuffer(blasScratchBuffer, scratchSize,
                                               VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                   | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                               m_queueInfo.queue, VK_NULL_HANDLE,
                                               ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE, VMA_MEMORY_USAGE_AUTO, {},
                                               m_accelStructProps.minAccelerationStructureScratchOffsetAlignment);

  if(result != VK_SUCCESS)
  {
    // Cleanup and return error
    blasBuilder.deinit();
    return result;
  }

  // Start the build and compaction of the BLAS
  VkDeviceSize hintBuildBudget = m_blasAccelerationStructureBudget;  // Limiting the size of the scratch buffer to 2MB
  bool         finished        = false;

  std::span<AccelerationStructureBuildData> buildDataSpan = blasBuildData;
  std::span<AccelerationStructure>          blasSpan      = blasSet;

  do
  {
    {
      VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);

      VkResult cmdResult = blasBuilder.cmdCreateBlas(cmd, buildDataSpan, blasSpan, blasScratchBuffer.address,
                                                     blasScratchBuffer.bufferSize, hintBuildBudget);
      if(cmdResult == VK_SUCCESS)
      {
        finished = true;
      }
      else if(cmdResult != VK_INCOMPLETE)
      {
        // Any result other than VK_SUCCESS or VK_INCOMPLETE is an error
        blasBuilder.deinit();
        return cmdResult;
      }

      result = endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue);
      if(result != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return result;
      }
    }
    // compact BLAS if needed
    if(buildFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
    {
      // Compacting the BLAS, and destroy the previous ones
      VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);
      blasBuilder.cmdCompactBlas(cmd, buildDataSpan, blasSpan);

      result = endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue);
      if(result != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return result;
      }

      blasBuilder.destroyNonCompactedBlas();
    }
  } while(!finished);

  blasBuildStatistics = blasBuilder.getStatistics();

  // Giving a name to the BLAS
  for(size_t i = 0; i < blasSet.size(); i++)
  {
    NVVK_DBG_NAME(blasSet[i].accel);
  }

  // Cleanup
  blasBuilder.deinit();

  return VK_SUCCESS;
}

VkResult AccelerationStructureHelperLB::tlasSubmitBuildAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances,
                                                               VkBuildAccelerationStructureFlagsKHR buildFlags)
{
  VkDevice device = m_alloc->getDevice();

  // we must not invoke build if already built. use update.
  assert((tlasInstancesBuffer.buffer == VK_NULL_HANDLE)
         && "Do not invoke build if already built. build with VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR, then use tlasUpdate");

  VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);

  // Create the instances buffer (using LargeBuffer for large scenes with many instances)
  VkDeviceSize instancesSize = std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances).size_bytes();
  VkResult     result        = m_alloc->createLargeBuffer(tlasInstancesBuffer, instancesSize,
                                                          VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                              | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR,
                                                          m_queueInfo.queue);

  if(result != VK_SUCCESS)
  {
    return result;
  }
  NVVK_DBG_NAME(tlasInstancesBuffer.buffer);

  // Upload via chunked staging (supports >4GB, 256MB chunks)
  result = m_uploader->appendLargeBuffer(tlasInstancesBuffer, 0, instancesSize, tlasInstances.data());
  if(result != VK_SUCCESS)
  {
    // Cleanup instances buffer
    m_alloc->destroyLargeBuffer(tlasInstancesBuffer);
    tlasInstancesBuffer = {};
    return result;
  }

  m_uploader->cmdUploadAppended(cmd);

  // Barrier to ensure transfer write completes before acceleration structure build.
  // VK_ACCESS_2_SHADER_READ_BIT is required because the acceleration structure build
  // operation reads the instance data from the buffer (not just writes to the AS).
  // Without this flag, validation layers report READ_AFTER_WRITE hazards.
  accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);


  tlasBuildData = {VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
  AccelerationStructureGeometryInfo geometryInfo =
      tlasBuildData.makeInstanceGeometry(tlasInstances.size(), tlasInstancesBuffer.address);
  tlasBuildData.addGeometry(geometryInfo);

  // Get the size of the TLAS
  auto sizeInfo = tlasBuildData.finalizeGeometry(m_alloc->getDevice(), buildFlags);

  // Create the scratch buffer (using LargeBuffer for large TLAS builds)
  result = m_alloc->createLargeBuffer(tlasScratchBuffer, sizeInfo.buildScratchSize,
                                      VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                          | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                      m_queueInfo.queue, VK_NULL_HANDLE, ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE,
                                      VMA_MEMORY_USAGE_AUTO, {}, m_accelStructProps.minAccelerationStructureScratchOffsetAlignment);

  if(result != VK_SUCCESS)
  {
    // Cleanup instances buffer
    m_alloc->destroyLargeBuffer(tlasInstancesBuffer);
    tlasInstancesBuffer = {};
    return result;
  }
  NVVK_DBG_NAME(tlasScratchBuffer.buffer);

  // Create the TLAS
  result = m_alloc->createAcceleration(tlas, tlasBuildData.makeCreateInfo());
  if(result != VK_SUCCESS)
  {
    // Cleanup buffers
    m_alloc->destroyLargeBuffer(tlasInstancesBuffer);
    m_alloc->destroyLargeBuffer(tlasScratchBuffer);
    tlasInstancesBuffer = {};
    tlasScratchBuffer   = {};
    return result;
  }
  NVVK_DBG_NAME(tlas.accel);

  tlasBuildData.cmdBuildAccelerationStructure(cmd, tlas.accel, tlasScratchBuffer.address);

  tlasSize = tlasInstances.size();

  result = endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue);
  if(result != VK_SUCCESS)
  {
    // Cleanup everything
    m_alloc->destroyAcceleration(tlas);
    m_alloc->destroyLargeBuffer(tlasInstancesBuffer);
    m_alloc->destroyLargeBuffer(tlasScratchBuffer);
    tlas                = {};
    tlasInstancesBuffer = {};
    tlasScratchBuffer   = {};
    return result;
  }

  m_uploader->releaseStaging();

  return VK_SUCCESS;
}

VkResult AccelerationStructureHelperLB::tlasSubmitUpdateAndWait(const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances)
{
  VkDevice device = m_alloc->getDevice();

  bool sizeChanged = (tlasInstances.size() != tlasSize);

  VkCommandBuffer cmd = createSingleTimeCommands(device, m_transientPool);

  // Update the instance buffer via chunked staging upload (256MB chunks)
  VkResult result =
      m_uploader->appendLargeBuffer(tlasInstancesBuffer, 0, std::span(tlasInstances).size_bytes(), tlasInstances.data());
  if(result != VK_SUCCESS)
  {
    return result;
  }

  m_uploader->cmdUploadAppended(cmd);

  // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
  accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);

  if(tlasScratchBuffer.buffer == VK_NULL_HANDLE)
  {
    result = m_alloc->createLargeBuffer(tlasScratchBuffer, tlasBuildData.sizeInfo.buildScratchSize,
                                        VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                        m_queueInfo.queue, VK_NULL_HANDLE, ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE,
                                        VMA_MEMORY_USAGE_AUTO, {}, m_accelStructProps.minAccelerationStructureScratchOffsetAlignment);
    if(result != VK_SUCCESS)
    {
      return result;
    }
    NVVK_DBG_NAME(tlasScratchBuffer.buffer);
  }

  // Building or updating the top-level acceleration structure
  if(sizeChanged)
  {
    tlasBuildData.cmdBuildAccelerationStructure(cmd, tlas.accel, tlasScratchBuffer.address);
    tlasSize = tlasInstances.size();
  }
  else
  {
    tlasBuildData.cmdUpdateAccelerationStructure(cmd, tlas.accel, tlasScratchBuffer.address);
  }

  // Make sure to have the TLAS ready before using it
  result = endSingleTimeCommands(cmd, device, m_transientPool, m_queueInfo.queue);
  if(result != VK_SUCCESS)
  {
    return result;
  }

  m_uploader->releaseStaging();

  return VK_SUCCESS;
}
