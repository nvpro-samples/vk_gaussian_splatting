/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "particle_acceleration_structures_gpu.hpp"

#include <cassert>
#include <iostream>

#include <nvutils/timers.hpp>
#include <nvvk/debug_util.hpp>

namespace vk_gaussian_splatting {

void ParticleAccelerationStructureHelperGpu::init(nvvk::ResourceAllocator* alloc, nvvk::QueueInfo queueInfo)
{
  assert(!m_transientPool && "init() called multiple times");
  m_alloc         = alloc;
  m_queueInfo     = queueInfo;
  m_transientPool = nvvk::createTransientCommandPool(alloc->getDevice(), queueInfo.familyIndex);

  m_accelStructProps                 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext                       = &m_accelStructProps;
  vkGetPhysicalDeviceProperties2(alloc->getPhysicalDevice(), &props2);
}

void ParticleAccelerationStructureHelperGpu::deinit()
{
  deinitAccelerationStructures();

  if(m_transientPool)
  {
    vkDestroyCommandPool(m_alloc->getDevice(), m_transientPool, nullptr);
  }
  m_transientPool    = nullptr;
  m_alloc            = nullptr;
  m_queueInfo        = {};
  m_createInfo       = {};
  m_accelStructProps = {};
}

void ParticleAccelerationStructureHelperGpu::deinitAccelerationStructures()
{
  if(m_blas.accel)
    m_alloc->destroyAcceleration(m_blas);
  if(m_tlas.accel)
    m_alloc->destroyAcceleration(m_tlas);

  if(m_vertexBuffer.buffer)
    m_alloc->destroyLargeBuffer(m_vertexBuffer);
  if(m_indexBuffer.buffer)
    m_alloc->destroyLargeBuffer(m_indexBuffer);
  if(m_aabbBuffer.buffer)
    m_alloc->destroyLargeBuffer(m_aabbBuffer);
  if(m_tlasInstancesBuffer.buffer)
    m_alloc->destroyLargeBuffer(m_tlasInstancesBuffer);
  if(m_blasScratchBuffer.buffer)
    m_alloc->destroyLargeBuffer(m_blasScratchBuffer);
  if(m_tlasScratchBuffer.buffer)
    m_alloc->destroyLargeBuffer(m_tlasScratchBuffer);

  m_blasBuildData       = {};
  m_tlasBuildData       = {};
  m_blas                = {};
  m_tlas                = {};
  m_vertexBuffer        = {};
  m_indexBuffer         = {};
  m_aabbBuffer          = {};
  m_tlasInstancesBuffer = {};
  m_blasScratchBuffer   = {};
  m_tlasScratchBuffer   = {};
}

VkResult ParticleAccelerationStructureHelperGpu::create(const CreateInfo& info, const RecordComputeFn& recordComputeInit)
{
  nvutils::ScopedTimer timer("GPU AS: create (BLAS+TLAS)\n");
  assert(m_alloc && "init() must be called before create()");
  assert(m_blas.accel == VK_NULL_HANDLE && "create() called when BLAS already exists");
  assert(m_tlas.accel == VK_NULL_HANDLE && "create() called when TLAS already exists");

  if(info.instanceCount == 0)
    return VK_ERROR_INITIALIZATION_FAILED;

  m_createInfo = info;

  VkResult result = allocateGeometryBuffers(info);
  if(result != VK_SUCCESS)
    return result;

  result = allocateTlasInstancesBuffer(info.instanceCount);
  if(result != VK_SUCCESS)
    return result;

  m_blasBuildData = {VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
  m_blasBuildData.addGeometry(makeGeometryInfo(info));
  m_blasBuildData.finalizeGeometry(m_alloc->getDevice(), info.blasBuildFlags);

  m_tlasBuildData = {VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
  m_tlasBuildData.addGeometry(makeInstanceGeometry(info.instanceCount));
  m_tlasBuildData.finalizeGeometry(m_alloc->getDevice(), info.tlasBuildFlags);

  result = allocateScratchBuffers();
  if(result != VK_SUCCESS)
    return result;

  result = m_alloc->createAcceleration(m_tlas, m_tlasBuildData.makeCreateInfo());
  if(result != VK_SUCCESS)
    return result;
  NVVK_DBG_NAME(m_tlas.accel);

  const bool wantsCompaction = (info.blasBuildFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) != 0;
  if(wantsCompaction)
  {
    nvvk::AccelerationStructureBuilder blasBuilder;
    blasBuilder.init(m_alloc);

    std::vector<nvvk::AccelerationStructureBuildData> blasBuildDataVec{m_blasBuildData};
    std::vector<nvvk::AccelerationStructure>          blasSet(1);

    VkDeviceSize scratchBudget = m_blasBuildData.sizeInfo.buildScratchSize;
    VkDeviceSize scratchSize   = blasBuilder.getScratchSize(scratchBudget, blasBuildDataVec);
    if(m_blasScratchBuffer.buffer == VK_NULL_HANDLE || m_blasScratchBuffer.bufferSize < scratchSize)
    {
      if(m_blasScratchBuffer.buffer)
        m_alloc->destroyLargeBuffer(m_blasScratchBuffer);
      m_blasScratchBuffer = {};
      VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                       | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
      result = m_alloc->createLargeBuffer(m_blasScratchBuffer, scratchSize, usageFlags, m_queueInfo.queue, VK_NULL_HANDLE,
                                          nvvk::ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE, VMA_MEMORY_USAGE_AUTO, {},
                                          m_accelStructProps.minAccelerationStructureScratchOffsetAlignment);
      if(result != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return result;
      }
      NVVK_DBG_NAME(m_blasScratchBuffer.buffer);
    }

    VkResult buildResult = VK_INCOMPLETE;
    while(buildResult == VK_INCOMPLETE)
    {
      VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);

      if(recordComputeInit)
      {
        nvutils::ScopedTimer computeTimer("GPU AS: compute init (BLAS)\n");
        recordComputeInit(cmd);
        barrierComputeToAsBuild(cmd);
      }

      {
        nvutils::ScopedTimer                            buildTimer("GPU AS: build BLAS\n");
        std::span<nvvk::AccelerationStructureBuildData> buildSpan = blasBuildDataVec;
        std::span<nvvk::AccelerationStructure>          blasSpan  = blasSet;
        buildResult = blasBuilder.cmdCreateBlas(cmd, buildSpan, blasSpan, m_blasScratchBuffer.address,
                                                m_blasScratchBuffer.bufferSize, m_blasBuildData.sizeInfo.accelerationStructureSize);
      }

      VkResult submitResult = nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
      if(submitResult != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return submitResult;
      }
      if(buildResult != VK_SUCCESS && buildResult != VK_INCOMPLETE)
      {
        blasBuilder.deinit();
        return buildResult;
      }
    }

    VkResult compactResult = VK_INCOMPLETE;
    while(compactResult == VK_INCOMPLETE)
    {
      VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);
      std::span<nvvk::AccelerationStructureBuildData> buildSpan = blasBuildDataVec;
      std::span<nvvk::AccelerationStructure>          blasSpan  = blasSet;
      compactResult                                             = blasBuilder.cmdCompactBlas(cmd, buildSpan, blasSpan);
      VkResult submitResult = nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
      if(submitResult != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return submitResult;
      }
    }
    if(compactResult != VK_SUCCESS)
    {
      blasBuilder.deinit();
      return compactResult;
    }

    blasBuilder.destroyNonCompactedBlas();
    const auto stats = blasBuilder.getStatistics();
    if(stats.totalOriginalSize > 0)
    {
      LOGD("%s\n", stats.toString().c_str());
    }

    blasBuilder.deinit();

    m_blas          = blasSet[0];
    m_blasBuildData = blasBuildDataVec[0];

    // Re-run compute to update instance buffer with compacted BLAS address.
    VkCommandBuffer tlasCmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);
    if(recordComputeInit)
    {
      nvutils::ScopedTimer computeTimer("GPU AS: compute init (TLAS)\n");
      recordComputeInit(tlasCmd);
      barrierComputeToAsBuild(tlasCmd);
    }

    {
      nvutils::ScopedTimer buildTimer("GPU AS: build TLAS\n");
      m_tlasBuildData.cmdBuildAccelerationStructure(tlasCmd, m_tlas.accel, m_tlasScratchBuffer.address);
    }

    return nvvk::endSingleTimeCommands(tlasCmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
  }

  result = m_alloc->createAcceleration(m_blas, m_blasBuildData.makeCreateInfo());
  if(result != VK_SUCCESS)
    return result;
  NVVK_DBG_NAME(m_blas.accel);

  VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);

  if(recordComputeInit)
  {
    nvutils::ScopedTimer computeTimer("GPU AS: compute init\n");
    recordComputeInit(cmd);
    barrierComputeToAsBuild(cmd);
  }

  {
    nvutils::ScopedTimer buildTimer("GPU AS: build BLAS+TLAS\n");
    m_blasBuildData.cmdBuildAccelerationStructure(cmd, m_blas.accel, m_blasScratchBuffer.address);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
  }

  return nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
}

VkResult ParticleAccelerationStructureHelperGpu::createBlasOnly(const BlasCreateInfo& info, const RecordComputeFn& recordComputeInit)
{
  nvutils::ScopedTimer timer("GPU AS: create BLAS\n");
  assert(m_alloc && "init() must be called before createBlasOnly()");
  assert(m_blas.accel == VK_NULL_HANDLE && "createBlasOnly() called when BLAS already exists");

  CreateInfo temp{};
  temp.geometryType     = info.geometryType;
  temp.vertexBufferSize = info.vertexBufferSize;
  temp.indexBufferSize  = info.indexBufferSize;
  temp.aabbBufferSize   = info.aabbBufferSize;
  temp.vertexCount      = info.vertexCount;
  temp.indexCount       = info.indexCount;
  temp.aabbCount        = info.aabbCount;
  temp.vertexStride     = info.vertexStride;
  temp.vertexFormat     = info.vertexFormat;
  temp.blasBuildFlags   = info.blasBuildFlags;
  m_createInfo          = temp;

  VkResult result = allocateGeometryBuffers(temp);
  if(result != VK_SUCCESS)
    return result;

  m_blasBuildData = {VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
  m_blasBuildData.addGeometry(makeGeometryInfo(temp));
  m_blasBuildData.finalizeGeometry(m_alloc->getDevice(), info.blasBuildFlags);

  result = allocateScratchBuffers();
  if(result != VK_SUCCESS)
    return result;

  const bool wantsCompaction = (info.blasBuildFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) != 0;
  if(wantsCompaction)
  {
    nvvk::AccelerationStructureBuilder blasBuilder;
    blasBuilder.init(m_alloc);

    std::vector<nvvk::AccelerationStructureBuildData> blasBuildDataVec{m_blasBuildData};
    std::vector<nvvk::AccelerationStructure>          blasSet(1);

    VkDeviceSize scratchBudget = m_blasBuildData.sizeInfo.buildScratchSize;
    VkDeviceSize scratchSize   = blasBuilder.getScratchSize(scratchBudget, blasBuildDataVec);
    if(m_blasScratchBuffer.buffer == VK_NULL_HANDLE || m_blasScratchBuffer.bufferSize < scratchSize)
    {
      if(m_blasScratchBuffer.buffer)
        m_alloc->destroyLargeBuffer(m_blasScratchBuffer);
      m_blasScratchBuffer = {};
      VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                       | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
      result = m_alloc->createLargeBuffer(m_blasScratchBuffer, scratchSize, usageFlags, m_queueInfo.queue, VK_NULL_HANDLE,
                                          nvvk::ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE, VMA_MEMORY_USAGE_AUTO, {},
                                          m_accelStructProps.minAccelerationStructureScratchOffsetAlignment);
      if(result != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return result;
      }
      NVVK_DBG_NAME(m_blasScratchBuffer.buffer);
    }

    VkResult buildResult = VK_INCOMPLETE;
    while(buildResult == VK_INCOMPLETE)
    {
      VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);

      if(recordComputeInit)
      {
        nvutils::ScopedTimer computeTimer("GPU AS: compute init (BLAS)\n");
        recordComputeInit(cmd);
        barrierComputeToAsBuild(cmd);
      }

      {
        nvutils::ScopedTimer                            buildTimer("GPU AS: build BLAS\n");
        std::span<nvvk::AccelerationStructureBuildData> buildSpan = blasBuildDataVec;
        std::span<nvvk::AccelerationStructure>          blasSpan  = blasSet;
        buildResult = blasBuilder.cmdCreateBlas(cmd, buildSpan, blasSpan, m_blasScratchBuffer.address,
                                                m_blasScratchBuffer.bufferSize, m_blasBuildData.sizeInfo.accelerationStructureSize);
      }

      VkResult submitResult = nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
      if(submitResult != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return submitResult;
      }
      if(buildResult != VK_SUCCESS && buildResult != VK_INCOMPLETE)
      {
        blasBuilder.deinit();
        return buildResult;
      }
    }

    VkResult compactResult = VK_INCOMPLETE;
    while(compactResult == VK_INCOMPLETE)
    {
      VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);
      std::span<nvvk::AccelerationStructureBuildData> buildSpan = blasBuildDataVec;
      std::span<nvvk::AccelerationStructure>          blasSpan  = blasSet;
      compactResult                                             = blasBuilder.cmdCompactBlas(cmd, buildSpan, blasSpan);
      VkResult submitResult = nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
      if(submitResult != VK_SUCCESS)
      {
        blasBuilder.deinit();
        return submitResult;
      }
    }
    if(compactResult != VK_SUCCESS)
    {
      blasBuilder.deinit();
      return compactResult;
    }

    blasBuilder.destroyNonCompactedBlas();
    const auto stats = blasBuilder.getStatistics();
    if(stats.totalOriginalSize > 0)
    {
      LOGD("%s\n", stats.toString().c_str());
    }

    blasBuilder.deinit();

    m_blas          = blasSet[0];
    m_blasBuildData = blasBuildDataVec[0];
    return VK_SUCCESS;
  }

  result = m_alloc->createAcceleration(m_blas, m_blasBuildData.makeCreateInfo());
  if(result != VK_SUCCESS)
    return result;
  NVVK_DBG_NAME(m_blas.accel);

  VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);

  if(recordComputeInit)
  {
    nvutils::ScopedTimer computeTimer("GPU AS: compute init (BLAS)\n");
    recordComputeInit(cmd);
    barrierComputeToAsBuild(cmd);
  }

  {
    nvutils::ScopedTimer buildTimer("GPU AS: build BLAS\n");
    m_blasBuildData.cmdBuildAccelerationStructure(cmd, m_blas.accel, m_blasScratchBuffer.address);
  }

  return nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
}

VkResult ParticleAccelerationStructureHelperGpu::createTlasOnly(const TlasCreateInfo& info, const RecordComputeFn& recordComputeInit)
{
  nvutils::ScopedTimer timer("GPU AS: create TLAS\n");
  assert(m_alloc && "init() must be called before createTlasOnly()");
  assert(m_tlas.accel == VK_NULL_HANDLE && "createTlasOnly() called when TLAS already exists");

  if(info.instanceCount == 0)
    return VK_ERROR_INITIALIZATION_FAILED;

  // Persist build flags for update path
  m_createInfo                = {};
  m_createInfo.tlasBuildFlags = info.tlasBuildFlags;

  VkResult result = allocateTlasInstancesBuffer(info.instanceCount);
  if(result != VK_SUCCESS)
    return result;

  m_tlasBuildData = {VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
  m_tlasBuildData.addGeometry(makeInstanceGeometry(info.instanceCount));
  m_tlasBuildData.finalizeGeometry(m_alloc->getDevice(), info.tlasBuildFlags);

  result = allocateScratchBuffers();
  if(result != VK_SUCCESS)
    return result;

  result = m_alloc->createAcceleration(m_tlas, m_tlasBuildData.makeCreateInfo());
  if(result != VK_SUCCESS)
    return result;
  NVVK_DBG_NAME(m_tlas.accel);

  VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);

  if(recordComputeInit)
  {
    nvutils::ScopedTimer computeTimer("GPU AS: compute init (TLAS)\n");
    recordComputeInit(cmd);
    barrierComputeToAsBuild(cmd);
  }

  {
    nvutils::ScopedTimer buildTimer("GPU AS: build TLAS\n");
    m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
  }

  return nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
}

VkResult ParticleAccelerationStructureHelperGpu::updateTlasOnly(const RecordComputeFn& recordComputeUpdate)
{
  nvutils::ScopedTimer timer("GPU AS: update TLAS\n");
  assert(m_tlas.accel && "updateTlasOnly() called without TLAS");

  if(!(m_createInfo.tlasBuildFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR))
    return VK_ERROR_FEATURE_NOT_PRESENT;

  VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);

  if(recordComputeUpdate)
  {
    nvutils::ScopedTimer computeTimer("GPU AS: compute update (TLAS)\n");
    recordComputeUpdate(cmd);
    barrierComputeToAsBuild(cmd);
  }

  {
    nvutils::ScopedTimer updateTimer("GPU AS: update TLAS build\n");
    m_tlasBuildData.cmdUpdateAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
  }

  return nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
}

VkResult ParticleAccelerationStructureHelperGpu::updateBlasAndTlas(const RecordComputeFn& recordComputeUpdate)
{
  nvutils::ScopedTimer timer("GPU AS: update BLAS+TLAS\n");
  assert(m_blas.accel && m_tlas.accel && "updateBlasAndTlas() called without BLAS/TLAS");

  if(!(m_createInfo.blasBuildFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR))
    return VK_ERROR_FEATURE_NOT_PRESENT;
  if(!(m_createInfo.tlasBuildFlags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR))
    return VK_ERROR_FEATURE_NOT_PRESENT;

  VkCommandBuffer cmd = nvvk::createSingleTimeCommands(m_alloc->getDevice(), m_transientPool);

  if(recordComputeUpdate)
  {
    nvutils::ScopedTimer computeTimer("GPU AS: compute update (BLAS+TLAS)\n");
    recordComputeUpdate(cmd);
    barrierComputeToAsBuild(cmd);
  }

  {
    nvutils::ScopedTimer updateTimer("GPU AS: update BLAS+TLAS build\n");
    m_blasBuildData.cmdUpdateAccelerationStructure(cmd, m_blas.accel, m_blasScratchBuffer.address);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    m_tlasBuildData.cmdUpdateAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
  }

  return nvvk::endSingleTimeCommands(cmd, m_alloc->getDevice(), m_transientPool, m_queueInfo.queue);
}

VkResult ParticleAccelerationStructureHelperGpu::allocateGeometryBuffers(const CreateInfo& info)
{
  nvutils::ScopedTimer timer("GPU AS: allocate geometry buffers\n");
  VkBufferUsageFlags2  usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                   | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  VkQueue queue = m_queueInfo.queue;

  if(info.geometryType == GeometryType::eTriangles)
  {
    if(info.vertexBufferSize == 0 || info.indexBufferSize == 0)
      return VK_ERROR_INITIALIZATION_FAILED;

    VkResult result = m_alloc->createLargeBuffer(m_vertexBuffer, info.vertexBufferSize, usageFlags, queue);
    if(result != VK_SUCCESS)
      return result;
    NVVK_DBG_NAME(m_vertexBuffer.buffer);

    result = m_alloc->createLargeBuffer(m_indexBuffer, info.indexBufferSize, usageFlags, queue);
    if(result != VK_SUCCESS)
      return result;
    NVVK_DBG_NAME(m_indexBuffer.buffer);
  }
  else
  {
    if(info.aabbBufferSize == 0)
      return VK_ERROR_INITIALIZATION_FAILED;

    VkResult result = m_alloc->createLargeBuffer(m_aabbBuffer, info.aabbBufferSize, usageFlags, queue);
    if(result != VK_SUCCESS)
      return result;
    NVVK_DBG_NAME(m_aabbBuffer.buffer);
  }

  return VK_SUCCESS;
}

VkResult ParticleAccelerationStructureHelperGpu::allocateTlasInstancesBuffer(uint32_t instanceCount)
{
  nvutils::ScopedTimer timer("GPU AS: allocate TLAS instances buffer\n");
  VkDeviceSize        sizeBytes = static_cast<VkDeviceSize>(instanceCount) * sizeof(VkAccelerationStructureInstanceKHR);
  VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                   | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  VkResult result = m_alloc->createLargeBuffer(m_tlasInstancesBuffer, sizeBytes, usageFlags, m_queueInfo.queue);
  if(result != VK_SUCCESS)
    return result;
  NVVK_DBG_NAME(m_tlasInstancesBuffer.buffer);
  return VK_SUCCESS;
}

VkResult ParticleAccelerationStructureHelperGpu::allocateScratchBuffers()
{
  nvutils::ScopedTimer timer("GPU AS: allocate scratch buffers\n");
  VkDeviceSize         blasScratchSize = (m_blasBuildData.asType == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR) ?
                                             ((m_blasBuildData.sizeInfo.buildScratchSize > m_blasBuildData.sizeInfo.updateScratchSize) ?
                                                  m_blasBuildData.sizeInfo.buildScratchSize :
                                                  m_blasBuildData.sizeInfo.updateScratchSize) :
                                             0;
  VkDeviceSize         tlasScratchSize = (m_tlasBuildData.asType == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) ?
                                             ((m_tlasBuildData.sizeInfo.buildScratchSize > m_tlasBuildData.sizeInfo.updateScratchSize) ?
                                                  m_tlasBuildData.sizeInfo.buildScratchSize :
                                                  m_tlasBuildData.sizeInfo.updateScratchSize) :
                                             0;

  VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                   | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;

  VkResult result = VK_SUCCESS;
  if(blasScratchSize > 0)
  {
    result = m_alloc->createLargeBuffer(m_blasScratchBuffer, blasScratchSize, usageFlags, m_queueInfo.queue, VK_NULL_HANDLE,
                                        nvvk::ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE, VMA_MEMORY_USAGE_AUTO, {},
                                        m_accelStructProps.minAccelerationStructureScratchOffsetAlignment);
    if(result != VK_SUCCESS)
      return result;
    NVVK_DBG_NAME(m_blasScratchBuffer.buffer);
  }

  if(tlasScratchSize > 0)
  {
    result = m_alloc->createLargeBuffer(m_tlasScratchBuffer, tlasScratchSize, usageFlags, m_queueInfo.queue, VK_NULL_HANDLE,
                                        nvvk::ResourceAllocator::DEFAULT_LARGE_CHUNK_SIZE, VMA_MEMORY_USAGE_AUTO, {},
                                        m_accelStructProps.minAccelerationStructureScratchOffsetAlignment);
    if(result != VK_SUCCESS)
      return result;
    NVVK_DBG_NAME(m_tlasScratchBuffer.buffer);
  }

  return VK_SUCCESS;
}

nvvk::AccelerationStructureGeometryInfo ParticleAccelerationStructureHelperGpu::makeGeometryInfo(const CreateInfo& info) const
{
  nvvk::AccelerationStructureGeometryInfo geoInfo{};

  if(info.geometryType == GeometryType::eTriangles)
  {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat             = info.vertexFormat;
    triangles.vertexData.deviceAddress = m_vertexBuffer.address;
    triangles.vertexStride             = info.vertexStride;
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = m_indexBuffer.address;
    triangles.maxVertex                = info.vertexCount > 0 ? info.vertexCount - 1 : 0;

    geoInfo.geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geoInfo.geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geoInfo.geometry.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geoInfo.geometry.geometry.triangles = triangles;

    geoInfo.rangeInfo.primitiveCount  = info.indexCount / 3;
    geoInfo.rangeInfo.primitiveOffset = 0;
    geoInfo.rangeInfo.firstVertex     = 0;
    geoInfo.rangeInfo.transformOffset = 0;
  }
  else
  {
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{};
    aabbs.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data.deviceAddress = m_aabbBuffer.address;
    aabbs.stride             = sizeof(VkAabbPositionsKHR);

    geoInfo.geometry.sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geoInfo.geometry.geometryType   = VK_GEOMETRY_TYPE_AABBS_KHR;
    geoInfo.geometry.flags          = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geoInfo.geometry.geometry.aabbs = aabbs;

    geoInfo.rangeInfo.primitiveCount  = info.aabbCount;
    geoInfo.rangeInfo.primitiveOffset = 0;
    geoInfo.rangeInfo.firstVertex     = 0;
    geoInfo.rangeInfo.transformOffset = 0;
  }

  return geoInfo;
}

nvvk::AccelerationStructureGeometryInfo ParticleAccelerationStructureHelperGpu::makeInstanceGeometry(uint32_t instanceCount) const
{
  nvvk::AccelerationStructureBuildData temp{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
  return temp.makeInstanceGeometry(instanceCount, m_tlasInstancesBuffer.address);
}

void ParticleAccelerationStructureHelperGpu::barrierComputeToAsBuild(VkCommandBuffer cmd) const
{
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

}  // namespace vk_gaussian_splatting
