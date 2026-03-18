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

#pragma once

#include <functional>

#include <volk.h>

#include "nvvk/acceleration_structures.hpp"
#include "nvvk/commands.hpp"
#include "nvvk/resource_allocator.hpp"

namespace vk_gaussian_splatting {

class ParticleAccelerationStructureHelperGpu
{
public:
  enum class GeometryType : uint32_t
  {
    eTriangles = 0,
    eAabbs     = 1,
  };

  struct CreateInfo
  {
    GeometryType geometryType = GeometryType::eTriangles;

    VkDeviceSize vertexBufferSize = 0;
    VkDeviceSize indexBufferSize  = 0;
    VkDeviceSize aabbBufferSize   = 0;

    uint32_t vertexCount = 0;
    uint32_t indexCount  = 0;
    uint32_t aabbCount   = 0;

    VkDeviceSize vertexStride = 0;
    VkFormat     vertexFormat = VK_FORMAT_UNDEFINED;

    uint32_t instanceCount = 0;

    VkBuildAccelerationStructureFlagsKHR blasBuildFlags = 0;
    VkBuildAccelerationStructureFlagsKHR tlasBuildFlags = 0;
  };

  struct BlasCreateInfo
  {
    GeometryType                         geometryType     = GeometryType::eTriangles;
    VkDeviceSize                         vertexBufferSize = 0;
    VkDeviceSize                         indexBufferSize  = 0;
    VkDeviceSize                         aabbBufferSize   = 0;
    uint32_t                             vertexCount      = 0;
    uint32_t                             indexCount       = 0;
    uint32_t                             aabbCount        = 0;
    VkDeviceSize                         vertexStride     = 0;
    VkFormat                             vertexFormat     = VK_FORMAT_UNDEFINED;
    VkBuildAccelerationStructureFlagsKHR blasBuildFlags   = 0;
  };

  struct TlasCreateInfo
  {
    uint32_t                             instanceCount  = 0;
    VkBuildAccelerationStructureFlagsKHR tlasBuildFlags = 0;
  };

  using RecordComputeFn = std::function<void(VkCommandBuffer)>;

  ~ParticleAccelerationStructureHelperGpu() { assert(!m_transientPool && "deinit missing"); }

  void init(nvvk::ResourceAllocator* alloc, nvvk::QueueInfo queueInfo);
  void deinit();

  VkResult create(const CreateInfo& info, const RecordComputeFn& recordComputeInit);
  VkResult createBlasOnly(const BlasCreateInfo& info, const RecordComputeFn& recordComputeInit);
  VkResult createTlasOnly(const TlasCreateInfo& info, const RecordComputeFn& recordComputeInit);
  VkResult updateTlasOnly(const RecordComputeFn& recordComputeUpdate);
  VkResult updateBlasAndTlas(const RecordComputeFn& recordComputeUpdate);

  void deinitAccelerationStructures();

  VkDeviceAddress getVertexBufferAddress() const { return m_vertexBuffer.address; }
  VkDeviceAddress getIndexBufferAddress() const { return m_indexBuffer.address; }
  VkDeviceAddress getAabbBufferAddress() const { return m_aabbBuffer.address; }
  VkDeviceAddress getInstanceBufferAddress() const { return m_tlasInstancesBuffer.address; }
  VkDeviceSize    getInstanceBufferSize() const { return m_tlasInstancesBuffer.bufferSize; }
  VkDeviceSize    getTlasScratchBufferSize() const { return m_tlasScratchBuffer.bufferSize; }
  VkDeviceSize    getBlasScratchBufferSize() const { return m_blasScratchBuffer.bufferSize; }

  const nvvk::AccelerationStructure& getBlas() const { return m_blas; }
  const nvvk::AccelerationStructure& getTlas() const { return m_tlas; }

private:
  VkResult allocateGeometryBuffers(const CreateInfo& info);
  VkResult allocateTlasInstancesBuffer(uint32_t instanceCount);
  VkResult allocateScratchBuffers();

  nvvk::AccelerationStructureGeometryInfo makeGeometryInfo(const CreateInfo& info) const;
  nvvk::AccelerationStructureGeometryInfo makeInstanceGeometry(uint32_t instanceCount) const;

  void barrierComputeToAsBuild(VkCommandBuffer cmd) const;

private:
  nvvk::QueueInfo                                    m_queueInfo{};
  nvvk::ResourceAllocator*                           m_alloc{nullptr};
  VkCommandPool                                      m_transientPool{};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelStructProps{};

  nvvk::LargeBuffer m_vertexBuffer{};
  nvvk::LargeBuffer m_indexBuffer{};
  nvvk::LargeBuffer m_aabbBuffer{};

  nvvk::LargeBuffer m_tlasInstancesBuffer{};
  nvvk::LargeBuffer m_blasScratchBuffer{};
  nvvk::LargeBuffer m_tlasScratchBuffer{};

  nvvk::AccelerationStructureBuildData m_blasBuildData{};
  nvvk::AccelerationStructure          m_blas{};

  nvvk::AccelerationStructureBuildData m_tlasBuildData{};
  nvvk::AccelerationStructure          m_tlas{};

  CreateInfo m_createInfo{};
};

}  // namespace vk_gaussian_splatting
