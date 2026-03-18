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

#ifndef _SPLAT_SET_VK_H_
#define _SPLAT_SET_VK_H_

#include <vector>
#include <algorithm>

#include <vulkan/vulkan_core.h>

#include <nvapp/application.hpp>

#include <nvvk/staging.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/physical_device.hpp>

#include "../shaders/shaderio.h"  // For SplatSetDesc
#include "acceleration_structures_lb.hpp"
#include "splat_set.h"
#include "utilities.h"
#include "obj_loader.h"

#include "splat_set.h"  // Base class with RAM data
#include "memory_statistics.h"

namespace vk_gaussian_splatting {

// RTX initialization status for each splat set
enum class RtxStatus : uint32_t
{
  eDelayed,  // RTX not yet initialized/attempted (or deferred until needed)
  eSuccess,  // RTX successfully initialized and all allocations succeeded
  eError     // RTX initialization failed (allocation error, out of memory, etc.)
};

// Format and upload 3DGS data to VRAM
// Inherits from SplatSet to own both RAM data and GPU resources
class SplatSetVk : public SplatSet
{
public:
  enum class Flags : uint32_t
  {
    eNone            = 0,
    eDelete          = 1 << 0,  // Remove from GPU + delete from RAM
    eNew             = 1 << 1,  // Just created, needs GPU upload
    eDataChanged     = 1 << 2,  // Data storage changed (format/storage mode)
    eGeometryChanged = 1 << 3,  // Geometry changed (BLAS rebuild needed)
  };

  // Query methods for state
  bool isMarkedForDeletion() const { return static_cast<uint32_t>(flags) & static_cast<uint32_t>(Flags::eDelete); }
  bool shouldShowInUI() const { return !isMarkedForDeletion(); }

  size_t index{0};                         // Position in SplatSetManagerVk::m_splatSets vector
  Flags  flags            = Flags::eNone;  // Set by manager methods only
  size_t instanceRefCount = 0;             // Number of instances referencing this splat set

  SplatSetVk() = default;

  ~SplatSetVk(void) {}

  void init(nvapp::Application*                                 app,
            nvvk::ResourceAllocator*                            alloc,
            nvvk::StagingUploader*                              uploader,
            VkSampler*                                          sampler,
            nvvk::PhysicalDeviceInfo*                           deviceInfo,
            VkPhysicalDeviceAccelerationStructurePropertiesKHR* accelStructProps)
  {
    m_app              = app;
    m_alloc            = alloc;
    m_uploader         = uploader;
    m_sampler          = sampler;
    m_deviceInfo       = deviceInfo;
    m_accelStructProps = accelStructProps;
    // Note: rtAccelerationStructures is now managed by SplatSetManagerVk
  }

  void deinit()
  {
    // Note: rtAccelerationStructures is now managed by SplatSetManagerVk (no deinit needed here)
    m_app              = nullptr;
    m_alloc            = nullptr;
    m_uploader         = nullptr;
    m_sampler          = nullptr;
    m_deviceInfo       = nullptr;
    m_accelStructProps = nullptr;
    rtxStatus          = RtxStatus::eDelayed;  // Reset to delayed state
  }


  // uploads the splatSet into VRAM
  // format/rgbaFormat in  [FORMAT_FLOAT32, FORMAT_FLOAT16, FORMAT_UINT8]
  // Note: Uses inherited SplatSet data (this->positions, etc.)
  void initDataStorage(uint32_t shFormat, uint32_t rgbaFormat);

  // destroy all buffers from VRAM
  // a new initDataStorage can be invoked afterward
  void deinitDataStorage();

  // initDataStorage must be invoked prior to creation of the splat model
  // Note: Uses inherited SplatSet data
  // Returns: true on success, false on allocation failure (sets rtxStatus to eError)
  bool rtxInitSplatModel(bool useInstances, bool useAABBs, bool compressBlas, int kernelDegree, float kernelMinResponse, bool kernelAdaptiveClamping);

  void rtxDeinitSplatModel()
  {
    // Destroy splat model buffers (now LargeBuffers)
    if(m_splatModel.vertexBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyLargeBuffer(m_splatModel.vertexBuffer);
    if(m_splatModel.indexBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyLargeBuffer(m_splatModel.indexBuffer);
    if(m_splatModel.aabbBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyLargeBuffer(m_splatModel.aabbBuffer);

    m_splatModel = {};
    rtxStatus    = RtxStatus::eDelayed;  // Reset to delayed state (not error, can be retried)

    // Reset RTX buffer memory stats
    memoryStats.rtxVertexBuffer = 0;
    memoryStats.rtxIndexBuffer  = 0;
    memoryStats.rtxAabbBuffer   = 0;
  }

  void rtxDeinitAccelerationStructures()
  {
    // Deinitialize geometry only (BLAS/TLAS managed by SplatSetManagerVk)
    rtxDeinitSplatModel();
    rtxStatus = RtxStatus::eDelayed;  // Reset to delayed (not error, can be retried)
    blasIndex = UINT32_MAX;
  }

  // Compute transform matrix for a specific splat (used by manager for TLAS building)
  // Note: Uses inherited SplatSet data
  glm::mat4 rtxComputeTransformMatrix(uint64_t splatIdx);

  // Get geometry info for BLAS building (used by manager for batch BLAS builds)
  nvvk::AccelerationStructureGeometryInfo rtxCreateSplatModelAccelerationStructureGeometryInfo();

  // Accessors for private members (needed by instances to build descriptors)
  inline uint32_t getStorage() const { return dataStorage; }
  inline uint32_t getShFormat() const { return shFormat; }
  inline uint32_t getRgbaFormat() const { return rgbaFormat; }


public:
  VkSampler* m_sampler = nullptr;

  // Data textures

  nvvk::Image centersMap;
  nvvk::Image scalesMap;       // RTX specific
  nvvk::Image rotationsMap;    // RTX specific
  nvvk::Image covariancesMap;  // Raster specific
  nvvk::Image colorsMap;
  nvvk::Image sphericalHarmonicsMap;

  // Data buffers (LargeBuffer to support >4GB via sparse binding)

  nvvk::LargeBuffer centersBuffer;
  nvvk::LargeBuffer scalesBuffer;       // RTX specific
  nvvk::LargeBuffer rotationsBuffer;    // RTX specific
  nvvk::LargeBuffer covariancesBuffer;  // Raster specific
  nvvk::LargeBuffer colorsBuffer;
  nvvk::LargeBuffer sphericalHarmonicsBuffer;

  // Material properties for splat set (default material for new instances)
  shaderio::ObjMaterial splatMaterial;

  // Splat set metadata (needed by instances to build descriptors)
  uint32_t splatCount  = 0;  // Number of splats in this set
  uint32_t shDegree    = 0;  // Maximum SH degree (0-3)
  uint32_t dataStorage = 0;  // STORAGE_BUFFERS or STORAGE_TEXTURES (set via global MACRO)
  uint32_t shFormat    = 0;  // FORMAT_FLOAT32, FORMAT_FLOAT16, or FORMAT_UINT8 (for SH)
  uint32_t rgbaFormat  = 0;  // FORMAT_FLOAT32, FORMAT_FLOAT16, or FORMAT_UINT8 (for RGBA colors)
  // Note: name and path are inherited from base SplatSet class

  // Texture indices (for STORAGE_TEXTURES mode, bindless texture array)
  // These are set during initDataTextures() and used by instances to build descriptors
  uint32_t textureIndexCenters     = 0;
  uint32_t textureIndexScales      = 0;
  uint32_t textureIndexRotations   = 0;
  uint32_t textureIndexColors      = 0;
  uint32_t textureIndexCovariances = 0;
  uint32_t textureIndexSH          = 0;

  // Note: descriptor and descriptorBuffer moved to SplatSetInstanceVk (per-instance data)

  ////////////////////////
  // Ray tracing specifics

  // The Splat model (contains only one splat if instanced or all splats)
  struct SplatModel
  {
    // Icosa related

    uint32_t nbVertices{0};  // total number of vec3 vertex positions stored in vertexBuffer
    uint32_t nbIndices{0};   // total number of vertex indices stored in vertexBuffer
    nvvk::LargeBuffer vertexBuffer;  // Device buffer of the vertices (LargeBuffer for non-instanced mode with millions of splats)
    nvvk::LargeBuffer indexBuffer;  // Device buffer of the indices forming triangles

    // AABB related
    uint32_t          nbAABB;
    nvvk::LargeBuffer aabbBuffer;

  } m_splatModel;

  // aligned with VkAabbPositionsKHR
  struct SplatAabb
  {
    glm::vec3 minimum;
    glm::vec3 maximum;
  };

  // RTX: Index into manager's BLAS set (manager owns the AccelerationStructureHelper)
  uint32_t blasIndex = UINT32_MAX;  // Index into SplatSetManagerVk::m_rtAccelerationStructures.blasSet

  // Data storage memory usage statistics (per splat set, local tracking)
  ModelMemoryStats memoryStats;

  // RTX initialization status (default to eDelayed - not attempted yet)
  RtxStatus rtxStatus = RtxStatus::eDelayed;

  // RTX AS memory stats
  uint64_t tlasSizeBytes = 0;  // Size of the TLAS in VRAM in bytes
  uint64_t blasSizeBytes = 0;  // Size of the BLAS in VRAM in bytes

private:
  // create the buffers on the device and upload
  // the splat set data from host to device
  // Note: Uses inherited SplatSet data
  void initDataBuffers();

  // release buffers
  void deinitDataBuffers(void);

  // create the texture maps on the device and upload
  // the splat set data from host to device
  // Note: Uses inherited SplatSet data
  void initDataTextures();

  // release textures
  void deinitDataTextures(void);

  // Create texture, upload data and assign sampler
  // sampler will be released by deinitTexture
  void initTexture(uint32_t width, uint32_t height, uint32_t bufsize, void* data, VkFormat format, const VkSampler& sampler, nvvk::Image& texture);

  // Destroy texture at once, texture must not be in use
  void deinitTexture(nvvk::Image& texture);

private:
  // RTX specifics

  void rtxCreateSplatIcosahedron(uint64_t                splatOffset,
                                 std::vector<glm::vec3>& vertices,
                                 std::vector<uint32_t>&  indices,
                                 std::vector<SplatAabb>& aabbs,
                                 glm::mat4               transform = glm::mat4(1.0));

private:
  bool  m_rtxUseAABBs;
  bool  m_rtxUseInstances;
  bool  m_rtxCompressBlas;
  int   m_rtxKernelDegree;
  float m_rtxKernelMinResponse;
  bool  m_rtxKernelAdaptiveClamping;

  nvapp::Application*                                 m_app              = nullptr;
  nvvk::ResourceAllocator*                            m_alloc            = nullptr;
  nvvk::StagingUploader*                              m_uploader         = nullptr;
  nvvk::PhysicalDeviceInfo*                           m_deviceInfo       = nullptr;
  VkPhysicalDeviceAccelerationStructurePropertiesKHR* m_accelStructProps = nullptr;
};

// Bitwise operators for SplatSetVk::Flags
inline SplatSetVk::Flags operator|(SplatSetVk::Flags a, SplatSetVk::Flags b)
{
  return static_cast<SplatSetVk::Flags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline SplatSetVk::Flags operator&(SplatSetVk::Flags a, SplatSetVk::Flags b)
{
  return static_cast<SplatSetVk::Flags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline SplatSetVk::Flags& operator|=(SplatSetVk::Flags& a, SplatSetVk::Flags b)
{
  return a = a | b;
}
inline SplatSetVk::Flags& operator&=(SplatSetVk::Flags& a, SplatSetVk::Flags b)
{
  return a = a & b;
}
inline SplatSetVk::Flags operator~(SplatSetVk::Flags a)
{
  return static_cast<SplatSetVk::Flags>(~static_cast<uint32_t>(a));
}

}  // namespace vk_gaussian_splatting

#endif
