/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <nvvk/acceleration_structures.hpp>

#include "splat_set.h"
#include "utilities.h"

namespace vk_gaussian_splatting {

// Format and upload 3DGS data to VRAM
class SplatSetVk
{
public:
  SplatSetVk()
      : transform(1.0f)         // identity
      , transformInverse(1.0f)  // identity
  {
    computeTransform(scale, rotation, translation, transform, transformInverse);
  };

  ~SplatSetVk(void) {}

  void init(nvapp::Application*                                 app,
            nvvk::ResourceAllocator*                            alloc,
            nvvk::StagingUploader*                              uploader,
            VkSampler*                                          sampler,
            PhysicalDeviceInfo*                                 deviceInfo,
            VkPhysicalDeviceAccelerationStructurePropertiesKHR* accelStructProps)
  {
    m_app        = app;
    m_alloc      = alloc;
    m_uploader   = uploader;
    m_sampler    = sampler;
    m_deviceInfo = deviceInfo;
    rtAccelerationStructures.init(m_alloc, m_uploader, m_app->getQueue(0), 2000, 2000);
  }

  void deinit()
  {
    rtAccelerationStructures.deinit();
    m_app        = nullptr;
    m_alloc      = nullptr;
    m_uploader   = nullptr;
    m_sampler    = nullptr;
    m_deviceInfo = nullptr;
    rtxValid     = false;
  }

  void resetTransform()
  {
    translation = {0.0f, 0.0f, 0.0f};
    rotation    = {0.0f, 0.0f, 0.0f};
    scale       = {1.0f, 1.0f, 1.0f};  // INRIA models comes inverted
    computeTransform(scale, rotation, translation, transform, transformInverse);
  }

  // uploads the splatSet into VRAM
  // storage in [STORAGE_BUFFERS, STORAGE_TEXTURES]
  // format in  [FORMAT_FLOAT32, FORMAT_FLOAT16, FORMAT_UINT8]
  void initDataStorage(SplatSet& splatSet, uint32_t storage, uint32_t format);

  // destroy all buffers from VRAM
  // a new initDataStorage can be invoked afterward
  void deinitDataStorage();

  // initDataStorage must be invoked prior to creation of the splat model
  void rtxInitSplatModel(SplatSet& splatSet, bool useInstances, bool useAABBs, bool compressBlas, int kernelDegree, float kernelMinResponse, bool kernelAdaptiveClamping);

  void rtxDeinitSplatModel()
  {
    m_alloc->destroyBuffer(m_splatModel.vertexBuffer);
    m_alloc->destroyBuffer(m_splatModel.indexBuffer);
    m_alloc->destroyBuffer(m_splatModel.aabbBuffer);
    rtxValid = false;
  }

  // rtxInitSplatModel must be invoked prior to creation of acceleration structure
  void rtxInitAccelerationStructures(SplatSet& splatSet);

  void rtxDeinitAccelerationStructures()
  {
    rtAccelerationStructures.deinitAccelerationStructures();
    rtxValid = false;
  }

  // reset the memory usage stats
  inline void resetMemoryStats() { memoryStats = {}; }

public:
  // Model transformations

  glm::vec3 translation{0.0f};
  glm::vec3 rotation{0.0f};
  glm::vec3 scale{1.0f, 1.0f, 1.0f};
  glm::mat4 transform{};                // transformation matrix of the model
  glm::mat4 transformInverse{};         // inverseTransformation matrix of the model

  //
  VkSampler* m_sampler = nullptr;

  // Data textures

  nvvk::Image centersMap;
  nvvk::Image scalesMap;       // RTX specific
  nvvk::Image rotationsMap;    // RTX specific
  nvvk::Image covariancesMap;  // Raster specific
  nvvk::Image colorsMap;
  nvvk::Image sphericalHarmonicsMap;

  // Data buffers

  nvvk::Buffer centersBuffer;
  nvvk::Buffer scalesBuffer;       // RTX specific
  nvvk::Buffer rotationsBuffer;    // RTX specific
  nvvk::Buffer covariancesBuffer;  // Raster specific
  nvvk::Buffer colorsBuffer;
  nvvk::Buffer sphericalHarmonicsBuffer;

  ////////////////////////
  // Ray tracing specifics

  // The Splat model (contains only one splat if instanced or all splats)
  struct SplatModel
  {
    // Icosa related

    uint32_t     nbVertices{0};  // total number of vec3 vertex positions stored in vertexBuffer
    uint32_t     nbIndices{0};   // total number of vertex indices stored in vertexBuffer
    nvvk::Buffer vertexBuffer;   // Device buffer of the vertices
    nvvk::Buffer indexBuffer;    // Device buffer of the indices forming triangles

    // AABB related
    uint32_t     nbAABB;
    nvvk::Buffer aabbBuffer;

  } m_splatModel;

  // aligned with VkAabbPositionsKHR
  struct SplatAabb
  {
    glm::vec3 minimum;
    glm::vec3 maximum;
  };

  nvvk::AccelerationStructureHelper rtAccelerationStructures;  // provides access to BLAS and TLAS

  // data storage memory usage statistics
  struct ModelMemoryStats
  {
    // Memory footprint on host memory

    uint32_t srcAll     = 0;  // RAM bytes used for all the data of source model
    uint32_t srcCenters = 0;  // RAM bytes used for splat centers of source model
    // covariance
    uint32_t srcCov = 0;
    // spherical harmonics coeficients
    uint32_t srcShAll   = 0;  // RAM bytes used for all the SH coefs of source model
    uint32_t srcSh0     = 0;  // RAM bytes used for SH degree 0 of source model
    uint32_t srcShOther = 0;  // RAM bytes used for SH degree 1 of source model

    // Memory footprint on device memory (allocated)

    uint32_t devAll     = 0;  // GRAM bytes used for all the data of source model
    uint32_t devCenters = 0;  // GRAM bytes used for splat centers of source model
    // covariance
    uint32_t devCov = 0;
    // spherical harmonics coeficients
    uint32_t devShAll   = 0;  // GRAM bytes used for all the SH coefs of source model
    uint32_t devSh0     = 0;  // GRAM bytes used for SH degree 0 of source model
    uint32_t devShOther = 0;  // GRAM bytes used for SH degree 1 of source model

    // Actual data size within textures (a.k.a. mem footprint minus padding and
    // eventual unused components)

    uint32_t odevAll     = 0;  // GRAM bytes used for all the data of source model
    uint32_t odevCenters = 0;  // GRAM bytes used for splat centers of source model
    // covariance
    uint32_t odevCov = 0;
    // spherical harmonics coeficients
    uint32_t odevShAll   = 0;  // GRAM bytes used for all the SH coefs of source model
    uint32_t odevSh0     = 0;  // GRAM bytes used for SH degree 0 of source model
    uint32_t odevShOther = 0;  // GRAM bytes used for SH degree 1 of source model
  } memoryStats;

  // Is RTX valid
  bool rtxValid = false;  // This flag is set to false if some rtx AS allocations failed or before init
  // RTX AS memory stats
  uint32_t tlasSizeBytes;  // Size of the TLAS in VRAM in bytes
  uint32_t blasSizeBytes;  // Size of the BLAS in VRAM in bytes

private:
  // create the buffers on the device and upload
  // the splat set data from host to device
  void initDataBuffers(SplatSet& splatSet);

  // release buffers
  void deinitDataBuffers(void);

  // create the texture maps on the device and upload
  // the splat set data from host to device
  void initDataTextures(SplatSet& splatSet);

  // release textures
  void deinitDataTextures(void);

  // Create texture, upload data and assign sampler
  // sampler will be released by deinitTexture
  void initTexture(uint32_t width, uint32_t height, uint32_t bufsize, void* data, VkFormat format, const VkSampler& sampler, nvvk::Image& texture);

  // Destroy texture at once, texture must not be in use
  void deinitTexture(nvvk::Image& texture);

private:
  // RTX specifics

  void rtxCreateSplatIcosahedron(std::vector<glm::vec3>& vertices,
                                 std::vector<uint32_t>&  indices,
                                 std::vector<SplatAabb>& aabbs,
                                 glm::mat4               transform = glm::mat4(1.0));

  glm::mat4 rtxComputeTransformMatrix(SplatSet& splatSet, uint64_t splatIdx);

  nvvk::AccelerationStructureGeometryInfo rtxCreateSplatModelAccelerationStructureGeometryInfo();

private:
  uint32_t m_storage{};
  uint32_t m_format{};

  bool  m_rtxUseAABBs;
  bool  m_rtxUseInstances;
  bool  m_rtxCompressBlas;
  int   m_rtxKernelDegree;
  float m_rtxKernelMinResponse;
  bool  m_rtxKernelAdaptiveClamping;

  nvapp::Application*      m_app        = nullptr;
  nvvk::ResourceAllocator* m_alloc      = nullptr;
  nvvk::StagingUploader*   m_uploader   = nullptr;
  PhysicalDeviceInfo*      m_deviceInfo = nullptr;
};

}  // namespace vk_gaussian_splatting

#endif
