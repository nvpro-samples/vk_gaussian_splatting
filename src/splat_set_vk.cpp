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

#include "splat_set_vk.h"
#include "shaderio.h"
#include "utilities.h"
#include "memory_monitor_vk.h"

#include <vulkan/vk_enum_string_helper.h>  // For string_VkResult
#include <iostream>
#include <chrono>

// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/default_structs.hpp>

namespace vk_gaussian_splatting {

// Utility function to compute the texture size according to the size of the data to be stored
// By default use map of 4K Width and 1K height then adjust the height according to the data size
glm::ivec2 computeDataTextureSize(int elementsPerTexel, int elementsPerSplat, int maxSplatCount, glm::ivec2 texSize = {4096, 1024})
{
  while(texSize.x * texSize.y * elementsPerTexel < maxSplatCount * elementsPerSplat)
    texSize.y *= 2;
  return texSize;
};

// Check if the required texture size for a splat set would exceed device limits
// Returns the estimated required texture height (width is fixed at 4096)
uint32_t computeMaxDataTextureSize(uint32_t splatCount, uint32_t shDegree)
{
  // Estimate required texture height for the largest texture (spherical harmonics)
  int sphericalHarmonicsComponentCount = 15;  // SH degree 0-1 (default)
  if(shDegree == 2)
    sphericalHarmonicsComponentCount = 24;
  else if(shDegree == 3)
    sphericalHarmonicsComponentCount = 45;

  // Round up to multiple of 4
  int paddedCount = (sphericalHarmonicsComponentCount + 3) & ~3;

  // Estimate required height: we need (splatCount * paddedCount) / (4096 * 4) rows
  // Start with 1024 and keep doubling
  uint32_t estimatedHeight = 1024;
  int64_t  requiredTexels  = static_cast<int64_t>(splatCount) * paddedCount / 4;  // 4 elements per texel
  int64_t  availableTexels = 4096LL * estimatedHeight;

  while(availableTexels < requiredTexels)
  {
    estimatedHeight *= 2;
    availableTexels = 4096LL * estimatedHeight;
  }

  return estimatedHeight;
}

// quantize a float onto a uint8
uint8_t toUint8(float v, float rangeMin, float rangeMax)
{
  float normalized = (v - rangeMin) / (rangeMax - rangeMin);
  return static_cast<uint8_t>(std::clamp(std::round(normalized * 255.0f), 0.0f, 255.0f));
};

// Returns the size in bytes for a given format enum
int formatSize(uint32_t format)
{
  if(format == FORMAT_FLOAT32)
    return 4;
  if(format == FORMAT_FLOAT16)
    return 2;
  if(format == FORMAT_UINT8)
    return 1;
  return 0;
}

// convert SH coef to given format on the flight and store into dstBuffer
void storeSh(int format, float* srcBuffer, uint64_t srcIndex, void* dstBuffer, uint64_t dstIndex)
{
  if(format == FORMAT_FLOAT32)
    static_cast<float*>(dstBuffer)[dstIndex] = srcBuffer[srcIndex];
  else if(format == FORMAT_FLOAT16)
    static_cast<uint16_t*>(dstBuffer)[dstIndex] = glm::packHalf1x16(srcBuffer[srcIndex]);
  else if(format == FORMAT_UINT8)
    static_cast<uint8_t*>(dstBuffer)[dstIndex] = toUint8(srcBuffer[srcIndex], -1., 1.);
}

///////////////////
// class definition

void SplatSetVk::initDataStorage(uint32_t _shFormat, uint32_t _rgbaFormat)
{
  // store the parameters for further usage
  shFormat   = _shFormat;
  rgbaFormat = _rgbaFormat;

  // Store metadata (from inherited SplatSet data)
  splatCount = static_cast<uint32_t>(size());
  shDegree   = static_cast<uint32_t>(maxShDegree());

  // Initialize default material: fully emissive, no lighting interaction
  splatMaterial.ambient       = glm::vec3(0.0f);
  splatMaterial.diffuse       = glm::vec3(0.0f);
  splatMaterial.specular      = glm::vec3(0.0f);
  splatMaterial.transmittance = glm::vec3(0.0f);
  splatMaterial.emission      = glm::vec3(1.0f);  // Fully emissive
  splatMaterial.shininess     = 0.0f;
  splatMaterial.ior           = 1.0f;
  splatMaterial.illum         = 0;

  // Note: Material is now stored per-instance in SplatSetInstanceVk::descriptor
  // This splatMaterial serves as the default for new instances

  if(dataStorage == STORAGE_BUFFERS)
  {
    initDataBuffers();
  }
  else if(dataStorage == STORAGE_TEXTURES)
  {
    // Check if scene is too large for texture storage mode
    if(m_deviceInfo)
    {
      uint32_t maxTextureDim   = m_deviceInfo->properties10.limits.maxImageDimension2D;
      uint32_t estimatedHeight = computeMaxDataTextureSize(splatCount, shDegree);

      if(estimatedHeight > maxTextureDim)
      {
        LOGE("Failed to create texture storage for splat set (index=%zu): Scene too large for texture mode\n", index);
        LOGE("  Splat count: %.2f M, SH degree: %u, Estimated height needed: %u, Device max: %u\n",
             splatCount / 1000000.0, shDegree, estimatedHeight, maxTextureDim);
        LOGW("  Falling back to BUFFER storage mode.\n");

        // Fall back to buffer storage
        dataStorage = STORAGE_BUFFERS;
        initDataBuffers();
        return;
      }
    }

    initDataTextures();
  }
  else
    LOGE("Invalid storage format");
}

void SplatSetVk::deinitDataStorage()
{
  // Note: Material is now stored per-instance, no buffer to destroy here

  // Unconditionally deinit both buffers and textures.
  // This is safe because nvvk::LargeBuffer/Image default-initialize to null,
  // and destroyLargeBuffer/destroyImage check for null handles.
  // This prevents leaks if dataStorage changed between init and deinit
  // (e.g., UI toggled from textures to buffers, or fallback occurred).
  deinitDataBuffers();
  deinitDataTextures();
}

///////////////////
// using data buffers to store splatset in VRAM

void SplatSetVk::initDataBuffers()
{
  if(false)
  {
    // dump splat info for debug
    uint32_t splatId = 4178424;
    if(splatId < size())
    {

      std::cout << positions[splatId * 3 + 0] << " ";
      std::cout << positions[splatId * 3 + 1] << " ";
      std::cout << positions[splatId * 3 + 2] << "  0 1 0  ";
      std::cout << f_dc[splatId * 3 + 0] << " ";
      std::cout << f_dc[splatId * 3 + 1] << " ";
      std::cout << f_dc[splatId * 3 + 2] << "  ";
      for(int i = 0; i < 45; ++i)
      {
        std::cout << f_rest[splatId * 45 + i] << " ";
      }
      std::cout << " " << opacity[splatId] << "  ";
      std::cout << scale[splatId * 3 + 0] << " ";
      std::cout << scale[splatId * 3 + 1] << " ";
      std::cout << scale[splatId * 3 + 2] << "  ";
      std::cout << rotation[splatId * 4 + 0] << " ";
      std::cout << rotation[splatId * 4 + 1] << " ";
      std::cout << rotation[splatId * 4 + 2] << " ";
      std::cout << rotation[splatId * 4 + 3] << std::endl;
    }
  }
  auto       startTime  = std::chrono::high_resolution_clock::now();
  const auto splatCount = (uint32_t)positions.size() / 3;

  // Queue for sparse binding (required for LargeBuffer)
  VkQueue queue = m_app->getQueue(0).queue;

  // Device buffer usage flags
  VkBufferUsageFlagBits2 deviceBufferUsageFlags =
      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
      | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT;

  // Centers and Scales (scales and rotations are only for raytrace, raster
  // uses pre-computed covariances, see covariance section hereafter)
  {
    const VkDeviceSize bufferSize3Comp = VkDeviceSize(splatCount) * 3 * sizeof(float);
    const VkDeviceSize bufferSize4Comp = VkDeviceSize(splatCount) * 4 * sizeof(float);

    std::cout << "Allocating splat buffers: centers/scales=" << bufferSize3Comp << " B, rotations=" << bufferSize4Comp
              << " B (" << splatCount << " splats)" << std::endl;

    NVVK_CHECK(m_alloc->createLargeBuffer(centersBuffer, bufferSize3Comp, deviceBufferUsageFlags, queue));
    NVVK_DBG_NAME(centersBuffer.buffer);
    NVVK_CHECK(m_alloc->createLargeBuffer(scalesBuffer, bufferSize3Comp, deviceBufferUsageFlags, queue));
    NVVK_DBG_NAME(scalesBuffer.buffer);
    NVVK_CHECK(m_alloc->createLargeBuffer(rotationsBuffer, bufferSize4Comp, deviceBufferUsageFlags, queue));
    NVVK_DBG_NAME(rotationsBuffer.buffer);

    NVVK_CHECK(m_uploader->appendLargeBuffer(centersBuffer, 0, bufferSize3Comp, positions.data()));
    NVVK_CHECK(m_uploader->appendLargeBuffer(scalesBuffer, 0, bufferSize3Comp, scale.data()));
    NVVK_CHECK(m_uploader->appendLargeBuffer(rotationsBuffer, 0, bufferSize4Comp, rotation.data()));

    // memory statistics
    memoryStats.hostCenters        = bufferSize3Comp;
    memoryStats.deviceUsedCenters  = bufferSize3Comp;
    memoryStats.deviceAllocCenters = bufferSize3Comp;

    memoryStats.hostScales        = bufferSize3Comp;
    memoryStats.deviceUsedScales  = bufferSize3Comp;
    memoryStats.deviceAllocScales = bufferSize3Comp;

    memoryStats.hostRotations        = bufferSize4Comp;
    memoryStats.deviceUsedRotations  = bufferSize4Comp;
    memoryStats.deviceAllocRotations = bufferSize4Comp;
  }

  // covariances (for raster only)
  {
    const VkDeviceSize bufferSize = VkDeviceSize(splatCount) * 2 * 3 * sizeof(float);

    std::cout << "Allocating covariance buffers: " << bufferSize << " B (" << splatCount << " splats)" << std::endl;

    NVVK_CHECK(m_alloc->createLargeBuffer(covariancesBuffer, bufferSize, deviceBufferUsageFlags, queue));
    NVVK_DBG_NAME(covariancesBuffer.buffer);

    // Compute covariances into temporary CPU buffer
    std::vector<float> covData(size_t(splatCount) * 6);

    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto stride3 = splatIdx * 3;
      const auto stride4 = splatIdx * 4;
      const auto stride6 = splatIdx * 6;
      glm::vec3  scl{std::exp(scale[stride3 + 0]), std::exp(scale[stride3 + 1]), std::exp(scale[stride3 + 2])};

      glm::quat rot{rotation[stride4 + 0], rotation[stride4 + 1], rotation[stride4 + 2], rotation[stride4 + 3]};
      rot = glm::normalize(rot);

      const glm::mat3 scaleMatrix           = glm::mat3(glm::scale(scl));
      const glm::mat3 rotationMatrix        = glm::mat3_cast(rot);
      const glm::mat3 covarianceMatrix      = rotationMatrix * scaleMatrix;
      glm::mat3       transformedCovariance = covarianceMatrix * glm::transpose(covarianceMatrix);

      covData[stride6 + 0] = glm::value_ptr(transformedCovariance)[0];
      covData[stride6 + 1] = glm::value_ptr(transformedCovariance)[3];
      covData[stride6 + 2] = glm::value_ptr(transformedCovariance)[6];

      covData[stride6 + 3] = glm::value_ptr(transformedCovariance)[4];
      covData[stride6 + 4] = glm::value_ptr(transformedCovariance)[7];
      covData[stride6 + 5] = glm::value_ptr(transformedCovariance)[8];
    }
    END_PAR_LOOP();

    NVVK_CHECK(m_uploader->appendLargeBuffer(covariancesBuffer, 0, bufferSize, covData.data()));

    // memory statistics
    memoryStats.hostCov        = (splatCount * (4 + 3)) * sizeof(float);
    memoryStats.deviceUsedCov  = bufferSize;
    memoryStats.deviceAllocCov = bufferSize;
  }

  // Colors. SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  {
    const VkDeviceSize bufferSize = VkDeviceSize(splatCount) * 4 * formatSize(rgbaFormat);

    std::cout << "Allocating color buffers: " << bufferSize << " B (" << splatCount << " splats, format=" << rgbaFormat << ")" << std::endl;

    NVVK_CHECK(m_alloc->createLargeBuffer(colorsBuffer, bufferSize, deviceBufferUsageFlags, queue));
    NVVK_DBG_NAME(colorsBuffer.buffer);

    // Compute colors into temporary CPU buffer
    std::vector<uint8_t> colorData(bufferSize);
    void*                colorPtr = colorData.data();

    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto  stride3 = splatIdx * 3;
      const auto  stride4 = splatIdx * 4;
      const float SH_C0   = 0.28209479177387814f;
      const float r       = glm::clamp(0.5f + SH_C0 * f_dc[stride3 + 0], 0.0f, 1.0f);
      const float g       = glm::clamp(0.5f + SH_C0 * f_dc[stride3 + 1], 0.0f, 1.0f);
      const float b       = glm::clamp(0.5f + SH_C0 * f_dc[stride3 + 2], 0.0f, 1.0f);
      const float a       = glm::clamp(1.0f / (1.0f + std::exp(-opacity[splatIdx])), 0.0f, 1.0f);

      if(rgbaFormat == FORMAT_FLOAT32)
      {
        static_cast<float*>(colorPtr)[stride4 + 0] = r;
        static_cast<float*>(colorPtr)[stride4 + 1] = g;
        static_cast<float*>(colorPtr)[stride4 + 2] = b;
        static_cast<float*>(colorPtr)[stride4 + 3] = a;
      }
      else if(rgbaFormat == FORMAT_FLOAT16)
      {
        static_cast<uint16_t*>(colorPtr)[stride4 + 0] = glm::packHalf1x16(r);
        static_cast<uint16_t*>(colorPtr)[stride4 + 1] = glm::packHalf1x16(g);
        static_cast<uint16_t*>(colorPtr)[stride4 + 2] = glm::packHalf1x16(b);
        static_cast<uint16_t*>(colorPtr)[stride4 + 3] = glm::packHalf1x16(a);
      }
      else if(rgbaFormat == FORMAT_UINT8)
      {
        static_cast<uint8_t*>(colorPtr)[stride4 + 0] = toUint8(r, 0.f, 1.f);
        static_cast<uint8_t*>(colorPtr)[stride4 + 1] = toUint8(g, 0.f, 1.f);
        static_cast<uint8_t*>(colorPtr)[stride4 + 2] = toUint8(b, 0.f, 1.f);
        static_cast<uint8_t*>(colorPtr)[stride4 + 3] = toUint8(a, 0.f, 1.f);
      }
    }
    END_PAR_LOOP()

    NVVK_CHECK(m_uploader->appendLargeBuffer(colorsBuffer, 0, bufferSize, colorData.data()));

    // memory statistics
    memoryStats.hostSh0        = splatCount * 4 * sizeof(float);  // original data is always float
    memoryStats.deviceUsedSh0  = bufferSize;
    memoryStats.deviceAllocSh0 = bufferSize;
  }

  // Spherical harmonics of degree 1 to 3
  if(!f_rest.empty())
  {
    const uint32_t totalSphericalHarmonicsComponentCount    = (uint32_t)f_rest.size() / splatCount;
    const uint32_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    int sphericalHarmonicsDegree = 0;
    int splatStride              = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
    {
      sphericalHarmonicsDegree = 1;
      splatStride += 3 * 3;
    }
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
    {
      sphericalHarmonicsDegree = 2;
      splatStride += 5 * 3;
    }
    if(sphericalHarmonicsCoefficientsPerChannel == 15)
    {
      sphericalHarmonicsDegree = 3;
      splatStride += 7 * 3;
    }

    // same for the time beeing, would be less if we do not upload all src degrees
    int targetSplatStride = splatStride;

    const VkDeviceSize bufferSize = VkDeviceSize(splatCount) * splatStride * formatSize(shFormat);

    std::cout << "Allocating SH buffers: " << bufferSize << " B (" << splatCount << " splats, stride=" << splatStride << ")" << std::endl;

    NVVK_CHECK(m_alloc->createLargeBuffer(sphericalHarmonicsBuffer, bufferSize, deviceBufferUsageFlags, queue));
    NVVK_DBG_NAME(sphericalHarmonicsBuffer.buffer);

    // Compute SH data into temporary CPU buffer (bufferSize bytes, element size depends on shFormat)
    std::vector<uint8_t> shData(bufferSize);
    void*                shPtr = shData.data();

    auto startShTime = std::chrono::high_resolution_clock::now();

    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto srcBase   = splatStride * splatIdx;
      const auto destBase  = targetSplatStride * splatIdx;
      int        dstOffset = 0;
      // degree 1, three coefs per component
      for(auto i = 0; i < 3; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + i);
          const auto dstIndex = destBase + dstOffset++;

          storeSh(shFormat, f_rest.data(), srcIndex, shPtr, dstIndex);
        }
      }
      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const auto dstIndex = destBase + dstOffset++;

          storeSh(shFormat, f_rest.data(), srcIndex, shPtr, dstIndex);
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const auto dstIndex = destBase + dstOffset++;

          storeSh(shFormat, f_rest.data(), srcIndex, shPtr, dstIndex);
        }
      }
    }
    END_PAR_LOOP()

    auto      endShTime   = std::chrono::high_resolution_clock::now();
    long long buildShTime = std::chrono::duration_cast<std::chrono::milliseconds>(endShTime - startShTime).count();
    std::cout << "Sh data updated in " << buildShTime << "ms" << std::endl;

    NVVK_CHECK(m_uploader->appendLargeBuffer(sphericalHarmonicsBuffer, 0, bufferSize, shData.data()));

    // memory statistics
    memoryStats.hostShOther        = (uint32_t)f_rest.size() * sizeof(float);
    memoryStats.deviceUsedShOther  = bufferSize;
    memoryStats.deviceAllocShOther = bufferSize;
  }

  // Record all staged uploads and submit
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  m_uploader->cmdUploadAppended(cmd);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);

  m_app->submitAndWaitTempCmdBuffer(cmd);
  m_uploader->releaseStaging();

  // update statistics totals
  memoryStats.hostShAll        = memoryStats.hostSh0 + memoryStats.hostShOther;
  memoryStats.deviceUsedShAll  = memoryStats.deviceUsedSh0 + memoryStats.deviceUsedShOther;
  memoryStats.deviceAllocShAll = memoryStats.deviceAllocSh0 + memoryStats.deviceAllocShOther;

  memoryStats.hostAll = memoryStats.hostCenters + memoryStats.hostScales + memoryStats.hostRotations
                        + memoryStats.hostCov + memoryStats.hostShAll;
  memoryStats.deviceUsedAll = memoryStats.deviceUsedCenters + memoryStats.deviceUsedScales
                              + memoryStats.deviceUsedRotations + memoryStats.deviceUsedCov + memoryStats.deviceUsedShAll;
  memoryStats.deviceAllocAll = memoryStats.deviceAllocCenters + memoryStats.deviceAllocScales + memoryStats.deviceAllocRotations
                               + memoryStats.deviceAllocCov + memoryStats.deviceAllocShAll;

  // Note: Descriptor initialization removed - now handled per-instance by SplatSetInstanceVk::rebuildDescriptor()

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Data buffers updated in " << buildTime << "ms" << std::endl;
}

void SplatSetVk::deinitDataBuffers()
{
  if(centersBuffer.buffer)
    m_alloc->destroyLargeBuffer(centersBuffer);
  if(scalesBuffer.buffer)
    m_alloc->destroyLargeBuffer(scalesBuffer);
  if(rotationsBuffer.buffer)
    m_alloc->destroyLargeBuffer(rotationsBuffer);
  if(colorsBuffer.buffer)
    m_alloc->destroyLargeBuffer(colorsBuffer);
  if(covariancesBuffer.buffer)
    m_alloc->destroyLargeBuffer(covariancesBuffer);
  if(sphericalHarmonicsBuffer.buffer)
    m_alloc->destroyLargeBuffer(sphericalHarmonicsBuffer);

  centersBuffer            = {};
  scalesBuffer             = {};
  rotationsBuffer          = {};
  colorsBuffer             = {};
  covariancesBuffer        = {};
  sphericalHarmonicsBuffer = {};
}

///////////////////
// using texture maps to store splatset in VRAM

void SplatSetVk::initDataTextures()
{
  auto startTime = std::chrono::high_resolution_clock::now();

  const auto splatCount = (uint32_t)positions.size() / 3;

  // centers (3 components but texture map is only allowed with 4 components)
  // TODO: May pack as done for covariances not to waste alpha chanel ? but must
  // compare performance (1 lookup vs 2 lookups due to packing)
  {
    glm::ivec2         centersMapSize = computeDataTextureSize(3, 3, splatCount);
    std::vector<float> centers(centersMapSize.x * centersMapSize.y * 4);  // includes some padding and unused w channel

    glm::ivec2         scalesMapSize = computeDataTextureSize(3, 3, splatCount);
    std::vector<float> scales(scalesMapSize.x * scalesMapSize.y * 4);  // includes some padding and unused w channel

    glm::ivec2         rotationsMapSize = computeDataTextureSize(4, 4, splatCount);
    std::vector<float> rotations(rotation);
    rotations.resize(rotationsMapSize.x * rotationsMapSize.y * 4);  // includes some padding

    //for(uint32_t i = 0; i < splatCount; ++i)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      // we skip the alpha channel that is left undefined and not used in the shader
      for(uint32_t cmp = 0; cmp < 3; ++cmp)
      {
        centers[splatIdx * 4 + cmp] = positions[splatIdx * 3 + cmp];
        scales[splatIdx * 4 + cmp]  = scale[splatIdx * 3 + cmp];
      }
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    initTexture(centersMapSize.x, centersMapSize.y, (uint32_t)centers.size() * sizeof(float), (void*)centers.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, centersMap);

    initTexture(scalesMapSize.x, scalesMapSize.y, (uint32_t)scales.size() * sizeof(float), (void*)scales.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, scalesMap);

    initTexture(rotationsMapSize.x, rotationsMapSize.y, (uint32_t)rotations.size() * sizeof(float),
                (void*)rotations.data(), VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, rotationsMap);

    // memory statistics
    memoryStats.hostCenters        = splatCount * 3 * sizeof(float);
    memoryStats.deviceUsedCenters  = splatCount * 3 * sizeof(float);  // no compression or quantization yet
    memoryStats.deviceAllocCenters = centersMapSize.x * centersMapSize.y * 4 * sizeof(float);

    memoryStats.hostScales        = splatCount * 3 * sizeof(float);
    memoryStats.deviceUsedScales  = splatCount * 3 * sizeof(float);
    memoryStats.deviceAllocScales = scalesMapSize.x * scalesMapSize.y * 4 * sizeof(float);

    memoryStats.hostRotations        = splatCount * 4 * sizeof(float);
    memoryStats.deviceUsedRotations  = splatCount * 4 * sizeof(float);
    memoryStats.deviceAllocRotations = rotationsMapSize.x * rotationsMapSize.y * 4 * sizeof(float);
  }
  // covariances
  {
    glm::ivec2         mapSize = computeDataTextureSize(4, 6, splatCount);
    std::vector<float> covariances(mapSize.x * mapSize.y * 4, 0.0f);
    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto stride3 = splatIdx * 3;
      const auto stride4 = splatIdx * 4;
      const auto stride6 = splatIdx * 6;
      glm::vec3  scl{std::exp(scale[stride3 + 0]), std::exp(scale[stride3 + 1]), std::exp(scale[stride3 + 2])};

      glm::quat rot{rotation[stride4 + 0], rotation[stride4 + 1], rotation[stride4 + 2], rotation[stride4 + 3]};
      rot = glm::normalize(rot);

      // computes the covariance
      const glm::mat3 scaleMatrix           = glm::mat3(glm::scale(scl));
      const glm::mat3 rotationMatrix        = glm::mat3_cast(rot);  // where rotation is a quaternion
      const glm::mat3 covarianceMatrix      = rotationMatrix * scaleMatrix;
      glm::mat3       transformedCovariance = covarianceMatrix * glm::transpose(covarianceMatrix);

      covariances[stride6 + 0] = glm::value_ptr(transformedCovariance)[0];
      covariances[stride6 + 1] = glm::value_ptr(transformedCovariance)[3];
      covariances[stride6 + 2] = glm::value_ptr(transformedCovariance)[6];

      covariances[stride6 + 3] = glm::value_ptr(transformedCovariance)[4];
      covariances[stride6 + 4] = glm::value_ptr(transformedCovariance)[7];
      covariances[stride6 + 5] = glm::value_ptr(transformedCovariance)[8];
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    initTexture(mapSize.x, mapSize.y, (uint32_t)covariances.size() * sizeof(float), (void*)covariances.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, covariancesMap);
    // memory statistics
    memoryStats.hostCov        = (splatCount * (4 + 3)) * sizeof(float);
    memoryStats.deviceUsedCov  = splatCount * 6 * sizeof(float);  // covariance takes less space than rotation + scale
    memoryStats.deviceAllocCov = mapSize.x * mapSize.y * 4 * sizeof(float);
  }
  // SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  {
    glm::ivec2           mapSize    = computeDataTextureSize(4, 4, splatCount);
    const uint32_t       elemSize   = formatSize(rgbaFormat);
    const uint32_t       bufferSize = mapSize.x * mapSize.y * 4 * elemSize;
    std::vector<uint8_t> colors(bufferSize, 0);
    void*                data = colors.data();
    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto  stride3 = splatIdx * 3;
      const auto  stride4 = splatIdx * 4;
      const float SH_C0   = 0.28209479177387814f;
      const float r       = glm::clamp(0.5f + SH_C0 * f_dc[stride3 + 0], 0.0f, 1.0f);
      const float g       = glm::clamp(0.5f + SH_C0 * f_dc[stride3 + 1], 0.0f, 1.0f);
      const float b       = glm::clamp(0.5f + SH_C0 * f_dc[stride3 + 2], 0.0f, 1.0f);
      const float a       = glm::clamp(1.0f / (1.0f + std::exp(-opacity[splatIdx])), 0.0f, 1.0f);

      if(rgbaFormat == FORMAT_FLOAT32)
      {
        static_cast<float*>(data)[stride4 + 0] = r;
        static_cast<float*>(data)[stride4 + 1] = g;
        static_cast<float*>(data)[stride4 + 2] = b;
        static_cast<float*>(data)[stride4 + 3] = a;
      }
      else if(rgbaFormat == FORMAT_FLOAT16)
      {
        static_cast<uint16_t*>(data)[stride4 + 0] = glm::packHalf1x16(r);
        static_cast<uint16_t*>(data)[stride4 + 1] = glm::packHalf1x16(g);
        static_cast<uint16_t*>(data)[stride4 + 2] = glm::packHalf1x16(b);
        static_cast<uint16_t*>(data)[stride4 + 3] = glm::packHalf1x16(a);
      }
      else if(rgbaFormat == FORMAT_UINT8)
      {
        static_cast<uint8_t*>(data)[stride4 + 0] = toUint8(r, 0.f, 1.f);
        static_cast<uint8_t*>(data)[stride4 + 1] = toUint8(g, 0.f, 1.f);
        static_cast<uint8_t*>(data)[stride4 + 2] = toUint8(b, 0.f, 1.f);
        static_cast<uint8_t*>(data)[stride4 + 3] = toUint8(a, 0.f, 1.f);
      }
    }
    END_PAR_LOOP()
    // place the result in the dedicated texture map
    VkFormat vkFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    if(rgbaFormat == FORMAT_FLOAT16)
      vkFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    else if(rgbaFormat == FORMAT_UINT8)
      vkFormat = VK_FORMAT_R8G8B8A8_UNORM;
    initTexture(mapSize.x, mapSize.y, bufferSize, data, vkFormat, *m_sampler, colorsMap);
    // memory statistics
    memoryStats.hostSh0        = splatCount * 4 * sizeof(float);  // original sh0 and opacity are floats
    memoryStats.deviceUsedSh0  = splatCount * 4 * elemSize;
    memoryStats.deviceAllocSh0 = bufferSize;
  }
  // Prepare the spherical harmonics of degree 1 to 3
  if(!f_rest.empty())
  {
    const uint32_t sphericalHarmonicsElementsPerTexel       = 4;
    const uint32_t totalSphericalHarmonicsComponentCount    = (uint32_t)f_rest.size() / splatCount;
    const uint32_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    int sphericalHarmonicsDegree = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
      sphericalHarmonicsDegree = 1;
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
      sphericalHarmonicsDegree = 2;
    if(sphericalHarmonicsCoefficientsPerChannel >= 15)
      sphericalHarmonicsDegree = 3;

    // add some padding at each splat if needed for easy texture lookups
    int sphericalHarmonicsComponentCount = 0;
    if(sphericalHarmonicsDegree == 1)
      sphericalHarmonicsComponentCount = 9;
    if(sphericalHarmonicsDegree == 2)
      sphericalHarmonicsComponentCount = 24;
    if(sphericalHarmonicsDegree == 3)
      sphericalHarmonicsComponentCount = 45;

    int paddedSphericalHarmonicsComponentCount = sphericalHarmonicsComponentCount;
    while(paddedSphericalHarmonicsComponentCount % 4 != 0)
      paddedSphericalHarmonicsComponentCount++;

    glm::ivec2 mapSize =
        computeDataTextureSize(sphericalHarmonicsElementsPerTexel, paddedSphericalHarmonicsComponentCount, splatCount);

    const uint32_t bufferSize = mapSize.x * mapSize.y * sphericalHarmonicsElementsPerTexel * formatSize(shFormat);

    std::vector<uint8_t> paddedSHArray(bufferSize, 0);

    void* data = (void*)paddedSHArray.data();

    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto srcBase   = totalSphericalHarmonicsComponentCount * splatIdx;
      const auto destBase  = paddedSphericalHarmonicsComponentCount * splatIdx;
      int        dstOffset = 0;
      // degree 1, three coefs per component
      for(auto i = 0; i < 3; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(shFormat, f_rest.data(), srcIndex, data, dstIndex);
        }
      }

      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(shFormat, f_rest.data(), srcIndex, data, dstIndex);
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(shFormat, f_rest.data(), srcIndex, data, dstIndex);
        }
      }
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    if(shFormat == FORMAT_FLOAT32)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R32G32B32A32_SFLOAT, *m_sampler, sphericalHarmonicsMap);
    }
    else if(shFormat == FORMAT_FLOAT16)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R16G16B16A16_SFLOAT, *m_sampler, sphericalHarmonicsMap);
    }
    else if(shFormat == FORMAT_UINT8)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R8G8B8A8_UNORM, *m_sampler, sphericalHarmonicsMap);
    }

    // memory statistics
    memoryStats.hostShOther        = (uint32_t)f_rest.size() * sizeof(float);
    memoryStats.deviceUsedShOther  = (uint32_t)f_rest.size() * formatSize(shFormat);
    memoryStats.deviceAllocShOther = bufferSize;
  }

  // update statistics totals
  memoryStats.hostShAll        = memoryStats.hostSh0 + memoryStats.hostShOther;
  memoryStats.deviceUsedShAll  = memoryStats.deviceUsedSh0 + memoryStats.deviceUsedShOther;
  memoryStats.deviceAllocShAll = memoryStats.deviceAllocSh0 + memoryStats.deviceAllocShOther;

  memoryStats.hostAll = memoryStats.hostCenters + memoryStats.hostScales + memoryStats.hostRotations
                        + memoryStats.hostCov + memoryStats.hostShAll;
  memoryStats.deviceUsedAll = memoryStats.deviceUsedCenters + memoryStats.deviceUsedScales
                              + memoryStats.deviceUsedRotations + memoryStats.deviceUsedCov + memoryStats.deviceUsedShAll;
  memoryStats.deviceAllocAll = memoryStats.deviceAllocCenters + memoryStats.deviceAllocScales + memoryStats.deviceAllocRotations
                               + memoryStats.deviceAllocCov + memoryStats.deviceAllocShAll;

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Data textures updated in " << buildTime << "ms" << std::endl;

  // Store texture indices for bindless texture array access
  // In single-instance mode, we use a simple sequential scheme starting from 0
  // TODO Use enums
  textureIndexCenters     = 0;
  textureIndexScales      = 1;
  textureIndexRotations   = 2;
  textureIndexColors      = 3;
  textureIndexCovariances = 4;
  textureIndexSH          = 5;
}

void SplatSetVk::deinitDataTextures()
{
  deinitTexture(centersMap);
  deinitTexture(scalesMap);
  deinitTexture(rotationsMap);
  deinitTexture(covariancesMap);

  deinitTexture(colorsMap);
  deinitTexture(sphericalHarmonicsMap);

  // Note: descriptorBuffer removed - now managed per-instance by SplatSetManagerVk
}

void SplatSetVk::initTexture(uint32_t width, uint32_t height, uint32_t bufsize, void* data, VkFormat format, const VkSampler& sampler, nvvk::Image& texture)
{
  const VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkImageCreateInfo createInfo = DEFAULT_VkImageCreateInfo;
  createInfo.mipLevels         = 1;
  createInfo.extent            = {width, height, 1};
  createInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  createInfo.format            = format;

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_alloc->createImage(texture, createInfo, DEFAULT_VkImageViewCreateInfo));
  NVVK_DBG_NAME(texture.image);
  NVVK_DBG_NAME(texture.descriptor.imageView);

  NVVK_CHECK(m_uploader->appendImage(texture, std::span<uint8_t>((uint8_t*)data, bufsize), imageLayout));
  m_uploader->cmdUploadAppended(cmd);

  texture.descriptor.sampler = sampler;

  m_app->submitAndWaitTempCmdBuffer(cmd);
  m_uploader->releaseStaging();
}

void SplatSetVk::deinitTexture(nvvk::Image& texture)
{
  m_alloc->destroyImage(texture);
}

/////////////// section related to RTX


// Comment from Paper code
//
// phi = golden ratio = (1 + sqrt(5)) / 2
// r = radius of the inscribed circle = 1 = (phi^2 * s) / ( 2 * sqrt(3))
// s = edge length = ( 2 * sqrt(3) ) / phi^2
// V = (5/12) * ( 3 + sqrt(5) ) * s^3 = 8.0
constexpr float goldenRatio   = 1.618033988749895f;  // cast to float discards a bit of precision
constexpr float icosaEdge     = 1.323169076499215f;  // cast to float discards a bit of precision
constexpr float icosaVrtScale = 0.5 * icosaEdge;

float kernelScale(float density, float modulatedMinResponse, float kernelDegree, bool adaptiveClamping)
{
  const float responseModulation = adaptiveClamping ? density : 1.0f;
  const float minResponse        = fminf(modulatedMinResponse / responseModulation, 0.97f);

  // linear kernel
  if(kernelDegree == 0)
  {
    return ((1.0f - minResponse) / 3.0f) / -0.329630334487f;
  }

  /// generalized gaussian of degree b : scaling a = -4.5/3^b
  /// e^{a*|x|^b}
  const float b = kernelDegree;
  const float a = -4.5f / powf(3.0f, static_cast<float>(b));
  /// find distance r (>0) st e^{a*r^b} = minResponse
  /// TODO : reshuffle the math to call powf only once
  return powf(logf(minResponse) / a, 1.0f / b);
}

glm::mat4 SplatSetVk::rtxComputeTransformMatrix(uint64_t splatIdx)
{
  const auto stride3 = splatIdx * 3;
  const auto stride4 = splatIdx * 4;

  // compute the transformation matrix
  glm::vec3 scl{std::exp(scale[stride3 + 0]), std::exp(scale[stride3 + 1]), std::exp(scale[stride3 + 2])};

  glm::quat rot{rotation[stride4 + 0], rotation[stride4 + 1], rotation[stride4 + 2], rotation[stride4 + 3]};
  rot = glm::normalize(rot);

  glm::vec3 position{positions[stride3 + 0], positions[stride3 + 1], positions[stride3 + 2]};

  const float density = 1.0f / (1.0f + std::exp(-opacity[splatIdx]));

  const float kerScale = kernelScale(density, m_rtxKernelMinResponse, float(m_rtxKernelDegree), m_rtxKernelAdaptiveClamping);

  const glm::vec3 totalScale = scl * icosaVrtScale * kerScale;

  const glm::mat4 translateMatrix = glm::translate(position);
  const glm::mat4 scaleMatrix     = glm::scale(totalScale);
  const glm::mat4 rotationMatrix  = glm::mat4_cast(rot);  // where rotation is a quaternion

  // Note: This is the LOCAL splat transform. Instance transform is applied by caller in rtxInitAccelerationStructures
  return translateMatrix * rotationMatrix * scaleMatrix;
}

// transformed unit regular icosahedron
void SplatSetVk::rtxCreateSplatIcosahedron(uint64_t                offset,
                                           std::vector<glm::vec3>& vertices,
                                           std::vector<uint32_t>&  indices,
                                           std::vector<SplatAabb>& aabbs,
                                           glm::mat4               transform)  // = glm::mat4(1.0))
{
  // Golden ratio for icosahedron
  const float t = (1.0f + sqrt(5.0f)) / 2.0f;

  // Define the 12 vertices of an icosahedron
  static const std::vector<glm::vec4> s_vertices = {{-1, t, 0, 1},  {1, t, 0, 1},   {0, 1, -t, 1}, {-t, 0, -1, 1},
                                                    {-t, 0, 1, 1},  {0, 1, t, 1},   {t, 0, 1, 1},  {0, -1, t, 1},
                                                    {-1, -t, 0, 1}, {0, -1, -t, 1}, {t, 0, -1, 1}, {1, -t, 0, 1}};

  // Define the 20 triangular faces by vertex indices
  static const std::vector<uint32_t> s_indices = {0,  1, 2, 0,  2, 3,  0,  3, 4,  0, 4,  5, 0,  5,  1, 6, 1, 5, 6,  5,
                                                  7,  6, 7, 11, 6, 11, 10, 6, 10, 1, 8,  4, 3,  8,  3, 9, 8, 9, 11, 8,
                                                  11, 7, 8, 7,  4, 9,  3,  2, 9,  2, 10, 9, 10, 11, 5, 4, 7, 1, 10, 2};

  SplatAabb aabb{.minimum = glm::vec3(std::numeric_limits<float>::max()), .maximum = glm::vec3(std::numeric_limits<float>::min())};

  const auto vertexOffset = offset * 12;
  const auto indexOffset  = offset * 20 * 3;

  for(auto i = 0; i < s_vertices.size(); ++i)
  {
    const auto pos             = glm::vec3(transform * s_vertices[i]);
    vertices[vertexOffset + i] = pos;
    aabb.maximum               = glm::max(pos, aabb.maximum);
    aabb.minimum               = glm::min(pos, aabb.minimum);
  }
  aabbs[offset] = aabb;
  for(auto i = 0; i < s_indices.size(); ++i)
  {
    indices[indexOffset + i] = uint32_t(vertexOffset) + s_indices[i];
  }
}

bool SplatSetVk::rtxInitSplatModel(bool useInstances, bool useAABBs, bool compressBlas, int kernelDegree, float kernelMinResponse, bool kernelAdaptiveClamping)
{
  // CRITICAL: Destroy old buffers first if they exist (e.g., when switching modes)
  // Without this, we'll have memory leaks and potential crashes
  if(m_splatModel.vertexBuffer.buffer != VK_NULL_HANDLE || m_splatModel.indexBuffer.buffer != VK_NULL_HANDLE
     || m_splatModel.aabbBuffer.buffer != VK_NULL_HANDLE)
  {
    std::cout << "  Destroying old splat model buffers before recreating..." << std::endl;
    rtxDeinitSplatModel();
  }

  // Start in delayed state (will be updated to success or error at end)
  rtxStatus = RtxStatus::eDelayed;

  // stored for later use by rtxComputeTransformMatrix and rtxInitAccelerationStructures
  m_rtxUseAABBs               = useAABBs;
  m_rtxUseInstances           = useInstances;
  m_rtxCompressBlas           = compressBlas;
  m_rtxKernelDegree           = kernelDegree;
  m_rtxKernelMinResponse      = kernelMinResponse;
  m_rtxKernelAdaptiveClamping = kernelAdaptiveClamping;

  std::vector<glm::vec3> vertices;
  std::vector<uint32_t>  indices;
  std::vector<SplatAabb> aabbs;

  const uint64_t splatCount = size();

  if(useInstances)
  {
    // One unit icosahedron for instancing
    vertices.resize(12);
    indices.resize(20 * 3);
    aabbs.resize(1);
    rtxCreateSplatIcosahedron(0, vertices, indices, aabbs);
  }
  else
  {
    // One icosahedron per splat
    vertices.resize(splatCount * 12);
    indices.resize(splatCount * 20 * 3);
    aabbs.resize(splatCount);
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const glm::mat4 transform = rtxComputeTransformMatrix(splatIdx);
      rtxCreateSplatIcosahedron(splatIdx, vertices, indices, aabbs, transform);
    }
    END_PAR_LOOP()
  }

  // Store vertex and index count and aabbs
  m_splatModel.nbVertices = static_cast<uint32_t>(vertices.size());
  m_splatModel.nbIndices  = static_cast<uint32_t>(indices.size());
  m_splatModel.nbAABB     = static_cast<uint32_t>(aabbs.size());

  // Create the buffers on Device and copy vertices and indices
  VkCommandBuffer    cmd             = m_app->createTempCmdBuffer();
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  // Get queue for sparse binding (required for LargeBuffer)
  VkQueue queue = m_app->getQueue(0).queue;

  // Log buffer sizes for debugging
  VkDeviceSize vertexSize = vertices.size() * sizeof(glm::vec3);
  VkDeviceSize indexSize  = indices.size() * sizeof(uint32_t);
  VkDeviceSize aabbSize   = aabbs.size() * sizeof(SplatAabb);

  std::cout << "Creating splat model buffers (mode: " << (useAABBs ? "AABB" : "Icosahedron") << "):" << std::endl;

  VkDeviceSize totalSize = 0;
  VkResult     result;

  // Create buffers based on mode
  if(useAABBs)
  {
    // AABB mode: Only create AABB buffer
    std::cout << "  AABB buffer:   " << (aabbSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    totalSize = aabbSize;

    std::cout << "  Creating AABB buffer..." << std::endl;
    result = m_alloc->createLargeBuffer(m_splatModel.aabbBuffer, aabbSize,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | rayTracingFlags,
                                        queue);
    if(result != VK_SUCCESS)
    {
      LOGE("Failed to allocate AABB buffer (%.2f GB): %s\n", aabbSize / (1024.0 * 1024.0 * 1024.0), string_VkResult(result));
      queryVRAMInfo(m_app->getPhysicalDevice());
      rtxDeinitSplatModel();  // Cleanup any partial allocations
      rtxStatus = RtxStatus::eError;
      return false;
    }
    NVVK_DBG_NAME(m_splatModel.aabbBuffer.buffer);

    std::cout << "  AABB buffer created, uploading via chunked staging (256MB chunks)..." << std::endl;
    result = m_uploader->appendLargeBuffer(m_splatModel.aabbBuffer, 0, aabbSize, aabbs.data());
    if(result != VK_SUCCESS)
    {
      LOGE("Failed to upload AABB data: %s\n", string_VkResult(result));
      rtxDeinitSplatModel();
      m_uploader->cancelAppended();
      m_uploader->releaseStaging();
      rtxStatus = RtxStatus::eError;
      return false;
    }

    // Track memory
    memoryStats.rtxAabbBuffer = aabbSize;
  }
  else
  {
    // Icosahedron mode: Create vertex and index buffers
    std::cout << "  Vertex buffer: " << (vertexSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "  Index buffer:  " << (indexSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    totalSize = vertexSize + indexSize;

    std::cout << "  Creating vertex buffer..." << std::endl;
    result = m_alloc->createLargeBuffer(m_splatModel.vertexBuffer, vertexSize,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | rayTracingFlags,
                                        queue);
    if(result != VK_SUCCESS)
    {
      LOGE("Failed to allocate vertex buffer (%.2f GB): %s\n", vertexSize / (1024.0 * 1024.0 * 1024.0), string_VkResult(result));
      queryVRAMInfo(m_app->getPhysicalDevice());
      rtxDeinitSplatModel();
      m_uploader->cancelAppended();
      m_uploader->releaseStaging();
      rtxStatus = RtxStatus::eError;
      return false;
    }
    NVVK_DBG_NAME(m_splatModel.vertexBuffer.buffer);

    std::cout << "  Vertex buffer created, uploading via chunked staging (256MB chunks)..." << std::endl;
    result = m_uploader->appendLargeBuffer(m_splatModel.vertexBuffer, 0, vertexSize, vertices.data());
    if(result != VK_SUCCESS)
    {
      LOGE("Failed to upload vertex data: %s\n", string_VkResult(result));
      rtxDeinitSplatModel();
      m_uploader->cancelAppended();
      m_uploader->releaseStaging();
      rtxStatus = RtxStatus::eError;
      return false;
    }

    std::cout << "  Creating index buffer..." << std::endl;
    result = m_alloc->createLargeBuffer(m_splatModel.indexBuffer, indexSize,
                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | rayTracingFlags, queue);
    if(result != VK_SUCCESS)
    {
      LOGE("Failed to allocate index buffer (%.2f GB): %s\n", indexSize / (1024.0 * 1024.0 * 1024.0), string_VkResult(result));
      queryVRAMInfo(m_app->getPhysicalDevice());
      rtxDeinitSplatModel();
      m_uploader->cancelAppended();
      m_uploader->releaseStaging();
      rtxStatus = RtxStatus::eError;
      return false;
    }
    NVVK_DBG_NAME(m_splatModel.indexBuffer.buffer);

    std::cout << "  Index buffer created, uploading via chunked staging (256MB chunks)..." << std::endl;
    result = m_uploader->appendLargeBuffer(m_splatModel.indexBuffer, 0, indexSize, indices.data());
    if(result != VK_SUCCESS)
    {
      LOGE("Failed to upload index data: %s\n", string_VkResult(result));
      rtxDeinitSplatModel();
      m_uploader->cancelAppended();
      m_uploader->releaseStaging();
      rtxStatus = RtxStatus::eError;
      return false;
    }

    // Track memory
    memoryStats.rtxVertexBuffer = vertexSize;
    memoryStats.rtxIndexBuffer  = indexSize;
  }

  std::cout << "  Total:         " << (totalSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
  std::cout << "  All buffers created and uploaded via chunked staging." << std::endl;

  m_uploader->cmdUploadAppended(cmd);
  m_app->submitAndWaitTempCmdBuffer(cmd);
  m_uploader->releaseStaging();  // Release staging buffers after submission

  // Success! Set status and return
  rtxStatus = RtxStatus::eSuccess;
  return true;
}

nvvk::AccelerationStructureGeometryInfo SplatSetVk::rtxCreateSplatModelAccelerationStructureGeometryInfo()
{
  // use icosa mesh
  if(!m_rtxUseAABBs)
  {
    // Describe buffer as array of glm::vec3 wint uint32 indices.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = m_splatModel.vertexBuffer.address;
    triangles.vertexStride             = sizeof(glm::vec3);
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = m_splatModel.indexBuffer.address;
    triangles.transformData            = {};  // identity
    triangles.maxVertex                = m_splatModel.nbVertices - 1;

    // Identify the data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR geometry{};  // do not forget default init {} so pNext is NULL
    geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geometry.geometry.triangles = triangles;

    // The entire array will be used to build the BLAS.
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.firstVertex     = 0;
    rangeInfo.primitiveCount  = m_splatModel.nbIndices / 3;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.transformOffset = 0;

    return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
  }
  // Use AABBs
  else
  {
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{};
    aabbs.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data.deviceAddress = m_splatModel.aabbBuffer.address;
    aabbs.stride             = sizeof(SplatAabb);

    // Identify the data as containing opaque AABBs.
    VkAccelerationStructureGeometryKHR geometry{};  // do not forget default init {} so pNext is NULL
    geometry.sType          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType   = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags          = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    geometry.geometry.aabbs = aabbs;

    // The entire array will be used to build the BLAS.
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.firstVertex     = 0;
    rangeInfo.primitiveCount  = m_splatModel.nbAABB;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.transformOffset = 0;

    return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
  }
}

}  // namespace vk_gaussian_splatting
