/*
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

#include "splat_set_manager_vk.h"
#include "utilities.h"
#include "parameters.h"
#include "memory_monitor_vk.h"
#include <vulkan/vk_enum_string_helper.h>  // For string_VkResult
#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <fmt/format.h>

// GPU radix sort
#include <vk_radix_sort.h>

namespace vk_gaussian_splatting {

//-----------------------------------------------------------------------------
// SplatSetInstanceVk Methods
//-----------------------------------------------------------------------------

void SplatSetInstanceVk::rebuildDescriptor(const SplatSetVk* splatSet, shaderio::SplatSetDesc& descriptor)
{
  if(!splatSet)
  {
    std::cerr << "ERROR: Cannot rebuild descriptor: splatSet is null" << std::endl;
    return;
  }

  // Populate asset data (GPU buffer addresses, metadata)
  descriptor.centersAddress     = reinterpret_cast<float*>(splatSet->centersBuffer.address);
  descriptor.colorsAddress      = splatSet->colorsBuffer.address;
  descriptor.scalesAddress      = reinterpret_cast<float*>(splatSet->scalesBuffer.address);
  descriptor.rotationsAddress   = reinterpret_cast<float*>(splatSet->rotationsBuffer.address);
  descriptor.shAddress          = splatSet->sphericalHarmonicsBuffer.address;
  descriptor.covariancesAddress = reinterpret_cast<float*>(splatSet->covariancesBuffer.address);

  // Texture handles (for STORAGE_TEXTURES mode)
  // These are indices into the bindless texture array, stored in the splat set (shared by all instances)
  descriptor.centersTexture     = splatSet->textureIndexCenters;
  descriptor.scalesTexture      = splatSet->textureIndexScales;
  descriptor.rotationsTexture   = splatSet->textureIndexRotations;
  descriptor.colorsTexture      = splatSet->textureIndexColors;
  descriptor.covariancesTexture = splatSet->textureIndexCovariances;
  descriptor.shTexture          = splatSet->textureIndexSH;

  // Metadata from asset
  descriptor.splatCount = splatSet->splatCount;
  descriptor.shDegree   = splatSet->shDegree;

  // Default bases (overridden for split-BLAS RTX descriptors).
  // Raster uses per-instance descriptors with global base set from the index table.
  descriptor.splatBase       = 0;
  descriptor.globalSplatBase = 0;

  // Data storage and format
  descriptor.storage    = splatSet->dataStorage;
  descriptor.format     = splatSet->shFormat;
  descriptor.rgbaFormat = splatSet->rgbaFormat;

  // Instance-specific data (transform, material)
  descriptor.transform                = transform;
  descriptor.transformInverse         = transformInverse;
  descriptor.transformRotScaleInverse = transformRotScaleInverse;

  // Update needShading flag before copying to GPU descriptor
  shaderio::updateMaterialNeedsShading(splatMaterial);
  descriptor.material = splatMaterial;
}

//-----------------------------------------------------------------------------
// Lifecycle
//-----------------------------------------------------------------------------

void SplatSetManagerVk::init(nvapp::Application*                                 app,
                             nvvk::ResourceAllocator*                            alloc,
                             nvvk::StagingUploader*                              uploader,
                             VkSampler*                                          sampler,
                             nvvk::PhysicalDeviceInfo*                           deviceInfo,
                             VkPhysicalDeviceAccelerationStructurePropertiesKHR* accelStructProps,
                             nvutils::ProfilerTimeline*                          profilerTimeline)
{
  m_app              = app;
  m_alloc            = alloc;
  m_uploader         = uploader;
  m_sampler          = sampler;
  m_deviceInfo       = deviceInfo;
  m_accelStructProps = accelStructProps;
  m_profilerTimeline = profilerTimeline;

  m_rtAccelerationStructures.helper.init(m_alloc, m_uploader, m_app->getQueue(0), 2000, 2000);
  m_particleAsHelper.init(m_alloc, m_app->getQueue(0));

  // Initialize CPU async sorter (application lifetime)
  m_cpuSorter.initialize(m_profilerTimeline);

  std::cout << "SplatSetManagerVk initialized" << std::endl;
}

// Deinitialize manager (app exit only)
// NOTE: vkDeviceWaitIdle shall be invoked before calling this method
void SplatSetManagerVk::deinit()
{
  // Shutdown CPU async sorter thread (application lifetime)
  m_cpuSorter.shutdown();

  // Reset all scene assets first (marks for deletion)
  reset();

  // Explicitly free all remaining splat set GPU resources (data buffers, RTX geometry).
  // reset() only marks for deferred deletion via processVramUpdates(), but processVramUpdates()
  // will NOT run after deinit(). We must free the data buffers (centers, scales, rotations,
  // colors, SH) directly here to avoid leaks.
  for(auto& splatSet : m_splatSets)
  {
    if(splatSet)
    {
      splatSet->rtxDeinitAccelerationStructures();
      splatSet->deinitDataStorage();
      splatSet->deinit();
    }
  }
  m_splatSets.clear();
  m_instances.clear();

  // Deinitialize RTX helper (destroys transient command pool)
  m_rtAccelerationStructures.helper.deinit();
  m_particleAsHelper.deinit();
  for(auto& tlasHelper : m_particleAsTlasHelpers)
  {
    tlasHelper.deinit();
  }
  m_particleAsTlasHelpers.clear();
  for(auto& blasHelper : m_particleAsBlasHelpers)
  {
    blasHelper.deinit();
  }
  m_particleAsBlasHelpers.clear();
  for(auto& chunk : m_particleAsBlasChunks)
  {
    chunk.helper.deinitAccelerationStructures();
    chunk.helper.deinit();
  }
  m_particleAsBlasChunks.clear();
  m_particleAsBlasChunkRanges.clear();

  clearSceneGpuBuffers();

  // Ensure TLAS address buffer is destroyed
  if(m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasAddressBuffer);
    m_rtAccelerationStructures.tlasAddressBuffer = {};
  }

  if(m_rtAccelerationStructures.tlasOffsetBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasOffsetBuffer);
    m_rtAccelerationStructures.tlasOffsetBuffer = {};
  }

  // Clear allocator pointers
  m_app                       = nullptr;
  m_alloc                     = nullptr;
  m_uploader                  = nullptr;
  m_sampler                   = nullptr;
  m_deviceInfo                = nullptr;
  m_accelStructProps          = nullptr;
  m_particleAsComputePipeline = VK_NULL_HANDLE;
  m_particleAsPipelineLayout  = VK_NULL_HANDLE;
  m_particleAsDescriptorSet   = VK_NULL_HANDLE;

  std::cout << "SplatSetManagerVk deinitialized" << std::endl;
}

// Reset all splat sets and instances (scene reset, not app exit)
// NOTE: vkDeviceWaitIdle shall be invoked before calling this method
void SplatSetManagerVk::reset()
{
  // CRITICAL: Use the SAME deferred deletion flow as DELETE (which works perfectly)
  // Call deleteInstance() for each instance, then deleteSplatSet() for each splat set
  // This reuses the existing, tested deletion logic

  // Delete all instances (iterate backwards to avoid issues with index changes)
  for(size_t i = m_instances.size(); i-- > 0;)
  {
    if(m_instances[i])
    {
      deleteInstance(m_instances[i]);
    }
  }

  // Delete all splat sets (iterate backwards, in case some weren't referenced by instances)
  for(size_t i = m_splatSets.size(); i-- > 0;)
  {
    if(m_splatSets[i])
    {
      deleteSplatSet(m_splatSets[i]);
    }
  }

  // Reset naming counter (will be recreated from 0 when new scene loads)
  m_nextInstanceNumber = 0;

  // Reset RTX state (no error, just nothing initialized yet)
  m_rtxState = RtxState::eRtxNone;

  std::cout << "SplatSetManagerVk reset: Marked all assets for deletion" << std::endl;
}

//-----------------------------------------------------------------------------
// Asset Management
//-----------------------------------------------------------------------------

std::shared_ptr<SplatSetVk> SplatSetManagerVk::createSplatSet(const std::string& path, std::shared_ptr<SplatSetVk> splatSetVk)
{
  splatSetVk->init(m_app, m_alloc, m_uploader, m_sampler, m_deviceInfo, m_accelStructProps);
  splatSetVk->path = path;

  // Set metadata from RAM data (needed for global index table and logging before GPU upload)
  splatSetVk->splatCount = static_cast<uint32_t>(splatSetVk->size());
  splatSetVk->shDegree   = static_cast<uint32_t>(splatSetVk->maxShDegree());

  // Initialize default material (will be set again in initDataStorage, but needed now for instances)
  splatSetVk->splatMaterial.ambient       = glm::vec3(0.0f);
  splatSetVk->splatMaterial.diffuse       = glm::vec3(0.0f);
  splatSetVk->splatMaterial.specular      = glm::vec3(0.0f);
  splatSetVk->splatMaterial.transmittance = glm::vec3(0.0f);
  splatSetVk->splatMaterial.emission      = glm::vec3(1.0f);  // Fully emissive
  splatSetVk->splatMaterial.shininess     = 0.0f;

  // Set flag: needs GPU upload (deferred to processVramUpdates)
  splatSetVk->flags |= SplatSetVk::Flags::eNew;

  // Set index and store
  splatSetVk->index = m_splatSets.size();
  m_splatSets.push_back(splatSetVk);

  // Update max SH degree immediately after adding splat set
  // This ensures the UI can read the correct value before processVramUpdates runs
  updateMaxShDegree();

  // Request GPU sync (deferred to processVramUpdates)
  // Will use prmData.dataStorage and prmData.shFormat
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildBLAS;

  std::cout << "Created splat set '" << path << "' (index=" << splatSetVk->index
            << ", splats=" << splatSetVk->splatCount << ") - deferred upload" << std::endl;

  return splatSetVk;
}

void SplatSetManagerVk::deleteSplatSet(std::shared_ptr<SplatSetVk> splatSet)
{
  if(!splatSet)
  {
    std::cerr << "Warning: deleteSplatSet called with null pointer" << std::endl;
    return;
  }

  // Mark asset for deletion (deferred to processVramUpdates)
  splatSet->flags |= SplatSetVk::Flags::eDelete;

  // Mark all instances using this asset for deletion
  for(auto& instance : m_instances)
  {
    if(instance && instance->splatSet == splatSet)
    {
      instance->flags |= SplatSetInstanceVk::Flags::eDelete;
    }
  }

  pendingRequests |= Request::eProcessDeletions;

  std::cout << "Marked splat set (index=" << splatSet->index << ") for deletion" << std::endl;
}

// Note: getSplatSet() now inline in header (direct vector access)

//-----------------------------------------------------------------------------
// Instance Management
//-----------------------------------------------------------------------------

std::shared_ptr<SplatSetInstanceVk> SplatSetManagerVk::createInstance(std::shared_ptr<SplatSetVk> splatSet, const glm::mat4& transform)
{
  if(!splatSet)
  {
    std::cerr << "Warning: createInstance called with null splatSet" << std::endl;
    return nullptr;
  }

  // Create new instance
  auto instance      = std::make_shared<SplatSetInstanceVk>();
  instance->splatSet = splatSet;
  instance->index    = m_instances.size();

  // Increment reference count for this splat set
  splatSet->instanceRefCount++;

  // Set transform (decompose or use directly)
  instance->transform        = transform;
  instance->transformInverse = glm::inverse(transform);

  // Extract TRS components from transform
  glm::vec3 skew;
  glm::vec4 perspective;
  glm::quat orientation;
  glm::decompose(transform, instance->scale, orientation, instance->translation, skew, perspective);
  instance->rotation = glm::degrees(glm::eulerAngles(orientation));

  // Set rotation-scale inverse (for normal transformation and ray direction transformation)
  // Note: Must use inverse() for non-uniform scaling (transpose only works for orthogonal matrices)
  glm::mat4 R                        = glm::mat4_cast(orientation);
  glm::mat4 S                        = glm::scale(glm::mat4(1.0f), instance->scale);
  instance->transformRotScaleInverse = glm::mat3(glm::inverse(R * S));

  // Set default material
  instance->splatMaterial = splatSet->splatMaterial;

  // Generate display name (use path for filename extraction)
  std::filesystem::path filepath(splatSet->path);
  std::string           filename = truncateFilename(filepath.filename().string());
  instance->displayName          = fmt::format("Splat set {} - {}", m_nextInstanceNumber, filename);
  ++m_nextInstanceNumber;

  // Mark as new (needs GPU descriptor/TLAS update)
  instance->flags |= SplatSetInstanceVk::Flags::eNew;

  m_instances.push_back(instance);

  // Request GPU updates
  // Force full BLAS+TLAS rebuild (not just TLAS) so that subsequent TLAS updates work correctly.
  // The GPU TLAS-only rebuild path creates a TLAS that cannot be updated in-place reliably.
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildBLAS;
  pendingRequests |= Request::eUpdateGlobalIndexTable;

  std::cout << "Created instance '" << instance->displayName << "' (index=" << instance->index << ")" << std::endl;

  return instance;
}

std::shared_ptr<SplatSetInstanceVk> SplatSetManagerVk::registerInstance(std::shared_ptr<SplatSetVk>         splatSet,
                                                                        std::shared_ptr<SplatSetInstanceVk> instance)
{
  if(!splatSet || !instance)
  {
    std::cerr << "Warning: registerInstance called with null pointer" << std::endl;
    return nullptr;
  }

  // Associate instance with splat set
  instance->splatSet = splatSet;
  instance->index    = m_instances.size();

  // Increment reference count for this splat set
  splatSet->instanceRefCount++;

  // Generate display name if not already set (e.g., loaded from file)
  if(instance->displayName.empty())
  {
    std::filesystem::path filepath(splatSet->path);
    std::string           filename = truncateFilename(filepath.filename().string());
    instance->displayName          = fmt::format("Splat set {} - {}", m_nextInstanceNumber, filename);
    ++m_nextInstanceNumber;
  }

  // Ensure transform matrices are consistent with TRS (project-loaded instances rely on this)
  computeTransform(instance->scale, instance->rotation, instance->translation, instance->transform,
                   instance->transformInverse, instance->transformRotScaleInverse);

  // Mark as new (needs GPU descriptor/TLAS update)
  instance->flags |= SplatSetInstanceVk::Flags::eNew;

  m_instances.push_back(instance);

  // Request GPU updates
  // Force full BLAS+TLAS rebuild (not just TLAS) so that subsequent TLAS updates work correctly.
  // The GPU TLAS-only rebuild path creates a TLAS that cannot be updated in-place reliably.
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildBLAS;
  pendingRequests |= Request::eUpdateGlobalIndexTable;

  std::cout << "Registered instance '" << instance->displayName << "' (index=" << instance->index << ")" << std::endl;

  return instance;
}

std::shared_ptr<SplatSetInstanceVk> SplatSetManagerVk::duplicateInstance(std::shared_ptr<SplatSetInstanceVk> sourceInstance)
{
  if(!sourceInstance || !sourceInstance->splatSet)
  {
    std::cerr << "Warning: duplicateInstance called with invalid source instance" << std::endl;
    return nullptr;
  }

  // Create new instance as a copy of the source (copy all fields via copy constructor)
  auto newInstance = std::make_shared<SplatSetInstanceVk>(*sourceInstance);

  // Generate NEW display name (don't copy source name, use path for filename extraction)
  std::filesystem::path filepath(sourceInstance->splatSet->path);
  std::string           filename = truncateFilename(filepath.filename().string());
  newInstance->displayName       = fmt::format("Splat set {} - {}", m_nextInstanceNumber, filename);
  ++m_nextInstanceNumber;

  // Register the new instance (this will set index and add to vector)
  // Note: registerInstance may reallocate vector, so sourceInstance reference could be invalidated after this
  newInstance = registerInstance(sourceInstance->splatSet, newInstance);

  if(!newInstance)
  {
    std::cerr << "Warning: duplicateInstance failed to register new instance" << std::endl;
    return nullptr;
  }

  std::cout << "Duplicated instance: source='" << sourceInstance->displayName << "' -> new='"
            << newInstance->displayName << "'" << std::endl;

  return newInstance;
}

// Note: Old duplicate code removed

void SplatSetManagerVk::deleteInstance(std::shared_ptr<SplatSetInstanceVk> instance)
{
  if(!instance)
  {
    std::cerr << "Warning: deleteInstance called with null pointer" << std::endl;
    return;
  }

  // Set Delete flag (deferred deletion in processVramUpdates)
  instance->flags |= SplatSetInstanceVk::Flags::eDelete;
  pendingRequests |= Request::eProcessDeletions;

  std::cout << "Marked instance (index=" << instance->index << ") for deletion" << std::endl;
}

// Note: getInstance() now inline in header (direct vector access)

void SplatSetManagerVk::updateInstanceTransform(std::shared_ptr<SplatSetInstanceVk> instance)
{
  if(!instance)
  {
    std::cerr << "Warning: updateInstanceTransform called with null pointer" << std::endl;
    return;
  }

  // UI has already modified instance->translation/rotation/scale
  // Just set flag and request GPU update
  instance->flags |= SplatSetInstanceVk::Flags::eTransformChanged;
  markGpuDescriptorsDirty();
  pendingRequests |= Request::eUpdateTransformsOnly;
}

void SplatSetManagerVk::updateInstanceMaterial(std::shared_ptr<SplatSetInstanceVk> instance)
{
  if(!instance)
  {
    std::cerr << "Warning: updateInstanceMaterial called with null pointer" << std::endl;
    return;
  }

  // UI has already modified instance->splatMaterial
  // Just set flag and request GPU update
  instance->flags |= SplatSetInstanceVk::Flags::eMaterialChanged;
  pendingRequests |= Request::eUpdateDescriptors;  // Material embedded in descriptor
}

//-----------------------------------------------------------------------------
// VRAM SYNC - Process all deferred updates
//-----------------------------------------------------------------------------

void SplatSetManagerVk::processVramUpdates(bool processRtx)
{
  bool instanceCountChanged   = false;
  bool descriptorsNeedRebuild = false;

  // Phase 1: Remove from GPU + delete from RAM
  processRamVramDeletionsIfNeeded(instanceCountChanged, descriptorsNeedRebuild);

  // Phase 2: Process RAM → GPU data uploads (new splat sets, instances, transforms, materials, data storage)
  bool hasTransformChanges = false;
  processRamToVramDataUploads(instanceCountChanged, descriptorsNeedRebuild, hasTransformChanges);

  // Phase 3: RTX Acceleration Structures (BLAS first, then TLAS)
  if(!processRtxAccelerationStructures(processRtx, hasTransformChanges, descriptorsNeedRebuild))
    return;  // Early exit requested by Phase 3 - skip Phase 4

  // Phase 4: Rebuild descriptors & index tables (after all GPU resources ready)
  processDescriptorsAndIndexTables(descriptorsNeedRebuild);

  // In raster mode (processRtx=false), Phase 3 skips RTX processing entirely,
  // so eUpdateTransformsOnly is never cleared there. Clear it here since the
  // transform was already applied via descriptor update in Phase 4.
  // The deferred RTX rebuild (m_deferredRtxRebuildPending) will handle the actual
  // AS update when switching back to RTX pipeline.
  if(!processRtx && static_cast<uint32_t>(pendingRequests & Request::eUpdateTransformsOnly))
  {
    pendingRequests &= ~Request::eUpdateTransformsOnly;
  }
}

//-----------------------------------------------------------------------------
// processVramUpdates sub-methods
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Phase 1: Remove instances/splat sets from GPU + delete from RAM
//-----------------------------------------------------------------------------
void SplatSetManagerVk::processRamVramDeletionsIfNeeded(bool& instanceCountChanged, bool& descriptorsNeedRebuild)
{
  if(!static_cast<uint32_t>(pendingRequests & Request::eProcessDeletions))
    return;

  // Ensure the CPU sorter is not reading positions that are about to be freed
  m_cpuSorter.waitUntilIdleAndReset();

  // 1a. Delete instances (shift-left compaction)
  {
    size_t originalSize = m_instances.size();
    size_t shiftLeft    = 0;

    for(size_t i = 0; i < originalSize; i++)
    {
      if(m_instances[i]->isMarkedForDeletion())
      {
        std::cout << "processVramUpdates: Deleting instance (index=" << i << ")" << std::endl;

        // Decrement reference count for this splat set
        auto splatSet = m_instances[i]->splatSet;
        if(splatSet)
        {
          if(splatSet->instanceRefCount > 0)
            splatSet->instanceRefCount--;

          // Check if this was the last instance referencing the splat set
          if(splatSet->instanceRefCount == 0)
          {
            std::cout << "  Last instance deleted, marking splat set for deletion (index=" << splatSet->index << ")" << std::endl;
            splatSet->flags |= SplatSetVk::Flags::eDelete;
          }
        }

        // Instance will be destroyed when shared_ptr is released
        shiftLeft++;
        instanceCountChanged = true;
      }
      else
      {
        // Keep instance - shift it left if needed
        m_instances[i - shiftLeft]        = m_instances[i];
        m_instances[i - shiftLeft]->index = i - shiftLeft;
      }
    }

    m_instances.resize(originalSize - shiftLeft);

    if(shiftLeft > 0)
      std::cout << "Deleted " << shiftLeft << " splat set instances" << std::endl;
  }

  // 1b. Delete splat sets (shift-left compaction)
  {
    size_t originalSize = m_splatSets.size();
    size_t shiftLeft    = 0;

    for(size_t i = 0; i < originalSize; i++)
    {
      if(m_splatSets[i]->isMarkedForDeletion())
      {
        // Verify no instances still reference this splat set
        bool hasReferences = false;
        for(const auto& inst : m_instances)
        {
          if(inst->splatSet == m_splatSets[i])
          {
            hasReferences = true;
            break;
          }
        }

        if(hasReferences)
        {
          // Clear delete flag - still in use
          m_splatSets[i]->flags &= ~SplatSetVk::Flags::eDelete;
          // Keep this splat set
          m_splatSets[i - shiftLeft]        = m_splatSets[i];
          m_splatSets[i - shiftLeft]->index = i - shiftLeft;
        }
        else
        {
          // Safe to delete
          std::cout << "processVramUpdates: Deleting splat set (index=" << i << ")" << std::endl;

          // Deinitialize GPU resources (vkDeviceWaitIdle already called by GaussianSplatting::processUpdateRequests)
          m_splatSets[i]->rtxDeinitAccelerationStructures();  // Deinit BLAS + any TLAS
          m_splatSets[i]->deinitDataStorage();
          m_splatSets[i]->deinit();

          shiftLeft++;
        }
      }
      else
      {
        m_splatSets[i - shiftLeft]        = m_splatSets[i];
        m_splatSets[i - shiftLeft]->index = i - shiftLeft;
      }
    }

    m_splatSets.resize(originalSize - shiftLeft);

    if(shiftLeft > 0)
      std::cout << "Deleted " << shiftLeft << " splat sets" << std::endl;
  }

  if(instanceCountChanged)
  {
    descriptorsNeedRebuild = true;
    // Force full BLAS+TLAS rebuild (not just TLAS) so that subsequent TLAS updates work correctly.
    pendingRequests |= Request::eRebuildBLAS;
    pendingRequests |= Request::eUpdateGlobalIndexTable;
  }

  pendingRequests &= ~Request::eProcessDeletions;

  // Update max SH degree after deletions
  updateMaxShDegree();

  if(m_instances.empty() && m_splatSets.empty())
  {
    rtxDeinitAccelerationStructures();
    clearSceneGpuBuffers();
    m_rtxState = RtxState::eRtxNone;
  }
}

//-----------------------------------------------------------------------------
// Phase 2: Process RAM → GPU data uploads
//-----------------------------------------------------------------------------
void SplatSetManagerVk::processRamToVramDataUploads(bool& instanceCountChanged, bool& descriptorsNeedRebuild, bool& hasTransformChanges)
{
  // 2a. Process New splat sets (upload geometry to GPU using global prmData settings)
  for(const auto& splatSet : m_splatSets)
  {
    if(!splatSet)
      continue;
    if(static_cast<uint32_t>(splatSet->flags & SplatSetVk::Flags::eNew))
    {
      std::cout << "processVramUpdates: Uploading new splat set to GPU (index=" << splatSet->index
                << ", storage=" << splatSet->dataStorage << ", format=" << prmData.shFormat << ")" << std::endl;

      // Upload data to GPU
      splatSet->initDataStorage(prmData.shFormat, prmData.rgbaFormat);

      descriptorsNeedRebuild = true;
      pendingRequests |= Request::eRebuildBLAS;

      splatSet->flags &= ~SplatSetVk::Flags::eNew;  // Clear flag
    }
  }

  // Update consolidated memory stats if any new splat sets were uploaded
  if(static_cast<uint32_t>(pendingRequests & Request::eRebuildBLAS))
  {
    updateConsolidatedMemoryStats();
    updateMaxShDegree();  // Update max SH degree after new splat sets uploaded
  }

  // 2b. Process New instances (add to descriptors)
  for(const auto& instance : m_instances)
  {
    if(!instance)
      continue;
    if(static_cast<uint32_t>(instance->flags & SplatSetInstanceVk::Flags::eNew))
    {
      std::cout << "processVramUpdates: Adding new instance to descriptors (index=" << instance->index << ")" << std::endl;

      instanceCountChanged   = true;
      descriptorsNeedRebuild = true;
      instance->flags &= ~SplatSetInstanceVk::Flags::eNew;
    }
  }

  // 2c. Process transform changes
  for(const auto& instance : m_instances)
  {
    if(!instance)
      continue;
    if(static_cast<uint32_t>(instance->flags & SplatSetInstanceVk::Flags::eTransformChanged))
    {
      // Recompute transform matrix from components (UI modified translation/rotation/scale)
      computeTransform(instance->scale, instance->rotation, instance->translation, instance->transform,
                       instance->transformInverse, instance->transformRotScaleInverse);

      hasTransformChanges = true;
      instance->flags &= ~SplatSetInstanceVk::Flags::eTransformChanged;
    }
  }

  if(hasTransformChanges)
  {
    std::cout << "processVramUpdates: Transform changes detected" << std::endl;
    descriptorsNeedRebuild = true;  // Transform embedded in descriptor
  }

  // 2d. Process material changes
  bool hasMaterialChanges = false;
  for(const auto& instance : m_instances)
  {
    if(!instance)
      continue;
    if(static_cast<uint32_t>(instance->flags & SplatSetInstanceVk::Flags::eMaterialChanged))
    {
      hasMaterialChanges = true;
      instance->flags &= ~SplatSetInstanceVk::Flags::eMaterialChanged;
    }
  }

  if(hasMaterialChanges)
  {
    std::cout << "processVramUpdates: Material changes detected" << std::endl;
    descriptorsNeedRebuild = true;  // Material embedded in descriptor
  }

  // 2e. Process data storage changes (regenerate ALL marked splat sets with global prmData settings)
  bool dataStorageRegenerated = false;
  for(const auto& splatSet : m_splatSets)
  {
    if(!splatSet)
      continue;
    if(static_cast<uint32_t>(splatSet->flags & SplatSetVk::Flags::eDataChanged))
    {
      std::cout << "processVramUpdates: Regenerating data storage (index=" << splatSet->index
                << ", storage=" << splatSet->dataStorage << ", format=" << prmData.shFormat << ")" << std::endl;

      // Deinitialize old data
      splatSet->deinitDataStorage();

      // Reinitialize with global prmData settings
      splatSet->initDataStorage(prmData.shFormat, prmData.rgbaFormat);

      descriptorsNeedRebuild = true;
      dataStorageRegenerated = true;
      splatSet->flags &= ~SplatSetVk::Flags::eDataChanged;
    }
  }

  // IMPORTANT: Wait for all texture/buffer uploads to complete before proceeding
  // initDataStorage() uses staging buffers (StagingUploader) for async GPU uploads.
  // Without this wait, descriptor binding in initPipelines() might reference incomplete GPU resources.
  // TODO: Could be optimized with targeted fences/semaphores instead of global device wait
  if(dataStorageRegenerated)
  {
    VkDevice device = m_alloc->getDevice();
    vkDeviceWaitIdle(device);

    // Update consolidated memory stats after regeneration
    updateConsolidatedMemoryStats();
    updateMaxShDegree();  // Update max SH degree after data storage changes

    // Signal that Vulkan texture descriptors (BINDING_SPLAT_TEXTURES) need rebinding
    m_textureDescriptorsDirty = true;
  }
}

//-----------------------------------------------------------------------------
// Phase 4: Rebuild descriptors & index tables
//-----------------------------------------------------------------------------
void SplatSetManagerVk::processDescriptorsAndIndexTables(bool descriptorsNeedRebuild)
{
  // 4a. Rebuild global index tables if needed
  if(static_cast<uint32_t>(pendingRequests & Request::eUpdateGlobalIndexTable) || m_globalIndexTableDirty)
  {
    std::cout << "processVramUpdates: Rebuilding global index tables" << std::endl;

    rebuildGlobalIndexTables();
    uploadGlobalIndexTablesToGPU();
    m_globalIndexTableDirty = false;
    pendingRequests &= ~Request::eUpdateGlobalIndexTable;
  }

  // 4b. Rebuild descriptors if needed
  if(descriptorsNeedRebuild || static_cast<uint32_t>(pendingRequests & Request::eUpdateDescriptors) || m_gpuDescriptorsDirty)
  {
    std::cout << "processVramUpdates: Rebuilding GPU descriptor array" << std::endl;

    updateGpuDescriptorArray();
    uploadGpuDescriptorArray();
    m_gpuDescriptorsDirty = false;
    pendingRequests &= ~Request::eUpdateDescriptors;
  }
}

//-----------------------------------------------------------------------------
// Phase 3: RTX Acceleration Structures (BLAS first, then TLAS)
// Returns false if Phase 4 should be skipped (early exit), true to continue.
// Dispatches to sub-methods: rtxRebuildBlasAndTlas, rtxRebuildTlas, rtxUpdateTlasTransforms.
//-----------------------------------------------------------------------------
bool SplatSetManagerVk::processRtxAccelerationStructures(bool processRtx, bool hasTransformChanges, bool& descriptorsNeedRebuild)
{
  const bool gpuPipelineReady = (m_particleAsComputePipeline != VK_NULL_HANDLE) && (m_particleAsPipelineLayout != VK_NULL_HANDLE);
  const bool processRtxGpu = processRtx && gpuPipelineReady;
  if(processRtx && !gpuPipelineReady)
  {
    LOGW("GPU particle AS pipeline not ready; deferring RTX build/update this frame.\n");
  }

  if(processRtxGpu)
  {
    // 3a. Full BLAS + TLAS rebuild
    if(static_cast<uint32_t>(pendingRequests & Request::eRebuildBLAS))
    {
      if(!rtxRebuildBlasAndTlas())
        return false;
    }
    // 3b. TLAS-only rebuild (instance count changed, BLAS already exists)
    else if(static_cast<uint32_t>(pendingRequests & Request::eRebuildTLAS))
    {
      if(!rtxRebuildTlas())
        return false;
    }
    // 3c. TLAS update only (fast path for transform changes)
    else if(static_cast<uint32_t>(pendingRequests & Request::eUpdateTransformsOnly))
    {
      // WORKAROUND: Use counter to force multiple rebuilds before allowing update path
      //
      // The update path (cmdUpdateAccelerationStructure/refit) requires the TLAS to be rebuilt
      // multiple times first (currently 20) to stabilize internal state before refit operations
      // work correctly. This prevents progressive performance degradation during continuous
      // transforms after copy/import operations.
      //
      // See rtxRebuildBlasAndTlas() for where the counter is initialized.
      if(m_tlasNeedsFullRebuild > 0)
      {
        pendingRequests &= ~Request::eUpdateTransformsOnly;
        pendingRequests |= Request::eRebuildTLAS;
        m_tlasNeedsFullRebuild--;  // Decrement counter
        if(!rtxRebuildTlas())
          return false;
      }
      else
      {
        // Counter reached 0: use normal update path
        if(!rtxUpdateTlasTransforms(descriptorsNeedRebuild))
          return false;
      }
    }
    else
    {
      // No RTX request matched while pipeline is ready.
      // Clear transform-only flag if transforms were already processed via descriptor updates.
      if(hasTransformChanges && static_cast<uint32_t>(pendingRequests & Request::eUpdateTransformsOnly))
      {
        pendingRequests &= ~Request::eUpdateTransformsOnly;
      }
    }
  }

  return true;
}

//-----------------------------------------------------------------------------
// Marks all splat sets as error, deinitializes AS, and clears pending RTX requests.
// Used as a centralized failure handler for all RTX build paths.
//-----------------------------------------------------------------------------
void SplatSetManagerVk::handleRtxBuildFailure(const char* reason)
{
  LOGE("%s\n", reason);
  for(auto& splatSet : m_splatSets)
  {
    if(splatSet)
    {
      splatSet->rtxStatus = RtxStatus::eError;
    }
  }
  rtxDeinitAccelerationStructures();
  m_rtxState = RtxState::eRtxError;
  pendingRequests &= ~Request::eRebuildBLAS;
  pendingRequests &= ~Request::eRebuildTLAS;
  pendingRequests &= ~Request::eUpdateTransformsOnly;
}

//-----------------------------------------------------------------------------
// Clears the RTX-specific descriptor array and associated GPU buffer.
// Called when split-BLAS mode is not active or before switching descriptor layouts.
//-----------------------------------------------------------------------------
void SplatSetManagerVk::clearRtxDescriptorArray()
{
  m_gpuRtxDescriptorArray.clear();
  if(m_rtxDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtxDescriptorBuffer);
    m_rtxDescriptorBuffer = {};
  }
  m_useSplitBlasRtxDescriptors = false;
}

//-----------------------------------------------------------------------------
// Section 3a: Full BLAS + TLAS rebuild for all splat sets.
// Handles both per-splat (instanced) and per-splat-set (non-instanced) modes.
// Returns false to skip Phase 4 (early exit on error or completion), true to continue.
//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildBlasAndTlas()
{
  std::cout << "processVramUpdates: Rebuilding BLAS for all splat sets" << std::endl;

  // Clean up any prior GPU particle AS helpers before rebuilding
  for(auto& tlasHelper : m_particleAsTlasHelpers)
  {
    tlasHelper.deinitAccelerationStructures();
    tlasHelper.deinit();
  }
  m_particleAsTlasHelpers.clear();
  for(auto& blasHelper : m_particleAsBlasHelpers)
  {
    blasHelper.deinitAccelerationStructures();
    blasHelper.deinit();
  }
  m_particleAsBlasHelpers.clear();
  for(auto& chunk : m_particleAsBlasChunks)
  {
    chunk.helper.deinitAccelerationStructures();
    chunk.helper.deinit();
  }
  m_particleAsBlasChunks.clear();
  m_particleAsBlasChunkRanges.clear();
  if(m_rtxDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtxDescriptorBuffer);
    m_rtxDescriptorBuffer = {};
  }
  m_gpuRtxDescriptorArray.clear();

  // Early exit if no instances — nothing to build regardless of mode
  if(m_instances.empty())
  {
    LOGW("GPU particle AS build skipped (no instances). RTX disabled.\n");
    rtxDeinitAccelerationStructures();
    m_rtxState = RtxState::eRtxNone;
    pendingRequests &= ~Request::eRebuildBLAS;
    pendingRequests &= ~Request::eRebuildTLAS;
    pendingRequests &= ~Request::eUpdateTransformsOnly;
    return false;
  }

  bool result;
  if(prmRtxData.useTlasInstances)
  {
    result = rtxRebuildBlasAndTlasPerSplat();
  }
  else
  {
    // Shared per-instance setup: global index tables, descriptor arrays, BLAS flags

    // Rebuild global index tables BEFORE building TLAS (needed for shader-side splat ID resolution)
    if(static_cast<uint32_t>(pendingRequests & Request::eUpdateGlobalIndexTable) || m_globalIndexTableDirty)
    {
      std::cout << "processVramUpdates: Rebuilding global index tables (pre-TLAS)" << std::endl;
      rebuildGlobalIndexTables();
      uploadGlobalIndexTablesToGPU();
      m_globalIndexTableDirty = false;
      pendingRequests &= ~Request::eUpdateGlobalIndexTable;
    }

    // Ensure descriptor arrays are current
    updateGpuDescriptorArray();
    uploadGpuDescriptorArray();
    m_gpuDescriptorsDirty = false;

    updateGpuSplatSetDescriptorArray();
    uploadGpuSplatSetDescriptorArray();

    VkBuildAccelerationStructureFlagsKHR blasFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    if(prmRtxData.compressBlas)
      blasFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;

    clearRtxDescriptorArray();
    m_particleAsBlasChunks.clear();
    m_particleAsBlasChunkRanges.clear();

    // Detect if any splat set needs multi-BLAS splitting
    bool needsSplit = false;
    for(const auto& splatSet : m_splatSets)
    {
      if(!splatSet)
        continue;
      const uint32_t maxSplats = computeMaxSplatsPerGpuBlas(prmRtxData.useAABBs, blasFlags, splatSet->splatCount);
      if(maxSplats < splatSet->splatCount)
      {
        needsSplit = true;
        break;
      }
    }

    if(!needsSplit)
      result = rtxRebuildBlasAndTlasPerInstanceSingleBlas(blasFlags);
    else
      result = rtxRebuildBlasAndTlasPerInstanceMultiBlas(blasFlags);

    if(result)
    {
      // CRITICAL: Wait for device idle after complete RTX rebuild
      // Ensures all GPU work is complete before returning control to main loop
      std::cout << "processVramUpdates: Waiting for device idle after RTX rebuild..." << std::endl;
      vkDeviceWaitIdle(m_app->getDevice());
      std::cout << "processVramUpdates: Device idle confirmed, RTX rebuild complete" << std::endl;
    }
  }

  pendingRequests &= ~Request::eRebuildBLAS;
  pendingRequests &= ~Request::eRebuildTLAS;
  pendingRequests &= ~Request::eUpdateTransformsOnly;

  // WORKAROUND: Set counter to force multiple rebuilds before allowing update path
  //
  // PROBLEM: The update path (cmdUpdateAccelerationStructure/refit) causes progressive performance
  // degradation when used immediately after copy/import, even though it works fine after Reset.
  // Testing revealed that the TLAS needs to be rebuilt multiple times before refit operations
  // work correctly. This is NOT about swapchain delay (3 frames) but about the TLAS internal
  // state needing multiple rebuilds to converge to a stable state.
  //
  // SOLUTION: Force full rebuilds (cmdBuildAccelerationStructure) for the first N transforms after
  // a copy/import, then allow the update path (refit) to be used. Testing showed:
  // - 3-4 rebuilds: Still causes performance leak
  // - 10 rebuilds: Works correctly
  // - 20 rebuilds: Safe margin for stability
  //
  // TODO: Investigate root cause - why does TLAS need multiple rebuilds to stabilize?
  // Possible causes: BVH topology convergence, GPU internal state, scratch buffer state, etc.
  m_tlasNeedsFullRebuild = 20;

  return result;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildBlasAndTlasPerSplat()
{
  // Per-splat TLAS path does not use split-BLAS descriptors.
  clearRtxDescriptorArray();
  m_useGpuBlasForSplatSets = false;

  uint64_t totalSplats = 0;
  for(const auto& instance : m_instances)
  {
    if(instance && instance->splatSet)
      totalSplats += instance->splatSet->splatCount;
  }
  const uint32_t maxInstances = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);

  if(totalSplats == 0)
  {
    LOGW("GPU particle AS path disabled (totalSplats=0). RTX disabled.\n");
    rtxDeinitAccelerationStructures();
    m_rtxState = RtxState::eRtxNone;
    return false;
  }

  LOGI("GPU particle AS build path enabled (mode=%s, totalSplats=%u, maxInstanceCount=%u)\n",
       prmRtxData.useAABBs ? "AABB" : "Icosahedron", static_cast<uint32_t>(totalSplats), maxInstances);

  // Deinitialize old acceleration structures (both BLAS and TLAS)
  rtxDeinitAccelerationStructures();

  // Ensure global index tables are up to date for GPU instance generation
  if(static_cast<uint32_t>(pendingRequests & Request::eUpdateGlobalIndexTable) || m_globalIndexTableDirty)
  {
    rebuildGlobalIndexTables();
    uploadGlobalIndexTablesToGPU();
    m_globalIndexTableDirty = false;
    pendingRequests &= ~Request::eUpdateGlobalIndexTable;
  }

  // Ensure GPU descriptor array exists (for splat data addresses and transforms)
  updateGpuDescriptorArray();
  uploadGpuDescriptorArray();
  m_gpuDescriptorsDirty = false;

  // Build BLAS (unit AABB or unit icosahedron) on GPU
  ParticleAccelerationStructureHelperGpu::BlasCreateInfo blasInfo{};
  if(prmRtxData.useAABBs)
  {
    blasInfo.geometryType   = ParticleAccelerationStructureHelperGpu::GeometryType::eAabbs;
    blasInfo.aabbBufferSize = sizeof(VkAabbPositionsKHR);
    blasInfo.aabbCount      = 1;
  }
  else
  {
    blasInfo.geometryType     = ParticleAccelerationStructureHelperGpu::GeometryType::eTriangles;
    blasInfo.vertexBufferSize = sizeof(glm::vec3) * 12;
    blasInfo.indexBufferSize  = sizeof(uint32_t) * 60;
    blasInfo.vertexCount      = 12;
    blasInfo.indexCount       = 60;
    blasInfo.vertexStride     = sizeof(glm::vec3);
    blasInfo.vertexFormat     = VK_FORMAT_R32G32B32_SFLOAT;
  }
  blasInfo.blasBuildFlags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  if(prmRtxData.compressBlas)
    blasInfo.blasBuildFlags |=
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;

  auto recordComputeBlas = [&](VkCommandBuffer cmd) {
    if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
      return;

    shaderio::ParticleAsBuildPushConstants pc{};
    pc.splatSetDescriptorAddress = getGPUDescriptorArrayAddress();
    pc.globalIndexTableAddress   = getGlobalIndexTableAddress();
    pc.aabbBufferAddress         = m_particleAsHelper.getAabbBufferAddress();
    pc.vertexBufferAddress       = m_particleAsHelper.getVertexBufferAddress();
    pc.indexBufferAddress        = m_particleAsHelper.getIndexBufferAddress();
    pc.instanceCount             = 1;
    pc.geometryType              = prmRtxData.useAABBs ? 1u : 0u;
    pc.writeGeometry             = 1u;
    pc.kernelDegree              = static_cast<uint32_t>(prmRtx.kernelDegree);
    pc.kernelMinResponse         = prmRtx.kernelMinResponse;
    pc.kernelAdaptiveClamping    = prmRtx.kernelAdaptiveClamping ? 1u : 0u;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                            &m_particleAsDescriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
  };

  VkResult blasResult = m_particleAsHelper.createBlasOnly(blasInfo, recordComputeBlas);
  if(blasResult != VK_SUCCESS)
  {
    handleRtxBuildFailure("GPU particle BLAS build failed.");
    return false;
  }

  // Build TLAS array (multi-TLAS if needed)
  const uint64_t tlasCount = (totalSplats + maxInstances - 1) / maxInstances;
  LOGI("GPU particle AS TLAS build (count=%llu)\n", static_cast<unsigned long long>(tlasCount));
  m_particleAsTlasHelpers.clear();
  m_particleAsTlasHelpers.resize(tlasCount);

  m_rtAccelerationStructures.tlasList.clear();
  m_rtAccelerationStructures.tlasInstancesArrays.clear();

  m_rtAccelerationStructures.tlasCount      = static_cast<uint32_t>(tlasCount);
  m_rtAccelerationStructures.totalSizeBytes = 0;

  std::vector<uint64_t> tlasAddresses(tlasCount);
  std::vector<uint32_t> tlasOffsets(tlasCount);

  for(uint64_t tlasIdx = 0; tlasIdx < tlasCount; ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalSplats - baseIndex));

    auto& tlasHelper = m_particleAsTlasHelpers[tlasIdx];
    tlasHelper.init(m_alloc, m_app->getQueue(0));

    ParticleAccelerationStructureHelperGpu::TlasCreateInfo tlasInfo{};
    tlasInfo.instanceCount = instanceCount;
    tlasInfo.tlasBuildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

    auto recordComputeTlas = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getGPUDescriptorArrayAddress();
      pc.globalIndexTableAddress   = getGlobalIndexTableAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.aabbBufferAddress         = m_particleAsHelper.getAabbBufferAddress();
      pc.blasAddress               = m_particleAsHelper.getBlas().address;
      pc.vertexBufferAddress       = m_particleAsHelper.getVertexBufferAddress();
      pc.indexBufferAddress        = m_particleAsHelper.getIndexBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.geometryType              = prmRtxData.useAABBs ? 1u : 0u;
      pc.kernelDegree              = static_cast<uint32_t>(prmRtx.kernelDegree);
      pc.kernelMinResponse         = prmRtx.kernelMinResponse;
      pc.kernelAdaptiveClamping    = prmRtx.kernelAdaptiveClamping ? 1u : 0u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                              &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    VkResult tlasResult = tlasHelper.createTlasOnly(tlasInfo, recordComputeTlas);
    if(tlasResult != VK_SUCCESS)
    {
      handleRtxBuildFailure("GPU particle TLAS build failed.");
      return false;
    }

    tlasAddresses[tlasIdx] = tlasHelper.getTlas().address;
    tlasOffsets[tlasIdx]   = baseIndex;
    m_rtAccelerationStructures.totalSizeBytes += tlasHelper.getTlas().buffer.bufferSize;
  }

  if(m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasAddressBuffer);
  if(m_rtAccelerationStructures.tlasOffsetBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasOffsetBuffer);

  const VkDeviceSize tlasAddressBufferSize = tlasCount * sizeof(uint64_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasAddressBuffer, tlasAddressBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasAddressBuffer.buffer);

  const VkDeviceSize tlasOffsetBufferSize = tlasCount * sizeof(uint32_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, tlasOffsetBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasOffsetBuffer.buffer);

  VkCommandBuffer uploadCmd = m_app->createTempCmdBuffer();
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasAddressBuffer, 0, std::span<uint64_t>(tlasAddresses));
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, 0, std::span<uint32_t>(tlasOffsets));
  m_uploader->cmdUploadAppended(uploadCmd);
  m_app->submitAndWaitTempCmdBuffer(uploadCmd);
  m_uploader->releaseStaging();

  // Update per-splat set status (single BLAS used for all instances)
  for(auto& splatSet : m_splatSets)
  {
    if(splatSet)
    {
      splatSet->rtxStatus     = RtxStatus::eSuccess;
      splatSet->blasSizeBytes = m_particleAsHelper.getBlas().buffer.bufferSize;
    }
  }

  memRaytracing.usedBlas             = m_particleAsHelper.getBlas().buffer.bufferSize;
  memRaytracing.blasScratchBuffer    = m_particleAsHelper.getBlasScratchBufferSize();
  memRaytracing.usedTlas             = m_rtAccelerationStructures.totalSizeBytes;
  memRaytracing.tlasInstancesBuffers = 0;
  memRaytracing.tlasScratchBuffers   = 0;
  for(const auto& tlasHelper : m_particleAsTlasHelpers)
  {
    memRaytracing.tlasInstancesBuffers += tlasHelper.getInstanceBufferSize();
    memRaytracing.tlasScratchBuffers += tlasHelper.getTlasScratchBufferSize();
  }

  m_rtxState = RtxState::eRtxValid;
  return false;  // Skip Phase 4 — descriptors already rebuilt
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildBlasAndTlasPerInstanceSingleBlas(VkBuildAccelerationStructureFlagsKHR blasFlags)
{
  clearRtxDescriptorArray();
  const uint32_t totalInstances = static_cast<uint32_t>(m_instances.size());
  const uint32_t maxInstances   = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);
  const uint64_t tlasCount      = (totalInstances + maxInstances - 1) / maxInstances;

  // Build per-splat-set BLAS on GPU (single BLAS per splat set)
  m_useGpuBlasForSplatSets = true;
  m_particleAsBlasHelpers.clear();
  m_particleAsBlasHelpers.resize(m_splatSets.size());
  for(size_t splatSetIdx = 0; splatSetIdx < m_splatSets.size(); ++splatSetIdx)
  {
    auto& splatSet = m_splatSets[splatSetIdx];
    if(!splatSet)
      continue;

    auto& blasHelper = m_particleAsBlasHelpers[splatSetIdx];
    blasHelper.init(m_alloc, m_app->getQueue(0));

    ParticleAccelerationStructureHelperGpu::BlasCreateInfo blasInfo{};
    const uint32_t                                         splatCount = splatSet->splatCount;
    LOGI("GPU particle BLAS build (per-splat-set, splatSet=%zu, splats=%u, mode=%s)\n", splatSetIdx, splatCount,
         prmRtxData.useAABBs ? "AABB" : "Icosahedron");
    if(prmRtxData.useAABBs)
    {
      blasInfo.geometryType   = ParticleAccelerationStructureHelperGpu::GeometryType::eAabbs;
      blasInfo.aabbBufferSize = sizeof(VkAabbPositionsKHR) * splatCount;
      blasInfo.aabbCount      = splatCount;
    }
    else
    {
      blasInfo.geometryType     = ParticleAccelerationStructureHelperGpu::GeometryType::eTriangles;
      blasInfo.vertexBufferSize = sizeof(glm::vec3) * 12 * splatCount;
      blasInfo.indexBufferSize  = sizeof(uint32_t) * 60 * splatCount;
      blasInfo.vertexCount      = 12 * splatCount;
      blasInfo.indexCount       = 60 * splatCount;
      blasInfo.vertexStride     = sizeof(glm::vec3);
      blasInfo.vertexFormat     = VK_FORMAT_R32G32B32_SFLOAT;
    }
    blasInfo.blasBuildFlags = blasFlags;

    auto recordComputeBlas = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getSplatSetDescriptorArrayAddress();
      pc.aabbBufferAddress         = blasHelper.getAabbBufferAddress();
      pc.vertexBufferAddress       = blasHelper.getVertexBufferAddress();
      pc.indexBufferAddress        = blasHelper.getIndexBufferAddress();
      pc.instanceCount             = splatCount;
      pc.instanceBaseIndex         = static_cast<uint32_t>(splatSetIdx);
      pc.geometryType              = prmRtxData.useAABBs ? 1u : 0u;
      pc.writeGeometry             = 1u;
      pc.geometryMode              = 2u;
      pc.kernelDegree              = static_cast<uint32_t>(prmRtx.kernelDegree);
      pc.kernelMinResponse         = prmRtx.kernelMinResponse;
      pc.kernelAdaptiveClamping    = prmRtx.kernelAdaptiveClamping ? 1u : 0u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                              &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t totalVerts = prmRtxData.useAABBs ? splatCount : (splatCount * 12);
      const uint32_t totalInds  = prmRtxData.useAABBs ? 0u : (splatCount * 60);
      const uint32_t totalCount = totalVerts > totalInds ? totalVerts : totalInds;
      const uint32_t groupCount = (totalCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    VkResult blasResult = blasHelper.createBlasOnly(blasInfo, recordComputeBlas);
    if(blasResult != VK_SUCCESS)
    {
      handleRtxBuildFailure("GPU particle BLAS build failed (per-splat-set).");
      return false;
    }

    splatSet->rtxStatus     = RtxStatus::eSuccess;
    splatSet->blasSizeBytes = blasHelper.getBlas().buffer.bufferSize;
  }

  // Ensure GPU descriptor array exists (includes BLAS address per splat set)
  updateGpuDescriptorArray();
  uploadGpuDescriptorArray();
  m_gpuDescriptorsDirty = false;

  // Validate BLAS addresses before building TLAS
  bool allBlasValid = true;
  for(const auto& desc : m_gpuDescriptorArray)
  {
    if(desc.blasAddress == 0)
    {
      allBlasValid = false;
      break;
    }
  }
  if(!allBlasValid)
  {
    LOGW("GPU particle AS per-instance TLAS build failed (missing BLAS address). RTX disabled.\n");
    rtxDeinitAccelerationStructures();
    m_rtxState = RtxState::eRtxError;
    return false;
  }

  LOGI("GPU particle AS TLAS build (per-instance, count=%llu)\n", static_cast<unsigned long long>(tlasCount));
  for(auto& tlasHelper : m_particleAsTlasHelpers)
  {
    tlasHelper.deinitAccelerationStructures();
    tlasHelper.deinit();
  }
  m_particleAsTlasHelpers.clear();
  m_particleAsTlasHelpers.resize(tlasCount);

  m_rtAccelerationStructures.tlasList.clear();
  m_rtAccelerationStructures.tlasInstancesArrays.clear();
  m_rtAccelerationStructures.tlasCount      = static_cast<uint32_t>(tlasCount);
  m_rtAccelerationStructures.totalSizeBytes = 0;

  std::vector<uint64_t> tlasAddresses(tlasCount);
  std::vector<uint32_t> tlasOffsets(tlasCount);

  for(uint64_t tlasIdx = 0; tlasIdx < tlasCount; ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalInstances - baseIndex));
    if(instanceCount == 0)
      continue;

    auto& tlasHelper = m_particleAsTlasHelpers[tlasIdx];
    tlasHelper.init(m_alloc, m_app->getQueue(0));

    ParticleAccelerationStructureHelperGpu::TlasCreateInfo tlasInfo{};
    tlasInfo.instanceCount = instanceCount;
    tlasInfo.tlasBuildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

    auto recordComputeTlas = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getGPUDescriptorArrayAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.instanceMode              = 1u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                              &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    VkResult tlasResult = tlasHelper.createTlasOnly(tlasInfo, recordComputeTlas);
    if(tlasResult != VK_SUCCESS)
    {
      handleRtxBuildFailure("GPU particle TLAS build failed.");
      return false;
    }

    tlasAddresses[tlasIdx] = tlasHelper.getTlas().address;
    tlasOffsets[tlasIdx]   = baseIndex;
    m_rtAccelerationStructures.totalSizeBytes += tlasHelper.getTlas().buffer.bufferSize;
  }

  if(m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasAddressBuffer);
  if(m_rtAccelerationStructures.tlasOffsetBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasOffsetBuffer);

  const VkDeviceSize tlasAddressBufferSize = tlasCount * sizeof(uint64_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasAddressBuffer, tlasAddressBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasAddressBuffer.buffer);

  const VkDeviceSize tlasOffsetBufferSize = tlasCount * sizeof(uint32_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, tlasOffsetBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasOffsetBuffer.buffer);

  VkCommandBuffer uploadCmd = m_app->createTempCmdBuffer();
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasAddressBuffer, 0, std::span<uint64_t>(tlasAddresses));
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, 0, std::span<uint32_t>(tlasOffsets));
  m_uploader->cmdUploadAppended(uploadCmd);
  m_app->submitAndWaitTempCmdBuffer(uploadCmd);
  m_uploader->releaseStaging();

  memRaytracing.usedBlas          = 0;
  memRaytracing.blasScratchBuffer = 0;
  for(const auto& blasHelper : m_particleAsBlasHelpers)
  {
    memRaytracing.usedBlas += blasHelper.getBlas().buffer.bufferSize;
    memRaytracing.blasScratchBuffer += blasHelper.getBlasScratchBufferSize();
  }

  memRaytracing.usedTlas             = m_rtAccelerationStructures.totalSizeBytes;
  memRaytracing.tlasInstancesBuffers = 0;
  memRaytracing.tlasScratchBuffers   = 0;
  for(const auto& tlasHelper : m_particleAsTlasHelpers)
  {
    memRaytracing.tlasInstancesBuffers += tlasHelper.getInstanceBufferSize();
    memRaytracing.tlasScratchBuffers += tlasHelper.getTlasScratchBufferSize();
  }

  m_rtxState = RtxState::eRtxValid;
  return true;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildBlasAndTlasPerInstanceMultiBlas(VkBuildAccelerationStructureFlagsKHR blasFlags)
{
  LOGI("GPU particle BLAS build (multi-BLAS per splat set)\n");

  const uint32_t maxInstances = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);

  m_useGpuBlasForSplatSets = false;
  m_particleAsBlasChunks.clear();
  m_particleAsBlasChunkRanges.clear();
  m_particleAsBlasChunkRanges.resize(m_splatSets.size());

  // Pre-reserve to avoid moves (ParticleAccelerationStructureHelperGpu must be deinit'd before destruction)
  uint32_t totalChunkCount = 0;
  for(const auto& splatSet : m_splatSets)
  {
    if(!splatSet)
      continue;
    const uint32_t splatCount       = splatSet->splatCount;
    const uint32_t maxSplatsPerBlas = computeMaxSplatsPerGpuBlas(prmRtxData.useAABBs, blasFlags, splatCount);
    const uint32_t chunkCount       = (splatCount + maxSplatsPerBlas - 1) / maxSplatsPerBlas;
    totalChunkCount += chunkCount;
  }
  m_particleAsBlasChunks.reserve(totalChunkCount);

  for(size_t splatSetIdx = 0; splatSetIdx < m_splatSets.size(); ++splatSetIdx)
  {
    auto& splatSet = m_splatSets[splatSetIdx];
    if(!splatSet)
      continue;

    const uint32_t splatCount       = splatSet->splatCount;
    const uint32_t maxSplatsPerBlas = computeMaxSplatsPerGpuBlas(prmRtxData.useAABBs, blasFlags, splatCount);
    const uint32_t chunkCount       = (splatCount + maxSplatsPerBlas - 1) / maxSplatsPerBlas;

    LOGI("GPU particle BLAS split (splatSet=%zu, splats=%u, maxSplatsPerBLAS=%u, chunks=%u)\n", splatSetIdx, splatCount,
         maxSplatsPerBlas, chunkCount);

    ParticleAsBlasChunkRange range{};
    range.first                              = static_cast<uint32_t>(m_particleAsBlasChunks.size());
    range.count                              = chunkCount;
    m_particleAsBlasChunkRanges[splatSetIdx] = range;

    for(uint32_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx)
    {
      const uint32_t chunkBase        = chunkIdx * maxSplatsPerBlas;
      const uint32_t chunkCountSplats = std::min<uint32_t>(maxSplatsPerBlas, splatCount - chunkBase);

      m_particleAsBlasChunks.emplace_back();
      auto& chunk         = m_particleAsBlasChunks.back();
      chunk.splatSetIndex = static_cast<uint32_t>(splatSetIdx);
      chunk.splatBase     = chunkBase;
      chunk.splatCount    = chunkCountSplats;
      chunk.helper.init(m_alloc, m_app->getQueue(0));

      ParticleAccelerationStructureHelperGpu::BlasCreateInfo blasInfo{};
      if(prmRtxData.useAABBs)
      {
        blasInfo.geometryType   = ParticleAccelerationStructureHelperGpu::GeometryType::eAabbs;
        blasInfo.aabbBufferSize = sizeof(VkAabbPositionsKHR) * chunkCountSplats;
        blasInfo.aabbCount      = chunkCountSplats;
      }
      else
      {
        blasInfo.geometryType     = ParticleAccelerationStructureHelperGpu::GeometryType::eTriangles;
        blasInfo.vertexBufferSize = sizeof(glm::vec3) * 12 * chunkCountSplats;
        blasInfo.indexBufferSize  = sizeof(uint32_t) * 60 * chunkCountSplats;
        blasInfo.vertexCount      = 12 * chunkCountSplats;
        blasInfo.indexCount       = 60 * chunkCountSplats;
        blasInfo.vertexStride     = sizeof(glm::vec3);
        blasInfo.vertexFormat     = VK_FORMAT_R32G32B32_SFLOAT;
      }
      blasInfo.blasBuildFlags = blasFlags;

      // --- VRAM budget pre-check: bail out before GPU build if insufficient free VRAM ---
      {
        auto         blasSizes     = estimateBlasBuildSizes(prmRtxData.useAABBs, blasFlags, chunkCountSplats);
        VkDeviceSize geometrySize  = blasInfo.aabbBufferSize + blasInfo.vertexBufferSize + blasInfo.indexBufferSize;
        VkDeviceSize estimatedPeak = geometrySize + blasSizes.accelerationStructureSize + blasSizes.buildScratchSize;

        VRAMSummary  vram     = queryVRAMSummary(m_app->getPhysicalDevice());
        VkDeviceSize freeVram = (vram.budgetBytes > vram.usedBytes) ? (vram.budgetBytes - vram.usedBytes) : 0;

        if(estimatedPeak > freeVram)
        {
          LOGE("GPU particle BLAS chunk %u/%u aborted: estimated peak VRAM %s exceeds free VRAM %s (budget %s, used %s).\n",
               chunkIdx + 1, chunkCount, formatMemorySize(estimatedPeak).c_str(), formatMemorySize(freeVram).c_str(),
               formatMemorySize(vram.budgetBytes).c_str(), formatMemorySize(vram.usedBytes).c_str());
          for(auto& builtChunk : m_particleAsBlasChunks)
          {
            builtChunk.helper.deinitAccelerationStructures();
            builtChunk.helper.deinit();
          }
          m_particleAsBlasChunks.clear();
          m_particleAsBlasChunkRanges.clear();
          handleRtxBuildFailure("GPU particle BLAS build aborted: insufficient VRAM for next chunk.");
          return false;
        }
      }

      auto recordComputeBlas = [&](VkCommandBuffer cmd) {
        if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
          return;

        shaderio::ParticleAsBuildPushConstants pc{};
        pc.splatSetDescriptorAddress = getSplatSetDescriptorArrayAddress();
        pc.aabbBufferAddress         = chunk.helper.getAabbBufferAddress();
        pc.vertexBufferAddress       = chunk.helper.getVertexBufferAddress();
        pc.indexBufferAddress        = chunk.helper.getIndexBufferAddress();
        pc.instanceCount             = chunkCountSplats;
        pc.instanceBaseIndex         = static_cast<uint32_t>(splatSetIdx);
        pc.splatBaseIndex            = chunkBase;
        pc.geometryType              = prmRtxData.useAABBs ? 1u : 0u;
        pc.writeGeometry             = 1u;
        pc.geometryMode              = 2u;
        pc.kernelDegree              = static_cast<uint32_t>(prmRtx.kernelDegree);
        pc.kernelMinResponse         = prmRtx.kernelMinResponse;
        pc.kernelAdaptiveClamping    = prmRtx.kernelAdaptiveClamping ? 1u : 0u;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                                &m_particleAsDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        const uint32_t totalVerts = prmRtxData.useAABBs ? chunkCountSplats : (chunkCountSplats * 12);
        const uint32_t totalInds  = prmRtxData.useAABBs ? 0u : (chunkCountSplats * 60);
        const uint32_t totalCount = totalVerts > totalInds ? totalVerts : totalInds;
        const uint32_t groupCount = (totalCount + 255) / 256;
        vkCmdDispatch(cmd, groupCount, 1, 1);
      };

      VkResult blasResult = chunk.helper.createBlasOnly(blasInfo, recordComputeBlas);
      if(blasResult != VK_SUCCESS)
      {
        for(auto& builtChunk : m_particleAsBlasChunks)
        {
          builtChunk.helper.deinitAccelerationStructures();
          builtChunk.helper.deinit();
        }
        m_particleAsBlasChunks.clear();
        m_particleAsBlasChunkRanges.clear();
        handleRtxBuildFailure("GPU particle BLAS build failed (split).");
        return false;
      }
    }

    splatSet->rtxStatus     = RtxStatus::eSuccess;
    splatSet->blasSizeBytes = 0;
  }

  // Build per-TLAS-instance RTX descriptor array (one descriptor per BLAS chunk).
  // This removes the need for descIndex/globalOffset mapping buffers.
  m_gpuRtxDescriptorArray.clear();

  // Build/upload RTX descriptor array for the split-BLAS path.
  rebuildRtxDescriptorArrayFromChunks();

  const uint32_t totalInstances = static_cast<uint32_t>(m_gpuRtxDescriptorArray.size());
  const uint64_t tlasCount      = (totalInstances + maxInstances - 1) / maxInstances;

  if(totalInstances == 0)
  {
    LOGW("GPU particle AS multi-BLAS TLAS build skipped (no instances)\n");
    m_rtxState = RtxState::eRtxNone;
    return false;
  }

  LOGI("GPU particle AS TLAS build (multi-BLAS, instances=%u, tlasCount=%llu)\n", totalInstances,
       static_cast<unsigned long long>(tlasCount));

  for(auto& tlasHelper : m_particleAsTlasHelpers)
  {
    tlasHelper.deinitAccelerationStructures();
    tlasHelper.deinit();
  }
  m_particleAsTlasHelpers.clear();
  m_particleAsTlasHelpers.resize(tlasCount);

  m_rtAccelerationStructures.tlasList.clear();
  m_rtAccelerationStructures.tlasInstancesArrays.clear();
  m_rtAccelerationStructures.tlasCount      = static_cast<uint32_t>(tlasCount);
  m_rtAccelerationStructures.totalSizeBytes = 0;

  std::vector<uint64_t> tlasAddresses(tlasCount);
  std::vector<uint32_t> tlasOffsets(tlasCount);

  for(uint64_t tlasIdx = 0; tlasIdx < tlasCount; ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalInstances - baseIndex));
    if(instanceCount == 0)
      continue;

    auto& tlasHelper = m_particleAsTlasHelpers[tlasIdx];
    tlasHelper.init(m_alloc, m_app->getQueue(0));

    ParticleAccelerationStructureHelperGpu::TlasCreateInfo tlasInfo{};
    tlasInfo.instanceCount = instanceCount;
    tlasInfo.tlasBuildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

    auto recordComputeTlas = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getRtxDescriptorArrayAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.instanceMode              = 1u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                              &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    VkResult tlasResult = tlasHelper.createTlasOnly(tlasInfo, recordComputeTlas);
    if(tlasResult != VK_SUCCESS)
    {
      handleRtxBuildFailure("GPU particle TLAS build failed (split).");
      return false;
    }

    tlasAddresses[tlasIdx] = tlasHelper.getTlas().address;
    tlasOffsets[tlasIdx]   = baseIndex;
    m_rtAccelerationStructures.totalSizeBytes += tlasHelper.getTlas().buffer.bufferSize;
  }

  if(m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasAddressBuffer);
  if(m_rtAccelerationStructures.tlasOffsetBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasOffsetBuffer);

  const VkDeviceSize tlasAddressBufferSize = tlasCount * sizeof(uint64_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasAddressBuffer, tlasAddressBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasAddressBuffer.buffer);

  const VkDeviceSize tlasOffsetBufferSize = tlasCount * sizeof(uint32_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, tlasOffsetBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasOffsetBuffer.buffer);

  VkCommandBuffer uploadCmd = m_app->createTempCmdBuffer();
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasAddressBuffer, 0, std::span<uint64_t>(tlasAddresses));
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, 0, std::span<uint32_t>(tlasOffsets));
  m_uploader->cmdUploadAppended(uploadCmd);
  m_app->submitAndWaitTempCmdBuffer(uploadCmd);
  m_uploader->releaseStaging();

  memRaytracing.usedBlas          = 0;
  memRaytracing.blasScratchBuffer = 0;
  for(const auto& chunk : m_particleAsBlasChunks)
  {
    memRaytracing.usedBlas += chunk.helper.getBlas().buffer.bufferSize;
    memRaytracing.blasScratchBuffer += chunk.helper.getBlasScratchBufferSize();
  }

  for(auto& splatSet : m_splatSets)
  {
    if(splatSet)
    {
      const uint32_t splatSetIdx = static_cast<uint32_t>(splatSet->index);
      if(splatSetIdx < m_particleAsBlasChunkRanges.size())
      {
        const auto& range      = m_particleAsBlasChunkRanges[splatSetIdx];
        size_t      totalBytes = 0;
        for(uint32_t i = 0; i < range.count; ++i)
        {
          totalBytes += m_particleAsBlasChunks[range.first + i].helper.getBlas().buffer.bufferSize;
        }
        splatSet->blasSizeBytes = totalBytes;
      }
    }
  }

  memRaytracing.usedTlas             = m_rtAccelerationStructures.totalSizeBytes;
  memRaytracing.tlasInstancesBuffers = 0;
  memRaytracing.tlasScratchBuffers   = 0;
  for(const auto& tlasHelper : m_particleAsTlasHelpers)
  {
    memRaytracing.tlasInstancesBuffers += tlasHelper.getInstanceBufferSize();
    memRaytracing.tlasScratchBuffers += tlasHelper.getTlasScratchBufferSize();
  }

  m_rtxState = RtxState::eRtxValid;
  return true;
}

//-----------------------------------------------------------------------------
// Section 3b: TLAS-only rebuild (BLAS already exists, instance count changed).
// Returns false to skip Phase 4 (early exit on error or completion), true to continue.
//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildTlas()
{
  if(m_instances.empty())
  {
    // No instances left - deinit RTX structures instead of rebuilding
    std::cout << "processVramUpdates: No instances, deinitializing RTX structures" << std::endl;
    rtxDeinitAccelerationStructures();
    m_rtxState = RtxState::eRtxNone;  // Not an error, just intentional cleanup
    pendingRequests &= ~Request::eRebuildTLAS;
    pendingRequests &= ~Request::eUpdateTransformsOnly;
    return true;
  }

  bool result;
  if(prmRtxData.useTlasInstances && m_particleAsHelper.getBlas().accel != VK_NULL_HANDLE)
  {
    result = rtxRebuildTlasPerSplat();
  }
  else if(!prmRtxData.useTlasInstances && !m_particleAsTlasHelpers.empty())
  {
    result = rtxRebuildTlasPerInstance();
  }
  else
  {
    result = rtxRebuildTlasFallbackCleanup();
  }

  pendingRequests &= ~Request::eRebuildTLAS;
  pendingRequests &= ~Request::eUpdateTransformsOnly;
  // Note: Don't clear m_tlasNeedsFullRebuild here - it's decremented in the dispatcher
  return result;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildTlasPerSplat()
{
  LOGI("GPU AS TLAS rebuild (per-splat)\n");

  const uint32_t totalSplats  = getTotalGlobalSplatCount();
  const uint32_t maxInstances = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);
  const uint64_t tlasCount    = (totalSplats + maxInstances - 1) / maxInstances;

  // Ensure global index tables are up to date for GPU instance generation
  if(static_cast<uint32_t>(pendingRequests & Request::eUpdateGlobalIndexTable) || m_globalIndexTableDirty)
  {
    rebuildGlobalIndexTables();
    uploadGlobalIndexTablesToGPU();
    m_globalIndexTableDirty = false;
    pendingRequests &= ~Request::eUpdateGlobalIndexTable;
  }

  // Ensure GPU descriptor array exists (for splat data addresses and transforms)
  updateGpuDescriptorArray();
  uploadGpuDescriptorArray();
  m_gpuDescriptorsDirty = false;

  for(auto& tlasHelper : m_particleAsTlasHelpers)
  {
    tlasHelper.deinitAccelerationStructures();
    tlasHelper.deinit();
  }
  m_particleAsTlasHelpers.clear();
  m_particleAsTlasHelpers.resize(tlasCount);

  m_rtAccelerationStructures.tlasList.clear();
  m_rtAccelerationStructures.tlasInstancesArrays.clear();
  m_rtAccelerationStructures.tlasCount      = static_cast<uint32_t>(tlasCount);
  m_rtAccelerationStructures.totalSizeBytes = 0;

  std::vector<uint64_t> tlasAddresses(tlasCount);
  std::vector<uint32_t> tlasOffsets(tlasCount);

  for(uint64_t tlasIdx = 0; tlasIdx < tlasCount; ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalSplats - baseIndex));

    auto& tlasHelper = m_particleAsTlasHelpers[tlasIdx];
    tlasHelper.init(m_alloc, m_app->getQueue(0));

    ParticleAccelerationStructureHelperGpu::TlasCreateInfo tlasInfo{};
    tlasInfo.instanceCount = instanceCount;
    tlasInfo.tlasBuildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

    auto recordComputeTlas = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getGPUDescriptorArrayAddress();
      pc.globalIndexTableAddress   = getGlobalIndexTableAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.aabbBufferAddress         = m_particleAsHelper.getAabbBufferAddress();
      pc.blasAddress               = m_particleAsHelper.getBlas().address;
      pc.vertexBufferAddress       = m_particleAsHelper.getVertexBufferAddress();
      pc.indexBufferAddress        = m_particleAsHelper.getIndexBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.geometryType              = prmRtxData.useAABBs ? 1u : 0u;
      pc.kernelDegree              = static_cast<uint32_t>(prmRtx.kernelDegree);
      pc.kernelMinResponse         = prmRtx.kernelMinResponse;
      pc.kernelAdaptiveClamping    = prmRtx.kernelAdaptiveClamping ? 1u : 0u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                              &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    VkResult tlasResult = tlasHelper.createTlasOnly(tlasInfo, recordComputeTlas);
    if(tlasResult != VK_SUCCESS)
    {
      LOGE("GPU particle TLAS rebuild failed: %s\n", string_VkResult(tlasResult));
      m_rtxState = RtxState::eRtxError;
      return false;
    }

    tlasAddresses[tlasIdx] = tlasHelper.getTlas().address;
    tlasOffsets[tlasIdx]   = baseIndex;
    m_rtAccelerationStructures.totalSizeBytes += tlasHelper.getTlas().buffer.bufferSize;
  }

  if(m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasAddressBuffer);
  if(m_rtAccelerationStructures.tlasOffsetBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasOffsetBuffer);

  const VkDeviceSize tlasAddressBufferSize = tlasCount * sizeof(uint64_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasAddressBuffer, tlasAddressBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasAddressBuffer.buffer);

  const VkDeviceSize tlasOffsetBufferSize = tlasCount * sizeof(uint32_t);
  NVVK_CHECK(m_alloc->createBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, tlasOffsetBufferSize,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_rtAccelerationStructures.tlasOffsetBuffer.buffer);

  VkCommandBuffer uploadCmd = m_app->createTempCmdBuffer();
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasAddressBuffer, 0, std::span<uint64_t>(tlasAddresses));
  m_uploader->appendBuffer(m_rtAccelerationStructures.tlasOffsetBuffer, 0, std::span<uint32_t>(tlasOffsets));
  m_uploader->cmdUploadAppended(uploadCmd);
  m_app->submitAndWaitTempCmdBuffer(uploadCmd);
  m_uploader->releaseStaging();

  return true;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildTlasPerInstance()
{
  // Detect multi-BLAS (chunk) mode vs single-BLAS per splat set
  const bool useSplitBlas = !m_particleAsBlasChunks.empty() && !m_particleAsBlasChunkRanges.empty();

  // Refresh standard descriptors (transforms) first
  updateGpuDescriptorArray();
  uploadGpuDescriptorArray();
  m_gpuDescriptorsDirty = false;

  // In multi-BLAS mode, also refresh RTX descriptors (one per chunk, with BLAS addresses + transforms)
  if(useSplitBlas)
  {
    rebuildRtxDescriptorArrayFromChunks();
  }

  // Total TLAS instances: in multi-BLAS mode = number of chunks, otherwise = number of user instances
  const uint32_t totalInstances =
      useSplitBlas ? static_cast<uint32_t>(m_gpuRtxDescriptorArray.size()) : static_cast<uint32_t>(m_instances.size());
  const uint32_t maxInstances = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);

  LOGI("GPU AS TLAS rebuild (per-instance, splitBlas=%d, instances=%u)\n", useSplitBlas ? 1 : 0, totalInstances);

  for(size_t tlasIdx = 0; tlasIdx < m_particleAsTlasHelpers.size(); ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalInstances - baseIndex));
    if(instanceCount == 0)
      continue;

    auto& tlasHelper    = m_particleAsTlasHelpers[tlasIdx];
    auto  recordCompute = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      // Use RTX descriptor array in multi-BLAS mode, standard array otherwise
      pc.splatSetDescriptorAddress = useSplitBlas ? getRtxDescriptorArrayAddress() : getGPUDescriptorArrayAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.instanceMode              = 1u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                               &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    tlasHelper.updateTlasOnly(recordCompute);
  }

  return false;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxRebuildTlasFallbackCleanup()
{
  std::cout << "processVramUpdates: Rebuilding unified TLAS (instance count changed)" << std::endl;

  // Rebuild global index tables BEFORE building TLAS (needed for shader-side splat ID resolution)
  if(static_cast<uint32_t>(pendingRequests & Request::eUpdateGlobalIndexTable) || m_globalIndexTableDirty)
  {
    std::cout << "processVramUpdates: Rebuilding global index tables (pre-TLAS)" << std::endl;
    rebuildGlobalIndexTables();
    uploadGlobalIndexTablesToGPU();
    m_globalIndexTableDirty = false;
    pendingRequests &= ~Request::eUpdateGlobalIndexTable;
  }

  // Must deinitialize existing TLAS before rebuilding (can't build over existing one)
  // Note: This only deinits TLAS, not BLAS (BLAS are still valid)
  // Clear old TLAS structures before rebuild
  for(auto& tlasHelper : m_rtAccelerationStructures.tlasList)
  {
    tlasHelper.deinitAccelerationStructures();  // Free TLAS/BLAS buffers
    tlasHelper.deinit();                        // Free command pool
  }
  m_rtAccelerationStructures.tlasList.clear();
  m_rtAccelerationStructures.tlasInstancesArrays.clear();
  if(m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasAddressBuffer);
    m_rtAccelerationStructures.tlasAddressBuffer = {};  // Reset to empty state
  }
  if(m_rtAccelerationStructures.tlasOffsetBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasOffsetBuffer);
    m_rtAccelerationStructures.tlasOffsetBuffer = {};  // Reset to empty state
  }
  LOGW("GPU particle AS TLAS rebuild skipped (no valid splat sets or instances). RTX disabled.\n");
  m_rtxState = RtxState::eRtxNone;

  return true;
}

//-----------------------------------------------------------------------------
// Section 3c: Fast TLAS update path for transform changes only.
// Updates existing TLAS instances with new transforms from descriptor array.
// Returns false to skip Phase 4 (early exit on error), true to continue.
//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxUpdateTlasTransforms(bool& descriptorsNeedRebuild)
{
  LOGI("[TRACE] Section 3c entered: instances=%zu useTlasInstances=%d tlasHelpers=%zu\n", m_instances.size(),
       prmRtxData.useTlasInstances, m_particleAsTlasHelpers.size());

  if(m_instances.empty())
  {
    pendingRequests &= ~Request::eUpdateTransformsOnly;
    return true;
  }

  if(m_particleAsTlasHelpers.empty())
  {
    assert(false && "rtxUpdateTlasTransforms: unexpected fallback - m_particleAsTlasHelpers is empty during transform-only update");
    pendingRequests &= ~Request::eUpdateTransformsOnly;
    return true;
  }

  // Refresh descriptor array (transforms) if needed — common to all paths
  if(descriptorsNeedRebuild || static_cast<uint32_t>(pendingRequests & Request::eUpdateDescriptors) || m_gpuDescriptorsDirty)
  {
    LOGI("GPU AS TLAS update: refreshing descriptor array\n");
    updateGpuDescriptorArray();
    uploadGpuDescriptorArray();
    m_gpuDescriptorsDirty = false;
    pendingRequests &= ~Request::eUpdateDescriptors;
    descriptorsNeedRebuild = false;
  }

  bool result;
  if(prmRtxData.useTlasInstances)
  {
    result = rtxUpdateTlasPerSplat();
  }
  else
  {
    const bool useSplitBlas = !m_particleAsBlasChunks.empty() && !m_particleAsBlasChunkRanges.empty();
    if(useSplitBlas)
      result = rtxUpdateTlasPerInstanceMultiBlas();
    else
      result = rtxUpdateTlasPerInstanceSingleBlas();
  }

  if(result)
  {
    // CRITICAL: Wait for device idle after TLAS update
    // Ensures all GPU work is complete before returning control to main loop.
    // Without this, continuous dragging can cause the GPU to fall behind, leading to
    // race conditions and progressive performance degradation.
    vkDeviceWaitIdle(m_app->getDevice());
  }

  pendingRequests &= ~Request::eUpdateTransformsOnly;
  return result;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxUpdateTlasPerSplat()
{
  // Ensure RTX does not use a stale split-BLAS descriptor array in per-splat TLAS mode.
  // Even though chunks don't exist in per-splat mode, calling this ensures proper cleanup:
  // clears m_gpuRtxDescriptorArray, destroys m_rtxDescriptorBuffer if it exists, and sets
  // m_useSplitBlasRtxDescriptors = false.
  rebuildRtxDescriptorArrayFromChunks();
  LOGI("GPU AS TLAS update (per-splat, count=%zu)\n", m_particleAsTlasHelpers.size());

  // Note: Global index tables and TLAS address/offset buffers are not updated here because:
  // - A rebuild always happens before any update (copy/import triggers eRebuildBLAS)
  // - The rebuild path already updates global index tables and creates TLAS address/offset buffers
  // - During a refit (update), only instance transforms change; TLAS addresses remain the same
  // - Therefore, these buffers are already correct and don't need to be refreshed

  const uint32_t totalSplats  = getTotalGlobalSplatCount();
  const uint32_t maxInstances = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);
  if(totalSplats == 0)
    return true;

  for(size_t tlasIdx = 0; tlasIdx < m_particleAsTlasHelpers.size(); ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalSplats - baseIndex));

    if(instanceCount == 0)
      continue;

    auto& tlasHelper    = m_particleAsTlasHelpers[tlasIdx];
    auto  recordCompute = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getGPUDescriptorArrayAddress();
      pc.globalIndexTableAddress   = getGlobalIndexTableAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.aabbBufferAddress         = m_particleAsHelper.getAabbBufferAddress();
      pc.blasAddress               = m_particleAsHelper.getBlas().address;
      pc.vertexBufferAddress       = m_particleAsHelper.getVertexBufferAddress();
      pc.indexBufferAddress        = m_particleAsHelper.getIndexBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.geometryType              = prmRtxData.useAABBs ? 1u : 0u;
      pc.kernelDegree              = static_cast<uint32_t>(prmRtx.kernelDegree);
      pc.kernelMinResponse         = prmRtx.kernelMinResponse;
      pc.kernelAdaptiveClamping    = prmRtx.kernelAdaptiveClamping ? 1u : 0u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                               &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    // Check for errors from updateTlasOnly
    VkResult updateResult = tlasHelper.updateTlasOnly(recordCompute);
    if(updateResult != VK_SUCCESS)
    {
      LOGE("GPU particle TLAS update failed: %s\n", string_VkResult(updateResult));
      m_rtxState = RtxState::eRtxError;
      return false;
    }
  }

  return true;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxUpdateTlasPerInstanceMultiBlas()
{
  // Multi-BLAS path: rebuild RTX descriptor array from chunks (picks up updated transforms)
  rebuildRtxDescriptorArrayFromChunks();

  const uint32_t totalInstances = static_cast<uint32_t>(m_gpuRtxDescriptorArray.size());
  const uint32_t maxInstances   = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);
  LOGI("GPU AS TLAS update (per-instance, multi-BLAS, instances=%u)\n", totalInstances);

  if(totalInstances == 0)
    return true;

  for(size_t tlasIdx = 0; tlasIdx < m_particleAsTlasHelpers.size(); ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalInstances - baseIndex));
    if(instanceCount == 0)
      continue;

    auto& tlasHelper    = m_particleAsTlasHelpers[tlasIdx];
    auto  recordCompute = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getRtxDescriptorArrayAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.instanceMode              = 1u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                               &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    tlasHelper.updateTlasOnly(recordCompute);
  }
  return true;
}

//-----------------------------------------------------------------------------
bool SplatSetManagerVk::rtxUpdateTlasPerInstanceSingleBlas()
{
  LOGI("GPU AS TLAS update (per-instance, single-BLAS, count=%zu)\n", m_particleAsTlasHelpers.size());

  // Validate BLAS addresses in standard descriptor array
  for(const auto& desc : m_gpuDescriptorArray)
  {
    if(desc.blasAddress == 0)
    {
      LOGW("GPU particle AS per-instance TLAS update skipped (missing BLAS address).\n");
      m_rtxState = RtxState::eRtxError;
      return false;
    }
  }

  const uint32_t totalInstances = static_cast<uint32_t>(m_instances.size());
  const uint32_t maxInstances   = static_cast<uint32_t>(m_accelStructProps->maxInstanceCount);
  if(totalInstances == 0)
    return true;

  for(size_t tlasIdx = 0; tlasIdx < m_particleAsTlasHelpers.size(); ++tlasIdx)
  {
    uint32_t baseIndex     = static_cast<uint32_t>(tlasIdx * maxInstances);
    uint32_t instanceCount = static_cast<uint32_t>(std::min<uint64_t>(maxInstances, totalInstances - baseIndex));
    if(instanceCount == 0)
      continue;

    auto& tlasHelper    = m_particleAsTlasHelpers[tlasIdx];
    auto  recordCompute = [&](VkCommandBuffer cmd) {
      if(m_particleAsComputePipeline == VK_NULL_HANDLE || m_particleAsPipelineLayout == VK_NULL_HANDLE)
        return;

      shaderio::ParticleAsBuildPushConstants pc{};
      pc.splatSetDescriptorAddress = getGPUDescriptorArrayAddress();
      pc.tlasInstanceBufferAddress = tlasHelper.getInstanceBufferAddress();
      pc.instanceCount             = instanceCount;
      pc.instanceBaseIndex         = baseIndex;
      pc.instanceMode              = 1u;

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsComputePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_particleAsPipelineLayout, 0, 1,
                               &m_particleAsDescriptorSet, 0, nullptr);
      vkCmdPushConstants(cmd, m_particleAsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

      const uint32_t groupCount = (instanceCount + 255) / 256;
      vkCmdDispatch(cmd, groupCount, 1, 1);
    };

    tlasHelper.updateTlasOnly(recordCompute);
  }
  return true;
}


//-----------------------------------------------------------------------------
// Global Index Table
//-----------------------------------------------------------------------------

bool SplatSetManagerVk::updateGlobalIndexTablesIfNeeded()
{
  if(!m_globalIndexTableDirty)
    return false;

  rebuildGlobalIndexTables();
  uploadGlobalIndexTablesToGPU();

  m_globalIndexTableDirty = false;

  // Only return true if we actually have data (otherwise assets buffer doesn't need updating)
  return !m_globalIndexTable.empty();
}

VkDeviceAddress SplatSetManagerVk::getGlobalIndexTableAddress() const
{
  return m_globalIndexTableBuffer.address;
}

VkDeviceAddress SplatSetManagerVk::getSplatSetGlobalIndexTableAddress() const
{
  return m_splatSetGlobalIndexTableBuffer.address;
}

void SplatSetManagerVk::rebuildGlobalIndexTables()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");
  m_totalGlobalSplatCount = 0;

  // If no instances, return early
  if(m_instances.empty())
  {
    m_instanceInfos.resize(0);
    m_globalIndexTable.clear();
    m_splatSetGlobalIndexTable.clear();
    std::cout << "Global Index Tables: 0 instances, 0 total splats" << std::endl;
    return;
  }

  // First pass: compute total size and per-instance offsets (reuse m_instanceInfos to avoid reallocation)
  m_instanceInfos.resize(0);  // Clear without deallocating

  uint32_t totalSplats = 0;
  uint32_t splatSetIdx = 0;
  for(const auto& instance : m_instances)
  {
    if(!instance || !instance->splatSet)
      continue;
    m_instanceInfos.push_back({splatSetIdx, instance->splatSet->splatCount, totalSplats});
    totalSplats += instance->splatSet->splatCount;
    ++splatSetIdx;
  }

  // Resize tables to exact size (avoids repeated push_back reallocations)
  m_globalIndexTable.resize(totalSplats);
  m_splatSetGlobalIndexTable.resize(m_instanceInfos.size());
  m_totalGlobalSplatCount = totalSplats;

  // Second pass: fill splatSetGlobalIndexTable offsets
  for(size_t i = 0; i < m_instanceInfos.size(); ++i)
  {
    m_splatSetGlobalIndexTable[i] = m_instanceInfos[i].globalOffset;
  }

  // Third pass: fill globalIndexTable in parallel (one parallel batch per instance)
  for(const auto& info : m_instanceInfos)
  {
    const uint32_t setIdx = info.splatSetIdx;
    const uint32_t offset = info.globalOffset;
    const uint32_t count  = info.splatCount;

    START_PAR_LOOP(count, splatIdx)
    {
      m_globalIndexTable[offset + splatIdx].splatSetIndex = setIdx;
      m_globalIndexTable[offset + splatIdx].splatIndex    = static_cast<uint32_t>(splatIdx);
    }
    END_PAR_LOOP();
  }

  std::cout << "Global Index Tables rebuilt: " << m_instances.size() << " instances, " << m_totalGlobalSplatCount
            << " total splats" << std::endl;
}

void SplatSetManagerVk::uploadGlobalIndexTablesToGPU()
{
  // CRITICAL: Destroy old buffers BEFORE creating new ones to avoid address reuse issues
  // These are large buffers (~400 MB each for 100M splats), same as sorting buffers
  if(m_globalIndexTableBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_globalIndexTableBuffer);
  }
  if(m_splatSetGlobalIndexTableBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSetGlobalIndexTableBuffer);
  }

  m_globalIndexTableBuffer         = {};
  m_splatSetGlobalIndexTableBuffer = {};

  // Upload globalIndexTable
  if(!m_globalIndexTable.empty())
  {
    VkDeviceSize requiredSize = m_globalIndexTable.size() * sizeof(GlobalSplatIndexEntry);

    NVVK_CHECK(m_alloc->createBuffer(m_globalIndexTableBuffer, requiredSize,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));

    // Upload data
    VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_uploader->appendBuffer(m_globalIndexTableBuffer, 0, std::span(m_globalIndexTable)));
    m_uploader->cmdUploadAppended(cmdBuf);
    m_app->submitAndWaitTempCmdBuffer(cmdBuf);
    m_uploader->releaseStaging();

    // Track memory
    memModels.globalIndexTableBuffer = requiredSize;
  }
  else
  {
    memModels.globalIndexTableBuffer = 0;
  }

  // Upload splatSetGlobalIndexTable
  if(!m_splatSetGlobalIndexTable.empty())
  {
    VkDeviceSize requiredSize = m_splatSetGlobalIndexTable.size() * sizeof(uint32_t);

    NVVK_CHECK(m_alloc->createBuffer(m_splatSetGlobalIndexTableBuffer, requiredSize,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));

    // Upload data
    VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_uploader->appendBuffer(m_splatSetGlobalIndexTableBuffer, 0, std::span(m_splatSetGlobalIndexTable)));
    m_uploader->cmdUploadAppended(cmdBuf);
    m_app->submitAndWaitTempCmdBuffer(cmdBuf);
    m_uploader->releaseStaging();

    // Track memory
    memModels.splatSetIndexTableBuffer = requiredSize;
  }
  else
  {
    memModels.splatSetIndexTableBuffer = 0;
  }

  // Update sorting buffers (same lifetime as global index tables)
  // Check if we need to resize sorting buffers
  if(m_totalGlobalSplatCount != m_sortingBuffersAllocatedCount)
  {
    std::cout << "Updating sorting buffers (" << m_sortingBuffersAllocatedCount << " -> " << m_totalGlobalSplatCount
              << " splats)" << std::endl;

    // Destroy old VRDX sorter first (must be destroyed before buffers)
    if(m_splatSortingVrdxSorter != VK_NULL_HANDLE)
    {
      vrdxDestroySorter(m_splatSortingVrdxSorter);
      m_splatSortingVrdxSorter = VK_NULL_HANDLE;
    }

    if(m_splatSortingIndicesHost.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(m_splatSortingIndicesHost);
    }
    if(m_splatSortingIndicesDevice.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(m_splatSortingIndicesDevice);
    }
    if(m_splatSortingDistancesDevice.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(m_splatSortingDistancesDevice);
    }
    if(m_splatSortingVrdxStorageBuffer.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(m_splatSortingVrdxStorageBuffer);
    }

    m_splatSortingIndicesHost       = {};
    m_splatSortingIndicesDevice     = {};
    m_splatSortingDistancesDevice   = {};
    m_splatSortingVrdxStorageBuffer = {};

    // Create new sorting buffers if we have splats
    if(m_totalGlobalSplatCount > 0)
    {
      const VkDeviceSize minBufferSize = 16;  // Minimum allocation (avoid 0-size buffers)
      const VkDeviceSize bufferSize =
          m_totalGlobalSplatCount > 0 ? ((m_totalGlobalSplatCount * sizeof(uint32_t) + 15) / 16) * 16 : minBufferSize;

      // Host buffer for CPU sorting (mapped, host visible)
      NVVK_CHECK(m_alloc->createBuffer(m_splatSortingIndicesHost, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));

      // Device indices buffer (used by shaders and CPU/GPU sorting)
      NVVK_CHECK(m_alloc->createBuffer(m_splatSortingIndicesDevice, bufferSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                           | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));

      // Device distances buffer (written by dist.comp.slang)
      NVVK_CHECK(m_alloc->createBuffer(m_splatSortingDistancesDevice, bufferSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                           | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));

      // Create VRDX sorter
      VrdxSorterCreateInfo gpuSorterInfo{.physicalDevice = m_app->getPhysicalDevice(), .device = m_app->getDevice()};
      vrdxCreateSorter(&gpuSorterInfo, &m_splatSortingVrdxSorter);

      // Create VRDX storage buffer
      const uint32_t                vrdxSplatCount = m_totalGlobalSplatCount;  // VRDX uses uint32_t
      VrdxSorterStorageRequirements requirements;
      vrdxGetSorterKeyValueStorageRequirements(m_splatSortingVrdxSorter, vrdxSplatCount, &requirements);

      NVVK_CHECK(m_alloc->createBuffer(m_splatSortingVrdxStorageBuffer, requirements.size, requirements.usage,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));

      std::cout << "Created sorting buffers: " << bufferSize << " bytes each, VRDX storage: " << requirements.size
                << " bytes" << std::endl;

      // Update sorting memory statistics (allocation sizes)
      memRasterization.hostAllocIndices        = bufferSize;
      memRasterization.hostAllocDistances      = 0;  // Distances not on host
      memRasterization.deviceAllocIndices      = bufferSize;
      memRasterization.deviceAllocDistances    = bufferSize;
      memRasterization.deviceAllocVrdxInternal = requirements.size;

      // Usage will be updated per-frame by updateRenderingMemoryStatistics()
    }
    else
    {
      // No splats: clear statistics
      memRasterization.hostAllocIndices        = 0;
      memRasterization.hostAllocDistances      = 0;
      memRasterization.deviceAllocIndices      = 0;
      memRasterization.deviceAllocDistances    = 0;
      memRasterization.deviceAllocVrdxInternal = 0;
    }

    m_sortingBuffersAllocatedCount = m_totalGlobalSplatCount;
  }
}

//-----------------------------------------------------------------------------
// GPU Descriptor Buffer
//-----------------------------------------------------------------------------

bool SplatSetManagerVk::updateGPUDescriptorsIfNeeded(bool forceUpdate)
{
  if(!m_gpuDescriptorsDirty && !forceUpdate)
    return false;

  updateGpuDescriptorArray();
  uploadGpuDescriptorArray();

  m_gpuDescriptorsDirty = false;

  // Only return true if we actually have descriptors (otherwise assets buffer doesn't need updating)
  return !m_gpuDescriptorArray.empty();
}

VkDeviceAddress SplatSetManagerVk::getDescriptorBufferAddress() const
{
  return m_descriptorBuffer.address;
}

VkDeviceAddress SplatSetManagerVk::getSplatSetDescriptorArrayAddress() const
{
  return m_splatSetDescriptorBuffer.address;
}

void SplatSetManagerVk::updateGpuDescriptorArray()
{
  m_gpuDescriptorArray.resize(m_instances.size());

  if(m_instances.empty())
  {
    std::cout << "GPU Descriptor Array: 0 instances" << std::endl;
    return;
  }

  // Assign texture indices to each splat set (shared by all instances)
  // Multi-splat-set texture mode: Each splat set using STORAGE_TEXTURES gets 6 consecutive texture indices
  // Splat sets using STORAGE_BUFFERS get 0 for all texture indices (unused)
  {
    uint32_t splatSetTextureIdx = 0;  // Only increments for texture-mode splat sets
    for(auto& splatSet : m_splatSets)
    {
      if(!splatSet)
        continue;

      if(splatSet->dataStorage == STORAGE_TEXTURES)
      {
        // This splat set uses textures - assign actual indices
        splatSet->textureIndexCenters     = splatSetTextureIdx * 6 + 0;
        splatSet->textureIndexScales      = splatSetTextureIdx * 6 + 1;
        splatSet->textureIndexRotations   = splatSetTextureIdx * 6 + 2;
        splatSet->textureIndexColors      = splatSetTextureIdx * 6 + 3;
        splatSet->textureIndexCovariances = splatSetTextureIdx * 6 + 4;
        splatSet->textureIndexSH          = splatSetTextureIdx * 6 + 5;
        ++splatSetTextureIdx;
      }
      else
      {
        // This splat set uses buffers - set texture indices to 0 (unused)
        splatSet->textureIndexCenters     = 0;
        splatSet->textureIndexScales      = 0;
        splatSet->textureIndexRotations   = 0;
        splatSet->textureIndexColors      = 0;
        splatSet->textureIndexCovariances = 0;
        splatSet->textureIndexSH          = 0;
      }
    }
  }

  size_t idx = 0;
  for(const auto& instance : m_instances)
  {
    if(!instance || !instance->splatSet)
      continue;
    const auto& splatSet = instance->splatSet;

    //std::cout << "  Building descriptor[" << idx << "] for instance index=" << instance->index
    //          << ", splatSet index=" << splatSet->index << ", splatCount=" << splatSet->splatCount
    //          << ", textureBaseIndex=" << splatSet->textureIndexCenters / 6 << std::endl;

    auto& descriptor = m_gpuDescriptorArray[idx];

    instance->rebuildDescriptor(splatSet.get(), descriptor);

    // Global splat base for RTX any-hit: global index table base per instance.
    if(idx < m_splatSetGlobalIndexTable.size())
    {
      descriptor.globalSplatBase = m_splatSetGlobalIndexTable[idx];
    }

    //std::cout << "    Descriptor: centersAddress=0x" << std::hex << (uint64_t)descriptor.centersAddress
    //          << ", colorsAddress=0x" << (uint64_t)descriptor.colorsAddress << ", storage=" << std::dec
    //          << descriptor.storage << ", format=" << descriptor.format << std::endl;

    // Fill per-splat-set BLAS address for non-instance TLAS mode
    descriptor.blasAddress = 0;
    if(m_useGpuBlasForSplatSets)
    {
      if(splatSet->index < m_particleAsBlasHelpers.size() && m_particleAsBlasHelpers[splatSet->index].getBlas().accel != VK_NULL_HANDLE)
      {
        descriptor.blasAddress = m_particleAsBlasHelpers[splatSet->index].getBlas().address;
      }
    }
    else if(splatSet->blasIndex != UINT32_MAX && splatSet->blasIndex < m_rtAccelerationStructures.helper.blasSet.size())
    {
      descriptor.blasAddress = m_rtAccelerationStructures.helper.blasSet[splatSet->blasIndex].address;
    }

    // Update instance's GPU descriptor index
    instance->gpuDescriptorIndex = static_cast<uint32_t>(idx);

    ++idx;  // Increment index for next iteration
  }

  std::cout << "GPU Descriptor Array rebuilt: " << m_gpuDescriptorArray.size() << " descriptors" << std::endl;
}

void SplatSetManagerVk::updateGpuSplatSetDescriptorArray()
{
  m_gpuSplatSetDescriptorArray.resize(m_splatSets.size());

  if(m_splatSets.empty())
  {
    std::cout << "GPU SplatSet Descriptor Array: 0 splat sets" << std::endl;
    return;
  }

  // Assign texture indices to each splat set (shared by all instances)
  uint32_t splatSetTextureIdx = 0;
  for(auto& splatSet : m_splatSets)
  {
    if(!splatSet)
      continue;
    if(splatSet->dataStorage == STORAGE_TEXTURES)
    {
      splatSet->textureIndexCenters     = splatSetTextureIdx * 6 + 0;
      splatSet->textureIndexScales      = splatSetTextureIdx * 6 + 1;
      splatSet->textureIndexRotations   = splatSetTextureIdx * 6 + 2;
      splatSet->textureIndexColors      = splatSetTextureIdx * 6 + 3;
      splatSet->textureIndexCovariances = splatSetTextureIdx * 6 + 4;
      splatSet->textureIndexSH          = splatSetTextureIdx * 6 + 5;
      ++splatSetTextureIdx;
    }
    else
    {
      splatSet->textureIndexCenters     = 0;
      splatSet->textureIndexScales      = 0;
      splatSet->textureIndexRotations   = 0;
      splatSet->textureIndexColors      = 0;
      splatSet->textureIndexCovariances = 0;
      splatSet->textureIndexSH          = 0;
    }
  }

  for(size_t idx = 0; idx < m_splatSets.size(); ++idx)
  {
    const auto& splatSet = m_splatSets[idx];
    if(!splatSet)
      continue;

    shaderio::SplatSetDesc desc{};
    desc.centersAddress     = reinterpret_cast<float*>(splatSet->centersBuffer.address);
    desc.colorsAddress      = splatSet->colorsBuffer.address;
    desc.scalesAddress      = reinterpret_cast<float*>(splatSet->scalesBuffer.address);
    desc.rotationsAddress   = reinterpret_cast<float*>(splatSet->rotationsBuffer.address);
    desc.shAddress          = splatSet->sphericalHarmonicsBuffer.address;
    desc.covariancesAddress = reinterpret_cast<float*>(splatSet->covariancesBuffer.address);

    desc.centersTexture     = splatSet->textureIndexCenters;
    desc.scalesTexture      = splatSet->textureIndexScales;
    desc.rotationsTexture   = splatSet->textureIndexRotations;
    desc.colorsTexture      = splatSet->textureIndexColors;
    desc.covariancesTexture = splatSet->textureIndexCovariances;
    desc.shTexture          = splatSet->textureIndexSH;

    desc.splatCount      = splatSet->splatCount;
    desc.shDegree        = splatSet->shDegree;
    desc.splatBase       = 0;
    desc.globalSplatBase = 0;
    desc.storage         = splatSet->dataStorage;
    desc.format          = splatSet->shFormat;
    desc.rgbaFormat      = splatSet->rgbaFormat;

    desc.transform                = glm::mat4(1.0f);
    desc.transformInverse         = glm::mat4(1.0f);
    desc.transformRotScaleInverse = glm::mat3(1.0f);

    m_gpuSplatSetDescriptorArray[idx] = desc;
  }

  std::cout << "GPU SplatSet Descriptor Array rebuilt: " << m_gpuSplatSetDescriptorArray.size() << " descriptors" << std::endl;
}

void SplatSetManagerVk::uploadGpuDescriptorArray()
{
  std::cout << "uploadGPUDescriptorArray: Starting (array size=" << m_gpuDescriptorArray.size() << ")" << std::endl;

  if(!m_gpuDescriptorArray.empty())
  {
    VkDeviceSize requiredSize = m_gpuDescriptorArray.size() * sizeof(shaderio::SplatSetDesc);
    std::cout << "  Required buffer size: " << requiredSize << " bytes" << std::endl;

    // Check if we can reuse the existing buffer (same size or larger)
    bool canReuseBuffer = (m_descriptorBuffer.buffer != VK_NULL_HANDLE) && (m_descriptorBuffer.bufferSize >= requiredSize);

    if(canReuseBuffer)
    {
      std::cout << "  Reusing existing buffer (size=" << m_descriptorBuffer.bufferSize << " bytes)" << std::endl;
    }
    else
    {
      // Need to allocate new buffer
      if(m_descriptorBuffer.buffer != VK_NULL_HANDLE)
      {
        m_alloc->destroyBuffer(m_descriptorBuffer);
      }

      m_descriptorBuffer = {};

      std::cout << "  Creating new descriptor buffer" << std::endl;
      NVVK_CHECK(m_alloc->createBuffer(m_descriptorBuffer, requiredSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
      NVVK_DBG_NAME(m_descriptorBuffer.buffer);
      std::cout << "  Buffer created: address=0x" << std::hex << m_descriptorBuffer.address << std::dec << std::endl;
    }

    // Track memory (always update, whether buffer is new or reused)
    memModels.descriptorBuffer = m_descriptorBuffer.bufferSize;

    // Upload data to buffer (new or reused)
    VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();
    std::cout << "  Uploading data to GPU..." << std::endl;
    NVVK_CHECK(m_uploader->appendBuffer(m_descriptorBuffer, 0, std::span(m_gpuDescriptorArray)));
    m_uploader->cmdUploadAppended(cmdBuf);
    m_app->submitAndWaitTempCmdBuffer(cmdBuf);
    m_uploader->releaseStaging();
    std::cout << "  Upload complete" << std::endl;
  }
  else
  {
    std::cout << "  Array empty" << std::endl;
    memModels.descriptorBuffer = 0;
    // No data needed, destroy buffer if it exists
    if(m_descriptorBuffer.buffer != VK_NULL_HANDLE)
    {
      std::cout << "  Destroying descriptor buffer (no longer needed)" << std::endl;
      m_alloc->destroyBuffer(m_descriptorBuffer);
      m_descriptorBuffer = {};
    }
  }

  std::cout << "uploadGPUDescriptorArray: Complete" << std::endl;
}

void SplatSetManagerVk::rebuildRtxDescriptorArrayFromChunks()
{
  // Build per-TLAS-instance descriptors for split-BLAS RTX path.
  // One descriptor per BLAS chunk, so InstanceID() maps directly to descriptor index.
  m_gpuRtxDescriptorArray.clear();

  for(uint32_t instanceIdx = 0; instanceIdx < m_instances.size(); ++instanceIdx)
  {
    const auto& instance = m_instances[instanceIdx];
    if(!instance || !instance->splatSet)
      continue;

    const uint32_t splatSetIdx = static_cast<uint32_t>(instance->splatSet->index);
    if(splatSetIdx >= m_particleAsBlasChunkRanges.size())
      continue;

    const auto& range = m_particleAsBlasChunkRanges[splatSetIdx];
    if(range.count == 0)
      continue;

    const uint32_t baseOffset = (instanceIdx < m_splatSetGlobalIndexTable.size()) ? m_splatSetGlobalIndexTable[instanceIdx] : 0u;

    for(uint32_t i = 0; i < range.count; ++i)
    {
      const auto& chunk = m_particleAsBlasChunks[range.first + i];

      shaderio::SplatSetDesc desc{};
      instance->rebuildDescriptor(instance->splatSet.get(), desc);

      // Per-chunk identity: local base within splat set and global base across all instances.
      // This replaces the TLAS instance mapping tables for RTX hit shaders.
      desc.splatBase       = chunk.splatBase;
      desc.globalSplatBase = baseOffset + chunk.splatBase;
      desc.splatCount      = chunk.splatCount;
      desc.blasAddress     = chunk.helper.getBlas().address;

      m_gpuRtxDescriptorArray.push_back(desc);
    }
  }

  if(m_gpuRtxDescriptorArray.empty())
  {
    if(m_rtxDescriptorBuffer.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(m_rtxDescriptorBuffer);
      m_rtxDescriptorBuffer = {};
    }
    m_useSplitBlasRtxDescriptors = false;
    return;
  }

  const VkDeviceSize requiredSize = m_gpuRtxDescriptorArray.size() * sizeof(shaderio::SplatSetDesc);
  if(m_rtxDescriptorBuffer.buffer != VK_NULL_HANDLE && m_rtxDescriptorBuffer.bufferSize < requiredSize)
  {
    m_alloc->destroyBuffer(m_rtxDescriptorBuffer);
    m_rtxDescriptorBuffer = {};
  }
  if(m_rtxDescriptorBuffer.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_rtxDescriptorBuffer, requiredSize,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_DBG_NAME(m_rtxDescriptorBuffer.buffer);
  }

  VkCommandBuffer mapCmd = m_app->createTempCmdBuffer();
  m_uploader->appendBuffer(m_rtxDescriptorBuffer, 0, std::span(m_gpuRtxDescriptorArray));
  m_uploader->cmdUploadAppended(mapCmd);
  m_app->submitAndWaitTempCmdBuffer(mapCmd);
  m_uploader->releaseStaging();

  // Mark RTX descriptors as active so ray tracing uses the per-TLAS-instance array.
  m_useSplitBlasRtxDescriptors = true;
}

void SplatSetManagerVk::uploadGpuSplatSetDescriptorArray()
{
  std::cout << "uploadGPU SplatSet Descriptor Array: Starting (array size=" << m_gpuSplatSetDescriptorArray.size()
            << ")" << std::endl;

  if(!m_gpuSplatSetDescriptorArray.empty())
  {
    VkDeviceSize requiredSize = m_gpuSplatSetDescriptorArray.size() * sizeof(shaderio::SplatSetDesc);

    bool canReuseBuffer =
        (m_splatSetDescriptorBuffer.buffer != VK_NULL_HANDLE) && (m_splatSetDescriptorBuffer.bufferSize >= requiredSize);
    if(!canReuseBuffer)
    {
      if(m_splatSetDescriptorBuffer.buffer != VK_NULL_HANDLE)
      {
        m_alloc->destroyBuffer(m_splatSetDescriptorBuffer);
      }
      m_splatSetDescriptorBuffer = {};
      NVVK_CHECK(m_alloc->createBuffer(m_splatSetDescriptorBuffer, requiredSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
      NVVK_DBG_NAME(m_splatSetDescriptorBuffer.buffer);
    }

    VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_uploader->appendBuffer(m_splatSetDescriptorBuffer, 0, std::span(m_gpuSplatSetDescriptorArray)));
    m_uploader->cmdUploadAppended(cmdBuf);
    m_app->submitAndWaitTempCmdBuffer(cmdBuf);
    m_uploader->releaseStaging();
  }
  else if(m_splatSetDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSetDescriptorBuffer);
    m_splatSetDescriptorBuffer = {};
  }

  std::cout << "uploadGPU SplatSet Descriptor Array: Complete" << std::endl;
}

void SplatSetManagerVk::clearSceneGpuBuffers()
{
  std::cout << "clearSceneGpuBuffers: releasing scene-scoped GPU buffers" << std::endl;
  if(m_descriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_descriptorBuffer);
    m_descriptorBuffer = {};
  }
  if(m_splatSetDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSetDescriptorBuffer);
    m_splatSetDescriptorBuffer = {};
  }
  if(m_rtxDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtxDescriptorBuffer);
    m_rtxDescriptorBuffer = {};
  }
  m_useSplitBlasRtxDescriptors = false;
  if(m_rtxDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtxDescriptorBuffer);
    m_rtxDescriptorBuffer = {};
  }

  if(m_globalIndexTableBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_globalIndexTableBuffer);
    m_globalIndexTableBuffer = {};
  }
  if(m_splatSetGlobalIndexTableBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSetGlobalIndexTableBuffer);
    m_splatSetGlobalIndexTableBuffer = {};
  }

  if(m_splatSortingVrdxSorter != VK_NULL_HANDLE)
  {
    vrdxDestroySorter(m_splatSortingVrdxSorter);
    m_splatSortingVrdxSorter = VK_NULL_HANDLE;
  }
  if(m_splatSortingIndicesHost.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSortingIndicesHost);
    m_splatSortingIndicesHost = {};
  }
  if(m_splatSortingIndicesDevice.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSortingIndicesDevice);
    m_splatSortingIndicesDevice = {};
  }
  if(m_splatSortingDistancesDevice.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSortingDistancesDevice);
    m_splatSortingDistancesDevice = {};
  }
  if(m_splatSortingVrdxStorageBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSortingVrdxStorageBuffer);
    m_splatSortingVrdxStorageBuffer = {};
  }
  m_sortingBuffersAllocatedCount = 0;

  m_gpuDescriptorArray.clear();
  m_gpuRtxDescriptorArray.clear();
  m_gpuSplatSetDescriptorArray.clear();
  m_globalIndexTable.clear();
  m_splatSetGlobalIndexTable.clear();
  m_splatIndices.clear();
  m_totalGlobalSplatCount = 0;

  memModels.descriptorBuffer         = 0;
  memModels.globalIndexTableBuffer   = 0;
  memModels.splatSetIndexTableBuffer = 0;

  memRasterization.hostAllocIndices        = 0;
  memRasterization.hostAllocDistances      = 0;
  memRasterization.deviceAllocIndices      = 0;
  memRasterization.deviceAllocDistances    = 0;
  memRasterization.deviceAllocVrdxInternal = 0;
}

//-----------------------------------------------------------------------------
// Ray Tracing
//-----------------------------------------------------------------------------

void SplatSetManagerVk::markSplatSetsForRegeneration(std::shared_ptr<SplatSetVk>& splatSet)
{
  if(!splatSet)
    return;

  std::cout << "markSplatSetForRegeneration: Marking one splat sets for regeneration" << std::endl;

  // Mark ALL splat sets for data regeneration
  splatSet->flags |= SplatSetVk::Flags::eDataChanged;

  // Request full regeneration (all splat sets need GPU reupload + BLAS rebuild)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildBLAS;
  pendingRequests |= Request::eUpdateGlobalIndexTable;
}

void SplatSetManagerVk::markAllSplatSetsForRegeneration()
{
  if(m_splatSets.empty())
    return;

  std::cout << "markAllSplatSetsForRegeneration: Marking " << m_splatSets.size() << " splat sets for regeneration" << std::endl;

  // Mark ALL splat sets for data regeneration
  for(auto& splatSet : m_splatSets)
  {
    if(!splatSet)
      continue;
    splatSet->flags |= SplatSetVk::Flags::eDataChanged;
  }

  // Request full regeneration (all splat sets need GPU reupload + BLAS rebuild)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildBLAS;
  pendingRequests |= Request::eUpdateGlobalIndexTable;
}

void SplatSetManagerVk::rtxDeinitAccelerationStructures()
{
  // Wait for GPU to finish using old acceleration structures before destroying them
  // This is critical when switching modes (instanced <-> non-instanced) or rebuilding BLAS
  vkDeviceWaitIdle(m_app->getDevice());

  // Destroy all TLAS in the array (must call both deinitAccelerationStructures AND deinit)
  for(auto& tlasHelper : m_rtAccelerationStructures.tlasList)
  {
    tlasHelper.deinitAccelerationStructures();  // Free TLAS/BLAS buffers
    tlasHelper.deinit();                        // Free command pool
  }
  m_rtAccelerationStructures.tlasList.clear();
  m_rtAccelerationStructures.tlasInstancesArrays.clear();

  // Destroy TLAS address buffer
  if(m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasAddressBuffer);
    m_rtAccelerationStructures.tlasAddressBuffer = {};  // Reset to empty state
  }

  if(m_rtAccelerationStructures.tlasOffsetBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_rtAccelerationStructures.tlasOffsetBuffer);
    m_rtAccelerationStructures.tlasOffsetBuffer = {};  // Reset to empty state
  }

  // Destroy all BLAS (required before rebuilding)
  m_rtAccelerationStructures.helper.deinitAccelerationStructures();

  // Destroy GPU-only particle AS (if used)
  m_particleAsHelper.deinitAccelerationStructures();
  for(auto& blasHelper : m_particleAsBlasHelpers)
  {
    blasHelper.deinitAccelerationStructures();
    blasHelper.deinit();
  }
  m_particleAsBlasHelpers.clear();
  for(auto& chunk : m_particleAsBlasChunks)
  {
    chunk.helper.deinitAccelerationStructures();
    chunk.helper.deinit();
  }
  m_particleAsBlasChunks.clear();
  m_particleAsBlasChunkRanges.clear();
  if(m_splatSetDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_splatSetDescriptorBuffer);
    m_splatSetDescriptorBuffer = {};
  }
  for(auto& tlasHelper : m_particleAsTlasHelpers)
  {
    tlasHelper.deinitAccelerationStructures();
    tlasHelper.deinit();
  }
  m_particleAsTlasHelpers.clear();
  m_useGpuBlasForSplatSets = false;

  m_gpuRtxDescriptorArray.clear();

  m_rtAccelerationStructures.tlasCount      = 0;
  m_rtAccelerationStructures.totalSizeBytes = 0;

  // Reset all splat sets' RTX status to eDelayed (not eError)
  // This clears any previous error states when RTX structures are destroyed
  // Note: m_rtxState is NOT set here - let the caller decide based on context
  // (intentional cleanup = eRtxNone vs error cleanup = eRtxError)
  for(auto& splatSet : m_splatSets)
  {
    if(splatSet)
      splatSet->rtxStatus = RtxStatus::eDelayed;
  }

  // Reset RTX acceleration structure memory tracking (but keep geometry buffers)
  memRaytracing.usedTlas             = 0;
  memRaytracing.usedBlas             = 0;
  memRaytracing.tlasAddressBuffer    = 0;
  memRaytracing.tlasOffsetBuffer     = 0;
  memRaytracing.blasScratchBuffer    = 0;
  memRaytracing.tlasInstancesBuffers = 0;
  memRaytracing.tlasScratchBuffers   = 0;
}

VkDeviceAddress SplatSetManagerVk::getTlasAddress() const
{
  // Return address of TLAS address buffer (for multi-TLAS bindless access)
  if(m_rtxState == RtxState::eRtxValid && m_rtAccelerationStructures.tlasAddressBuffer.buffer != VK_NULL_HANDLE)
  {
    return m_rtAccelerationStructures.tlasAddressBuffer.address;
  }
  return 0;
}

VkDeviceAddress SplatSetManagerVk::getRtxDescriptorArrayAddress() const
{
  if(m_useSplitBlasRtxDescriptors && m_rtxDescriptorBuffer.buffer != VK_NULL_HANDLE)
  {
    return m_rtxDescriptorBuffer.address;
  }
  return 0;
}

size_t SplatSetManagerVk::getTlasSizeBytes() const
{
  // Return total size of all TLAS in the array
  if(m_rtxState == RtxState::eRtxValid)
  {
    return m_rtAccelerationStructures.totalSizeBytes;
  }
  return 0;
}

size_t SplatSetManagerVk::getBlasSizeBytes() const
{
  // Sum up all BLAS sizes
  size_t totalSize = 0;
  for(const auto& splatSet : m_splatSets)
  {
    if(!splatSet)
      continue;
    totalSize += splatSet->blasSizeBytes;
  }
  return totalSize;
}

// Note: getInstanceCount() now inline in header

VkDeviceAddress SplatSetManagerVk::getGPUDescriptorArrayAddress() const
{
  return m_descriptorBuffer.address;
}

//-----------------------------------------------------------------------------
// UI Helpers
//-----------------------------------------------------------------------------

// Note: Handle list functions removed - use getSplatSets() and getInstances() directly

//-----------------------------------------------------------------------------
// Internal Helpers
//-----------------------------------------------------------------------------

void SplatSetManagerVk::markGlobalIndexTableDirty()
{
  m_globalIndexTableDirty = true;
}

void SplatSetManagerVk::markGpuDescriptorsDirty()
{
  m_gpuDescriptorsDirty = true;
}

void SplatSetManagerVk::updateMaxShDegree()
{
  m_maxShDegree = 0;
  for(const auto& splatSet : m_splatSets)
  {
    if(splatSet && !splatSet->isMarkedForDeletion())
    {
      m_maxShDegree = std::max(m_maxShDegree, splatSet->shDegree);
    }
  }
}

uint32_t SplatSetManagerVk::computeMaxSplatsPerGpuBlas(bool useAabbs, VkBuildAccelerationStructureFlagsKHR blasBuildFlags, uint32_t splatCount) const
{
  if(splatCount == 0 || !m_app)
  {
    return splatCount;
  }

  VkPhysicalDeviceMaintenance3Properties maint3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
  VkPhysicalDeviceProperties2            props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &maint3;
  vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &props2);
  const VkDeviceSize maxAlloc = maint3.maxMemoryAllocationSize;
  if(maxAlloc == 0)
  {
    return splatCount;
  }

  uint64_t maxSplatsByBuffer = splatCount;
  if(useAabbs)
  {
    const VkDeviceSize bytesPerSplat = sizeof(VkAabbPositionsKHR);
    maxSplatsByBuffer                = static_cast<uint64_t>(maxAlloc / bytesPerSplat);
  }
  else
  {
    const VkDeviceSize vertexBytesPerSplat = sizeof(glm::vec3) * 12;
    const VkDeviceSize indexBytesPerSplat  = sizeof(uint32_t) * 60;
    uint64_t           maxByVerts          = static_cast<uint64_t>(maxAlloc / vertexBytesPerSplat);
    uint64_t           maxByInds           = static_cast<uint64_t>(maxAlloc / indexBytesPerSplat);
    maxSplatsByBuffer                      = std::min(maxByVerts, maxByInds);
  }

  if(maxSplatsByBuffer == 0)
    return 1;

  uint32_t maxCandidate = static_cast<uint32_t>(std::min<uint64_t>(maxSplatsByBuffer, splatCount));

  uint32_t low  = 1;
  uint32_t high = maxCandidate;
  while(low < high)
  {
    uint32_t mid = low + (high - low + 1) / 2;
    if(estimateBlasBuildSizes(useAabbs, blasBuildFlags, mid).accelerationStructureSize <= maxAlloc)
    {
      low = mid;
    }
    else
    {
      high = mid - 1;
    }
  }

  return std::max<uint32_t>(1, low);
}

//-----------------------------------------------------------------------------
// Estimate BLAS build sizes (AS size + scratch) for a given splat count.
// Pure CPU query to the driver via vkGetAccelerationStructureBuildSizesKHR.
//-----------------------------------------------------------------------------
VkAccelerationStructureBuildSizesInfoKHR SplatSetManagerVk::estimateBlasBuildSizes(bool useAabbs,
                                                                                   VkBuildAccelerationStructureFlagsKHR blasBuildFlags,
                                                                                   uint32_t splatCount) const
{
  VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  if(useAabbs)
  {
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
    aabbs.data.deviceAddress = 0;
    aabbs.stride             = sizeof(VkAabbPositionsKHR);
    geometry.geometryType    = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.geometry.aabbs  = aabbs;
  }
  else
  {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexStride             = sizeof(glm::vec3);
    triangles.vertexData.deviceAddress = 0;
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = 0;
    triangles.maxVertex                = (splatCount * 12) > 0 ? (splatCount * 12 - 1) : 0;
    geometry.geometryType              = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles        = triangles;
  }
  geometry.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildInfo.flags         = blasBuildFlags;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries   = &geometry;

  uint32_t                                 primitiveCount = useAabbs ? splatCount : (splatCount * 20);
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(m_app->getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &buildInfo, &primitiveCount, &sizeInfo);
  return sizeInfo;
}

//-----------------------------------------------------------------------------
// Consolidated Memory Statistics
//-----------------------------------------------------------------------------

void SplatSetManagerVk::updateConsolidatedMemoryStats()
{
  // Reset global model memory stats
  memModels = {};

  // Reset RTX geometry (accumulated from splat sets)
  memRaytracing.vertexBuffer      = 0;
  memRaytracing.indexBuffer       = 0;
  memRaytracing.aabbBuffer        = 0;
  memRaytracing.vertexBufferAlloc = 0;
  memRaytracing.indexBufferAlloc  = 0;
  memRaytracing.aabbBufferAlloc   = 0;

  // Accumulate from each splat set's local memoryStats
  for(const auto& splatSet : m_splatSets)
  {
    if(!splatSet)
      continue;

    // Source (RAM) memory
    memModels.hostAll += splatSet->memoryStats.hostAll;
    memModels.hostCenters += splatSet->memoryStats.hostCenters;
    memModels.hostScales += splatSet->memoryStats.hostScales;
    memModels.hostRotations += splatSet->memoryStats.hostRotations;
    memModels.hostCov += splatSet->memoryStats.hostCov;
    memModels.hostShAll += splatSet->memoryStats.hostShAll;
    memModels.hostSh0 += splatSet->memoryStats.hostSh0;
    memModels.hostShOther += splatSet->memoryStats.hostShOther;

    // Device used (GPU actual) memory
    memModels.deviceAllocAll += splatSet->memoryStats.deviceAllocAll;
    memModels.deviceAllocCenters += splatSet->memoryStats.deviceAllocCenters;
    memModels.deviceAllocScales += splatSet->memoryStats.deviceAllocScales;
    memModels.deviceAllocRotations += splatSet->memoryStats.deviceAllocRotations;
    memModels.deviceAllocCov += splatSet->memoryStats.deviceAllocCov;
    memModels.deviceAllocShAll += splatSet->memoryStats.deviceAllocShAll;
    memModels.deviceAllocSh0 += splatSet->memoryStats.deviceAllocSh0;
    memModels.deviceAllocShOther += splatSet->memoryStats.deviceAllocShOther;

    // Optimally device (theoretical) memory
    memModels.deviceUsedAll += splatSet->memoryStats.deviceUsedAll;
    memModels.deviceUsedCenters += splatSet->memoryStats.deviceUsedCenters;
    memModels.deviceUsedScales += splatSet->memoryStats.deviceUsedScales;
    memModels.deviceUsedRotations += splatSet->memoryStats.deviceUsedRotations;
    memModels.deviceUsedCov += splatSet->memoryStats.deviceUsedCov;
    memModels.deviceUsedShAll += splatSet->memoryStats.deviceUsedShAll;
    memModels.deviceUsedSh0 += splatSet->memoryStats.deviceUsedSh0;
    memModels.deviceUsedShOther += splatSet->memoryStats.deviceUsedShOther;

    // RTX splat model buffers (accumulate to raytracing stats)
    memRaytracing.vertexBuffer += splatSet->memoryStats.rtxVertexBuffer;
    memRaytracing.indexBuffer += splatSet->memoryStats.rtxIndexBuffer;
    memRaytracing.aabbBuffer += splatSet->memoryStats.rtxAabbBuffer;
    memRaytracing.vertexBufferAlloc += splatSet->memoryStats.rtxVertexBuffer;  // Same as used for now
    memRaytracing.indexBufferAlloc += splatSet->memoryStats.rtxIndexBuffer;    // Same as used for now
    memRaytracing.aabbBufferAlloc += splatSet->memoryStats.rtxAabbBuffer;      // Same as used for now
  }
}

//--------------------------------------------------------------------------------------------------
// CPU Async Sorting
//--------------------------------------------------------------------------------------------------

void SplatSetManagerVk::tryConsumeAndUploadCpuSortingResult(VkCommandBuffer                     cmd,
                                                            const uint32_t                      splatCount,
                                                            const glm::vec3&                    viewDir,
                                                            const glm::vec3&                    eyePos,
                                                            bool                                cpuLazySort,
                                                            bool                                opacityGaussianDisabled,
                                                            std::shared_ptr<SplatSetInstanceVk> selectedInstance,
                                                            bool                                frontToBack)
{
  NVVK_DBG_SCOPE(cmd);

  // upload CPU sorted indices to the GPU if needed
  bool newIndexAvailable = false;

  if(!opacityGaussianDisabled)
  {
    // 1. Splatting/blending is on, we check for a newly sorted index table
    auto status = m_cpuSorter.getStatus();
    if(status != SplatSorterAsync::E_SORTING)
    {
      // sorter is sleeping, we can work on shared data
      // we take into account the result of the sort
      if(status == SplatSorterAsync::E_SORTED)
      {
        m_cpuSorter.consume(m_splatIndices);
        newIndexAvailable = true;
      }

      // Build per-instance sort inputs and trigger async sort
      std::vector<SplatSorterAsync::InstanceSortInput> instanceData;
      instanceData.reserve(m_instanceInfos.size());
      uint32_t infoIdx = 0;
      for(const auto& instance : m_instances)
      {
        if(!instance || !instance->splatSet)
          continue;
        const auto& info = m_instanceInfos[infoIdx++];
        instanceData.push_back({&instance->splatSet->positions, instance->transform, info.globalOffset, info.splatCount});
      }
      m_cpuSorter.sortAsync(viewDir, eyePos, instanceData, getTotalGlobalSplatCount(), cpuLazySort, frontToBack);
    }
  }
  else
  {
    // splatting off, we disable the sorting
    // indices would not be needed for non splatted points
    // however, using the same mechanism allows to use exactly the same shader
    // so if splatting/blending is off we provide an ordered table of indices
    // if not already filled by any other previous frames (sorted or not)
    const uint32_t totalGlobalSplatCount = getTotalGlobalSplatCount();
    bool           refill                = (m_splatIndices.size() != totalGlobalSplatCount);
    if(refill)
    {
      m_splatIndices.resize(totalGlobalSplatCount);
      for(uint32_t i = 0; i < totalGlobalSplatCount; ++i)
      {
        m_splatIndices[i] = i;
      }
      newIndexAvailable = true;
    }
  }

  // 2. upload to GPU if needed
  if(newIndexAvailable)
  {
    const auto& hostBuffer   = m_splatSortingIndicesHost;
    const auto& deviceBuffer = m_splatSortingIndicesDevice;

    // Prepare buffer on host using sorted indices
    memcpy(hostBuffer.mapping, m_splatIndices.data(), m_splatIndices.size() * sizeof(uint32_t));
    // copy buffer to device
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = splatCount * sizeof(uint32_t)};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, deviceBuffer.buffer, 1, &bc);
    // sync with end of copy to device
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }
}

//-----------------------------------------------------------------------------
// Debug state dump — writes all small buffers, descriptor values, parameters,
// and AS addresses to a timestamped text file for offline comparison.
//-----------------------------------------------------------------------------
void SplatSetManagerVk::dumpDebugState(const std::string& label) const
{
  // Build timestamped filename
  auto    now = std::chrono::system_clock::now();
  auto    tt  = std::chrono::system_clock::to_time_t(now);
  auto    ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &tt);
#else
  localtime_r(&tt, &tm);
#endif
  std::string filename =
      fmt::format("debug_dump_{}_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}_{:03d}.txt", label, tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms.count()));

  std::ofstream ofs(filename);
  if(!ofs.is_open())
  {
    LOGE("dumpDebugState: failed to open %s\n", filename.c_str());
    return;
  }

  ofs << "=== SplatSetManagerVk Debug Dump ===" << std::endl;
  ofs << "Label: " << label << std::endl;
  ofs << "Timestamp: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "." << std::setw(3) << std::setfill('0')
      << ms.count() << std::endl;
  ofs << std::endl;

  // --- Pending Requests ---
  ofs << "=== Pending Requests ===" << std::endl;
  uint32_t req = static_cast<uint32_t>(pendingRequests);
  ofs << "  raw value:            0x" << std::hex << req << std::dec << std::endl;
  ofs << "  eProcessDeletions:    " << ((req & 1) ? "YES" : "no") << std::endl;
  ofs << "  eUpdateGlobalIndexTable: " << ((req & 2) ? "YES" : "no") << std::endl;
  ofs << "  eUpdateDescriptors:   " << ((req & 4) ? "YES" : "no") << std::endl;
  ofs << "  eUpdateTransformsOnly:" << ((req & 8) ? "YES" : "no") << std::endl;
  ofs << "  eRebuildTLAS:         " << ((req & 16) ? "YES" : "no") << std::endl;
  ofs << "  eRebuildBLAS:         " << ((req & 32) ? "YES" : "no") << std::endl;
  ofs << std::endl;

  // --- Dirty Flags ---
  ofs << "=== Dirty Flags ===" << std::endl;
  ofs << "  m_globalIndexTableDirty: " << m_globalIndexTableDirty << std::endl;
  ofs << "  m_gpuDescriptorsDirty:   " << m_gpuDescriptorsDirty << std::endl;
  ofs << "  m_rtxState:              "
      << (m_rtxState == RtxState::eRtxNone  ? "eRtxNone" :
          m_rtxState == RtxState::eRtxValid ? "eRtxValid" :
                                              "eRtxError")
      << std::endl;
  ofs << "  m_useSplitBlasRtxDescriptors: " << m_useSplitBlasRtxDescriptors << std::endl;
  ofs << "  m_useGpuBlasForSplatSets:     " << m_useGpuBlasForSplatSets << std::endl;
  ofs << std::endl;

  // --- prmRtxData ---
  ofs << "=== prmRtxData ===" << std::endl;
  ofs << "  compressBlas:     " << prmRtxData.compressBlas << std::endl;
  ofs << "  useAABBs:         " << prmRtxData.useAABBs << std::endl;
  ofs << "  useTlasInstances: " << prmRtxData.useTlasInstances << std::endl;
  ofs << std::endl;

  // --- Counts ---
  ofs << "=== Counts ===" << std::endl;
  ofs << "  m_splatSets.size():   " << m_splatSets.size() << std::endl;
  ofs << "  m_instances.size():   " << m_instances.size() << std::endl;
  ofs << "  m_totalGlobalSplatCount: " << m_totalGlobalSplatCount << std::endl;
  ofs << "  m_maxShDegree:        " << m_maxShDegree << std::endl;
  ofs << std::endl;

  // --- Splat Sets ---
  ofs << "=== Splat Sets ===" << std::endl;
  for(size_t i = 0; i < m_splatSets.size(); ++i)
  {
    const auto& ss = m_splatSets[i];
    if(!ss)
    {
      ofs << "  [" << i << "] nullptr" << std::endl;
      continue;
    }
    ofs << "  [" << i << "] path=\"" << ss->path << "\" index=" << ss->index << " splatCount=" << ss->splatCount
        << " shDegree=" << ss->shDegree << " rtxStatus=" << static_cast<int>(ss->rtxStatus) << " blasSizeBytes=" << ss->blasSizeBytes
        << " flags=0x" << std::hex << static_cast<uint32_t>(ss->flags) << std::dec << std::endl;
  }
  ofs << std::endl;

  // --- Instances ---
  ofs << "=== Instances ===" << std::endl;
  for(size_t i = 0; i < m_instances.size(); ++i)
  {
    const auto& inst = m_instances[i];
    if(!inst)
    {
      ofs << "  [" << i << "] nullptr" << std::endl;
      continue;
    }
    ofs << "  [" << i << "] \"" << inst->displayName << "\" index=" << inst->index << " gpuDescIdx=" << inst->gpuDescriptorIndex
        << " flags=0x" << std::hex << static_cast<uint32_t>(inst->flags) << std::dec << std::endl;
    ofs << "    splatSet=" << (inst->splatSet ? inst->splatSet->path : "null")
        << " splatCount=" << (inst->splatSet ? std::to_string(inst->splatSet->splatCount) : "?") << std::endl;
    ofs << "    translation=(" << inst->translation.x << ", " << inst->translation.y << ", " << inst->translation.z
        << ")" << std::endl;
    ofs << "    rotation=(" << inst->rotation.x << ", " << inst->rotation.y << ", " << inst->rotation.z << ")" << std::endl;
    ofs << "    scale=(" << inst->scale.x << ", " << inst->scale.y << ", " << inst->scale.z << ")" << std::endl;
    // First row of transform matrix (enough to spot identity vs. actual transform)
    ofs << "    transform[0]=(" << inst->transform[0][0] << ", " << inst->transform[0][1] << ", "
        << inst->transform[0][2] << ", " << inst->transform[0][3] << ")" << std::endl;
    ofs << "    transform[3]=(" << inst->transform[3][0] << ", " << inst->transform[3][1] << ", "
        << inst->transform[3][2] << ", " << inst->transform[3][3] << ")" << std::endl;
  }
  ofs << std::endl;

  // --- GPU Descriptor Array ---
  ofs << "=== GPU Descriptor Array (m_gpuDescriptorArray, count=" << m_gpuDescriptorArray.size() << ") ===" << std::endl;
  for(size_t i = 0; i < m_gpuDescriptorArray.size(); ++i)
  {
    const auto& d = m_gpuDescriptorArray[i];
    ofs << "  [" << i << "]" << std::endl;
    ofs << "    centersAddr=0x" << std::hex << reinterpret_cast<uint64_t>(d.centersAddress) << std::dec << std::endl;
    ofs << "    colorsAddr=0x" << std::hex << d.colorsAddress << std::dec << std::endl;
    ofs << "    scalesAddr=0x" << std::hex << reinterpret_cast<uint64_t>(d.scalesAddress) << std::dec << std::endl;
    ofs << "    rotationsAddr=0x" << std::hex << reinterpret_cast<uint64_t>(d.rotationsAddress) << std::dec << std::endl;
    ofs << "    covariancesAddr=0x" << std::hex << reinterpret_cast<uint64_t>(d.covariancesAddress) << std::dec << std::endl;
    ofs << "    shAddr=0x" << std::hex << d.shAddress << std::dec << std::endl;
    ofs << "    blasAddress=0x" << std::hex << d.blasAddress << std::dec << std::endl;
    ofs << "    splatCount=" << d.splatCount << " shDegree=" << d.shDegree << std::endl;
    ofs << "    splatBase=" << d.splatBase << " globalSplatBase=" << d.globalSplatBase << std::endl;
    ofs << "    storage=" << d.storage << " format=" << d.format << " rgbaFormat=" << d.rgbaFormat << std::endl;
    // Transform columns (glm::mat4 is column-major: [col][row])
    ofs << "    transform col0=(" << d.transform[0][0] << ", " << d.transform[0][1] << ", " << d.transform[0][2] << ", "
        << d.transform[0][3] << ")" << std::endl;
    ofs << "    transform col3=(" << d.transform[3][0] << ", " << d.transform[3][1] << ", " << d.transform[3][2] << ", "
        << d.transform[3][3] << ")  [translation column]" << std::endl;
  }
  ofs << std::endl;

  // --- RTX Descriptor Array ---
  ofs << "=== RTX Descriptor Array (m_gpuRtxDescriptorArray, count=" << m_gpuRtxDescriptorArray.size() << ") ===" << std::endl;
  for(size_t i = 0; i < m_gpuRtxDescriptorArray.size(); ++i)
  {
    const auto& d = m_gpuRtxDescriptorArray[i];
    ofs << "  [" << i << "] blasAddress=0x" << std::hex << d.blasAddress << std::dec << " splatCount=" << d.splatCount
        << " splatBase=" << d.splatBase << " globalSplatBase=" << d.globalSplatBase << std::endl;
  }
  ofs << std::endl;

  // --- SplatSet Descriptor Array ---
  ofs << "=== SplatSet Descriptor Array (m_gpuSplatSetDescriptorArray, count=" << m_gpuSplatSetDescriptorArray.size()
      << ") ===" << std::endl;
  for(size_t i = 0; i < m_gpuSplatSetDescriptorArray.size(); ++i)
  {
    const auto& d = m_gpuSplatSetDescriptorArray[i];
    ofs << "  [" << i << "] blasAddress=0x" << std::hex << d.blasAddress << std::dec << " splatCount=" << d.splatCount
        << " splatBase=" << d.splatBase << std::endl;
  }
  ofs << std::endl;

  // --- GPU Buffers ---
  ofs << "=== GPU Buffers ===" << std::endl;
  auto dumpBuf = [&](const char* name, const nvvk::Buffer& buf) {
    ofs << "  " << name << ": buffer=" << (buf.buffer ? "VALID" : "NULL") << " address=0x" << std::hex << buf.address
        << std::dec << " size=" << buf.bufferSize << std::endl;
  };
  dumpBuf("m_descriptorBuffer", m_descriptorBuffer);
  dumpBuf("m_rtxDescriptorBuffer", m_rtxDescriptorBuffer);
  dumpBuf("m_splatSetDescriptorBuffer", m_splatSetDescriptorBuffer);
  dumpBuf("m_globalIndexTableBuffer", m_globalIndexTableBuffer);
  dumpBuf("m_splatSetGlobalIndexTableBuffer", m_splatSetGlobalIndexTableBuffer);
  dumpBuf("m_splatSortingIndicesHost", m_splatSortingIndicesHost);
  dumpBuf("m_splatSortingIndicesDevice", m_splatSortingIndicesDevice);
  dumpBuf("m_splatSortingDistancesDevice", m_splatSortingDistancesDevice);
  dumpBuf("m_splatSortingVrdxStorageBuffer", m_splatSortingVrdxStorageBuffer);
  ofs << "  m_sortingBuffersAllocatedCount: " << m_sortingBuffersAllocatedCount << std::endl;
  ofs << std::endl;

  // --- Global Index Table (RAM) ---
  ofs << "=== Instance Infos (m_instanceInfos, count=" << m_instanceInfos.size() << ") ===" << std::endl;
  for(size_t i = 0; i < m_instanceInfos.size(); ++i)
  {
    const auto& info = m_instanceInfos[i];
    ofs << "  [" << i << "] splatSetIdx=" << info.splatSetIdx << " splatCount=" << info.splatCount
        << " globalOffset=" << info.globalOffset << std::endl;
  }
  ofs << "  m_globalIndexTable.size(): " << m_globalIndexTable.size() << std::endl;
  ofs << "  m_splatSetGlobalIndexTable.size(): " << m_splatSetGlobalIndexTable.size() << std::endl;
  ofs << std::endl;

  // --- RT Acceleration Structures (SplatSetTlasArray) ---
  ofs << "=== RT Acceleration Structures (m_rtAccelerationStructures) ===" << std::endl;
  ofs << "  tlasCount:      " << m_rtAccelerationStructures.tlasCount << std::endl;
  ofs << "  totalSizeBytes: " << m_rtAccelerationStructures.totalSizeBytes << std::endl;
  ofs << "  tlasList.size(): " << m_rtAccelerationStructures.tlasList.size() << std::endl;
  ofs << "  tlasInstancesArrays.size(): " << m_rtAccelerationStructures.tlasInstancesArrays.size() << std::endl;
  dumpBuf("  tlasAddressBuffer", m_rtAccelerationStructures.tlasAddressBuffer);
  dumpBuf("  tlasOffsetBuffer", m_rtAccelerationStructures.tlasOffsetBuffer);
  ofs << std::endl;

  // --- Particle AS Helper (unit BLAS for per-splat mode) ---
  ofs << "=== Particle AS Helper (m_particleAsHelper) ===" << std::endl;
  ofs << "  BLAS accel:   " << (m_particleAsHelper.getBlas().accel ? "VALID" : "NULL") << std::endl;
  ofs << "  BLAS address: 0x" << std::hex << m_particleAsHelper.getBlas().address << std::dec << std::endl;
  ofs << "  BLAS size:    " << m_particleAsHelper.getBlas().buffer.bufferSize << std::endl;
  ofs << "  TLAS accel:   " << (m_particleAsHelper.getTlas().accel ? "VALID" : "NULL") << std::endl;
  ofs << "  TLAS address: 0x" << std::hex << m_particleAsHelper.getTlas().address << std::dec << std::endl;
  ofs << "  vertexBuf addr:   0x" << std::hex << m_particleAsHelper.getVertexBufferAddress() << std::dec << std::endl;
  ofs << "  indexBuf addr:    0x" << std::hex << m_particleAsHelper.getIndexBufferAddress() << std::dec << std::endl;
  ofs << "  aabbBuf addr:     0x" << std::hex << m_particleAsHelper.getAabbBufferAddress() << std::dec << std::endl;
  ofs << "  instanceBuf addr: 0x" << std::hex << m_particleAsHelper.getInstanceBufferAddress() << std::dec << std::endl;
  ofs << "  instanceBuf size: " << m_particleAsHelper.getInstanceBufferSize() << std::endl;
  ofs << "  tlasScratch size: " << m_particleAsHelper.getTlasScratchBufferSize() << std::endl;
  ofs << "  blasScratch size: " << m_particleAsHelper.getBlasScratchBufferSize() << std::endl;
  ofs << std::endl;

  // --- Per-splat-set BLAS helpers ---
  ofs << "=== Per-SplatSet BLAS Helpers (m_particleAsBlasHelpers, count=" << m_particleAsBlasHelpers.size() << ") ===" << std::endl;
  for(size_t i = 0; i < m_particleAsBlasHelpers.size(); ++i)
  {
    const auto& h = m_particleAsBlasHelpers[i];
    ofs << "  [" << i << "] BLAS accel=" << (h.getBlas().accel ? "VALID" : "NULL") << " addr=0x" << std::hex
        << h.getBlas().address << std::dec << " size=" << h.getBlas().buffer.bufferSize << std::endl;
    ofs << "    vertexBuf=0x" << std::hex << h.getVertexBufferAddress() << " indexBuf=0x" << h.getIndexBufferAddress()
        << " aabbBuf=0x" << h.getAabbBufferAddress() << std::dec << std::endl;
  }
  ofs << std::endl;

  // --- TLAS Helpers ---
  ofs << "=== TLAS Helpers (m_particleAsTlasHelpers, count=" << m_particleAsTlasHelpers.size() << ") ===" << std::endl;
  for(size_t i = 0; i < m_particleAsTlasHelpers.size(); ++i)
  {
    const auto& h = m_particleAsTlasHelpers[i];
    ofs << "  [" << i << "] TLAS accel=" << (h.getTlas().accel ? "VALID" : "NULL") << " addr=0x" << std::hex
        << h.getTlas().address << std::dec << " size=" << h.getTlas().buffer.bufferSize << std::endl;
    ofs << "    instanceBuf addr=0x" << std::hex << h.getInstanceBufferAddress() << std::dec
        << " size=" << h.getInstanceBufferSize() << std::endl;
    ofs << "    tlasScratch size=" << h.getTlasScratchBufferSize() << std::endl;
    ofs << "    BLAS accel=" << (h.getBlas().accel ? "VALID" : "NULL") << " addr=0x" << std::hex << h.getBlas().address
        << std::dec << std::endl;
  }
  ofs << std::endl;

  // --- BLAS Chunks ---
  ofs << "=== BLAS Chunks (m_particleAsBlasChunks, count=" << m_particleAsBlasChunks.size() << ") ===" << std::endl;
  for(size_t i = 0; i < m_particleAsBlasChunks.size(); ++i)
  {
    const auto& c = m_particleAsBlasChunks[i];
    ofs << "  [" << i << "] splatSetIndex=" << c.splatSetIndex << " splatBase=" << c.splatBase
        << " splatCount=" << c.splatCount << std::endl;
    ofs << "    BLAS accel=" << (c.helper.getBlas().accel ? "VALID" : "NULL") << " addr=0x" << std::hex
        << c.helper.getBlas().address << std::dec << " size=" << c.helper.getBlas().buffer.bufferSize << std::endl;
  }
  ofs << "  m_particleAsBlasChunkRanges.size(): " << m_particleAsBlasChunkRanges.size() << std::endl;
  for(size_t i = 0; i < m_particleAsBlasChunkRanges.size(); ++i)
  {
    ofs << "    [" << i << "] first=" << m_particleAsBlasChunkRanges[i].first
        << " count=" << m_particleAsBlasChunkRanges[i].count << std::endl;
  }
  ofs << std::endl;

  // --- Compute Pipeline State ---
  ofs << "=== Compute Pipeline State ===" << std::endl;
  ofs << "  m_particleAsComputePipeline: " << (m_particleAsComputePipeline ? "VALID" : "NULL") << std::endl;
  ofs << "  m_particleAsPipelineLayout:  " << (m_particleAsPipelineLayout ? "VALID" : "NULL") << std::endl;
  ofs << "  m_particleAsDescriptorSet:   " << (m_particleAsDescriptorSet ? "VALID" : "NULL") << std::endl;
  ofs << std::endl;

  // --- Computed Addresses (what push constants would see) ---
  ofs << "=== Computed Addresses (push constant inputs) ===" << std::endl;
  ofs << "  getGPUDescriptorArrayAddress(): 0x" << std::hex << getGPUDescriptorArrayAddress() << std::dec << std::endl;
  ofs << "  getGlobalIndexTableAddress():   0x" << std::hex << getGlobalIndexTableAddress() << std::dec << std::endl;
  ofs << "  getRtxDescriptorArrayAddress(): 0x" << std::hex << getRtxDescriptorArrayAddress() << std::dec << std::endl;
  ofs << "  getSplatSetDescriptorArrayAddress(): 0x" << std::hex << getSplatSetDescriptorArrayAddress() << std::dec << std::endl;
  if(m_accelStructProps)
  {
    ofs << "  maxInstanceCount: " << m_accelStructProps->maxInstanceCount << std::endl;
  }
  ofs << std::endl;

  ofs << "=== END ===" << std::endl;
  ofs.close();

  std::cout << "Debug state dumped to: " << filename << std::endl;
}

}  // namespace vk_gaussian_splatting
