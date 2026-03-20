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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>  // ProfilerTimeline

#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/physical_device.hpp>

// GPU radix sort (for sorting buffers)
#include <vk_radix_sort.h>

#include "acceleration_structures_lb.hpp"
#include "particle_acceleration_structures_gpu.hpp"
#include "splat_set_vk.h"  // Base asset class
#include "splat_set.h"
#include "shaderio.h"
#include "splat_sorter_async.h"  // CPU async sorter
#include "memory_statistics.h"

namespace vk_gaussian_splatting {

// RTX state for the manager's acceleration structures (BLAS/TLAS)
enum class RtxState : uint32_t
{
  eRtxNone,   // Not yet initialized or just reset (no content, no error)
  eRtxValid,  // Acceleration structures successfully built
  eRtxError   // An error occurred during RTX setup (allocation failure, build failure, etc.)
};

// NOTE: Handles removed - using direct shared_ptr references and vector indices

/**
 * SplatSetInstanceVk - Per-instance data for a splat set
 * 
 * Each instance references a base SplatSetVk asset and adds:
 * - Transform (position, rotation, scale)
 * - Material properties (lighting parameters)
 * - GPU descriptor index (for rendering)
 * 
 * This follows the same pattern as MeshInstanceVk.
 */
struct SplatSetInstanceVk
{
  enum class Flags : uint32_t
  {
    eNone             = 0,
    eDelete           = 1 << 0,  // Remove from GPU + delete from RAM
    eNew              = 1 << 1,  // Just created, needs descriptor entry
    eTransformChanged = 1 << 2,  // Transform in RAM changed, GPU needs update
    eMaterialChanged  = 1 << 3,  // Material in RAM changed, GPU needs update
  };

  // Query methods for state
  bool isMarkedForDeletion() const { return static_cast<uint32_t>(flags) & static_cast<uint32_t>(Flags::eDelete); }
  bool shouldShowInUI() const { return !isMarkedForDeletion(); }

  size_t                      index{0};            // Position in SplatSetManagerVk::m_instances vector
  std::shared_ptr<SplatSetVk> splatSet = nullptr;  // Direct reference to base asset (replaces splatSetHandle)

  // Transform parameters
  glm::vec3 translation = glm::vec3(0.0f);
  glm::vec3 rotation    = glm::vec3(0.0f);
  glm::vec3 scale       = glm::vec3(1.0f);

  // Transform matrices
  glm::mat4 transform                = glm::mat4(1.0f);
  glm::mat4 transformInverse         = glm::mat4(1.0f);
  glm::mat3 transformRotScaleInverse = glm::mat3(1.0f);  // Inverse of rotation-scale part

  // Material (per-instance)
  shaderio::ObjMaterial splatMaterial;

  // GPU descriptor index (index into gpuDescriptorArray)
  uint32_t gpuDescriptorIndex = 0;

  // Display name (e.g., "Splat set 0 - garden.ply")
  std::string displayName;

  Flags flags = Flags::eNone;  // Set by manager methods only

  // Rebuild descriptor from asset + instance data
  void rebuildDescriptor(const SplatSetVk* splatSet, shaderio::SplatSetDesc&);
};

// Bitwise operators for SplatSetInstanceVk::Flags
inline SplatSetInstanceVk::Flags operator|(SplatSetInstanceVk::Flags a, SplatSetInstanceVk::Flags b)
{
  return static_cast<SplatSetInstanceVk::Flags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline SplatSetInstanceVk::Flags operator&(SplatSetInstanceVk::Flags a, SplatSetInstanceVk::Flags b)
{
  return static_cast<SplatSetInstanceVk::Flags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline SplatSetInstanceVk::Flags& operator|=(SplatSetInstanceVk::Flags& a, SplatSetInstanceVk::Flags b)
{
  return a = a | b;
}
inline SplatSetInstanceVk::Flags& operator&=(SplatSetInstanceVk::Flags& a, SplatSetInstanceVk::Flags b)
{
  return a = a & b;
}
inline SplatSetInstanceVk::Flags operator~(SplatSetInstanceVk::Flags a)
{
  return static_cast<SplatSetInstanceVk::Flags>(~static_cast<uint32_t>(a));
}


class SplatSetManagerVk
{
public:
  enum class Request : uint32_t
  {
    eNone                   = 0,
    eProcessDeletions       = 1 << 0,  // Remove flagged instances/splat sets (GPU + RAM)
    eUpdateGlobalIndexTable = 1 << 1,  // Rebuild global index tables
    eUpdateDescriptors      = 1 << 2,  // Rebuild GPU descriptor buffer
    eUpdateTransformsOnly   = 1 << 3,  // Fast TLAS update (transforms only)
    eRebuildTLAS            = 1 << 4,  // Full TLAS rebuild (instance count changed)
    eRebuildBLAS            = 1 << 5,  // BLAS rebuild (geometry changed - rare)
  };

  // Multi-TLAS support for large scenes (maxInstanceCount limit)
  struct SplatSetTlasArray
  {
    // BLAS structures (one per unique splat set geometry)
    std::vector<nvvk::AccelerationStructure> blasSet;  // Bottom-level acceleration structures

    // TLAS structures (potentially multiple for large scenes)
    std::vector<AccelerationStructureHelperLB>                   tlasList;             // Multiple TLAS structures
    std::vector<std::vector<VkAccelerationStructureInstanceKHR>> tlasInstancesArrays;  // Instances per TLAS
    nvvk::Buffer tlasAddressBuffer;  // Buffer of TLAS addresses (uint64_t[])
    nvvk::Buffer tlasOffsetBuffer;   // Buffer of instance offsets (uint32_t[])
    uint32_t     tlasCount      = 0;
    uint64_t     totalSizeBytes = 0;

    // Helper for BLAS/TLAS building (single helper manages all)
    AccelerationStructureHelperLB helper;

    // Build statistics
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> blasBuildStatistics;
    std::vector<nvvk::AccelerationStructureBuildData>     blasBuildData;
  };

  SplatSetManagerVk()  = default;
  ~SplatSetManagerVk() = default;

  // Lifecycle
  void init(nvapp::Application*                                 app,
            nvvk::ResourceAllocator*                            alloc,
            nvvk::StagingUploader*                              uploader,
            VkSampler*                                          sampler,
            nvvk::PhysicalDeviceInfo*                           deviceInfo,
            VkPhysicalDeviceAccelerationStructurePropertiesKHR* accelStructProps,
            nvutils::ProfilerTimeline*                          profilerTimeline);

  void deinit();

  // Reset all splat sets and instances (for scene reset, not app exit)
  void reset();

  // ===== ASSET MANAGEMENT =====

  // Create a splat set from already-loaded RAM data (deferred upload to GPU)
  //
  // Note: File loading is handled by PlyLoaderAsync (async background thread).
  // This method takes ownership of the loaded SplatSet. GPU upload uses global settings
  // from prmData.dataStorage and prmData.shFormat (deferred to processVramUpdates).
  //
  // @param name Asset name (e.g., filename)
  // @param splatSet Already-loaded splat data in RAM (moved into SplatSetVk)
  // @return shared_ptr to the created splat set asset
  //
  std::shared_ptr<SplatSetVk> createSplatSet(const std::string& name, std::shared_ptr<SplatSetVk> splatSetVk);

  // Delete a splat set asset (deletes all instances using it)
  void deleteSplatSet(std::shared_ptr<SplatSetVk> splatSet);

  // Get splat set by index (direct vector access)
  std::shared_ptr<SplatSetVk> getSplatSet(size_t index) const
  {
    return (index < m_splatSets.size()) ? m_splatSets[index] : nullptr;
  }

  // Get number of splat sets
  size_t getSplatSetCount() const { return m_splatSets.size(); }

  // Get all splat sets (direct vector access for iteration)
  const std::vector<std::shared_ptr<SplatSetVk>>& getSplatSets() const { return m_splatSets; }

  // ===== INSTANCE MANAGEMENT =====

  // Create an instance of a splat set
  std::shared_ptr<SplatSetInstanceVk> createInstance(std::shared_ptr<SplatSetVk> splatSet,
                                                     const glm::mat4&            transform = glm::mat4(1.0f));

  // Register a pre-configured instance (for project loading)
  // Unlike createInstance(), this takes an already-configured instance
  // and just registers it with the manager.
  std::shared_ptr<SplatSetInstanceVk> registerInstance(std::shared_ptr<SplatSetVk>         splatSet,
                                                       std::shared_ptr<SplatSetInstanceVk> instance);

  // @brief Duplicate an existing instance (creates a copy with same transform and material)
  std::shared_ptr<SplatSetInstanceVk> duplicateInstance(std::shared_ptr<SplatSetInstanceVk> sourceInstance);

  // Delete a splat set instance
  void deleteInstance(std::shared_ptr<SplatSetInstanceVk> instance);

  std::shared_ptr<SplatSetInstanceVk> getInstance(size_t index) const
  {
    return (index < m_instances.size()) ? m_instances[index] : nullptr;
  }

  size_t getInstanceCount() const { return m_instances.size(); }

  // Get all instances (direct vector access for iteration)
  const std::vector<std::shared_ptr<SplatSetInstanceVk>>& getInstances() const { return m_instances; }

  // Update instance transform (UI has already modified RAM data)
  // This method just sets flags to trigger deferred GPU update
  void updateInstanceTransform(std::shared_ptr<SplatSetInstanceVk> instance);

  // Update instance material (UI has already modified RAM data)
  // This method just sets flags to trigger deferred GPU update
  void updateInstanceMaterial(std::shared_ptr<SplatSetInstanceVk> instance);

  // ===== DEFERRED UPDATE API =====
  // These methods set flags and requests - actual VRAM updates happen in processVramUpdates()

  // Process all deferred VRAM updates
  // Order: Delete → Update (RAM→GPU sync) → Upload (rebuild GPU structures)
  // @param processRtx If true, process RTX acceleration structure updates. If false, defer RTX updates.
  void processVramUpdates(bool processRtx = true);

  // Deferred update requests (set by modification methods)
  Request pendingRequests = Request::eNone;

  // ===== GLOBAL INDEX TABLE =====

  // Update global index tables if topology changed
  // Called once per frame before sorting (in processUpdateRequests)
  // @return true if tables were updated, false if no update was needed
  bool updateGlobalIndexTablesIfNeeded();

  // Get total splat count across all instances
  uint32_t getTotalGlobalSplatCount() const { return m_totalGlobalSplatCount; }

  // Get maximum SH degree across all splat sets
  uint32_t getMaxShDegree() const { return m_maxShDegree; }

  VkDeviceAddress getGlobalIndexTableAddress() const;
  VkDeviceAddress getSplatSetGlobalIndexTableAddress() const;

  // ===== SORTING BUFFERS =====

  VkDeviceAddress getSplatSortingDistancesAddress() const { return m_splatSortingDistancesDevice.address; }
  VkDeviceAddress getSplatSortingIndicesAddress() const { return m_splatSortingIndicesDevice.address; }
  VrdxSorter      getSplatSortingVrdxSorter() const { return m_splatSortingVrdxSorter; }

  const nvvk::LargeBuffer& getSplatSortingIndicesDevice() const { return m_splatSortingIndicesDevice; }
  const nvvk::LargeBuffer& getSplatSortingDistancesDevice() const { return m_splatSortingDistancesDevice; }
  const nvvk::LargeBuffer& getSplatSortingVrdxStorageBuffer() const { return m_splatSortingVrdxStorageBuffer; }

  // Try to consume CPU sorting result and upload to GPU
  // Called every frame for CPU async sorting
  // Supports single and multi-instance sorting (sorts all instances by global splat ID)
  void tryConsumeAndUploadCpuSortingResult(VkCommandBuffer                     cmd,
                                           const uint32_t                      splatCount,
                                           const glm::vec3&                    viewDir,
                                           const glm::vec3&                    eyePos,
                                           bool                                cpuLazySort,
                                           bool                                opacityGaussianDisabled,
                                           std::shared_ptr<SplatSetInstanceVk> selectedInstance,
                                           bool                                frontToBack = false);

  SplatSorterAsync::State getCpuSorterStatus() { return m_cpuSorter.getStatus(); }

  // ===== GPU DESCRIPTOR BUFFER =====

  // Update GPU descriptor buffer if instances changed
  // Called after instance creation/deletion/modification
  // return true if buffer was updated, false if no update was needed
  bool updateGPUDescriptorsIfNeeded(bool forceUpdate = false);

  VkDeviceAddress getGPUDescriptorArrayAddress() const;

  // Get GPU descriptor buffer address (for SceneAssets binding)
  VkDeviceAddress getDescriptorBufferAddress() const;

  // Get number of descriptors (number of instances)
  uint32_t getDescriptorCount() const { return static_cast<uint32_t>(m_gpuDescriptorArray.size()); }

  // Returns true (and clears the flag) when texture storage was (re)created
  // and the Vulkan descriptor set needs to rebind BINDING_SPLAT_TEXTURES.
  bool consumeTextureDescriptorsDirty()
  {
    bool dirty                = m_textureDescriptorsDirty;
    m_textureDescriptorsDirty = false;
    return dirty;
  }

  // ===== RAY TRACING =====

  // Mark one splat sets for data regeneration
  // Called when global dataStorage changes
  void markSplatSetsForRegeneration(std::shared_ptr<SplatSetVk>& splatSet);

  // Mark all splat sets for data regeneration
  // Called when global prmData.shFormat changes
  // This triggers full regeneration of ALL splat set VRAM buffers and acceleration structures
  void markAllSplatSetsForRegeneration();

  void rtxDeinitAccelerationStructures();

  VkDeviceAddress          getTlasAddress() const;
  const SplatSetTlasArray& getTlasArray() const { return m_rtAccelerationStructures; }
  VkDeviceAddress          getRtxDescriptorArrayAddress() const;

  // Provide particle AS compute pipeline handles and descriptor set (set by renderer).
  // The descriptor set is required so the AS build compute shader can access splat
  // textures via the storage accessors (BINDING_SPLAT_TEXTURES).
  void setParticleAsComputeState(VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet descriptorSet)
  {
    m_particleAsComputePipeline = pipeline;
    m_particleAsPipelineLayout  = layout;
    m_particleAsDescriptorSet   = descriptorSet;
  }

  // Check if RTX acceleration structures are valid and ready for use
  // Returns true only if manager state is eRtxValid AND no splat set has eError status
  bool isRtxValid() const
  {
    if(m_rtxState != RtxState::eRtxValid)
      return false;
    // Check if any splat set has an error
    for(const auto& splatSet : m_splatSets)
    {
      if(splatSet && splatSet->rtxStatus == RtxStatus::eError)
        return false;
    }
    return true;
  }

  // Check if RTX has encountered an actual error (allocation failure, build failure, etc.)
  // Returns false for eRtxNone (not yet initialized) — that is not an error
  bool isRtxError() const
  {
    if(m_rtxState == RtxState::eRtxError)
      return true;
    for(const auto& splatSet : m_splatSets)
    {
      if(splatSet && splatSet->rtxStatus == RtxStatus::eError)
        return true;
    }
    return false;
  }

  size_t getTlasSizeBytes() const;
  size_t getBlasSizeBytes() const;

  // ===== RTX STATISTICS (for UI display) =====

  // Number of TLAS structures (multi-TLAS for large scenes)
  uint32_t getRtxTlasCount() const { return static_cast<uint32_t>(m_particleAsTlasHelpers.size()); }

  // Total number of TLAS entries (instances written into TLAS)
  // In instanced mode (useTlasInstances): one entry per splat
  // In per-splat-set mode with chunks: one entry per BLAS chunk
  // In per-splat-set mode without chunks: one entry per user instance
  uint32_t getRtxTlasEntryCount() const
  {
    if(!m_particleAsBlasChunks.empty())
      return static_cast<uint32_t>(m_gpuRtxDescriptorArray.size());  // chunk mode
    if(!m_particleAsBlasHelpers.empty())
      return static_cast<uint32_t>(m_instances.size());  // non-instanced, single BLAS per set
    return m_totalGlobalSplatCount;                      // instanced mode: one entry per splat
  }

  // Number of BLAS structures
  uint32_t getRtxBlasCount() const
  {
    if(!m_particleAsBlasChunks.empty())
      return static_cast<uint32_t>(m_particleAsBlasChunks.size());  // chunk mode
    if(!m_particleAsBlasHelpers.empty())
      return static_cast<uint32_t>(m_particleAsBlasHelpers.size());  // non-instanced, single BLAS per set
    if(m_particleAsHelper.getBlas().accel != VK_NULL_HANDLE)
      return 1;  // instanced mode (single unit BLAS)
    return 0;
  }

  // True if BLAS are split into chunks (large models exceeding maxMemoryAllocationSize)
  bool isUsingBlasChunks() const { return !m_particleAsBlasChunks.empty(); }

  // True if multi-TLAS is active (large scenes exceeding maxInstanceCount)
  bool isUsingMultiTlas() const { return m_particleAsTlasHelpers.size() > 1; }

private:
  // ===== processVramUpdates sub-methods (called in order) =====

  // Phase 1: Remove instances/splat sets from GPU + delete from RAM.
  // Updates instanceCountChanged and descriptorsNeedRebuild flags.
  void processRamVramDeletionsIfNeeded(bool& instanceCountChanged, bool& descriptorsNeedRebuild);

  // Phase 2: Process RAM → GPU data uploads (new splat sets, instances, transforms, materials, data storage).
  // Updates instanceCountChanged, descriptorsNeedRebuild, and hasTransformChanges flags.
  void processRamToVramDataUploads(bool& instanceCountChanged, bool& descriptorsNeedRebuild, bool& hasTransformChanges);

  // Phase 3: RTX Acceleration Structures (BLAS first, then TLAS).
  // Returns false if Phase 4 should be skipped (early exit due to RTX build failure or rebuild completing all work).
  // Dispatches to sub-methods: rtxRebuildBlasAndTlas, rtxRebuildTlas, rtxUpdateTlasTransforms.
  bool processRtxAccelerationStructures(bool processRtx, bool hasTransformChanges, bool& descriptorsNeedRebuild);

  // Phase 3 helper: Marks all splat sets as error, deinitializes AS, clears pending RTX requests.
  void handleRtxBuildFailure(const char* reason);

  // Phase 3 helper: Clears the RTX-specific descriptor array and associated GPU buffer.
  void clearRtxDescriptorArray();

  // Phase 3a: Full BLAS + TLAS rebuild for all splat sets (instanced and non-instanced modes).
  // Dispatches to one of the 3 sub-methods below based on mode.
  bool rtxRebuildBlasAndTlas();

  // Phase 3a sub-methods:

  // Per-splat mode (useTlasInstances): builds unit BLAS + per-splat TLAS array.
  bool rtxRebuildBlasAndTlasPerSplat();

  // Per-instance mode, single-BLAS: one BLAS per splat set + per-instance TLAS.
  bool rtxRebuildBlasAndTlasPerInstanceSingleBlas(VkBuildAccelerationStructureFlagsKHR blasFlags);

  // Per-instance mode, multi-BLAS: split large splat sets into BLAS chunks + per-instance TLAS.
  bool rtxRebuildBlasAndTlasPerInstanceMultiBlas(VkBuildAccelerationStructureFlagsKHR blasFlags);

  // Phase 3b: TLAS-only rebuild (BLAS already exists, instance count changed).
  // Dispatches to one of the 3 sub-methods below based on mode.
  bool rtxRebuildTlas();

  // Phase 3b sub-methods:

  // Per-splat TLAS rebuild (useTlasInstances): shared BLAS, each splat is a TLAS instance.
  bool rtxRebuildTlasPerSplat();

  // Per-instance TLAS rebuild: handles both multi-BLAS and single-BLAS modes.
  bool rtxRebuildTlasPerInstance();

  // Fallback cleanup: no valid BLAS or TLAS helpers — cleans up stale structures.
  bool rtxRebuildTlasFallbackCleanup();

  // Phase 3c: Fast TLAS update path for transform changes only.
  // Dispatches to one of the 3 sub-methods below based on mode.
  bool rtxUpdateTlasTransforms(bool& descriptorsNeedRebuild);

  // Phase 3c sub-methods:

  // Per-splat TLAS mode (useTlasInstances): shared BLAS, each splat is a TLAS instance.
  bool rtxUpdateTlasPerSplat();

  // Per-instance TLAS, multi-BLAS: multiple BLAS chunks per splat set, uses RTX descriptor array.
  bool rtxUpdateTlasPerInstanceMultiBlas();

  // Per-instance TLAS, single-BLAS: one BLAS per splat set, uses standard descriptor array.
  bool rtxUpdateTlasPerInstanceSingleBlas();

  // Phase 4: Rebuild GPU descriptor array and global index tables after all resources are ready.
  void processDescriptorsAndIndexTables(bool descriptorsNeedRebuild);

  // ===== HELPERS =====

  // Update consolidated memory statistics by summing all splat set stats
  // Called after data storage operations (upload, regeneration, deletion)
  void updateConsolidatedMemoryStats();

  // ===== STORAGE =====

  // CPU-side storage (vector for cache-friendly direct access)
  std::vector<std::shared_ptr<SplatSetVk>>         m_splatSets;
  std::vector<std::shared_ptr<SplatSetInstanceVk>> m_instances;

  // GPU-side storage (contiguous for rendering)
  std::vector<shaderio::SplatSetDesc> m_gpuDescriptorArray;
  nvvk::Buffer                        m_descriptorBuffer;
  // RTX-only descriptor array (per-TLAS-instance, used for split BLAS)
  std::vector<shaderio::SplatSetDesc> m_gpuRtxDescriptorArray;
  nvvk::Buffer                        m_rtxDescriptorBuffer;
  bool                                m_useSplitBlasRtxDescriptors = false;
  // GPU-side storage for per-splat-set descriptors (BLAS generation)
  std::vector<shaderio::SplatSetDesc> m_gpuSplatSetDescriptorArray;
  nvvk::Buffer                        m_splatSetDescriptorBuffer;

  // Global Index Tables (RAM + GPU)
  struct GlobalSplatIndexEntry
  {
    uint32_t splatSetIndex;  // Which instance
    uint32_t splatIndex;     // Which splat within that instance
  };
  struct InstanceInfo
  {
    uint32_t splatSetIdx;   // Index into descriptor array
    uint32_t splatCount;    // Number of splats in this instance
    uint32_t globalOffset;  // Starting global splat index
  };
  std::vector<InstanceInfo>          m_instanceInfos;                   // RAM: cached per-instance offsets
  std::vector<GlobalSplatIndexEntry> m_globalIndexTable;                // RAM: authoritative
  std::vector<uint32_t>              m_splatSetGlobalIndexTable;        // RAM: authoritative
  nvvk::Buffer                       m_globalIndexTableBuffer;          // GPU: synchronized
  nvvk::Buffer                       m_splatSetGlobalIndexTableBuffer;  // GPU: synchronized
  uint32_t                           m_totalGlobalSplatCount = 0;
  uint32_t                           m_maxShDegree           = 0;  // Maximum SH degree across all splat sets

  // Sorting Buffers (same lifetime as global index tables)
  nvvk::Buffer      m_splatSortingIndicesHost;                        // Host buffer for CPU sorting upload
  nvvk::LargeBuffer m_splatSortingIndicesDevice;                      // Sorted indices buffer (GPU device)
  nvvk::LargeBuffer m_splatSortingDistancesDevice;                    // Distance buffer for depth sorting (GPU)
  nvvk::LargeBuffer m_splatSortingVrdxStorageBuffer;                  // VRDX internal storage (GPU)
  VrdxSorter   m_splatSortingVrdxSorter       = VK_NULL_HANDLE;  // GPU radix sorter
  uint32_t     m_sortingBuffersAllocatedCount = 0;               // Track buffer size for resize detection

  // CPU Async Sorting (application lifetime)
  SplatSorterAsync           m_cpuSorter;                   // Async CPU sorter thread
  std::vector<uint32_t>      m_splatIndices;                // CPU-side indices array
  nvutils::ProfilerTimeline* m_profilerTimeline = nullptr;  // For CPU sorter profiling

  // RTX
  SplatSetTlasArray m_rtAccelerationStructures;
  RtxState          m_rtxState = RtxState::eRtxNone;  // Manager-level RTX state (not yet initialized on startup)

  // GPU-only particle AS helper (AABB + TLAS instances)
  ParticleAccelerationStructureHelperGpu              m_particleAsHelper{};
  std::vector<ParticleAccelerationStructureHelperGpu> m_particleAsBlasHelpers{};
  std::vector<ParticleAccelerationStructureHelperGpu> m_particleAsTlasHelpers{};
  VkPipeline                                          m_particleAsComputePipeline = VK_NULL_HANDLE;
  VkPipelineLayout                                    m_particleAsPipelineLayout  = VK_NULL_HANDLE;
  VkDescriptorSet                                     m_particleAsDescriptorSet   = VK_NULL_HANDLE;
  bool                                                m_useGpuBlasForSplatSets    = false;

  struct ParticleAsBlasChunk
  {
    ParticleAccelerationStructureHelperGpu helper{};
    uint32_t                               splatSetIndex = 0;
    uint32_t                               splatBase     = 0;
    uint32_t                               splatCount    = 0;
  };
  struct ParticleAsBlasChunkRange
  {
    uint32_t first = 0;
    uint32_t count = 0;
  };
  std::vector<ParticleAsBlasChunk>      m_particleAsBlasChunks{};
  std::vector<ParticleAsBlasChunkRange> m_particleAsBlasChunkRanges{};

  // Dirty flags
  bool m_globalIndexTableDirty   = true;   // Set when topology changes
  bool m_gpuDescriptorsDirty     = true;   // Set when instances change
  bool m_textureDescriptorsDirty = false;  // Set when texture data storage is (re)created
  uint32_t m_tlasNeedsFullRebuild = 0;  // Counter: forces rebuilds before allowing update path (set to 10 on copy/import)

  // Vulkan resources
  nvapp::Application*                                 m_app              = nullptr;
  nvvk::ResourceAllocator*                            m_alloc            = nullptr;
  nvvk::StagingUploader*                              m_uploader         = nullptr;
  VkSampler*                                          m_sampler          = nullptr;
  nvvk::PhysicalDeviceInfo*                           m_deviceInfo       = nullptr;
  VkPhysicalDeviceAccelerationStructurePropertiesKHR* m_accelStructProps = nullptr;

  // ===== INTERNAL HELPERS =====

  void rebuildGlobalIndexTables();
  void uploadGlobalIndexTablesToGPU();

  void            updateGpuDescriptorArray();
  void            uploadGpuDescriptorArray();
  void            rebuildRtxDescriptorArrayFromChunks();
  void            updateGpuSplatSetDescriptorArray();
  void            uploadGpuSplatSetDescriptorArray();
  VkDeviceAddress getSplatSetDescriptorArrayAddress() const;

  // Release scene-scoped GPU buffers (descriptor/index/sorting) when empty.
  void clearSceneGpuBuffers();

  void markGlobalIndexTableDirty();
  void markGpuDescriptorsDirty();

  void updateMaxShDegree();  // Recompute max SH degree from all splat sets
  uint32_t computeMaxSplatsPerGpuBlas(bool useAabbs, VkBuildAccelerationStructureFlagsKHR blasBuildFlags, uint32_t splatCount) const;

  // Estimate BLAS build sizes (AS size + scratch) for a given splat count using vkGetAccelerationStructureBuildSizesKHR.
  // Used by both computeMaxSplatsPerGpuBlas (chunk sizing) and VRAM budget pre-check.
  VkAccelerationStructureBuildSizesInfoKHR estimateBlasBuildSizes(bool                                 useAabbs,
                                                                  VkBuildAccelerationStructureFlagsKHR blasBuildFlags,
                                                                  uint32_t splatCount) const;

public:
  // ===== DEBUG =====

  // Dump internal state (buffers, descriptors, parameters) to a timestamped file for debugging.
  // @param label  Short label embedded in the filename (e.g. "after_copy", "after_reset")
  void dumpDebugState(const std::string& label) const;

  // Last created instance (for UI convenience)
  std::shared_ptr<SplatSetInstanceVk> m_lastCreatedInstance = nullptr;

  // Asset naming counter (reset on reset())
  uint32_t m_nextInstanceNumber = 0;
};

// Bitwise operator implementations for SplatSetManagerVk::Request
inline SplatSetManagerVk::Request operator|(SplatSetManagerVk::Request a, SplatSetManagerVk::Request b)
{
  return static_cast<SplatSetManagerVk::Request>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline SplatSetManagerVk::Request operator&(SplatSetManagerVk::Request a, SplatSetManagerVk::Request b)
{
  return static_cast<SplatSetManagerVk::Request>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline SplatSetManagerVk::Request& operator|=(SplatSetManagerVk::Request& a, SplatSetManagerVk::Request b)
{
  return a = a | b;
}
inline SplatSetManagerVk::Request& operator&=(SplatSetManagerVk::Request& a, SplatSetManagerVk::Request b)
{
  return a = a & b;
}
inline SplatSetManagerVk::Request operator~(SplatSetManagerVk::Request a)
{
  return static_cast<SplatSetManagerVk::Request>(~static_cast<uint32_t>(a));
}

}  // namespace vk_gaussian_splatting
