/*/*
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

#include <string>
#include <vector>
#include <set>
#include <memory>
#include <algorithm>

#include <glm/glm.hpp>

#include <nvapp/application.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/staging.hpp>

#include "acceleration_structures_lb.hpp"

#include "shaderio.h"

#include "obj_loader.h"
#include "light_manager_vk.h"

namespace vk_gaussian_splatting {

// Forward declarations
struct MeshVk;
struct MeshInstanceVk;

// Import MeshType enum for convenience
using shaderio::MeshType;

// Custom comparator for std::set (orders by pointer address for stable iteration)
struct SharedPtrCompare
{
  template <typename T>
  bool operator()(const std::shared_ptr<T>& a, const std::shared_ptr<T>& b) const
  {
    return a.get() < b.get();
  }
};

// The OBJ model (Vk suffix for consistency with Vulkan class naming convention)
struct MeshVk
{
  enum class Flags : uint32_t
  {
    eNone             = 0,
    eDelete           = 1 << 0,  // Remove from GPU + delete from RAM
    eNew              = 1 << 1,  // Just created, needs descriptor entry (geometry already uploaded)
    eMaterialsChanged = 1 << 2,  // Materials buffer needs upload to GPU
  };

  // Query methods for state
  bool isMarkedForDeletion() const { return static_cast<uint32_t>(flags) & static_cast<uint32_t>(Flags::eDelete); }
  bool shouldShowInUI() const { return !isMarkedForDeletion(); }
  bool shouldRender() const { return !isMarkedForDeletion(); }

  size_t       index{0};  // Position in MeshManagerVk::meshes vector
  std::string  path;      // Full file path (e.g., "C:/models/teapot.obj")
  uint32_t     nbIndices{0};
  uint32_t     nbVertices{0};
  nvvk::Buffer vertexBuffer;     // Device buffer of all 'Vertex' (allocated in processVramUpdates)
  nvvk::Buffer indexBuffer;      // Device buffer of the indices forming triangles
  nvvk::Buffer materialsBuffer;  // Device buffer of 'Wavefront' materials
  nvvk::Buffer matIndexBuffer;   // Device buffer of per face material IDs

  // RAM storage (for upload to GPU in processVramUpdates)
  std::vector<ObjVertex>   vertexData;    // RAM copy for upload
  std::vector<uint32_t>    indexData;     // RAM copy for upload
  std::vector<ObjMaterial> materials;     // RAM copy for upload
  std::vector<uint32_t>    matIndexData;  // RAM copy for upload
  std::vector<std::string> matNames;      // name of each material stored in materials

  Flags flags = Flags::eNone;  // Set by manager methods only

  // Buffer management methods (called by MeshManagerVk)
  void initBuffers(nvvk::ResourceAllocator* alloc, nvvk::StagingUploader* uploader);
  void deinitBuffers(nvvk::ResourceAllocator* alloc);
};

// Bitwise operators for MeshVk::Flags
inline MeshVk::Flags operator|(MeshVk::Flags a, MeshVk::Flags b)
{
  return static_cast<MeshVk::Flags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline MeshVk::Flags operator&(MeshVk::Flags a, MeshVk::Flags b)
{
  return static_cast<MeshVk::Flags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline MeshVk::Flags& operator|=(MeshVk::Flags& a, MeshVk::Flags b)
{
  return a = a | b;
}
inline MeshVk::Flags& operator&=(MeshVk::Flags& a, MeshVk::Flags b)
{
  return a = a & b;
}
inline MeshVk::Flags operator~(MeshVk::Flags a)
{
  return static_cast<MeshVk::Flags>(~static_cast<uint32_t>(a));
}

// Per-instance mesh data (pointer-based architecture)
struct MeshInstanceVk
{
  enum class Flags : uint32_t
  {
    eNone             = 0,
    eDelete           = 1 << 0,  // Remove from GPU + delete from RAM
    eNew              = 1 << 1,  // Just created, needs descriptor entry
    eTransformChanged = 1 << 2,  // Transform in RAM changed, GPU needs update
    eMaterialChanged  = 1 << 3,  // Material in RAM changed, GPU needs update (unused for now)
  };

  size_t                  index{0};   // Position in MeshManagerVk::instances vector
  std::shared_ptr<MeshVk> mesh;       // Direct reference to mesh (replaces uint32_t objIndex)
  std::string             name;       // Display name (e.g., "Model 0 - teapot.obj")
  MeshType  type{MeshType::eObject};  // Instance type (Object, LightProxy, etc.) - used for RTX mask and UI filtering
  glm::vec3 translation{0.0f};
  glm::vec3 rotation{0.0f};
  glm::vec3 scale{1.0f, 1.0f, 1.0f};
  glm::mat4 transform;                   // Matrix of the instance
  glm::mat4 transformInverse{};          // Inverse Matrix of the instance
  glm::mat3 transformRotScaleInverse{};  // Inverse of rotation-scale part (for normals)

  Flags flags = Flags::eNone;  // Set by manager methods only

  // Query methods for state
  bool isMarkedForDeletion() const { return static_cast<uint32_t>(flags) & static_cast<uint32_t>(Flags::eDelete); }
  bool shouldShowInUI() const { return !isMarkedForDeletion(); }
  bool shouldRender() const { return !isMarkedForDeletion(); }

  // Note: mesh shared_ptr ensures mesh stays alive as long as instance exists
  // This provides automatic lifetime management via reference counting
};

// Bitwise operators for MeshInstanceVk::Flags
inline MeshInstanceVk::Flags operator|(MeshInstanceVk::Flags a, MeshInstanceVk::Flags b)
{
  return static_cast<MeshInstanceVk::Flags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline MeshInstanceVk::Flags operator&(MeshInstanceVk::Flags a, MeshInstanceVk::Flags b)
{
  return static_cast<MeshInstanceVk::Flags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline MeshInstanceVk::Flags& operator|=(MeshInstanceVk::Flags& a, MeshInstanceVk::Flags b)
{
  return a = a | b;
}
inline MeshInstanceVk::Flags& operator&=(MeshInstanceVk::Flags& a, MeshInstanceVk::Flags b)
{
  return a = a & b;
}
inline MeshInstanceVk::Flags operator~(MeshInstanceVk::Flags a)
{
  return static_cast<MeshInstanceVk::Flags>(~static_cast<uint32_t>(a));
}

class MeshManagerVk
{
public:
  enum class Request : uint32_t
  {
    eNone                 = 0,
    eProcessDeletions     = 1 << 0,  // Remove flagged instances/meshes (GPU + RAM)
    eUpdateDescriptors    = 1 << 1,  // Rebuild instance descriptor buffer
    eUpdateMaterials      = 1 << 2,  // Upload changed materials to GPU
    eUpdateTransformsOnly = 1 << 3,  // Fast TLAS update (transforms only)
    eRebuildTLAS          = 1 << 4,  // Full TLAS rebuild (instance count changed)
    eRebuildBLAS          = 1 << 5,  // BLAS rebuild (geometry changed - rare)
  };

  // Meshes: Store as shared_ptr for automatic lifetime management
  std::vector<std::shared_ptr<MeshVk>> meshes;

  // Instances: Vector for cache-friendly direct access
  std::vector<std::shared_ptr<MeshInstanceVk>> instances;

  // GPU-side data (rebuilt from instances on demand)
  std::vector<shaderio::MeshDesc> objectDescriptions;        // Per-instance descriptors for device access
  nvvk::Buffer                    objectDescriptionsBuffer;  // Device buffer of the mesh instance descriptors

  // RTX specifics
  AccelerationStructureHelperLB rtAccelerationStructures;

  // Deferred update requests
  Request pendingRequests = Request::eNone;

public:
  inline void init(nvapp::Application*                                 app,
                   nvvk::ResourceAllocator*                            alloc,
                   nvvk::StagingUploader*                              uploader,
                   VkPhysicalDeviceAccelerationStructurePropertiesKHR* accelStructProps)
  {
    m_app      = app;
    m_alloc    = alloc;
    m_uploader = uploader;
    rtAccelerationStructures.init(m_alloc, m_uploader, m_app->getQueue(0), 2000, 2000);
  };

  inline void deinit(void)
  {
    reset();
    rtAccelerationStructures.deinit();
    m_app      = {};
    m_alloc    = {};
    m_uploader = {};
  }

  // Reset all meshes and instances (for scene reset, not app exit)
  inline void reset()
  {
    // CRITICAL: Use the SAME deferred deletion flow as DELETE (which works perfectly)
    // Call deleteInstance() for each instance, then deleteMesh() for each mesh
    // This reuses the existing, tested deletion logic

    // Delete all instances (iterate backwards to avoid issues with index changes)
    for(size_t i = instances.size(); i-- > 0;)
    {
      if(instances[i])
      {
        deleteInstance(instances[i]);
      }
    }

    // Delete all meshes (iterate backwards, in case some weren't referenced by instances)
    for(size_t i = meshes.size(); i-- > 0;)
    {
      if(meshes[i])
      {
        deleteMesh(meshes[i]);
      }
    }

    // Reset naming counter (will be recreated from 0 when new scene loads)
    m_nextInstanceNumber = 0;

    // Reset convenience tracking
    m_lastCreatedInstance = nullptr;
  }

  // ===== MESH MANAGEMENT =====

  // High-level API: Load model from file, returns shared_ptr to mesh
  // Creates both the mesh and a default instance at origin
  std::shared_ptr<MeshVk> loadModel(const std::filesystem::path& filename);

  // Low-level API: Create mesh from raw data and upload to VRAM
  // Returns shared_ptr to the created mesh (can be used to create instances)
  // This is the reusable core that loadModel() calls internally
  std::shared_ptr<MeshVk> createMesh(const std::string&              name,
                                     const std::vector<ObjVertex>&   vertices,
                                     const std::vector<uint32_t>&    indices,
                                     const std::vector<ObjMaterial>& materials,
                                     const std::vector<uint32_t>&    matIndices);

  // ===== INSTANCE MANAGEMENT =====

  // Create an instance of an existing mesh with optional transform and type
  // Returns shared_ptr to the created instance
  std::shared_ptr<MeshInstanceVk> createInstance(std::shared_ptr<MeshVk> mesh,
                                                 const glm::mat4&        transform = glm::mat4(1.0f),
                                                 MeshType                type      = MeshType::eObject);

  /**
   * @brief Register a pre-configured instance with the manager
   * @param instance Pre-configured instance to register
   * @return shared_ptr to the registered instance
   * @note Sets the index field and adds to the instances vector
   */
  std::shared_ptr<MeshInstanceVk> registerInstance(std::shared_ptr<MeshInstanceVk> instance);

  /**
   * @brief Duplicate an existing mesh instance (creates a copy with same transform and material)
   * @param sourceInstance The instance to duplicate
   * @return shared_ptr to the new duplicated instance
   * @note This method safely handles vector reallocation that would invalidate the source reference
   */
  std::shared_ptr<MeshInstanceVk> duplicateInstance(std::shared_ptr<MeshInstanceVk> sourceInstance);

  // ===== DEFERRED UPDATE API =====
  // All modification methods set flags and requests - actual VRAM updates happen in processVramUpdates()

  // Delete an instance (sets Delete flag, actual deletion in processVramUpdates)
  // Also deletes the mesh if this was the last instance referencing it
  void deleteInstance(std::shared_ptr<MeshInstanceVk> instance);

  // Delete an instance without destroying the mesh (sets Delete flag)
  // Used for light proxies where meshes are persistent
  void deleteInstanceOnly(std::shared_ptr<MeshInstanceVk> instance);

  // Delete a mesh (sets Delete flag, actual deletion in processVramUpdates)
  void deleteMesh(std::shared_ptr<MeshVk> mesh);

  // Update instance transform (caller has already modified instance->transform in RAM)
  // This method just sets flags to trigger deferred GPU update
  void updateInstanceTransform(std::shared_ptr<MeshInstanceVk> instance);

  // Update instance material (sets MaterialChanged flag - currently unused)
  void updateInstanceMaterial(std::shared_ptr<MeshInstanceVk> instance);

  // Update mesh materials (materials already modified in RAM, sets MaterialsChanged flag)
  void updateMeshMaterials(std::shared_ptr<MeshVk> mesh);

  // Process all deferred VRAM updates
  // Order: Delete → Update (RAM→GPU sync) → Upload (rebuild GPU structures)
  // @param processRtx If true, process RTX acceleration structure updates. If false, defer RTX updates.
  void processVramUpdates(bool processRtx = true);

  // ===== UTILITY =====

  // Direct vector access helpers
  std::shared_ptr<MeshVk> getMesh(size_t index) const { return (index < meshes.size()) ? meshes[index] : nullptr; }

  std::shared_ptr<MeshInstanceVk> getInstance(size_t index) const
  {
    return (index < instances.size()) ? instances[index] : nullptr;
  }

  size_t getMeshCount() const { return meshes.size(); }
  size_t getInstanceCount() const { return instances.size(); }

  const std::vector<std::shared_ptr<MeshVk>>&         getMeshes() const { return meshes; }
  const std::vector<std::shared_ptr<MeshInstanceVk>>& getInstances() const { return instances; }

  // update the buffer that stores the list of object IDs
  // and some per object information
  void updateObjDescriptionBuffer();

  // init BLAS and TLAS for all the loaded models
  void rtxInitAccelerationStructures();

  // update TLAS transforms from instances to device
  void rtxUpdateTopLevelAccelerationStructure();

  void rtxDeinitAccelerationStructures() { rtAccelerationStructures.deinitAccelerationStructures(); }


private:
  // ===== INTERNAL HELPERS =====

  // Upload materials buffer for a mesh (called by processVramUpdates)
  // Note: Assumes materialsBuffer already exists (created by initBuffers)
  void uploadMaterialsBufferInternal(std::shared_ptr<MeshVk> mesh);

  // RTX specifics
  nvvk::AccelerationStructureGeometryInfo rtxCreateMeshVkKHR(const MeshVk& model);

private:
  nvapp::Application*      m_app{};
  nvvk::ResourceAllocator* m_alloc{};
  nvvk::StagingUploader*   m_uploader{};

public:
  // Last created instance pointer (for UI convenience after loadModel)
  std::shared_ptr<MeshInstanceVk> m_lastCreatedInstance = nullptr;

  // Asset naming counter (reset on reset(), only for user objects)
  uint32_t m_nextInstanceNumber = 0;
};

// Bitwise operator implementations for MeshManagerVk::Request
inline MeshManagerVk::Request operator|(MeshManagerVk::Request a, MeshManagerVk::Request b)
{
  return static_cast<MeshManagerVk::Request>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline MeshManagerVk::Request operator&(MeshManagerVk::Request a, MeshManagerVk::Request b)
{
  return static_cast<MeshManagerVk::Request>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline MeshManagerVk::Request& operator|=(MeshManagerVk::Request& a, MeshManagerVk::Request b)
{
  return a = a | b;
}
inline MeshManagerVk::Request& operator&=(MeshManagerVk::Request& a, MeshManagerVk::Request b)
{
  return a = a & b;
}
inline MeshManagerVk::Request operator~(MeshManagerVk::Request a)
{
  return static_cast<MeshManagerVk::Request>(~static_cast<uint32_t>(a));
}

}  // namespace vk_gaussian_splatting
