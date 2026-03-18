/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mesh_manager_vk.h"
#include "utilities.h"

//#define STB_IMAGE_IMPLEMENTATION
//#include <stb/stb_image.h>

#include <algorithm>  // for std::find
#include <fmt/format.h>

#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

namespace vk_gaussian_splatting {

// =============================================================================
// MeshVk Buffer Management
// =============================================================================

void MeshVk::initBuffers(nvvk::ResourceAllocator* alloc, nvvk::StagingUploader* uploader)
{
  // Allocate GPU buffers
  VkBufferUsageFlags flag            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBufferUsageFlags rayTracingFlags =  // used also for building acceleration structures
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  NVVK_CHECK(alloc->createBuffer(vertexBuffer, vertexData.size() * sizeof(ObjVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(vertexBuffer.buffer);

  NVVK_CHECK(alloc->createBuffer(indexBuffer, indexData.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(indexBuffer.buffer);

  NVVK_CHECK(alloc->createBuffer(materialsBuffer, materials.size() * sizeof(ObjMaterial),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(materialsBuffer.buffer);

  NVVK_CHECK(alloc->createBuffer(matIndexBuffer, matIndexData.size() * sizeof(uint32_t),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | rayTracingFlags));
  NVVK_DBG_NAME(matIndexBuffer.buffer);

  // Update needShading flags before uploading materials to GPU
  for(auto& mat : materials)
  {
    shaderio::updateMaterialNeedsShading(mat);
  }

  // Upload data to GPU
  NVVK_CHECK(uploader->appendBuffer(vertexBuffer, 0, std::span(vertexData)));
  NVVK_CHECK(uploader->appendBuffer(indexBuffer, 0, std::span(indexData)));
  NVVK_CHECK(uploader->appendBuffer(materialsBuffer, 0, std::span(materials)));
  NVVK_CHECK(uploader->appendBuffer(matIndexBuffer, 0, std::span(matIndexData)));

  // Note: uploader->cmdUploadAppended() and staging release handled by caller
}

void MeshVk::deinitBuffers(nvvk::ResourceAllocator* alloc)
{
  // Destroy all buffers
  alloc->destroyBuffer(vertexBuffer);
  alloc->destroyBuffer(indexBuffer);
  alloc->destroyBuffer(materialsBuffer);
  alloc->destroyBuffer(matIndexBuffer);

  // Reset to empty state
  vertexBuffer    = {};
  indexBuffer     = {};
  materialsBuffer = {};
  matIndexBuffer  = {};
}

// =============================================================================
// MeshManagerVk Implementation
// =============================================================================

// High-level API: Load from file
std::shared_ptr<MeshVk> MeshManagerVk::loadModel(const std::filesystem::path& filename)
{
  LOGI("Loading File:  %s \n", filename.string().c_str());
  ObjLoader loader;
  if(!loader.load(filename))
    return nullptr;

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = glm::pow(m.ambient, glm::vec3(2.2f));
    m.diffuse  = glm::pow(m.diffuse, glm::vec3(2.2f));
    m.specular = glm::pow(m.specular, glm::vec3(2.2f));
  }

  // Create mesh using low-level API (reusable!)
  auto mesh = createMesh(loader.filename.filename().string(),  // name
                         loader.m_vertices,                    // vertices
                         loader.m_indices,                     // indices
                         loader.m_materials,                   // materials
                         loader.m_matIndices                   // matIndices
  );

  //
  mesh->path = filename.string();

  // Create default instance at origin
  m_lastCreatedInstance = createInstance(mesh);

  return mesh;
}

// Low-level API: Create mesh from raw data (reusable for light proxies, procedural geometry, etc.)
std::shared_ptr<MeshVk> MeshManagerVk::createMesh(const std::string&              name,
                                                  const std::vector<ObjVertex>&   vertices,
                                                  const std::vector<uint32_t>&    indices,
                                                  const std::vector<ObjMaterial>& materials,
                                                  const std::vector<uint32_t>&    matIndices)
{
  auto mesh        = std::make_shared<MeshVk>();
  mesh->path       = "";  // Path is only meaningful for loaded files
  mesh->nbIndices  = static_cast<uint32_t>(indices.size());
  mesh->nbVertices = static_cast<uint32_t>(vertices.size());

  // Store RAM copies - buffers will be allocated AND uploaded in processVramUpdates()
  mesh->vertexData   = vertices;
  mesh->indexData    = indices;
  mesh->materials    = materials;
  mesh->matIndexData = matIndices;

  // Generate material names if not provided
  mesh->matNames.resize(materials.size());
  for(size_t i = 0; i < materials.size(); ++i)
  {
    mesh->matNames[i] = "material_" + std::to_string(i);
  }

  // Set flag: needs GPU upload (buffers NOT allocated yet)
  mesh->flags |= MeshVk::Flags::eNew;

  // Set index and store mesh
  mesh->index = meshes.size();
  meshes.push_back(mesh);

  // Request GPU sync (deferred to processVramUpdates)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildBLAS;

  LOGI("createMesh: Created mesh '%s' (meshes.size=%zu)\n", name.c_str(), meshes.size());

  return mesh;
}

// Create instance of existing mesh (returns shared_ptr to instance)
std::shared_ptr<MeshInstanceVk> MeshManagerVk::createInstance(std::shared_ptr<MeshVk> mesh, const glm::mat4& transform, MeshType type)
{
  if(!mesh)
  {
    LOGE("createInstance: Null mesh pointer\n");
    return nullptr;
  }

  auto instance                      = std::make_shared<MeshInstanceVk>();
  instance->mesh                     = mesh;  // Store mesh shared_ptr
  instance->type                     = type;
  instance->transform                = transform;
  instance->transformInverse         = glm::inverse(transform);
  instance->transformRotScaleInverse = glm::inverse(glm::mat3(transform));  // Extract and invert rotation-scale part

  // Generate display name (only for user objects, not light proxies)
  if(type == MeshType::eObject)
  {
    std::filesystem::path filepath(mesh->path);
    std::string           filename = filepath.filename().string();

    // Handle empty path (e.g., procedural meshes)
    if(filename.empty() && !mesh->path.empty())
      filename = mesh->path;  // Use whatever path string we have
    else if(filename.empty())
      filename = "Procedural";

    instance->name = fmt::format("Model {} - {}", m_nextInstanceNumber, truncateFilename(filename));
    ++m_nextInstanceNumber;
  }
  else
  {
    // Internal mesh types (light proxies, etc.) don't get numbered
    instance->name = mesh->path.empty() ? "Internal" : mesh->path;
  }

  // Set flag: needs descriptor entry
  instance->flags |= MeshInstanceVk::Flags::eNew;

  // Set index and store instance
  instance->index = instances.size();
  instances.push_back(instance);     // Add to vector
  m_lastCreatedInstance = instance;  // Store for UI convenience

  // Request GPU sync (deferred to processVramUpdates)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildTLAS;

  LOGI("createInstance: Created instance '%s' (instances.size=%zu)\n", instance->name.c_str(), instances.size());

  return instance;
}

std::shared_ptr<MeshInstanceVk> MeshManagerVk::registerInstance(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
  {
    LOGE("registerInstance: Null instance pointer\n");
    return nullptr;
  }

  // Set index and add to vector
  instance->index = instances.size();

  // Mark as new (needs descriptor entry)
  instance->flags |= MeshInstanceVk::Flags::eNew;

  instances.push_back(instance);
  m_lastCreatedInstance = instance;

  // Request GPU sync (deferred to processVramUpdates)
  pendingRequests |= Request::eUpdateDescriptors;
  pendingRequests |= Request::eRebuildTLAS;

  LOGI("registerInstance: Registered instance '%s' (instances.size=%zu)\n", instance->name.c_str(), instances.size());

  return instance;
}

std::shared_ptr<MeshInstanceVk> MeshManagerVk::duplicateInstance(std::shared_ptr<MeshInstanceVk> sourceInstance)
{
  if(!sourceInstance || !sourceInstance->mesh)
  {
    LOGE("duplicateInstance: Invalid source instance\n");
    return nullptr;
  }

  // Create new instance as a copy of the source (copy all fields via copy constructor)
  auto newInstance = std::make_shared<MeshInstanceVk>(*sourceInstance);

  // Generate NEW display name (don't copy source name) - only for user objects
  if(sourceInstance->type == MeshType::eObject)
  {
    std::filesystem::path filepath(sourceInstance->mesh->path);
    std::string           filename = filepath.filename().string();
    if(filename.empty())
      filename = sourceInstance->name;  // Fallback to source name

    newInstance->name = fmt::format("Model {} - {}", m_nextInstanceNumber, truncateFilename(filename));
    ++m_nextInstanceNumber;
  }
  // else: Keep copied name for internal instances

  // Register the new instance (this will set index and add to vector)
  // Note: registerInstance may reallocate vector, so sourceInstance reference could be invalidated after this
  newInstance = registerInstance(newInstance);

  if(!newInstance)
  {
    LOGE("duplicateInstance: Failed to register new instance\n");
    return nullptr;
  }

  LOGI("duplicateInstance: Duplicated instance (source='%s' -> new='%s')\n", sourceInstance->name.c_str(),
       newInstance->name.c_str());

  return newInstance;
}

void MeshManagerVk::updateObjDescriptionBuffer()
{
  // Rebuild objectDescriptions from instances set
  objectDescriptions.clear();

  for(const auto& instance : instances)
  {
    if(!instance || !instance->mesh)
      continue;  // Skip invalid instances

    shaderio::MeshDesc desc{};

    // Geometry addresses (from mesh)
    desc.vertexAddress        = (shaderio::ObjVertex*)instance->mesh->vertexBuffer.address;
    desc.indexAddress         = (uint32_t*)instance->mesh->indexBuffer.address;
    desc.materialAddress      = (shaderio::ObjMaterial*)instance->mesh->materialsBuffer.address;
    desc.materialIndexAddress = (uint32_t*)instance->mesh->matIndexBuffer.address;

    // Instance transform
    desc.transform                = instance->transform;
    desc.transformInverse         = instance->transformInverse;
    desc.transformRotScaleInverse = instance->transformRotScaleInverse;

    objectDescriptions.push_back(desc);
  }

  // Save old buffer to destroy after new one is ready
  nvvk::Buffer oldBuffer   = objectDescriptionsBuffer;
  objectDescriptionsBuffer = {};

  if(objectDescriptions.empty())
  {
    // Destroy old buffer if we're going to empty state
    if(oldBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(oldBuffer);
    return;
  }

  // Create buffer
  NVVK_CHECK(m_alloc->createBuffer(objectDescriptionsBuffer, objectDescriptions.size() * sizeof(shaderio::MeshDesc),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
  NVVK_DBG_NAME(objectDescriptionsBuffer.buffer);

  // Upload buffer
  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(objectDescriptionsBuffer, 0, std::span(objectDescriptions)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();

  // Destroy old buffer AFTER new one is submitted
  if(oldBuffer.buffer != VK_NULL_HANDLE)
    m_alloc->destroyBuffer(oldBuffer);
}

void MeshManagerVk::uploadMaterialsBufferInternal(std::shared_ptr<MeshVk> mesh)
{
  if(!mesh)
    return;

  // Note: This is an internal method called by processVramUpdates()
  // Assumes materialsBuffer already exists (created by initBuffers)

  // Update needShading flags before uploading materials to GPU
  for(auto& mat : mesh->materials)
  {
    shaderio::updateMaterialNeedsShading(mat);
  }

  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();

  NVVK_CHECK(m_uploader->appendBuffer(mesh->materialsBuffer, 0, std::span(mesh->materials)));

  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();
}

nvvk::AccelerationStructureGeometryInfo MeshManagerVk::rtxCreateMeshVkKHR(const MeshVk& model)
{
  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
  triangles.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
  triangles.vertexData.deviceAddress = model.vertexBuffer.address;
  triangles.vertexStride             = sizeof(ObjVertex);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = model.indexBuffer.address;
  triangles.transformData            = {};  // Identity
  triangles.maxVertex                = model.nbVertices - 1;

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR geometry{};
  geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geometry.geometry.triangles = triangles;

  // The entire array will be used to build the BLAS.
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
  rangeInfo.firstVertex     = 0;
  rangeInfo.primitiveCount  = model.nbIndices / 3;
  rangeInfo.primitiveOffset = 0;
  rangeInfo.transformOffset = 0;

  return nvvk::AccelerationStructureGeometryInfo{.geometry = geometry, .rangeInfo = rangeInfo};
}

void MeshManagerVk::rtxInitAccelerationStructures()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  // Mesh BLAS - each obj mesh is stored in a BLAS
  if(!meshes.empty())
  {
    std::vector<nvvk::AccelerationStructureGeometryInfo> asGeoInfo;
    asGeoInfo.reserve(meshes.size());

    for(const auto& mesh : meshes)
    {
      asGeoInfo.emplace_back(rtxCreateMeshVkKHR(*mesh));
    }
    // build the blas set
    NVVK_CHECK(rtAccelerationStructures.blasSubmitBuildAndWait(
        asGeoInfo, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
                       | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR));

    // Statistics
    LOGI("%s%s\n", nvutils::ScopedTimer::indent().c_str(), rtAccelerationStructures.blasBuildStatistics.toString().c_str());
  }

  // Mesh TLAS - one entry/node per instance
  if(!instances.empty())
  {
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(instances.size());
    uint32_t descriptorIndex = 0;  // Track descriptor array index

    for(const auto& instance : instances)
    {
      if(!instance || !instance->mesh)
        continue;

      size_t meshIndex = instance->mesh->index;
      if(meshIndex >= rtAccelerationStructures.blasSet.size())
        continue;

      VkAccelerationStructureInstanceKHR asInst{};
      asInst.transform           = nvvk::toTransformMatrixKHR(instance->transform);  // Position of the instance
      asInst.instanceCustomIndex = descriptorIndex;                                  // Index in descriptor array
      asInst.accelerationStructureReference = rtAccelerationStructures.blasSet[meshIndex].address;
      asInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      // Use instance type for mask (MeshType enum values match RTX masks)
      asInst.mask                                   = static_cast<uint32_t>(instance->type);
      asInst.instanceShaderBindingTableRecordOffset = 1;  // We will use the same closest hit hit group for all objects
      tlasInstances.emplace_back(asInst);
      descriptorIndex++;
    }
    // then build the dynamic TLAS, add allow update flag so we can update mesh matrices and use tlasUpade
    NVVK_CHECK(rtAccelerationStructures.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                                  | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
  }
}

void MeshManagerVk::rtxUpdateTopLevelAccelerationStructure()
{
  // Prepare TLAS for Incrusted meshes, different from the splat one.
  if(!instances.empty())
  {
    // TODO could be a class member to prevent reallocation
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(instances.size());
    uint32_t descriptorIndex = 0;  // Track descriptor array index

    for(const auto& instance : instances)
    {
      if(!instance || !instance->mesh)
        continue;

      size_t meshIndex = instance->mesh->index;
      if(meshIndex >= rtAccelerationStructures.blasSet.size())
        continue;

      VkAccelerationStructureInstanceKHR asInst{};
      asInst.transform           = nvvk::toTransformMatrixKHR(instance->transform);  // Position of the instance
      asInst.instanceCustomIndex = descriptorIndex;                                  // Index in descriptor array
      asInst.accelerationStructureReference = rtAccelerationStructures.blasSet[meshIndex].address;
      asInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      // Use instance type for mask (MeshType enum values match RTX masks)
      asInst.mask                                   = static_cast<uint32_t>(instance->type);
      asInst.instanceShaderBindingTableRecordOffset = 1;  // We will use the same closest hit hit group for all objects
      tlasInstances.emplace_back(asInst);
      descriptorIndex++;
    }

    // Check if instance count changed - if so, rebuild TLAS from scratch
    if(tlasInstances.size() != rtAccelerationStructures.tlasSize)
    {
      LOGI("Instance count changed (%zu -> %zu), rebuilding TLAS\n", rtAccelerationStructures.tlasSize, tlasInstances.size());

      // Wait for GPU to finish using old TLAS resources before destroying them
      vkDeviceWaitIdle(m_app->getDevice());

      // Manually destroy old TLAS resources (but keep BLAS)
      if(rtAccelerationStructures.tlas.accel)
        m_alloc->destroyAcceleration(rtAccelerationStructures.tlas);
      if(rtAccelerationStructures.tlasInstancesBuffer.buffer)
        m_alloc->destroyLargeBuffer(rtAccelerationStructures.tlasInstancesBuffer);
      if(rtAccelerationStructures.tlasScratchBuffer.buffer)
        m_alloc->destroyLargeBuffer(rtAccelerationStructures.tlasScratchBuffer);

      rtAccelerationStructures.tlas                = {};
      rtAccelerationStructures.tlasInstancesBuffer = {};
      rtAccelerationStructures.tlasScratchBuffer   = {};
      rtAccelerationStructures.tlasBuildData       = {};
      rtAccelerationStructures.tlasSize            = 0;

      // Rebuild with new instance count
      NVVK_CHECK(rtAccelerationStructures.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                                    | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
    }
    else
    {
      // Just update existing TLAS (transforms only)
      rtAccelerationStructures.tlasSubmitUpdateAndWait(tlasInstances);
    }
  }
}

void MeshManagerVk::deleteInstance(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
  {
    LOGE("deleteInstance: Null instance pointer\n");
    return;
  }

  // Set Delete flag (deferred deletion in processVramUpdates)
  instance->flags |= MeshInstanceVk::Flags::eDelete;
  pendingRequests |= Request::eProcessDeletions;

  LOGI("deleteInstance: Marked instance for deletion ('%s')\n", instance->name.c_str());
}

void MeshManagerVk::deleteInstanceOnly(std::shared_ptr<MeshInstanceVk> instance)
{
  // Delete instance without destroying the mesh (used for light proxies where meshes are persistent)
  // Note: Now uses deferred mechanism like deleteInstance() - processVramUpdates handles both
  if(!instance)
  {
    LOGE("deleteInstanceOnly: Null instance pointer\n");
    return;
  }

  // Set Delete flag (deferred deletion in processVramUpdates)
  instance->flags |= MeshInstanceVk::Flags::eDelete;
  pendingRequests |= Request::eProcessDeletions;

  LOGI("deleteInstanceOnly: Marked instance for deletion ('%s')\n", instance->name.c_str());
}

// =============================================================================
// DEFERRED UPDATE API - Methods that set flags and requests
// =============================================================================

void MeshManagerVk::deleteMesh(std::shared_ptr<MeshVk> mesh)
{
  if(!mesh)
    return;

  mesh->flags |= MeshVk::Flags::eDelete;
  pendingRequests |= Request::eProcessDeletions;
}

void MeshManagerVk::updateInstanceTransform(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
    return;

  // Caller has already modified instance->transform in RAM
  // Just set flag and request GPU update
  instance->flags |= MeshInstanceVk::Flags::eTransformChanged;
  pendingRequests |= Request::eUpdateTransformsOnly;
}

void MeshManagerVk::updateInstanceMaterial(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
    return;

  instance->flags |= MeshInstanceVk::Flags::eMaterialChanged;
  pendingRequests |= Request::eUpdateDescriptors;  // May need descriptor rebuild
}

void MeshManagerVk::updateMeshMaterials(std::shared_ptr<MeshVk> mesh)
{
  if(!mesh)
    return;

  // Materials already modified directly in RAM by caller
  // Just mark for GPU upload
  mesh->flags |= MeshVk::Flags::eMaterialsChanged;
  pendingRequests |= Request::eUpdateMaterials;
}

// =============================================================================
// VRAM SYNC - Process all deferred updates
// =============================================================================

void MeshManagerVk::processVramUpdates(bool processRtx)
{
  bool instanceCountChanged   = false;
  bool descriptorsNeedRebuild = false;

  // =========================================================================
  // Phase 1 - Remove from GPU + delete from RAM using shift-left compaction
  // =========================================================================

  if(static_cast<uint32_t>(pendingRequests & Request::eProcessDeletions))
  {
    // Step 1.1: Delete instances (shift-left compaction)
    {
      size_t originalSize = instances.size();
      size_t shiftLeft    = 0;

      for(size_t i = 0; i < originalSize; i++)
      {
        if(instances[i]->isMarkedForDeletion())
        {
          // Instance will be destroyed when shared_ptr is released
          shiftLeft++;
          instanceCountChanged = true;
        }
        else
        {
          // Keep instance - shift it left if needed
          instances[i - shiftLeft]        = instances[i];
          instances[i - shiftLeft]->index = i - shiftLeft;
        }
      }

      instances.resize(originalSize - shiftLeft);

      if(shiftLeft > 0)
        LOGI("Deleted %zu mesh instances\n", shiftLeft);
    }

    // Step 1.2: Delete meshes (only if no instances reference them)
    {
      size_t originalSize = meshes.size();
      size_t shiftLeft    = 0;

      for(size_t i = 0; i < originalSize; i++)
      {
        if(meshes[i]->isMarkedForDeletion())
        {
          // Verify no instances still reference this mesh
          bool hasReferences = false;
          for(const auto& inst : instances)
          {
            if(inst->mesh == meshes[i])
            {
              hasReferences = true;
              break;
            }
          }

          if(hasReferences)
          {
            // Clear delete flag - still in use
            meshes[i]->flags &= ~MeshVk::Flags::eDelete;
            // Keep this mesh
            meshes[i - shiftLeft]        = meshes[i];
            meshes[i - shiftLeft]->index = i - shiftLeft;
          }
          else
          {
            // Safe to delete
            vkDeviceWaitIdle(m_app->getDevice());
            meshes[i]->deinitBuffers(m_alloc);
            shiftLeft++;
          }
        }
        else
        {
          meshes[i - shiftLeft]        = meshes[i];
          meshes[i - shiftLeft]->index = i - shiftLeft;
        }
      }

      meshes.resize(originalSize - shiftLeft);

      if(shiftLeft > 0)
        LOGI("Deleted %zu meshes\n", shiftLeft);
    }

    if(instanceCountChanged)
    {
      descriptorsNeedRebuild = true;
      pendingRequests |= Request::eRebuildTLAS;
    }

    pendingRequests &= ~Request::eProcessDeletions;
  }

  // =========================================================================
  // PHASE 2: UPDATES (RAM → GPU sync)
  // =========================================================================

  // Process New meshes (allocate buffers + upload geometry)
  bool needsUpload = false;
  for(const auto& mesh : meshes)
  {
    if(static_cast<uint32_t>(mesh->flags & MeshVk::Flags::eNew))
    {
      // Allocate buffers and append to upload queue
      mesh->initBuffers(m_alloc, m_uploader);
      needsUpload = true;

      // Clear RAM copies after upload is queued (optional optimization)
      mesh->vertexData.clear();
      mesh->vertexData.shrink_to_fit();
      mesh->indexData.clear();
      mesh->indexData.shrink_to_fit();
      mesh->matIndexData.clear();
      mesh->matIndexData.shrink_to_fit();

      descriptorsNeedRebuild = true;
      pendingRequests |= Request::eRebuildBLAS;
      mesh->flags &= ~MeshVk::Flags::eNew;  // Clear flag
    }
  }

  // Execute all uploads in a single command buffer
  if(needsUpload)
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_uploader->cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_uploader->releaseStaging();
  }

  // Process New instances (add to descriptors)
  for(const auto& instance : instances)
  {
    if(static_cast<uint32_t>(instance->flags & MeshInstanceVk::Flags::eNew))
    {
      // Mesh geometry already uploaded above or in previous frame
      // Just needs descriptor entry
      instanceCountChanged   = true;
      descriptorsNeedRebuild = true;
      instance->flags &= ~MeshInstanceVk::Flags::eNew;  // Clear flag
    }
  }

  // Process material changes
  if(static_cast<uint32_t>(pendingRequests & Request::eUpdateMaterials))
  {
    for(const auto& mesh : meshes)
    {
      if(static_cast<uint32_t>(mesh->flags & MeshVk::Flags::eMaterialsChanged))
      {
        uploadMaterialsBufferInternal(mesh);
        mesh->flags &= ~MeshVk::Flags::eMaterialsChanged;
      }
    }

    // Update descriptor buffer so raster pipeline sees the new material data
    descriptorsNeedRebuild = true;

    pendingRequests &= ~Request::eUpdateMaterials;
  }

  // =========================================================================
  // PHASE 3: RTX ACCELERATION STRUCTURES (BLAS first, then TLAS)
  // =========================================================================

  // Only process RTX if in RTX pipeline mode
  // In raster mode, defer RTX builds until pipeline switch
  if(processRtx)
  {
    // IMPORTANT: Rebuild BLAS BEFORE any TLAS operations!
    // BLAS must exist before TLAS can reference them

    // Rebuild BLAS if needed
    if(static_cast<uint32_t>(pendingRequests & Request::eRebuildBLAS))
    {
      rtxDeinitAccelerationStructures();  // Clean up old structures first
      rtxInitAccelerationStructures();    // Rebuilds both BLAS and TLAS
      pendingRequests &= ~Request::eRebuildBLAS;
      pendingRequests &= ~Request::eRebuildTLAS;           // Already rebuilt
      pendingRequests &= ~Request::eUpdateTransformsOnly;  // Already applied in TLAS build
    }
    // Rebuild TLAS if needed (and BLAS wasn't rebuilt)
    else if(static_cast<uint32_t>(pendingRequests & Request::eRebuildTLAS))
    {
      rtxDeinitAccelerationStructures();  // Clean up old structures first
      rtxInitAccelerationStructures();    // Rebuilds both BLAS and TLAS
      pendingRequests &= ~Request::eRebuildTLAS;
      pendingRequests &= ~Request::eUpdateTransformsOnly;  // Already applied in TLAS build
    }
    // Update TLAS only (fast path for transform changes)
    else if(static_cast<uint32_t>(pendingRequests & Request::eUpdateTransformsOnly))
    {
      // Update TLAS with new transforms (for RTX pipeline - no rebuild, just update)
      rtxUpdateTopLevelAccelerationStructure();

      // Transform changes also need descriptor updates (for raster pipeline)
      // MeshDesc contains transform matrices that shaders read
      pendingRequests |= Request::eUpdateDescriptors;

      // Clear flags
      for(const auto& instance : instances)
      {
        instance->flags &= ~MeshInstanceVk::Flags::eTransformChanged;
      }

      pendingRequests &= ~Request::eUpdateTransformsOnly;
    }
  }
  else
  {
    // Raster mode: Keep BLAS/TLAS rebuild flags for deferred processing
    // When switching to RTX pipeline, these accumulated flags will trigger rebuild
    // Do NOT clear RebuildBLAS or RebuildTLAS flags

    // However, transform changes MUST update descriptor buffers immediately
    // (raster shaders read MeshDesc transform matrices from descriptor buffer)
    if(static_cast<uint32_t>(pendingRequests & Request::eUpdateTransformsOnly))
    {
      // Clear instance flags to prevent duplicate processing
      for(const auto& instance : instances)
      {
        instance->flags &= ~MeshInstanceVk::Flags::eTransformChanged;
      }

      // Trigger descriptor buffer update (needed for raster pipeline)
      pendingRequests |= Request::eUpdateDescriptors;

      // Clear UpdateTransformsOnly (we've handled it by updating descriptors)
      // When switching to RTX, RebuildBLAS/RebuildTLAS flags will trigger full rebuild
      pendingRequests &= ~Request::eUpdateTransformsOnly;
    }
  }

  // =========================================================================
  // PHASE 4: DESCRIPTORS (After all RTX structures are ready)
  // =========================================================================

  // Rebuild descriptors if needed (must be after BLAS/TLAS so addresses are valid)
  if(descriptorsNeedRebuild || static_cast<uint32_t>(pendingRequests & Request::eUpdateDescriptors))
  {
    updateObjDescriptionBuffer();
    pendingRequests &= ~Request::eUpdateDescriptors;
  }
}

}  // namespace vk_gaussian_splatting
