/*/*
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

#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <nvapp/application.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/acceleration_structures.hpp>

#include "shaderio.h"

#include "obj_loader.h"

namespace vk_gaussian_splatting {

// The OBJ model
struct Mesh
{
  std::string              path;
  std::string              name;
  uint32_t                 nbIndices{0};
  uint32_t                 nbVertices{0};
  nvvk::Buffer             vertexBuffer;     // Device buffer of all 'Vertex'
  nvvk::Buffer             indexBuffer;      // Device buffer of the indices forming triangles
  nvvk::Buffer             materialsBuffer;  // Device buffer of 'Wavefront' materials
  nvvk::Buffer             matIndexBuffer;   // Device buffer of per face material IDs
  std::vector<ObjMaterial> materials;        // RAM storage of materials for updates
  std::vector<std::string> matNames;         // name of each material stored in materials
};

struct Instance
{
  uint32_t  objIndex{0};  // Model index reference
  glm::vec3 translation{0.0f};
  glm::vec3 rotation{0.0f};
  glm::vec3 scale{1.0f, 1.0f, 1.0f};
  glm::mat4 transform;           // Matrix of the instance
  glm::mat4 transformInverse{};  // Inverse Matrix of the instance
};

class MeshSetVk
{
public:
  // Array of objects and instances in the scene
  std::vector<Mesh>              meshes;                    // Model on device
  std::vector<Instance>          instances;                 // Scene model instances
  std::vector<shaderio::ObjDesc> objectDescriptions;        // Model description for device access
  nvvk::Buffer                   objectDescriptionsBuffer;  // Device buffer of the OBJ objectDescriptions

  // RTX specifics
  nvvk::AccelerationStructureHelper rtAccelerationStructures;

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
    rtAccelerationStructures.deinit();
    m_app      = {};
    m_alloc    = {};
    m_uploader = {};
  }

  // load model from file, add it to the model set and upload to VRAM
  // for convenience RTX acceleration structures are also updated
  bool loadModel(const std::filesystem::path& filename);

  inline void deinitDataStorage()
  {
    // all data common to RTX and raster
    for(auto& mesh : meshes)
    {
      deinitMeshBuffers(mesh);
    }
    m_alloc->destroyBuffer(objectDescriptionsBuffer);
    objectDescriptionsBuffer = {};
    objectDescriptions.clear();
    instances.clear();
    meshes.clear();
  };

  // update the buffer that stores the list of object IDs
  // and some per object information
  void updateObjDescriptionBuffer();

  // update the buffer that stores the materials of a specific model
  void updateObjMaterialsBuffer(int modelIndex);

  // init BLAS and TLAS for all the loaded models
  void rtxInitAccelerationStructures();

  // update TLAS transforms from instances to device
  void rtxUpdateTopLevelAccelerationStructure();

  void rtxDeinitAccelerationStructures() { rtAccelerationStructures.deinitAccelerationStructures(); }

  // delete an instance, and its related mesh and material if last instance using it.
  // object descriptions buffer and rtx acceleration structures must be updated afterward
  void deleteInstance(uint32_t instanceId);

private:
  void deinitMeshBuffers(Mesh& mesh)
  {
    // all data common to RTX and raster
    m_alloc->destroyBuffer(mesh.vertexBuffer);
    m_alloc->destroyBuffer(mesh.indexBuffer);
    m_alloc->destroyBuffer(mesh.materialsBuffer);
    m_alloc->destroyBuffer(mesh.matIndexBuffer);
  }

  // RTX specifics
  nvvk::AccelerationStructureGeometryInfo rtxCreateMeshVkKHR(const Mesh& model);

private:
  nvapp::Application*      m_app{};
  nvvk::ResourceAllocator* m_alloc{};
  nvvk::StagingUploader*   m_uploader{};
};

}  // namespace vk_gaussian_splatting