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

#include "light_manager_vk.h"
#include "mesh_manager_vk.h"
#include "utilities.h"

#include <nvvk/debug_util.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/check_error.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/primitives.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace vk_gaussian_splatting {

// =============================================================================
// Lifecycle Methods
// =============================================================================

void LightManagerVk::init(nvapp::Application* app, nvvk::ResourceAllocator* alloc, nvvk::StagingUploader* uploader)
{
  m_app      = app;
  m_alloc    = alloc;
  m_uploader = uploader;

  // Start with empty arrays (no pre-allocation)
  lightSources.clear();
  instances.clear();
  m_nextLightNumber = 0;

  // No buffer creation here - created on first light
  lightsBuffer = {};

  LOGD("LightManagerVk::init: Initialized (0 lights)\n");
}

void LightManagerVk::reset()
{
  // CRITICAL: Use the SAME deferred deletion flow as DELETE (which works perfectly)
  // Call deleteInstance() for each light instance
  // This reuses the existing, tested deletion logic (which also handles proxy cleanup)

  // Delete all light instances (iterate backwards to avoid issues with index changes)
  // deleteInstance() will handle proxy mesh/instance deletion through MeshManagerVk
  for(size_t i = instances.size(); i-- > 0;)
  {
    if(instances[i])
    {
      deleteInstance(instances[i]);
    }
  }

  // Note: lightSources (assets) will be deleted automatically by deleteInstance()
  // if they have no remaining instances (same as normal delete flow)

  // Reset naming counter (will be recreated from 0 when new lights are added)
  m_nextLightNumber = 0;

  LOGD("LightManagerVk::reset: Marked all assets for deletion\n");
}

void LightManagerVk::deinit()
{
  reset();  // Clears data and destroys buffer
}

// =============================================================================
// Asset/Instance Management
// =============================================================================

std::shared_ptr<LightSourceInstanceVk> LightManagerVk::createLight()
{
  // 1. Create light asset (shared properties only)
  auto lightAsset = std::make_shared<LightSourceVk>();

  // Initialize asset fields (shared across instances)
  lightAsset->type            = shaderio::LightType::ePointLight;  // Point light (default)
  lightAsset->color           = glm::vec3(1.0f);
  lightAsset->intensity       = 1.0f;
  lightAsset->range           = 10.0f;
  lightAsset->innerConeAngle  = 30.0f;
  lightAsset->outerConeAngle  = 45.0f;
  lightAsset->attenuationMode = 2;  // Quadratic
  lightAsset->proxyScale      = 1.0f;

  // Create proxy mesh for THIS asset based on light type (NOT shared!)
  // Each asset gets its own mesh so it can have its own material/color
  if(m_meshSetVk)
  {
    // Choose proxy mesh based on type
    switch(lightAsset->type)
    {
      case shaderio::LightType::eDirectionalLight:
        lightAsset->proxyMesh = createProxyQuad();
        break;
      case shaderio::LightType::ePointLight:
        lightAsset->proxyMesh = createProxySphere();
        break;
      case shaderio::LightType::eSpotLight:
        lightAsset->proxyMesh = createProxyCone();
        break;
      default:
        lightAsset->proxyMesh = createProxySphere();
        break;
    }

    // Set material as purely emissive (no lighting)
    if(lightAsset->proxyMesh && !lightAsset->proxyMesh->materials.empty())
    {
      lightAsset->proxyMesh->materials[0].emission = lightAsset->color;
      lightAsset->proxyMesh->materials[0].diffuse  = glm::vec3(0.0f);  // No lighting
      lightAsset->proxyMesh->materials[0].ambient  = glm::vec3(0.0f);  // No lighting
      lightAsset->proxyMesh->materials[0].specular = glm::vec3(0.0f);  // No lighting
      lightAsset->proxyMaterial                    = lightAsset->proxyMesh->materials[0];
    }
  }

  lightSources.push_back(lightAsset);

  // 2. Create instance (per-instance data)
  auto instance         = std::make_shared<LightSourceInstanceVk>();
  instance->lightSource = lightAsset;
  instance->name        = fmt::format("Light {}", m_nextLightNumber);
  instance->translation = glm::vec3(0.0f, 2.0f, 0.0f);    // Initial position (per-instance!)
  instance->rotation    = glm::vec3(-90.0f, 0.0f, 0.0f);  // Initial rotation: -90° around X to point down (-Y)
  instance->index       = instances.size();
  ++m_nextLightNumber;

  // Create proxy mesh instance for visualization
  if(m_meshSetVk && lightAsset->proxyMesh)
  {
    instance->proxyInstance       = m_meshSetVk->createInstance(lightAsset->proxyMesh,
                                                                glm::mat4(1.0f),  // identity transform (will be updated by updateProxyTransform)
                                                                MeshType::eLightProxy  // mark as light proxy so it doesn't appear in UI tree
          );
    instance->proxyInstance->name = instance->name + " (proxy)";
    updateProxyTransform(instance);
  }

  instances.push_back(instance);

  // Mark for GPU buffer rebuild
  pendingRequests |= Request::eRebuildBuffer;

  LOGD("Created light: %s\n", instance->name.c_str());
  return instance;
}

std::shared_ptr<LightSourceInstanceVk> LightManagerVk::duplicateInstance(std::shared_ptr<LightSourceInstanceVk> sourceInstance)
{
  if(!sourceInstance || !sourceInstance->lightSource)
  {
    LOGE("duplicateInstance: Invalid source instance\n");
    return nullptr;
  }

  // SHARE the asset (don't create new one!)
  // This allows editing color/intensity to affect all instances
  auto newInstance         = std::make_shared<LightSourceInstanceVk>();
  newInstance->lightSource = sourceInstance->lightSource;  // ✅ SHARED ASSET!
  newInstance->name        = fmt::format("Light {}", m_nextLightNumber);
  newInstance->translation = sourceInstance->translation;  // Copy translation
  newInstance->rotation    = sourceInstance->rotation;     // Copy rotation
  newInstance->index       = instances.size();
  ++m_nextLightNumber;

  // Create proxy mesh instance for this new instance
  if(m_meshSetVk && newInstance->lightSource->proxyMesh)
  {
    newInstance->proxyInstance =
        m_meshSetVk->createInstance(newInstance->lightSource->proxyMesh,
                                    glm::mat4(1.0f),  // identity transform (will be updated by updateProxyTransform)
                                    MeshType::eLightProxy  // mark as light proxy so it doesn't appear in UI tree
        );
    newInstance->proxyInstance->name = newInstance->name + " (proxy)";
    updateProxyTransform(newInstance);
  }

  instances.push_back(newInstance);

  // Mark for GPU buffer rebuild
  pendingRequests |= Request::eRebuildBuffer;

  LOGD("Duplicated light: %s\n", newInstance->name.c_str());
  return newInstance;
}

void LightManagerVk::deleteInstance(std::shared_ptr<LightSourceInstanceVk> instance)
{
  if(!instance)
  {
    LOGE("deleteInstance: Invalid instance\n");
    return;
  }

  uint64_t index = instance->index;

  // Validate index
  if(index >= instances.size() || instances[index] != instance)
  {
    LOGE("deleteInstance: Instance index mismatch or out of range\n");
    return;
  }

  // Delete proxy mesh instance
  if(m_meshSetVk && instance->proxyInstance)
  {
    m_meshSetVk->deleteInstanceOnly(instance->proxyInstance);
    instance->proxyInstance = nullptr;
  }

  // Check if this is the last instance using this light asset
  auto lightAsset = instance->lightSource;
  if(lightAsset)
  {
    int refCount = 0;
    for(const auto& inst : instances)
    {
      if(inst->lightSource == lightAsset)
      {
        refCount++;
      }
    }

    // If last reference, remove light asset and its proxy mesh
    if(refCount == 1)
    {
      // Delete proxy mesh (each asset has its own mesh)
      if(m_meshSetVk && lightAsset->proxyMesh)
      {
        m_meshSetVk->deleteMesh(lightAsset->proxyMesh);
        lightAsset->proxyMesh = nullptr;
      }

      auto assetIt = std::find(lightSources.begin(), lightSources.end(), lightAsset);
      if(assetIt != lightSources.end())
      {
        lightSources.erase(assetIt);
        LOGD("Deleted light asset (last reference)\n");
      }
    }
  }

  // Remove instance
  instances.erase(instances.begin() + index);

  // Update indices for remaining instances
  for(size_t i = index; i < instances.size(); ++i)
  {
    instances[i]->index = i;
  }

  // Mark for GPU buffer rebuild
  pendingRequests |= Request::eRebuildBuffer;

  LOGD("Deleted light instance (remaining: %zu)\n", instances.size());
}

void LightManagerVk::updateLight(std::shared_ptr<LightSourceInstanceVk> instance)
{
  if(!instance)
  {
    LOGE("updateLight: Invalid instance\n");
    return;
  }

  // Update this instance's proxy transform (position changed)
  updateProxyTransform(instance);

  // Mark for buffer update (data changed, but size unchanged)
  pendingRequests |= Request::eUpdateBuffer;
}

void LightManagerVk::updateLightAsset(std::shared_ptr<LightSourceVk> asset)
{
  if(!asset)
  {
    LOGE("updateLightAsset: Invalid asset\n");
    return;
  }

  // Update proxy material color (affects all instances sharing this asset)
  asset->proxyMaterial.emission = asset->color;

  // Update the mesh's material (stored in MeshVk) - keep it purely emissive!
  if(asset->proxyMesh && !asset->proxyMesh->materials.empty())
  {
    asset->proxyMesh->materials[0].emission = asset->color;
    asset->proxyMesh->materials[0].diffuse  = glm::vec3(0.0f);  // Ensure no lighting
    asset->proxyMesh->materials[0].ambient  = glm::vec3(0.0f);  // Ensure no lighting
    asset->proxyMesh->materials[0].specular = glm::vec3(0.0f);  // Ensure no lighting
    m_meshSetVk->updateMeshMaterials(asset->proxyMesh);         // ✅ Mark for GPU upload!
  }

  // Update ALL instances that share this asset (transforms/scale)
  for(auto& instance : instances)
  {
    if(instance->lightSource == asset)
    {
      updateProxyTransform(instance);  // Update transform/scale
    }
  }

  // Mark for buffer update (asset properties changed)
  pendingRequests |= Request::eUpdateBuffer;
}

void LightManagerVk::recreateProxyForAsset(std::shared_ptr<LightSourceVk> asset)
{
  if(!asset || !m_meshSetVk)
  {
    LOGE("recreateProxyForAsset: Invalid asset or mesh manager\n");
    return;
  }

  // Delete old proxy mesh and all its instances
  if(asset->proxyMesh)
  {
    // Delete all proxy instances using this mesh
    for(auto& instance : instances)
    {
      if(instance->lightSource == asset && instance->proxyInstance)
      {
        m_meshSetVk->deleteInstanceOnly(instance->proxyInstance);
        instance->proxyInstance = nullptr;
      }
    }

    // Delete the mesh itself
    m_meshSetVk->deleteMesh(asset->proxyMesh);
    asset->proxyMesh = nullptr;
  }

  // Create new proxy mesh based on current type
  switch(asset->type)
  {
    case shaderio::LightType::eDirectionalLight:
      asset->proxyMesh = createProxyQuad();
      break;
    case shaderio::LightType::ePointLight:
      asset->proxyMesh = createProxySphere();
      break;
    case shaderio::LightType::eSpotLight:
      asset->proxyMesh = createProxyCone();
      break;
    default:
      asset->proxyMesh = createProxySphere();
      break;
  }

  // Set material as purely emissive
  if(asset->proxyMesh && !asset->proxyMesh->materials.empty())
  {
    asset->proxyMesh->materials[0].emission = asset->color;
    asset->proxyMesh->materials[0].diffuse  = glm::vec3(0.0f);
    asset->proxyMesh->materials[0].ambient  = glm::vec3(0.0f);
    asset->proxyMesh->materials[0].specular = glm::vec3(0.0f);
    asset->proxyMaterial                    = asset->proxyMesh->materials[0];
  }

  // Recreate proxy instances for all instances using this asset
  for(auto& instance : instances)
  {
    if(instance->lightSource == asset)
    {
      instance->proxyInstance = m_meshSetVk->createInstance(asset->proxyMesh, glm::mat4(1.0f), MeshType::eLightProxy);
      instance->proxyInstance->name = instance->name + " (proxy)";
      updateProxyTransform(instance);
    }
  }

  LOGD("Recreated proxy mesh for light asset (new type: %d)\n", static_cast<int>(asset->type));
}

// =============================================================================
// Access Methods
// =============================================================================

const std::string& LightManagerVk::getLightName(uint64_t index) const
{
  if(index >= instances.size())
  {
    static std::string empty = "";
    LOGE("getLightName: Index %zu out of range (size=%zu)\n", static_cast<size_t>(index), instances.size());
    return empty;
  }
  return instances[index]->name;
}

// getLight() removed - use getInstance() instead

// =============================================================================
// Deferred Update System
// =============================================================================

bool LightManagerVk::processVramUpdates()
{
  bool instancesChanged = false;

  // Handle buffer rebuild (size changed: add/delete)
  if(static_cast<uint32_t>(pendingRequests & Request::eRebuildBuffer))
  {
    rebuildBuffer();
    pendingRequests &= ~Request::eRebuildBuffer;
    pendingRequests &= ~Request::eUpdateBuffer;  // Rebuild includes update
    instancesChanged = true;                     // Need to update assets buffer
  }
  // Handle buffer update (data changed, size unchanged)
  else if(static_cast<uint32_t>(pendingRequests & Request::eUpdateBuffer))
  {
    updateBuffer();
    pendingRequests &= ~Request::eUpdateBuffer;
  }

  // Note: Proxy instance updates are handled by MeshManagerVk::processVramUpdates()
  return instancesChanged;
}

// =============================================================================
// Internal Buffer Management
// =============================================================================

void LightManagerVk::rebuildBuffer()
{
  // Save old buffer
  nvvk::Buffer oldBuffer = lightsBuffer;
  lightsBuffer           = {};

  if(instances.empty())
  {
    // No lights - destroy old buffer
    if(oldBuffer.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(oldBuffer);
    }
    LOGD("Light buffer destroyed (0 lights)\n");
    return;
  }

  // Build CPU array: combine instance position with asset properties
  std::vector<shaderio::LightSource> shaderLights;
  shaderLights.reserve(instances.size());

  for(const auto& instance : instances)
  {
    if(instance && instance->lightSource)
    {
      shaderio::LightSource shaderLight;
      // From instance
      shaderLight.position = instance->translation;
      shaderLight.direction = rotateDirection(instance->rotation, glm::vec3(0.0f, 0.0f, -1.0f));  // Computed from rotation
      // From asset
      shaderLight.type            = instance->lightSource->type;
      shaderLight.color           = instance->lightSource->color;
      shaderLight.intensity       = instance->lightSource->intensity;
      shaderLight.range           = instance->lightSource->range;
      shaderLight.innerConeAngle  = instance->lightSource->innerConeAngle;
      shaderLight.outerConeAngle  = instance->lightSource->outerConeAngle;
      shaderLight.attenuationMode = instance->lightSource->attenuationMode;
      shaderLight.radius          = instance->lightSource->proxyScale;  // For soft shadows disk sampling
      shaderLights.push_back(shaderLight);
    }
  }

  // Allocate new buffer
  size_t bufferSize = shaderLights.size() * sizeof(shaderio::LightSource);
  NVVK_CHECK(m_alloc->createBuffer(lightsBuffer, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  NVVK_DBG_NAME(lightsBuffer.buffer);

  // Upload data
  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_uploader->appendBuffer(lightsBuffer, 0, std::span(shaderLights)));
  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();

  // Destroy old buffer AFTER upload
  if(oldBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(oldBuffer);
  }

  LOGD("Light buffer rebuilt: %zu lights, %zu bytes\n", instances.size(), bufferSize);
}

void LightManagerVk::updateBuffer()
{
  if(instances.empty() || lightsBuffer.buffer == VK_NULL_HANDLE)
  {
    return;  // No buffer to update
  }

  // Build CPU array: combine instance position with asset properties
  std::vector<shaderio::LightSource> shaderLights;
  shaderLights.reserve(instances.size());

  for(const auto& instance : instances)
  {
    if(instance && instance->lightSource)
    {
      shaderio::LightSource shaderLight;
      // From instance
      shaderLight.position = instance->translation;
      shaderLight.direction = rotateDirection(instance->rotation, glm::vec3(0.0f, 0.0f, -1.0f));  // Computed from rotation
      // From asset
      shaderLight.type            = instance->lightSource->type;
      shaderLight.color           = instance->lightSource->color;
      shaderLight.intensity       = instance->lightSource->intensity;
      shaderLight.range           = instance->lightSource->range;
      shaderLight.innerConeAngle  = instance->lightSource->innerConeAngle;
      shaderLight.outerConeAngle  = instance->lightSource->outerConeAngle;
      shaderLight.attenuationMode = instance->lightSource->attenuationMode;
      shaderLight.radius          = instance->lightSource->proxyScale;  // For soft shadows disk sampling
      shaderLights.push_back(shaderLight);
    }
  }

  // Upload data (buffer size unchanged)
  VkCommandBuffer cmdBuf = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_uploader->appendBuffer(lightsBuffer, 0, std::span(shaderLights)));
  m_uploader->cmdUploadAppended(cmdBuf);
  m_app->submitAndWaitTempCmdBuffer(cmdBuf);
  m_uploader->releaseStaging();

  LOGD("Light buffer updated: %zu lights\n", instances.size());
}

// =============================================================================
// Proxy Management
// =============================================================================

std::shared_ptr<MeshVk> LightManagerVk::createProxySphere()
{
  if(!m_meshSetVk)
  {
    LOGE("createProxySphere: MeshManagerVk not set\n");
    return nullptr;
  }

  // Create sphere geometry (simplified - icosahedron-based)
  const int   segments = 16;
  const int   rings    = 8;
  const float radius   = 0.1f;

  std::vector<ObjVertex> vertices;
  std::vector<uint32_t>  indices;

  // Generate sphere vertices
  for(int ring = 0; ring <= rings; ++ring)
  {
    float phi = glm::pi<float>() * float(ring) / float(rings);
    for(int seg = 0; seg <= segments; ++seg)
    {
      float theta = 2.0f * glm::pi<float>() * float(seg) / float(segments);

      ObjVertex vert{};
      vert.pos = glm::vec3(radius * sin(phi) * cos(theta), radius * cos(phi), radius * sin(phi) * sin(theta));
      vert.nrm = glm::normalize(vert.pos);
      vertices.push_back(vert);
    }
  }

  // Generate sphere indices
  for(int ring = 0; ring < rings; ++ring)
  {
    for(int seg = 0; seg < segments; ++seg)
    {
      int i0 = ring * (segments + 1) + seg;
      int i1 = i0 + 1;
      int i2 = (ring + 1) * (segments + 1) + seg;
      int i3 = i2 + 1;

      indices.push_back(i0);
      indices.push_back(i2);
      indices.push_back(i1);

      indices.push_back(i1);
      indices.push_back(i2);
      indices.push_back(i3);
    }
  }

  // Create purely emissive material (no lighting interaction)
  ObjMaterial material{};
  material.emission                  = glm::vec3(1.0f);  // White by default (will be updated)
  material.diffuse                   = glm::vec3(0.0f);  // ✅ No diffuse = no lighting
  material.ambient                   = glm::vec3(0.0f);  // ✅ No ambient = no lighting
  material.specular                  = glm::vec3(0.0f);  // ✅ No specular = no lighting
  std::vector<ObjMaterial> materials = {material};
  std::vector<uint32_t>    matIndices(indices.size() / 3, 0);  // All faces use material 0

  // Create mesh (each light gets its own mesh for independent materials)
  auto mesh = m_meshSetVk->createMesh("light_proxy_sphere", vertices, indices, materials, matIndices);

  LOGD("Created proxy sphere: %zu vertices, %zu indices\n", vertices.size(), indices.size());
  return mesh;
}

std::shared_ptr<MeshVk> LightManagerVk::createProxyCone(int segments)
{
  if(!m_meshSetVk)
  {
    LOGE("createProxyCone: MeshManagerVk not set\n");
    return nullptr;
  }

  // Create cone geometry: apex at origin, pointing along -Z axis (matching light direction), base at z=-height
  const float height     = 0.2f;  // Small cone (5x smaller than original)
  const float baseRadius = 0.1f;

  std::vector<ObjVertex> vertices;
  std::vector<uint32_t>  indices;

  // Apex vertex (at origin)
  ObjVertex apex{};
  apex.pos = glm::vec3(0.0f, 0.0f, 0.0f);
  apex.nrm = glm::vec3(0.0f, 0.0f, 1.0f);  // Point forward
  vertices.push_back(apex);

  // Base center vertex
  ObjVertex baseCenter{};
  baseCenter.pos = glm::vec3(0.0f, 0.0f, -height);
  baseCenter.nrm = glm::vec3(0.0f, 0.0f, -1.0f);
  vertices.push_back(baseCenter);

  // Base circle vertices
  for(int i = 0; i <= segments; ++i)
  {
    float theta = 2.0f * glm::pi<float>() * float(i) / float(segments);
    float x     = baseRadius * cos(theta);
    float y     = baseRadius * sin(theta);

    // Base vertex
    ObjVertex baseVert{};
    baseVert.pos = glm::vec3(x, y, -height);

    // Cone side normal (pointing outward from cone surface)
    glm::vec3 toApex  = glm::normalize(apex.pos - baseVert.pos);
    glm::vec3 tangent = glm::vec3(-sin(theta), cos(theta), 0.0f);
    baseVert.nrm      = glm::normalize(glm::cross(tangent, toApex));

    vertices.push_back(baseVert);
  }

  // Generate indices for cone sides
  int apexIdx = 0;
  for(int i = 0; i < segments; ++i)
  {
    int baseIdx1 = 2 + i;
    int baseIdx2 = 2 + i + 1;

    indices.push_back(apexIdx);
    indices.push_back(baseIdx1);
    indices.push_back(baseIdx2);
  }

  // Generate indices for base circle (reversed winding since normal flipped)
  int baseCenterIdx = 1;
  for(int i = 0; i < segments; ++i)
  {
    int baseIdx1 = 2 + i;
    int baseIdx2 = 2 + i + 1;

    indices.push_back(baseCenterIdx);
    indices.push_back(baseIdx1);
    indices.push_back(baseIdx2);
  }

  // Create purely emissive material
  ObjMaterial material{};
  material.emission                  = glm::vec3(1.0f);  // White by default (will be updated)
  material.diffuse                   = glm::vec3(0.0f);
  material.ambient                   = glm::vec3(0.0f);
  material.specular                  = glm::vec3(0.0f);
  std::vector<ObjMaterial> materials = {material};
  std::vector<uint32_t>    matIndices(indices.size() / 3, 0);

  auto mesh = m_meshSetVk->createMesh("light_proxy_cone", vertices, indices, materials, matIndices);

  LOGD("Created proxy cone: %zu vertices, %zu indices\n", vertices.size(), indices.size());
  return mesh;
}

std::shared_ptr<MeshVk> LightManagerVk::createProxyQuad()
{
  if(!m_meshSetVk)
  {
    LOGE("createProxyQuad: MeshManagerVk not set\n");
    return nullptr;
  }

  // Create quad geometry: facing +Z axis, with a cone indicator
  const float size     = 0.1f;  // Small quad (0.2x0.2 total, 5x smaller than original)
  const float coneSize = 0.06f;

  std::vector<ObjVertex> vertices;
  std::vector<uint32_t>  indices;

  // Quad vertices (square facing +Z)
  ObjVertex v0, v1, v2, v3;
  v0.pos = glm::vec3(-size, -size, 0.0f);
  v1.pos = glm::vec3(size, -size, 0.0f);
  v2.pos = glm::vec3(size, size, 0.0f);
  v3.pos = glm::vec3(-size, size, 0.0f);
  v0.nrm = v1.nrm = v2.nrm = v3.nrm = glm::vec3(0.0f, 0.0f, 1.0f);

  vertices.push_back(v0);  // 0
  vertices.push_back(v1);  // 1
  vertices.push_back(v2);  // 2
  vertices.push_back(v3);  // 3

  // Cone indicator vertices (pointing in +Z direction)
  // Cone shaft
  ObjVertex c0, c1, c2, c3;
  c0.pos = glm::vec3(-coneSize * 0.1f, 0.0f, 0.0f);
  c1.pos = glm::vec3(coneSize * 0.1f, 0.0f, 0.0f);
  c2.pos = glm::vec3(coneSize * 0.1f, coneSize * 0.6f, 0.0f);
  c3.pos = glm::vec3(-coneSize * 0.1f, coneSize * 0.6f, 0.0f);
  c0.nrm = c1.nrm = c2.nrm = c3.nrm = glm::vec3(0.0f, 0.0f, 1.0f);

  vertices.push_back(c0);  // 4
  vertices.push_back(c1);  // 5
  vertices.push_back(c2);  // 6
  vertices.push_back(c3);  // 7

  // Cone tip
  ObjVertex t0, t1, t2;
  t0.pos = glm::vec3(-coneSize * 0.3f, coneSize * 0.6f, 0.0f);
  t1.pos = glm::vec3(coneSize * 0.3f, coneSize * 0.6f, 0.0f);
  t2.pos = glm::vec3(0.0f, coneSize, 0.0f);
  t0.nrm = t1.nrm = t2.nrm = glm::vec3(0.0f, 0.0f, 1.0f);

  vertices.push_back(t0);  // 8
  vertices.push_back(t1);  // 9
  vertices.push_back(t2);  // 10

  // Quad indices
  indices.push_back(0);
  indices.push_back(1);
  indices.push_back(2);
  indices.push_back(0);
  indices.push_back(2);
  indices.push_back(3);

  // Cone shaft indices
  indices.push_back(4);
  indices.push_back(5);
  indices.push_back(6);
  indices.push_back(4);
  indices.push_back(6);
  indices.push_back(7);

  // Cone tip indices
  indices.push_back(8);
  indices.push_back(9);
  indices.push_back(10);

  // Create purely emissive material
  ObjMaterial material{};
  material.emission                  = glm::vec3(1.0f);  // White by default (will be updated)
  material.diffuse                   = glm::vec3(0.0f);
  material.ambient                   = glm::vec3(0.0f);
  material.specular                  = glm::vec3(0.0f);
  std::vector<ObjMaterial> materials = {material};
  std::vector<uint32_t>    matIndices(indices.size() / 3, 0);

  auto mesh = m_meshSetVk->createMesh("light_proxy_quad", vertices, indices, materials, matIndices);

  LOGD("Created proxy quad: %zu vertices, %zu indices\n", vertices.size(), indices.size());
  return mesh;
}

void LightManagerVk::updateProxyTransform(std::shared_ptr<LightSourceInstanceVk> instance)
{
  if(!instance || !instance->proxyInstance || !instance->lightSource)
  {
    return;
  }

  // Update proxy instance transform to match light instance translation and rotation
  glm::vec3 lightPos   = instance->translation;                         // From instance
  glm::vec3 lightRot   = instance->rotation;                            // From instance
  glm::vec3 proxyScale = glm::vec3(instance->lightSource->proxyScale);  // From asset

  instance->proxyInstance->translation = lightPos;
  instance->proxyInstance->rotation    = lightRot;
  instance->proxyInstance->scale       = proxyScale;

  // Use utility function (now fixed to use quaternion-based rotation)
  computeTransform(proxyScale, lightRot, lightPos, instance->proxyInstance->transform,
                   instance->proxyInstance->transformInverse, instance->proxyInstance->transformRotScaleInverse);

  // Note: Proxy material color is updated in updateLightAsset(), not here

  // Mark for GPU update
  m_meshSetVk->updateInstanceTransform(instance->proxyInstance);
}

}  // namespace vk_gaussian_splatting
