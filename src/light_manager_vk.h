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

#include <string>
#include <vector>
#include <memory>
#include <fmt/format.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <nvapp/application.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/staging.hpp>

#include "shaderio.h"
#include "obj_loader.h"

namespace vk_gaussian_splatting {
class MeshManagerVk;    // Forward declaration
struct MeshVk;          // Forward declaration
struct MeshInstanceVk;  // Forward declaration

// ============================================================================
// Light Asset/Instance Architecture (matches Mesh and SplatSet patterns)
// ============================================================================

// LightSourceVk: Light asset (shared properties across instances)
struct LightSourceVk
{
  // Shared properties (affect all instances using this asset)
  shaderio::LightType type            = shaderio::LightType::ePointLight;  // Light type
  glm::vec3           color           = glm::vec3(1.0f);
  float               intensity       = 100.0f;
  float               range           = 10.0f;  // Effective range for point/spot lights
  float               innerConeAngle  = 30.0f;  // Spot light: inner cone (degrees)
  float               outerConeAngle  = 45.0f;  // Spot light: outer cone (degrees)
  int                 attenuationMode = 2;      // 0=None, 1=Linear, 2=Quadratic, 3=Physical
  float               proxyScale      = 1.0f;   // Visualization scale (independent of range)

  // C++ management data (NOT uploaded to GPU)
  std::shared_ptr<MeshVk> proxyMesh;      // Visualization mesh (sphere/cone/quad)
  ObjMaterial             proxyMaterial;  // Emissive material (color = light color)
};

// LightSourceInstanceVk: Light instance (light in the scene)
struct LightSourceInstanceVk
{
  std::shared_ptr<LightSourceVk> lightSource;  // Reference to light asset (SHARED!)

  // Per-instance data
  std::string name;   // "Light 0", "Light 1", etc.
  uint64_t    index;  // Index in instances array

  // Transform parameters (consistent with mesh/splat instances)
  glm::vec3 translation = glm::vec3(0.0f, 2.0f, 0.0f);  // World position
  glm::vec3 rotation    = glm::vec3(0.0f);              // XYZ rotations (degrees)

  // Proxy visualization
  std::shared_ptr<MeshInstanceVk> proxyInstance;  // Proxy mesh instance for visualization
};

// ============================================================================
// LightManagerVk: Manages light assets and instances
// ============================================================================

class LightManagerVk
{
public:
  // Deferred update requests (bitfield)
  enum class Request : uint32_t
  {
    eNone          = 0,
    eUpdateBuffer  = 1 << 0,  // Upload lights buffer to VRAM (data changed)
    eRebuildBuffer = 1 << 1,  // Reallocate buffer (size changed: add/delete)
  };

  // ===== PUBLIC API =====

  // Lifecycle
  void init(nvapp::Application* app, nvvk::ResourceAllocator* alloc, nvvk::StagingUploader* uploader);
  void deinit();
  void reset();

  // Asset/Instance management
  std::shared_ptr<LightSourceInstanceVk> createLight();
  std::shared_ptr<LightSourceInstanceVk> duplicateInstance(std::shared_ptr<LightSourceInstanceVk> sourceInstance);
  void                                   deleteInstance(std::shared_ptr<LightSourceInstanceVk> instance);

  // Update/access
  void   updateLight(std::shared_ptr<LightSourceInstanceVk> instance);
  void   updateLightAsset(std::shared_ptr<LightSourceVk> asset);       // Update asset (affects all instances)
  void   recreateProxyForAsset(std::shared_ptr<LightSourceVk> asset);  // Recreate proxy mesh when type changes
  size_t size() const { return instances.size(); }
  const std::string&                     getLightName(uint64_t index) const;
  std::shared_ptr<LightSourceInstanceVk> getInstance(uint64_t index) { return instances[index]; }

  // Deferred updates
  bool processVramUpdates();

  // Cross-manager reference
  void setMeshSet(MeshManagerVk* meshSet) { m_meshSetVk = meshSet; }

  // ===== PUBLIC DATA (for serialization and external access) =====

  // Asset naming counter (reset on reset())
  uint64_t m_nextLightNumber = 0;

  // Light instances
  std::vector<std::shared_ptr<LightSourceInstanceVk>> instances;

  // GPU buffer (for asset buffer pointer access)
  nvvk::Buffer lightsBuffer;

  // Pending update requests (public for external checks)
  Request pendingRequests = Request::eNone;

public:
  // ===== PUBLIC DATA (for loading) =====

  // Light assets (unique light definitions) - public for project loading
  std::vector<std::shared_ptr<LightSourceVk>> lightSources;

private:
  // ===== PRIVATE DATA =====

  // Vulkan context
  nvapp::Application*      m_app       = nullptr;
  nvvk::ResourceAllocator* m_alloc     = nullptr;
  nvvk::StagingUploader*   m_uploader  = nullptr;
  MeshManagerVk*           m_meshSetVk = nullptr;

  // ===== PRIVATE METHODS =====

  // Buffer management
  void rebuildBuffer();
  void updateBuffer();

  // Proxy management (called internally and from project loading)
  std::shared_ptr<MeshVk> createProxySphere();                 // Creates a NEW sphere mesh (not shared!)
  std::shared_ptr<MeshVk> createProxyCone(int segments = 16);  // Creates a NEW cone mesh for spot lights
  std::shared_ptr<MeshVk> createProxyQuad();                   // Creates a NEW quad mesh for directional lights
  void                    updateProxyTransform(std::shared_ptr<LightSourceInstanceVk> instance);
};

// ============================================================================
// Bitwise operators for LightManagerVk::Request
// ============================================================================

inline LightManagerVk::Request operator|(LightManagerVk::Request a, LightManagerVk::Request b)
{
  return static_cast<LightManagerVk::Request>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline LightManagerVk::Request operator&(LightManagerVk::Request a, LightManagerVk::Request b)
{
  return static_cast<LightManagerVk::Request>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline LightManagerVk::Request operator~(LightManagerVk::Request a)
{
  return static_cast<LightManagerVk::Request>(~static_cast<uint32_t>(a));
}

inline LightManagerVk::Request& operator|=(LightManagerVk::Request& a, LightManagerVk::Request b)
{
  a = a | b;
  return a;
}

inline LightManagerVk::Request& operator&=(LightManagerVk::Request& a, LightManagerVk::Request b)
{
  a = a & b;
  return a;
}

}  // namespace vk_gaussian_splatting
