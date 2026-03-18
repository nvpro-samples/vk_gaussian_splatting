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

#ifndef VISUAL_HELPERS_H
#define VISUAL_HELPERS_H

#include <nvvk/gbuffers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/barriers.hpp>
#include <nvapp/application.hpp>
#include <nvslang/slang.hpp>

#include "transform_helper_vk.h"
#include "grid_helper_vk.h"

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// VisualHelpers: Manages 3D transforml and grid visualization helpers
//
// Consolidates all helper rendering logic (transform , infinite grid) into a single class.
// Provides clean public access to individual helpers via nested members:
//   - m_helpers.grid.<method>()
//   - m_helpers.transform.<method>()
//--------------------------------------------------------------------------------------------------
class VisualHelpers
{
public:
  // Public helper objects for direct UI access
  GridHelperVk      grid;
  TransformHelperVk transform;

  // Initialization resources
  struct Resources
  {
    nvapp::Application*      app;
    nvvk::ResourceAllocator* alloc;
    nvvk::StagingUploader*   uploader;
    VkDevice                 device;
    VkSampler                sampler;
    nvslang::SlangCompiler*  slangCompiler;
    VkFormat                 colorFormat;
    VkFormat                 depthFormat;
  };

  // Lifecycle
  void init(const Resources& res);
  void deinit();
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, VkImage sceneDepth, VkImageView sceneDepthView, VkSampler sampler);

  // Rendering control
  void setEditingMode(bool enabled) { m_editingMode = enabled; }
  bool isEditingMode() const { return m_editingMode; }

  bool shouldRender() const;  // Checks if any helpers need rendering (camera check + visibility)
  void render(VkCommandBuffer  cmd,
              VkImage          mainColorImage,      // Scene color to copy from
              VkDescriptorSet  sceneDescriptorSet,  // Main scene descriptor set (for transform helper)
              const glm::mat4& viewMatrix,
              const glm::mat4& projMatrix,
              const glm::vec2& viewportSize,
              const glm::vec2& depthBufferSize);  // For DLSS depth size adjustment

  // Frame management
  void resetFrameState() { m_helpersRenderedThisFrame = false; }

  // Output routing (returns helper buffer if rendered, else returns nullptr to signal "use main buffer")
  VkImage         getOutputColorImage() const;
  VkImageView     getOutputColorImageView() const;
  VkDescriptorSet getOutputDescriptorSet() const;
  bool            wasRenderedThisFrame() const { return m_helpersRenderedThisFrame; }

private:
  // Resources
  nvapp::Application*      m_app         = nullptr;
  nvvk::ResourceAllocator* m_alloc       = nullptr;
  VkDevice                 m_device      = VK_NULL_HANDLE;
  VkSampler                m_sampler     = VK_NULL_HANDLE;
  VkFormat                 m_colorFormat = VK_FORMAT_UNDEFINED;
  VkFormat                 m_depthFormat = VK_FORMAT_UNDEFINED;

  // Helper rendering infrastructure
  nvvk::GBuffer    m_helperGBuffers;                         // 1 color + 1 depth for compositing
  VkDescriptorPool m_helperDescriptorPool = VK_NULL_HANDLE;  // Pool for helper descriptors
  VkDescriptorSet  m_helperDescriptorSet  = VK_NULL_HANDLE;  // Scene depth texture + sampler

  // State
  bool m_editingMode              = true;   // Enable/disable gizmo rendering
  bool m_helpersRenderedThisFrame = false;  // Track if rendered this frame

  // Internal methods
  void initDescriptorSet(VkImage sceneDepth, VkImageView sceneDepthView);
  void deinitDescriptorSet();
};

}  // namespace vk_gaussian_splatting

#endif  // VISUAL_HELPERS_H
