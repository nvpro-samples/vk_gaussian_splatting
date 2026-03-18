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

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "nvvk/resource_allocator.hpp"
#include "nvvk/staging.hpp"
#include "nvapp/application.hpp"
#include <nvslang/slang.hpp>
#include "visual_helpers_shaderio.h.slang"

namespace vk_gaussian_splatting {

//-----------------------------------------------------------------------------
// GridHelperVk - "Infinite" grid visualization on X/Z plane
//-----------------------------------------------------------------------------
// Features:
// - Adaptive LOD grid lines (1/10/100/1000... unit spacing)
// - Screen-space constant line thickness
// - Colored axis indicators (X=red, Y=green, Z=blue)
// - Checkerboard occlusion when behind scene geometry
//-----------------------------------------------------------------------------

class GridHelperVk
{
public:
  GridHelperVk() = default;
  ~GridHelperVk() { deinit(); }

  // Non-copyable
  GridHelperVk(const GridHelperVk&)            = delete;
  GridHelperVk& operator=(const GridHelperVk&) = delete;

  //-----------------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------------

  struct Resources
  {
    nvapp::Application*      app                       = nullptr;
    nvvk::ResourceAllocator* alloc                     = nullptr;
    nvvk::StagingUploader*   uploader                  = nullptr;
    VkDevice                 device                    = VK_NULL_HANDLE;
    nvslang::SlangCompiler*  slangCompiler             = nullptr;
    VkFormat                 colorFormat               = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat                 depthFormat               = VK_FORMAT_D32_SFLOAT;
    VkDescriptorSetLayout    helperDescriptorSetLayout = VK_NULL_HANDLE;  // Shared layout for scene depth
  };

  void init(const Resources& res);
  void deinit();

  //-----------------------------------------------------------------------------
  // Visibility
  //-----------------------------------------------------------------------------

  void setVisible(bool visible) { m_visible = visible; }
  bool isVisible() const { return m_visible; }
  void toggleVisible() { m_visible = !m_visible; }

  //-----------------------------------------------------------------------------
  // Configuration
  //-----------------------------------------------------------------------------

  // Line thickness in pixels (screen-space)
  void  setLineThickness(float pixels) { m_lineThickness = pixels; }
  float getLineThickness() const { return m_lineThickness; }

  //-----------------------------------------------------------------------------
  // Rendering
  //-----------------------------------------------------------------------------

  // Render the grid (called from helper compositing pass)
  // helperDescriptorSet: Contains scene depth texture for occlusion
  void renderRaster(VkCommandBuffer  cmd,
                    VkDescriptorSet  helperDescriptorSet,
                    const glm::mat4& viewMatrix,
                    const glm::mat4& projMatrix,
                    const glm::vec2& viewportSize,
                    const glm::vec2& depthBufferSize);  // DLSS render size or viewport size

private:
  //-----------------------------------------------------------------------------
  // Internal Types
  //-----------------------------------------------------------------------------

  struct GridVertex
  {
    glm::vec3 position;  // Vertex position
    glm::vec3 color;     // Line color (RGB)
  };

  // Note: Using unified push constants from visual_helpers_shaderio.h.slang
  using PushConstants = shaderio::visual_helpers::PushConstantVisualHelpers;

  // Scale level info for LOD blending (3 levels for smooth transitions)
  struct ScaleLevelInfo
  {
    float baseScale;    // The current base scale level
    float fineScale;    // The finer scale level (baseScale / 10)
    float coarseScale;  // The coarser scale level (baseScale * 10)
    float fineBlend;    // 0 = fine invisible, 1 = fine fully visible
    float coarseBlend;  // 0 = coarse invisible, 1 = coarse fully visible
  };

  //-----------------------------------------------------------------------------
  // Grid Generation
  //-----------------------------------------------------------------------------

  // Calculate scale levels and blend factor for smooth LOD transitions
  ScaleLevelInfo calculateScaleLevels(const glm::vec3& cameraPos);

  // Generate grid geometry once (at scale=1)
  void generateGridGeometry();

  // Upload geometry to GPU
  void uploadGeometry();

  //-----------------------------------------------------------------------------
  // Pipeline
  //-----------------------------------------------------------------------------

  void createPipeline();
  bool compileSlangShader(const char* filename, VkShaderModule& outModule);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------

  // Vulkan resources (not owned)
  nvapp::Application*      m_app           = nullptr;
  nvvk::ResourceAllocator* m_alloc         = nullptr;
  nvvk::StagingUploader*   m_uploader      = nullptr;
  VkDevice                 m_device        = VK_NULL_HANDLE;
  nvslang::SlangCompiler*  m_slangCompiler = nullptr;
  VkFormat                 m_colorFormat   = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat                 m_depthFormat   = VK_FORMAT_D32_SFLOAT;

  // Pipeline resources
  VkPipeline            m_pipeline                  = VK_NULL_HANDLE;
  VkPipelineLayout      m_pipelineLayout            = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_helperDescriptorSetLayout = VK_NULL_HANDLE;
  VkShaderModule        m_vertexShader              = VK_NULL_HANDLE;
  VkShaderModule        m_fragmentShader            = VK_NULL_HANDLE;

  // Geometry buffers
  nvvk::Buffer m_vertexBuffer;
  nvvk::Buffer m_indexBuffer;
  uint32_t     m_indexCount     = 0;
  uint32_t     m_vertexCapacity = 0;
  uint32_t     m_indexCapacity  = 0;

  // CPU-side geometry (rebuilt each frame)
  std::vector<GridVertex> m_vertices;
  std::vector<uint32_t>   m_indices;

  // State
  bool  m_visible           = true;  // Visible by default
  bool  m_initialized       = false;
  bool  m_geometryGenerated = false;  // Grid geometry created once
  float m_lineThickness     = 3.0f;   // Pixels (screen-space thickness) - unused now
  float m_minCellPixels     = 8.0f;   // Minimum pixels per cell before LOD switch - unused now

  // Cached values
  float m_lastGridScale = 1.0f;
};

}  // namespace vk_gaussian_splatting
