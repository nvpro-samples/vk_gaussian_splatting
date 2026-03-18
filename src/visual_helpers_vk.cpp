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

#include "visual_helpers_vk.h"
#include "memory_statistics.h"
#include "camera_set.h"
#include "utilities.h"

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <iostream>

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// Lifecycle
//--------------------------------------------------------------------------------------------------

void VisualHelpers::init(const Resources& res)
{
  m_app     = res.app;
  m_alloc   = res.alloc;
  m_device  = res.device;
  m_sampler = res.sampler;

  // Initialize TransformHelper (3D gizmo system)
  TransformHelperVk::Resources transformHelperRes{
      .app           = res.app,
      .alloc         = res.alloc,
      .uploader      = res.uploader,
      .device        = res.device,
      .sampler       = res.sampler,
      .slangCompiler = res.slangCompiler,
      .colorFormat   = res.colorFormat,
      .depthFormat   = res.depthFormat,
  };
  transform.init(transformHelperRes);

  // Initialize grid helper (uses same descriptor set layout as transform helper for scene depth)
  GridHelperVk::Resources gridHelperRes = {
      .app                       = res.app,
      .alloc                     = res.alloc,
      .uploader                  = res.uploader,
      .device                    = res.device,
      .slangCompiler             = res.slangCompiler,
      .colorFormat               = res.colorFormat,
      .depthFormat               = res.depthFormat,
      .helperDescriptorSetLayout = transform.getDescriptorSetLayout(),
  };
  grid.init(gridHelperRes);

  // Store formats for later reinitialization
  m_colorFormat = res.colorFormat;
  m_depthFormat = res.depthFormat;

  // Initialize helper GBuffer: 1 color + depth for helper overlay rendering
  // Always at viewport resolution, prevents helpers from being accumulated in temporal
  m_helperGBuffers.init({
      .allocator      = res.alloc,
      .colorFormats   = {m_colorFormat},  // Single color buffer for scene + helpers
      .depthFormat    = m_depthFormat,
      .imageSampler   = res.sampler,
      .descriptorPool = res.app->getTextureDescriptorPool(),
  });

  std::cout << "VisualHelpers initialized" << std::endl;
}

void VisualHelpers::deinit()
{
  // Cleanup helper descriptor set
  deinitDescriptorSet();

  // Explicitly reset handles (safety: ensure no stale handles remain)
  m_helperDescriptorPool = VK_NULL_HANDLE;
  m_helperDescriptorSet  = VK_NULL_HANDLE;

  // Cleanup helper GBuffers
  m_helperGBuffers.deinit();

  // Cleanup transform and grid helpers
  transform.deinit();
  grid.deinit();

  // Clear pointers
  m_app     = nullptr;
  m_alloc   = nullptr;
  m_device  = VK_NULL_HANDLE;
  m_sampler = VK_NULL_HANDLE;

  std::cout << "VisualHelpers deinitialized" << std::endl;
}

void VisualHelpers::onResize(VkCommandBuffer cmd, const VkExtent2D& size, VkImage sceneDepth, VkImageView sceneDepthView, VkSampler sampler)
{
  m_sampler = sampler;  // Store sampler for descriptor set creation

  // Update helper GBuffer to match viewport size
  NVVK_CHECK(m_helperGBuffers.update(cmd, size));

  // Track helper GBuffer memory (updated on resize)
  uint32_t colorFormatSize      = getColorFormatBytesPerPixel(m_colorFormat);
  uint32_t depthFormatSize      = 4;  // D32_SFLOAT = 4 bytes per pixel
  memRender.helperGBuffersColor = uint64_t(size.width) * size.height * colorFormatSize;  // 1 color attachment
  memRender.helperGBuffersDepth = uint64_t(size.width) * size.height * depthFormatSize;

  // Recreate helper descriptor set (depends on scene depth texture)
  deinitDescriptorSet();
  initDescriptorSet(sceneDepth, sceneDepthView);
}

//--------------------------------------------------------------------------------------------------
// Rendering
//--------------------------------------------------------------------------------------------------

bool VisualHelpers::shouldRender() const
{
  // Check if any helper needs rendering
  // - Grid can show anytime (independent of editing mode)
  // - Gizmo requires editing mode AND attached entity
  bool gridVisible  = grid.isVisible();
  bool gizmoVisible = m_editingMode && transform.isAttached();

  return gridVisible || gizmoVisible;
}

void VisualHelpers::render(VkCommandBuffer  cmd,
                           VkImage          mainColorImage,
                           VkDescriptorSet  sceneDescriptorSet,
                           const glm::mat4& viewMatrix,
                           const glm::mat4& projMatrix,
                           const glm::vec2& viewportSize,
                           const glm::vec2& depthBufferSize)
{
  // Safety check: ensure descriptor set is valid before rendering
  if(m_helperDescriptorSet == VK_NULL_HANDLE)
    return;

  NVVK_DBG_SCOPE(cmd);

  const VkExtent2D viewportExtent = {static_cast<uint32_t>(viewportSize.x), static_cast<uint32_t>(viewportSize.y)};

  // Step 1: Copy main color buffer to helper color buffer
  // This keeps main buffer clean for temporal accumulation (helpers won't be accumulated)
  {
    // Transition source for copy
    nvvk::cmdImageMemoryBarrier(cmd, {mainColorImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

    // Transition destination for copy
    nvvk::cmdImageMemoryBarrier(cmd, {m_helperGBuffers.getColorImage(0), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.extent         = {viewportExtent.width, viewportExtent.height, 1};

    vkCmdCopyImage(cmd, mainColorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_helperGBuffers.getColorImage(0),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // Transition source back to general
    nvvk::cmdImageMemoryBarrier(cmd, {mainColorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});

    // Transition destination for rendering
    nvvk::cmdImageMemoryBarrier(cmd, {m_helperGBuffers.getColorImage(0), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
  }

  // Step 2: Render helpers to helper GBuffer
  // Color attachment - use helper color buffer (contains copied scene)
  VkRenderingAttachmentInfo colorAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  colorAttachment.imageView                 = m_helperGBuffers.getColorImageView(0);
  colorAttachment.imageLayout               = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_LOAD;  // Keep copied scene
  colorAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.clearValue.color          = {{0.0f, 0.0f, 0.0f, 0.0f}};

  // Depth attachment - use helper depth buffer (starts clear)
  VkRenderingAttachmentInfo depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAttachment.imageView                 = m_helperGBuffers.getDepthImageView();
  depthAttachment.imageLayout               = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.clearValue.depthStencil   = {1.0f, 0};

  // Begin dynamic rendering
  VkRenderingInfo renderingInfo      = {VK_STRUCTURE_TYPE_RENDERING_INFO};
  renderingInfo.renderArea           = {{0, 0}, viewportExtent};
  renderingInfo.layerCount           = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments    = &colorAttachment;
  renderingInfo.pDepthAttachment     = &depthAttachment;

  vkCmdBeginRendering(cmd, &renderingInfo);

  // Set viewport and scissor (using WithCount variants for dynamic state compatibility)
  VkViewport viewport{0.0f, 0.0f, viewportSize.x, viewportSize.y, 0.0f, 1.0f};
  vkCmdSetViewportWithCount(cmd, 1, &viewport);

  VkRect2D scissor{{0, 0}, viewportExtent};
  vkCmdSetScissorWithCount(cmd, 1, &scissor);

  // Render grid first (behind gizmo)
  grid.renderRaster(cmd, m_helperDescriptorSet, viewMatrix, projMatrix, viewportSize, depthBufferSize);

  // Render transform gizmo on top (only when editing mode is active)
  if(m_editingMode)
  {
    transform.renderRaster(cmd, sceneDescriptorSet, m_helperDescriptorSet, viewMatrix, projMatrix, viewportSize, depthBufferSize);
  }

  vkCmdEndRendering(cmd);

  // Transition helper color buffer back to general for display/capture
  nvvk::cmdImageMemoryBarrier(cmd, {m_helperGBuffers.getColorImage(0), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});

  // Mark that helpers were rendered this frame
  m_helpersRenderedThisFrame = true;
}

//--------------------------------------------------------------------------------------------------
// Output Routing
//--------------------------------------------------------------------------------------------------

VkImage VisualHelpers::getOutputColorImage() const
{
  return m_helpersRenderedThisFrame ? m_helperGBuffers.getColorImage(0) : VK_NULL_HANDLE;
}

VkImageView VisualHelpers::getOutputColorImageView() const
{
  return m_helpersRenderedThisFrame ? m_helperGBuffers.getColorImageView(0) : VK_NULL_HANDLE;
}

VkDescriptorSet VisualHelpers::getOutputDescriptorSet() const
{
  return m_helpersRenderedThisFrame ? m_helperGBuffers.getDescriptorSet(0) : VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Internal Methods
//--------------------------------------------------------------------------------------------------

void VisualHelpers::initDescriptorSet(VkImage sceneDepth, VkImageView sceneDepthView)
{
  // Create descriptor pool for helper pass (scene depth sampling)
  VkDescriptorSetLayout             helperLayout = transform.getDescriptorSetLayout();
  std::vector<VkDescriptorPoolSize> poolSizes    = {
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},  // Scene depth texture
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1},        // Sampler
  };

  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets       = 1;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes    = poolSizes.data();

  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_helperDescriptorPool));
  NVVK_DBG_NAME(m_helperDescriptorPool);

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool     = m_helperDescriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts        = &helperLayout;

  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_helperDescriptorSet));
  NVVK_DBG_NAME(m_helperDescriptorSet);

  // Update descriptor set with scene depth texture
  VkDescriptorImageInfo depthImageInfo = {};
  depthImageInfo.imageView             = sceneDepthView;
  depthImageInfo.imageLayout           = VK_IMAGE_LAYOUT_GENERAL;  // Depth buffer stays in GENERAL for sampling

  VkDescriptorImageInfo samplerInfo = {};
  samplerInfo.sampler               = m_sampler;

  std::array<VkWriteDescriptorSet, 2> writes{};

  // Binding 0: Scene depth texture
  writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet          = m_helperDescriptorSet;
  writes[0].dstBinding      = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  writes[0].pImageInfo      = &depthImageInfo;

  // Binding 1: Sampler
  writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet          = m_helperDescriptorSet;
  writes[1].dstBinding      = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
  writes[1].pImageInfo      = &samplerInfo;

  vkUpdateDescriptorSets(m_device, 2, writes.data(), 0, nullptr);
}

void VisualHelpers::deinitDescriptorSet()
{
  // Descriptor set is freed with pool, no need to free explicitly
  if(m_helperDescriptorPool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(m_device, m_helperDescriptorPool, nullptr);
    m_helperDescriptorPool = VK_NULL_HANDLE;
    m_helperDescriptorSet  = VK_NULL_HANDLE;
  }
}

}  // namespace vk_gaussian_splatting
