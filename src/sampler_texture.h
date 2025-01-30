/*
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _SAMPLER_TEXTURE_H_
#define _SAMPLER_TEXTURE_H_

//
#include <vulkan/vulkan_core.h>

//
#include "nvvk/commands_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"

// Utility class to manage texture creation and destruction
// TODO: remove this class and just add methods to main class. Will be nought for our needs.
struct SampleTexture
{
private:
  VkDevice          m_device{};
  uint32_t          m_queueIndex{0};
  VkExtent2D        m_size{0, 0};
  nvvk::Texture     m_texture;
  nvvkhl::AllocVma* m_alloc{nullptr};

public:
  SampleTexture(VkDevice device, uint32_t queueIndex, nvvkhl::AllocVma* a)
      : m_device(device)
      , m_queueIndex(queueIndex)
      , m_alloc(a)
  {
  }

  ~SampleTexture() { destroy(); }

  // Create the image, the sampler and the image view
  void create(uint32_t width, uint32_t height, uint32_t bufsize, void* data, VkFormat format)
  {
    const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_size                              = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    const VkImageCreateInfo create_info = nvvk::makeImage2DCreateInfo(m_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, false);

    nvvk::CommandPool cpool(m_device, m_queueIndex);
    VkCommandBuffer   cmd = cpool.createCommandBuffer();
    m_texture             = m_alloc->createTexture(cmd, bufsize, data, create_info, sampler_info);

    cpool.submitAndWait(cmd);
  }

  void destroy()
  {
    // Destroying in next frame, avoid deleting while using
    nvvkhl::Application::submitResourceFree(
        [tex = m_texture, a = m_alloc]() { a->destroy(const_cast<nvvk::Texture&>(tex)); });
  }

  void               setSampler(const VkSampler& sampler) { m_texture.descriptor.sampler = sampler; }
  [[nodiscard]] bool isValid() const { return m_texture.image != nullptr; }
  [[nodiscard]] const VkDescriptorImageInfo& descriptor() const { return m_texture.descriptor; }
  [[nodiscard]] const VkExtent2D&            getSize() const { return m_size; }
  [[nodiscard]] float getAspect() const { return static_cast<float>(m_size.width) / static_cast<float>(m_size.height); }
};


#endif
