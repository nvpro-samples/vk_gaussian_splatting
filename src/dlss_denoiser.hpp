/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

// This class adds support for DLSS denoising
// It initialize the NGX and create the G-Buffers for the denoiser
// It also provides the descriptor set for the denoiser

#include <glm/glm.hpp>

#include "dlss_util.h"

#include "nvvk/gbuffers.hpp"
#include "nvvk/resource_allocator.hpp"
#include "nvvk/sampler_pool.hpp"
#include "nvutils/parameter_registry.hpp"

// #DLSS
#if defined(USE_DLSS)
#include "dlss_wrapper.hpp"
#include "nvsdk_ngx_helpers_vk.h"
#endif

#include "shaderio.h"

class DlssDenoiser
{
public:
  enum class SizeMode
  {
    eMin,
    eOptimal,
    eMax
  };

  struct Settings
  {
    bool     enable   = false;
    SizeMode sizeMode = SizeMode::eOptimal;
  };

  struct InitResources
  {
    VkInstance               instance;
    nvvk::ResourceAllocator* allocator;
    nvvk::SamplerPool*       samplerPool;
    VkDescriptorPool         descriptorPool;
  };


  DlssDenoiser()  = default;
  ~DlssDenoiser() = default;

  void init(const InitResources& resources);
  void deinit(const InitResources& resources);

  void initDenoiser(const InitResources& resources);
  // Return the descriptor for the DLSS
  VkDescriptorImageInfo getDescriptorImageInfo(shaderio::DlssImages name) const;

  // Return if the DLSS is enabled
  bool isEnabled() const;

  // Set DLSS enabled state
  void setEnabled(bool enabled)
  {
    m_settings.enable = enabled;
    if(enabled)
      m_forceReset = true;
  }

  // Get/Set DLSS size mode
  SizeMode getSizeMode() const { return m_settings.sizeMode; }
  void     setSizeMode(SizeMode mode)
  {
    m_settings.sizeMode = mode;
    m_sizeModeChanged   = true;
  }

  // When the size of the rendering changes, we need to update the DLSS buffers
  VkExtent2D updateSize(VkCommandBuffer cmd, VkExtent2D size);

  // This is setting the guide resources for the DLSS
  void setResources();

  // To be called for the Input and Output image
  void setResource(DlssRayReconstruction::ResourceType resourceId, VkImage image, VkImageView imageView, VkFormat format);

  // This is the actual denoising call
  void denoise(VkCommandBuffer cmd, glm::vec2 jitter, const glm::mat4& modelView, const glm::mat4& projection, bool reset = false);

  // This is for the UI
  bool onUi();

  // Return the render size
  VkExtent2D getRenderSize() const { return m_dlssGBuffers.getSize(); }

  // Check if the rendering size needs to be updated based on current settings
  bool needsSizeUpdate() const { return m_sizeModeChanged; }

  const nvvk::GBuffer& getGBuffers() const { return m_dlssGBuffers; }

  bool ensureInitialized(const InitResources& resources);

  void registerParameters(nvutils::ParameterRegistry* paramReg);

private:
  Settings m_settings{};
  bool     m_initialized = false;

  // #DLSS - Wrapper for DLSS
  NgxContext            m_ngx{};
  DlssRayReconstruction m_dlss{};

  // dereferenced by enum DlssImages, must be aligned
  std::vector<VkFormat> m_bufferInfos = {
      {VK_FORMAT_R32G32B32A32_SFLOAT},  // DLSS - Rendered image       : eDlssInputImage
      {VK_FORMAT_R8G8B8A8_UNORM},       // DLSS - BaseColor            : eDlssAlbedo
      {VK_FORMAT_R16G16B16A16_SFLOAT},  // DLSS - SpecAlbedo           : eDlssSpecAlbedo
      {VK_FORMAT_R16G16B16A16_SFLOAT},  // DLSS - Normal / Roughness   : eDlssNormalRoughness
      {VK_FORMAT_R16G16_SFLOAT},        // DLSS - Motion vectors       : eDlssMotion
      {VK_FORMAT_R16_SFLOAT},           // DLSS - ViewZ                : eDlssDepth
      {VK_FORMAT_R8_UNORM},             // DLSS - Object ID            : eSelectImage (optional)
  };

  nvvk::GBuffer m_dlssGBuffers{};  // G-Buffers: for denoising
  bool          m_dlssSupported = false;
  VkExtent2D    m_renderingSize{};
  VkDevice      m_device{};
  VkSampler     m_linearSampler{};
  bool          m_sizeModeChanged = false;  // Track if size mode has changed
  bool          m_forceReset      = false;  // Force reset of the denoiser
};
