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

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/debug_util.hpp>
#include <imgui/imgui.h>
#include <nvgui/property_editor.hpp>

#include "dlss_denoiser.hpp"

bool DlssDenoiser::ensureInitialized(const InitResources& resources)
{
  if(!m_initialized)
  {
    initDenoiser(resources);
    return true;
  }
  return false;
}

void DlssDenoiser::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  paramReg->add({"dlssEnable", "DLSS Denoiser: Enable DLSS denoiser"}, &m_settings.enable);
}

void DlssDenoiser::init(const InitResources& resources)
{

  resources.samplerPool->acquireSampler(m_linearSampler);
  // G-Buffer
  m_dlssGBuffers.init({.allocator      = resources.allocator,
                       .colorFormats   = m_bufferInfos,
                       .imageSampler   = m_linearSampler,
                       .descriptorPool = resources.descriptorPool});
}

void DlssDenoiser::deinit(const InitResources& resources)
{
  resources.samplerPool->releaseSampler(m_linearSampler);
  m_dlssGBuffers.deinit();
  m_dlss.deinit();
  m_ngx.deinit();
  m_initialized = false;
}

void DlssDenoiser::initDenoiser(const InitResources& resources)
{
  if(m_initialized)
    return;
  SCOPED_TIMER("Initializing DLSS Denoiser");

  m_device = resources.allocator->getDevice();

  // #DLSS - Create the DLSS
  NgxContext::InitInfo ngxInitInfo{
      .instance       = resources.instance,
      .physicalDevice = resources.allocator->getPhysicalDevice(),
      .device         = resources.allocator->getDevice(),
  };
  // ngxInitInfo.loggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;

  NVSDK_NGX_Result ngxResult = m_ngx.init(ngxInitInfo);
  if(ngxResult == NVSDK_NGX_Result_Success)
  {
    m_dlssSupported = (m_ngx.isDlssRRAvailable() == NVSDK_NGX_Result_Success);
  }

  if(!m_dlssSupported)
  {
    LOGW("NGX init failed: %d - DLSS unsupported\n", ngxResult);
  }
  m_initialized = true;
}


VkDescriptorImageInfo DlssDenoiser::getDescriptorImageInfo(shaderio::DlssImages name) const
{
  return m_dlssGBuffers.getDescriptorImageInfo((int)name);
}

bool DlssDenoiser::isEnabled() const
{
  return m_settings.enable;
}

VkExtent2D DlssDenoiser::updateSize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(!m_dlssSupported || !m_initialized)
    return size;

  // Query the supported sizes
  DlssRayReconstruction::SupportedSizes supportedSizes{};
  NVSDK_NGX_Result                      result =
      DlssRayReconstruction::querySupportedInputSizes(m_ngx, {size, NVSDK_NGX_PerfQuality_Value_MaxQuality}, &supportedSizes);
  if(NVSDK_NGX_FAILED(result))
  {
    m_renderingSize = size;
    LOGE("DLSS: Failed to query supported input sizes: %d\n", result);
    return m_renderingSize;  // Return the original size if query fails
  }

  // Choose the size based on the selected mode
  switch(m_settings.sizeMode)
  {
    case SizeMode::eMin:
      m_renderingSize = supportedSizes.minSize;
      break;
    case SizeMode::eMax:
      m_renderingSize = supportedSizes.maxSize;
      break;
    case SizeMode::eOptimal:
    default:
      m_renderingSize = supportedSizes.optimalSize;
      break;
  }

  // Update the last used size mode and clear the change flag
  m_sizeModeChanged = false;

  DlssRayReconstruction::InitInfo initInfo{
      .inputSize  = m_renderingSize,
      .outputSize = size,
  };
  m_dlss.deinit();
  vkDeviceWaitIdle(m_device);
  NVSDK_NGX_Result initResult = m_dlss.cmdInit(cmd, m_ngx, initInfo);

  // Check if DLSS initialization succeeded
  if(NVSDK_NGX_FAILED(initResult))
  {
    LOGE("DLSS: Failed to initialize with result: %d - Temporarily disabling DLSS\n", initResult);
    // Temporarily disable DLSS to prevent subsequent denoise failures
    m_settings.enable = false;
    return size;  // Return original size since DLSS failed
  }

  // Recreate the G-Buffers (always, even if not enabled - needed when user enables later)
  m_dlssGBuffers.update(cmd, m_renderingSize);

  return m_renderingSize;
}

void DlssDenoiser::setResources()
{
  if(!m_dlssSupported || !m_initialized)
    return;

  auto dlssResourceFromGBufTexture = [&](DlssRayReconstruction::ResourceType resource, shaderio::DlssImages gbufIndex) {
    m_dlss.setResource({resource, m_dlssGBuffers.getColorImage((uint32_t)gbufIndex),
                        m_dlssGBuffers.getColorImageView((uint32_t)gbufIndex), m_dlssGBuffers.getColorFormat((uint32_t)gbufIndex)});
  };

  // #DLSS Fill the user pool with our textures
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eColorIn, shaderio::DlssImages::eDlssInputImage);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eDiffuseAlbedo, shaderio::DlssImages::eDlssAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eSpecularAlbedo, shaderio::DlssImages::eDlssSpecAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eNormalRoughness, shaderio::DlssImages::eDlssNormalRoughness);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eMotionVector, shaderio::DlssImages::eDlssMotion);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eDepth, shaderio::DlssImages::eDlssDepth);
}

void DlssDenoiser::setResource(DlssRayReconstruction::ResourceType resourceId, VkImage image, VkImageView imageView, VkFormat format)
{
  m_dlss.setResource({resourceId, image, imageView, format});
}

void DlssDenoiser::denoise(VkCommandBuffer cmd, glm::vec2 jitter, const glm::mat4& modelView, const glm::mat4& projection, bool reset /*= false*/)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  // Safety check: don't attempt denoise if DLSS is not properly initialized or enabled
  if(!m_dlssSupported || !m_initialized || !m_settings.enable)
  {
    return;
  }

  reset                          = reset || m_forceReset;
  NVSDK_NGX_Result denoiseResult = m_dlss.cmdDenoise(cmd, m_ngx, {jitter, modelView, projection, reset});

  // Check for denoise errors and temporarily disable DLSS if it fails
  if(NVSDK_NGX_FAILED(denoiseResult))
  {
    LOGE("DLSS: Denoise failed with result: %d - Temporarily disabling DLSS\n", denoiseResult);
    // Temporarily disable to prevent error spam, but allow re-enable
    m_settings.enable = false;
    return;
  }

  m_forceReset = false;
}

bool DlssDenoiser::onUi()
{
  namespace PE = nvgui::PropertyEditor;

  bool changed = false;

  if(!m_dlssSupported && m_initialized)
  {
    PE::Text("DLSS is not available", "");
    return changed;
  }

  if(PE::Checkbox("Enable DLSS", &m_settings.enable))
  {
    m_forceReset = true;  // Force a reset when enabling/disabling DLSS
    changed      = true;
  }
  if(!m_initialized)
    return changed;

  if(!m_settings.enable)
    return changed;

  // Size mode selection
  const char* sizeModes[]     = {"Min", "Optimal", "Max"};
  int         currentSizeMode = static_cast<int>(m_settings.sizeMode);

  if(PE::Combo("DLSS Size Mode", &currentSizeMode, sizeModes, IM_ARRAYSIZE(sizeModes)))
  {
    m_settings.sizeMode = static_cast<SizeMode>(currentSizeMode);
    m_sizeModeChanged   = true;  // Mark that size mode has changed
    changed             = true;  // Mark that changes were made
  }

  PE::Text("Current Resolution", "%d x %d", m_renderingSize.width, m_renderingSize.height);

  return changed;
}
