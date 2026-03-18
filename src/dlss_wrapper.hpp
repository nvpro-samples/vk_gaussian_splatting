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

#include <vulkan/vulkan_core.h>

#include "nvvk/resource_allocator.hpp"

#include "nvsdk_ngx_vk.h"

#include <glm/glm.hpp>

#include <vector>
#include <string>


class NgxContext
{
public:
  struct ApplicationInfo
  {
    std::string engineVersion   = "0.0";
    std::string projectId       = "nvpro-samples";
    std::string applicationPath = "";
    // WAR: custom type prevents creating the NGX API context, using Omniverse instead
    NVSDK_NGX_EngineType engineType = NVSDK_NGX_ENGINE_TYPE_OMNIVERSE;
  };

  struct InitInfo
  {
    VkInstance              instance       = VK_NULL_HANDLE;
    VkPhysicalDevice        physicalDevice = VK_NULL_HANDLE;
    VkDevice                device         = VK_NULL_HANDLE;
    NVSDK_NGX_Logging_Level loggingLevel   = NVSDK_NGX_LOGGING_LEVEL_OFF;
    ApplicationInfo         appInfo        = {};
  };

  NVSDK_NGX_Result init(const InitInfo& info);
  NVSDK_NGX_Result deinit();

  NVSDK_NGX_EngineType       getEngineType() const { return m_initInfo.appInfo.engineType; }
  const std::string&         getEngineVersion() const { return m_initInfo.appInfo.engineVersion; }
  const std::string&         getProjectId() const { return m_initInfo.appInfo.projectId; }
  const NVSDK_NGX_Parameter* getNgxParams() const { return m_ngxParams; }
  NVSDK_NGX_Parameter*       getNgxParams() { return m_ngxParams; }
  NVSDK_NGX_Result           isDlssRRAvailable();

  const std::string& getApplicationPath() const { return m_initInfo.appInfo.applicationPath; }

  VkDevice         getDevice() const { return m_initInfo.device; }
  VkInstance       getInstance() const { return m_initInfo.instance; }
  VkPhysicalDevice getPhysicalDevice() const { return m_initInfo.physicalDevice; }

private:
  InitInfo             m_initInfo  = {};
  NVSDK_NGX_Parameter* m_ngxParams = nullptr;
};


class DlssRayReconstruction
{
public:
  static NVSDK_NGX_Result getRequiredInstanceExtensions(const NgxContext::ApplicationInfo&  appInfo,
                                                        std::vector<VkExtensionProperties>& extensions);

  static NVSDK_NGX_Result getRequiredDeviceExtensions(const NgxContext::ApplicationInfo&  appInfo,
                                                      const VkInstance&                   instance,
                                                      const VkPhysicalDevice&             physicalDevice,
                                                      std::vector<VkExtensionProperties>& extensions);


  enum class ResourceType
  {
    eColorIn,
    eColorOut,
    eDiffuseAlbedo,
    eSpecularAlbedo,
    eSpecularHitDistance,
    eNormalRoughness,
    eRoughness,
    eMotionVector,
    eDepth,
    eResourceTypeCount
  };


  struct SupportedSizes
  {
    VkExtent2D minSize;
    VkExtent2D maxSize;
    VkExtent2D optimalSize;
  };


  struct SupportedSizeInfo
  {
    VkExtent2D                  outputSize;
    NVSDK_NGX_PerfQuality_Value perfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  };

  static bool querySupport(const NgxContext& context);

  static NVSDK_NGX_Result querySupportedInputSizes(NgxContext& context, const SupportedSizeInfo& info, SupportedSizes* sizes);

  struct InitInfo
  {
    bool       packedNormalRoughness = true;
    bool       hardwareDepth         = true;
    VkExtent2D inputSize{};
    VkExtent2D outputSize{};
    // GPU node mask, change only if using a system with multiple GPUs
    uint32_t creationNodeMask   = 0x1;
    uint32_t visibilityNodeMask = 0x1;
  };

  NVSDK_NGX_Result cmdInit(VkCommandBuffer cmd, NgxContext& context, const InitInfo& info);
  NVSDK_NGX_Result deinit();


  struct Resource
  {
    ResourceType            type;
    VkImage                 image;
    VkImageView             imageView;
    VkFormat                format;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  };

  void setResource(const Resource& resource);

  struct DenoiseInfo
  {
    glm::vec2 jitter;
    glm::mat4 modelView;
    glm::mat4 projection;
    bool      reset = false;
  };

  NVSDK_NGX_Result cmdDenoise(VkCommandBuffer cmd, NgxContext& context, const DenoiseInfo& info);


private:
  InitInfo                                                                 m_initInfo{};
  NVSDK_NGX_Handle*                                                        m_handle{};
  std::array<NVSDK_NGX_Resource_VK, int(ResourceType::eResourceTypeCount)> m_resources;
};
