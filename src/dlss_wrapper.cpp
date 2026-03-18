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

#define _CRT_SECURE_NO_WARNINGS
#include <volk.h>

#include "dlss_wrapper.hpp"


#define NGX_ENABLE_DEPRECATED_GET_PARAMETERS 1

#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>

#include <glm/ext.hpp>

#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <cstring>  // memset
#include "nvutils/logger.hpp"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t g_ApplicationID = 0x12345678ABCDEF01ULL;  // Dummy ULL for testing, replace with actual application ID for production (ask NVIDIA DevRel)


// Function taken from stack overflow https://stackoverflow.com/questions/4804298/how-to-convert-wstring-into-string
static std::string wstringToString(const std::wstring& wstr)
{
  std::mbstate_t state{};
  const wchar_t* src = wstr.c_str();
  std::size_t    len = std::wcsrtombs(nullptr, &src, 0, &state);
  if(len == static_cast<std::size_t>(-1))
  {
    throw std::runtime_error("Conversion failed");
  }

  std::string str(len, '\0');
  std::wcsrtombs(&str[0], &src, len, &state);
  return str;
}

static std::wstring stringToWstring(const std::string& str)
{
  size_t len = std::mbstowcs(nullptr, str.c_str(), 0);
  if(len == static_cast<size_t>(-1))
  {
    return L"";  // Conversion failed
  }
  std::wstring wstr(len, L'\0');
  std::mbstowcs(&wstr[0], str.c_str(), len);
  return wstr;
}


static NVSDK_NGX_Result checkNgxResult(NVSDK_NGX_Result result, const char* func, int line)
{
  if(NVSDK_NGX_FAILED(result))
  {
    std::ostringstream str;
    str << "NGX Error: " << wstringToString(GetNGXResultAsString(result)) << " at " << func << ":" << line;

    LOGE("%s\n", str.str().c_str());
  }

  return result;
}

#define CALL_NGX(x)                                                                                                    \
  {                                                                                                                    \
    NVSDK_NGX_Result r = checkNgxResult((x), __func__, __LINE__);                                                      \
    if(NVSDK_NGX_FAILED(r))                                                                                            \
      return r;                                                                                                        \
  }

void NVSDK_CONV NGX_AppLogCallback(const char* message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature sourceComponent)
{
  LOGI("%s", message);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Class code
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NVSDK_NGX_Result NgxContext::init(const NgxContext::InitInfo& initInfo)
{
  std::wstring   exeString  = stringToWstring(initInfo.appInfo.applicationPath);
  const wchar_t* exeWString = exeString.c_str();

  m_initInfo = initInfo;

  NVSDK_NGX_FeatureCommonInfo info     = {};
  info.LoggingInfo.MinimumLoggingLevel = initInfo.loggingLevel;

  CALL_NGX(NVSDK_NGX_VULKAN_Init(g_ApplicationID, exeWString, initInfo.instance, initInfo.physicalDevice,
                                 initInfo.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &info));

  CALL_NGX(NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_ngxParams));

  if(m_ngxParams == nullptr)
    return NVSDK_NGX_Result_Fail;

  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::deinit()
{
  if(m_ngxParams)
  {
    CALL_NGX(NVSDK_NGX_VULKAN_DestroyParameters(m_ngxParams));
    CALL_NGX(NVSDK_NGX_VULKAN_Shutdown1(m_initInfo.device));
    m_ngxParams = {};
  }
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::isDlssRRAvailable()
{
  if(m_ngxParams == nullptr)
    return NVSDK_NGX_Result_Fail;

  NVSDK_NGX_FeatureCommonInfo    commonInfo = {};
  NVSDK_NGX_FeatureDiscoveryInfo info       = {};
  info.SDKVersion                           = NVSDK_NGX_Version_API;
  info.FeatureID                            = NVSDK_NGX_Feature_RayReconstruction;
  info.Identifier.IdentifierType            = NVSDK_NGX_Application_Identifier_Type_Application_Id;
  info.Identifier.v.ApplicationId           = g_ApplicationID;
  info.ApplicationDataPath                  = L" ";  // Non-empty string required by NGX API
  info.FeatureInfo                          = &commonInfo;

  NVSDK_NGX_FeatureRequirement requirement = {};

  NVSDK_NGX_VULKAN_GetFeatureRequirements(m_initInfo.instance, m_initInfo.physicalDevice, &info, &requirement);

  if(requirement.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported)
  {
    return NVSDK_NGX_Result_FAIL_Denied;
  }

  return NVSDK_NGX_Result_Success;
}

bool DlssRayReconstruction::querySupport(const NgxContext& context)
{
  int32_t  supported             = 0;
  int32_t  needsUpdatedDriver    = 1;
  uint32_t minDriverVersionMajor = ~0u;
  uint32_t minDriverVersionMinor = ~0u;

  // Check if DLSS_D (which is the DLSS_RR RayReconstruction Denoiser) is available.
  NVSDK_NGX_Result resUpdatedDriver =
      context.getNgxParams()->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsUpdatedDriver);
  NVSDK_NGX_Result resVersionMajor =
      context.getNgxParams()->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor, &minDriverVersionMajor);
  NVSDK_NGX_Result resVersionMinor =
      context.getNgxParams()->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor, &minDriverVersionMinor);

  if(NVSDK_NGX_SUCCEED(resUpdatedDriver))
  {
    if(needsUpdatedDriver)
    {
      // NVIDIA DLSS cannot be loaded due to outdated driver.
      if(NVSDK_NGX_SUCCEED(resVersionMajor) && NVSDK_NGX_SUCCEED(resVersionMinor))
      {
        // Min Driver Version required: minDriverVersionMajor.minDriverVersionMinor
        LOGW("DLSS_RR: Minimum driver version required: %d.%d\n", minDriverVersionMajor, minDriverVersionMinor);
        return false;
      }
    }
  }

  NVSDK_NGX_Result resDlssSupported = context.getNgxParams()->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &supported);
  if(NVSDK_NGX_FAILED(resDlssSupported) || !supported)
  {
    LOGW("DLSS_RR: not available on this hardware/platform\n");
    return false;
  }
  resDlssSupported = context.getNgxParams()->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, &supported);
  if(NVSDK_NGX_FAILED(resDlssSupported) || !supported)
  {
    LOGW("DLSS_RR: denied for this application\n");
    return false;
  }

  return true;
}


NVSDK_NGX_Result DlssRayReconstruction::getRequiredInstanceExtensions(const NgxContext::ApplicationInfo&  appInfo,
                                                                      std::vector<VkExtensionProperties>& extensions)
{
  std::wstring appPath = stringToWstring(appInfo.applicationPath);
  extensions.clear();
  NVSDK_NGX_FeatureCommonInfo commonInfo = {};

  NVSDK_NGX_FeatureDiscoveryInfo info{};
  info.SDKVersion                             = NVSDK_NGX_Version_API;
  info.FeatureID                              = NVSDK_NGX_Feature_RayReconstruction;
  info.Identifier.IdentifierType              = NVSDK_NGX_Application_Identifier_Type_Project_Id;
  info.Identifier.v.ProjectDesc.EngineType    = appInfo.engineType;
  info.Identifier.v.ProjectDesc.EngineVersion = appInfo.engineVersion.c_str();
  info.Identifier.v.ProjectDesc.ProjectId     = appInfo.projectId.c_str();

  info.ApplicationDataPath = appPath.c_str();
  info.FeatureInfo         = &commonInfo;

  uint32_t               numExtensions = 0;
  VkExtensionProperties* props{};

  CALL_NGX(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &numExtensions, &props));

  extensions = std::vector<VkExtensionProperties>(props, props + numExtensions);
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DlssRayReconstruction::getRequiredDeviceExtensions(const NgxContext::ApplicationInfo&  appInfo,
                                                                    const VkInstance&                   instance,
                                                                    const VkPhysicalDevice&             physicalDevice,
                                                                    std::vector<VkExtensionProperties>& extensions)
{
  std::wstring appPath = stringToWstring(appInfo.applicationPath);
  extensions.clear();

  NVSDK_NGX_FeatureCommonInfo    commonInfo{};
  NVSDK_NGX_FeatureDiscoveryInfo info{};
  info.SDKVersion                             = NVSDK_NGX_Version_API;
  info.FeatureID                              = NVSDK_NGX_Feature_RayReconstruction;
  info.Identifier.IdentifierType              = NVSDK_NGX_Application_Identifier_Type_Project_Id;
  info.Identifier.v.ProjectDesc.EngineType    = appInfo.engineType;
  info.Identifier.v.ProjectDesc.EngineVersion = appInfo.engineVersion.c_str();
  info.Identifier.v.ProjectDesc.ProjectId     = appInfo.projectId.c_str();
  info.ApplicationDataPath                    = appPath.c_str();
  info.FeatureInfo                            = &commonInfo;


  uint32_t               numExtensions = 0;
  VkExtensionProperties* props{};

  CALL_NGX(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physicalDevice, &info, &numExtensions, &props));

  extensions = std::vector<VkExtensionProperties>(props, props + numExtensions);

  return NVSDK_NGX_Result_Success;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


NVSDK_NGX_Result DlssRayReconstruction::cmdInit(VkCommandBuffer cmd, NgxContext& context, const InitInfo& info)
{
  m_initInfo = info;


  NVSDK_NGX_DLSSD_Create_Params dlssdParams = {};

  dlssdParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;


  if(info.packedNormalRoughness)
  {
    dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;  // we pack roughness into the normal's w channel
  }
  else
  {
    dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
  }

  if(info.hardwareDepth)
  {
    dlssdParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;  // we're providing hardware (raster) depth
  }
  else
  {
    dlssdParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;  // we're providing linear depth
  }


  dlssdParams.InWidth        = info.inputSize.width;
  dlssdParams.InHeight       = info.inputSize.height;
  dlssdParams.InTargetWidth  = info.outputSize.width;
  dlssdParams.InTargetHeight = info.outputSize.height;

  // Though marked as 'optional', these are absolutely needed
  dlssdParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;


  // Using Preset_E (Latest transformer model, required for DoF guide support)
  // Preset_D = Default transformer, Preset_E = Latest transformer (only two valid options)
  NVSDK_NGX_RayReconstruction_Hint_Render_Preset dlssdModel = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E;
  context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, dlssdModel);
  context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, dlssdModel);
  context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, dlssdModel);
  context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, dlssdModel);


  CALL_NGX(NGX_VULKAN_CREATE_DLSSD_EXT1(context.getDevice(), cmd, info.creationNodeMask, info.visibilityNodeMask,
                                        &m_handle, context.getNgxParams(), &dlssdParams));

  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DlssRayReconstruction::deinit()
{
  if(m_handle)
  {
    CALL_NGX(NVSDK_NGX_VULKAN_ReleaseFeature(m_handle));
  }
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DlssRayReconstruction::querySupportedInputSizes(NgxContext& context, const SupportedSizeInfo& info, SupportedSizes* sizes)
{

  if(!sizes)
  {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  // FIXME: check if that's important
  float sharpness;


  CALL_NGX(NGX_DLSSD_GET_OPTIMAL_SETTINGS(context.getNgxParams(), info.outputSize.width, info.outputSize.height, info.perfQualityValue,
                                          &sizes->optimalSize.width, &sizes->optimalSize.height, &sizes->maxSize.width,
                                          &sizes->maxSize.height, &sizes->minSize.width, &sizes->minSize.height, &sharpness));

  return NVSDK_NGX_Result_Success;
}

void DlssRayReconstruction::setResource(const Resource& resource)
{
  VkExtent2D size = (resource.type == ResourceType::eColorOut) ? m_initInfo.outputSize : m_initInfo.inputSize;

  bool isReadWrite = (resource.type == ResourceType::eColorOut);

  NVSDK_NGX_Resource_VK r = NVSDK_NGX_Create_ImageView_Resource_VK(resource.imageView, resource.image, resource.range,
                                                                   resource.format, size.width, size.height, isReadWrite);

  m_resources[int(resource.type)] = r;
}

NVSDK_NGX_Result DlssRayReconstruction::cmdDenoise(VkCommandBuffer cmd, NgxContext& context, const DenoiseInfo& info)
{

  NVSDK_NGX_VK_DLSSD_Eval_Params evalParams = {};
  evalParams.pInDiffuseAlbedo               = &m_resources[int(ResourceType::eDiffuseAlbedo)];
  evalParams.pInDiffuseHitDistance          = nullptr;
  evalParams.pInSpecularAlbedo              = &m_resources[int(ResourceType::eSpecularAlbedo)];
  evalParams.pInSpecularHitDistance = m_resources[int(ResourceType::eSpecularHitDistance)].Resource.ImageViewInfo.ImageView ?
                                          &m_resources[int(ResourceType::eSpecularHitDistance)] :
                                          nullptr;

  evalParams.pInNormals = &m_resources[int(ResourceType::eNormalRoughness)];
  // Is this needed with NVSDK_NGX_DLSS_Roughness_Mode_Packed?
  if(m_initInfo.packedNormalRoughness)
  {
    evalParams.pInRoughness = &m_resources[int(ResourceType::eNormalRoughness)];
  }
  else
  {
    evalParams.pInRoughness = &m_resources[int(ResourceType::eRoughness)];
  }
  evalParams.pInColor         = &m_resources[int(ResourceType::eColorIn)];
  evalParams.pInOutput        = &m_resources[int(ResourceType::eColorOut)];
  evalParams.pInDepth         = &m_resources[int(ResourceType::eDepth)];
  evalParams.pInMotionVectors = &m_resources[int(ResourceType::eMotionVector)];

  evalParams.InJitterOffsetX = -info.jitter.x;
  evalParams.InJitterOffsetY = -info.jitter.y;
  evalParams.InMVScaleX      = 1.0f;
  evalParams.InMVScaleY      = 1.0f;

  evalParams.InRenderSubrectDimensions.Width  = m_initInfo.inputSize.width;
  evalParams.InRenderSubrectDimensions.Height = m_initInfo.inputSize.height;

  glm::mat4 modelViewRowMajor     = glm::transpose(info.modelView);
  glm::mat4 projectionRowMajor    = glm::transpose(info.projection);
  evalParams.pInWorldToViewMatrix = glm::value_ptr(modelViewRowMajor);
  evalParams.pInViewToClipMatrix  = glm::value_ptr(projectionRowMajor);

  evalParams.InReset = info.reset;

  CALL_NGX(NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, m_handle, context.getNgxParams(), &evalParams));
  return NVSDK_NGX_Result_Success;
}
