/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vkgs_project_reader.h"
#include "gaussian_splatting_ui.h"
#include "parameters.h"
#include "utilities.h"
#include "splat_set_vk.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <tinygltf/json.hpp>

using json   = nlohmann::json;
namespace fs = std::filesystem;

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// Helper macros to load JSON values only if they exist
//

#define LOAD1(val, item, name)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if((item).contains(name))                                                                                          \
      (val) = (item)[name];                                                                                            \
  } while(0)

#define LOAD2(val, item, name)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if((item).contains(name))                                                                                          \
      (val) = {(item)[name][0], (item)[name][1]};                                                                      \
  } while(0)

#define LOAD3(val, item, name)                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if((item).contains(name))                                                                                          \
      (val) = {(item)[name][0], (item)[name][1], (item)[name][2]};                                                     \
  } while(0)

// Load material from JSON — version 7+ uses maxBounces, version 6 uses illum, older converts from Phong
static void loadMaterialFromJson(Material& mat, const json& matItem, int fileVersion)
{
  if(fileVersion >= 7)
  {
    LOAD3(mat.baseColor, matItem, "baseColor");
    LOAD1(mat.metallic, matItem, "metallic");
    LOAD1(mat.roughness, matItem, "roughness");
    LOAD3(mat.emissive, matItem, "emissive");
    LOAD1(mat.emissiveStrength, matItem, "emissiveStrength");
    LOAD1(mat.maxBounces, matItem, "maxBounces");
    LOAD1(mat.ior, matItem, "ior");
    LOAD1(mat.transmission, matItem, "transmission");
    LOAD1(mat.opacity, matItem, "opacity");
    LOAD1(mat.specularFactor, matItem, "specularFactor");
    LOAD3(mat.specularColorFactor, matItem, "specularColorFactor");
    LOAD1(mat.clearcoatFactor, matItem, "clearcoatFactor");
    LOAD1(mat.clearcoatRoughness, matItem, "clearcoatRoughness");
  }
  else if(fileVersion >= 6)
  {
    LOAD3(mat.baseColor, matItem, "baseColor");
    LOAD1(mat.metallic, matItem, "metallic");
    LOAD1(mat.roughness, matItem, "roughness");
    LOAD3(mat.emissive, matItem, "emissive");
    LOAD1(mat.emissiveStrength, matItem, "emissiveStrength");
    LOAD1(mat.ior, matItem, "ior");
    LOAD1(mat.transmission, matItem, "transmission");
    LOAD1(mat.opacity, matItem, "opacity");

    // Convert legacy illum to maxBounces
    int illum = 0;
    LOAD1(illum, matItem, "illum");
    mat.maxBounces = (illum == 3) ? 3 : (illum >= 1) ? 1 : 0;
  }
  else
  {
    // Legacy Phong material — convert to PBR
    glm::vec3 diffuse(0.7f), specular(0.0f), transmittance(0.0f);
    float     shininess = 32.0f;

    LOAD3(diffuse, matItem, "diffuse");
    LOAD3(specular, matItem, "specular");
    LOAD3(mat.emissive, matItem, "emission");
    LOAD3(transmittance, matItem, "transmittance");
    LOAD1(shininess, matItem, "shininess");
    LOAD1(mat.ior, matItem, "ior");

    mat.baseColor    = diffuse;
    mat.roughness    = std::sqrt(2.0f / (std::max(shininess, 1.0f) + 2.0f));
    float specLum    = glm::length(specular);
    float diffLum    = glm::length(diffuse);
    mat.metallic     = (specLum > 0.5f && diffLum < 0.1f) ? 1.0f : 0.0f;
    mat.transmission = (glm::length(transmittance) > 0.001f) ? 1.0f : 0.0f;
    mat.opacity      = 1.0f;

    // Convert legacy illum to maxBounces
    int illum = 0;
    LOAD1(illum, matItem, "illum");
    mat.maxBounces = (illum == 3) ? 3 : (illum >= 1) ? 1 : 0;
  }
}

//--------------------------------------------------------------------------------------------------
// Helper function to convert relative path to absolute
//
// Projects authored on Windows store asset paths with backslash separators
// (e.g. "data\\scene\\cloud.ply"). On POSIX, std::filesystem treats '\\' as a
// regular filename character, so such a path fails to resolve and the asset
// loads empty. Normalize to '/' first; '/' is accepted on Windows too, so this
// keeps .vkgs projects portable across platforms.
static std::filesystem::path makeAbsolutePath(const std::filesystem::path& base, const std::string& relativePath)
{
  std::string normalized = relativePath;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  return std::filesystem::absolute(base / normalized);
}

//--------------------------------------------------------------------------------------------------
// Load project from JSON data
//
bool VkgsProjectReader::loadProject(const json& data, const std::string& path, GaussianSplattingUI* ui)
{
  try
  {
    // Parse file version to determine format
    int fileVersion = 0;
    if(data.contains("version"))
    {
      fileVersion = data["version"].get<int>();
    }

    // Load all sections using helper functions
    loadAssetNamingCounters(data, ui);
    loadRendererSettings(data, ui);
    loadSplatGlobalOptions(data);
    loadSplatSetsAndInstances(data, fileVersion, ui);
    loadMeshes(data, fileVersion, path, ui);
    loadCameras(data, ui);
    loadLights(data, fileVersion, ui);
    loadEnvironment(data, path, ui);
    loadSettings(data, ui);
    loadTonemapping(data, ui);

    prmScene.projectToLoadFilename = "";
    return true;
  }
  catch(...)
  {
    prmScene.projectToLoadFilename = "";
    return false;
  }
}

//--------------------------------------------------------------------------------------------------
// Load asset naming counters from file
//
void VkgsProjectReader::loadAssetNamingCounters(const json& data, GaussianSplattingUI* ui)
{
  if(data.contains("nextSplatSetNumber"))
    ui->m_assets.splatSets.m_nextInstanceNumber = data["nextSplatSetNumber"].get<uint32_t>();
  if(data.contains("nextMeshNumber"))
    ui->m_assets.meshes.m_nextInstanceNumber = data["nextMeshNumber"].get<uint32_t>();
  if(data.contains("nextLightNumber"))
    ui->m_assets.lights.m_nextLightNumber = data["nextLightNumber"].get<uint32_t>();
}

//--------------------------------------------------------------------------------------------------
// Load renderer settings (vsync, pipeline, rendering parameters)
//
void VkgsProjectReader::loadRendererSettings(const json& data, GaussianSplattingUI* ui)
{
  if(!data.contains("renderer"))
    return;

  const auto& item = data["renderer"];
  if(item.contains("vsync"))
    ui->m_app->setVsync(item["vsync"]);

  LOAD1(prmSelectedPipeline, item, "pipeline");
  LOAD1(prmFrame.shDegree, item, "maxShDegree");
  LOAD1(prmRender.opacityGaussianDisabled, item, "opacityGaussianDisabled");
  LOAD1(prmRender.showShOnly, item, "showShOnly");
  LOAD1(prmRender.visualize, item, "visualize");
  LOAD1(prmRender.wireframe, item, "wireframe");
  LOAD1(prmRaster.cpuLazySort, item, "cpuLazySort");
  LOAD1(prmRaster.distShaderWorkgroupSize, item, "distShaderWorkgroupSize");
  LOAD1(prmRaster.fragmentBarycentric, item, "fragmentBarycentric");
  LOAD1(prmRaster.frustumCulling, item, "frustumCulling");
  LOAD1(prmRaster.sizeCulling, item, "sizeCulling");
  LOAD1(prmFrame.sizeCullingMinPixels, item, "sizeCullingMinPixels");
  LOAD1(prmRaster.meshShaderWorkgroupSize, item, "meshShaderWorkgroupSize");
  LOAD1(prmRaster.pointCloudModeEnabled, item, "pointCloudModeEnabled");
  LOAD1(prmRaster.sortingMethod, item, "sortingMethod");
  LOAD1(prmRtx.temporalSampling, item, "temporalSampling");
  LOAD1(prmFrame.frameSampleMax, item, "temporalSamplesCount");
  LOAD1(prmRtx.kernelAdaptiveClamping, item, "kernelAdaptiveClamping");
  LOAD1(prmRtx.kernelDegree, item, "kernelDegree");
  LOAD1(prmRtx.kernelMinResponse, item, "kernelMinResponse");
  LOAD1(prmRtx.particleSamplesPerPass, item, "particleSamplesPerPass");
  LOAD1(prmRtx.particleSamplesPerPass, item, "payloadArraySize");  // backward compat with old project files
  LOAD1(prmRtx.rtxTraceStrategy, item, "rtxTraceStrategy");

  // Normal method (enum class, need explicit cast from int)
  if(item.contains("normalMethod"))
    prmRender.normalMethod = (NormalMethod)item["normalMethod"].get<int>();
  LOAD1(prmRender.thinParticleThreshold, item, "thinParticleThreshold");
  LOAD1(prmRtx.fireflyClampThreshold, item, "fireflyClampThreshold");
#if defined(USE_DLSS)
  LOAD1(prmRtx.dlssMinRadianceThreshold, item, "dlssMinRadianceThreshold");
  if(item.contains("dlssEnabled"))
    ui->m_dlss.setEnabled(item["dlssEnabled"].get<bool>());
  if(item.contains("dlssSizeMode"))
    ui->m_dlss.setSizeMode(static_cast<DlssDenoiser::SizeMode>(item["dlssSizeMode"].get<int>()));
#endif

  // Lighting: current format is bool "lightingEnabled"
  if(item.contains("lightingEnabled"))
    prmRender.lightingEnabled = item["lightingEnabled"].get<bool>() ? LIGHTING_ENABLED : LIGHTING_DISABLED;
  // Backward compat: old projects used int "lightingMode" (0=off, 1+=on)
  if(!item.contains("lightingEnabled") && item.contains("lightingMode"))
    prmRender.lightingEnabled = (item["lightingMode"].get<int>() > 0) ? LIGHTING_ENABLED : LIGHTING_DISABLED;

  if(item.contains("shadowsMode"))
    prmRender.shadowsMode = (ShadowsMode)item["shadowsMode"].get<int>();
  // Backward compat: old projects used bool "shadowsEnabled"
  if(!item.contains("shadowsMode") && item.contains("shadowsEnabled"))
    prmRender.shadowsMode = item["shadowsEnabled"].get<bool>() ? ShadowsMode::eShadowsHard : ShadowsMode::eShadowsDisabled;

  LOAD1(prmFrame.rtxMaxBounces, item, "rtxMaxBounces");
  LOAD1(prmFrame.rtxSecondaryRayOffset, item, "rtxSecondaryRayOffset");
  LOAD1(prmFrame.minSplatSetCompositeTransmittance, item, "splatSetCompositeTransmittance");
  LOAD1(prmRtx.temporalSamplingMode, item, "temporalSamplingMode");
  LOAD1(prmFrame.alphaCullThreshold, item, "alphaCullThreshold");
  LOAD1(prmFrame.splatScale, item, "splatScale");
  LOAD1(prmFrame.alphaClamp, item, "alphaClamp");
  LOAD1(prmFrame.minTransmittance, item, "minTransmittance");
  LOAD1(prmFrame.maxPasses, item, "maxPasses");
  LOAD1(prmRtx.particleDepth, item, "particleDepth");
  LOAD1(prmRtx.billboardFrustumCulling, item, "billboardFrustumCulling");
  LOAD1(prmRtx.shortenRay, item, "shortenRay");
  LOAD1(prmRtx.quantizeMeshPayload, item, "quantizeMeshPayload");
  LOAD1(prmRtx.particleShadowOffset, item, "particleShadowOffset");
  LOAD1(prmRtx.particleShadowTransmittanceThreshold, item, "particleShadowTransmittanceThreshold");
  LOAD1(prmRtx.particleShadowColorStrength, item, "particleShadowColorStrength");
  LOAD1(prmRtx.particleEmissiveAoEnabled, item, "particleEmissiveAoEnabled");
  LOAD1(prmRtx.particleEmissiveAoRadius, item, "particleEmissiveAoRadius");
  LOAD1(prmRtx.particleEmissiveAoStrength, item, "particleEmissiveAoStrength");
  LOAD1(prmRtx.depthIsoThresholdRTX, item, "depthIsoThresholdRTX");
  LOAD1(prmRaster.depthIsoThreshold, item, "depthIsoThreshold");
  LOAD1(prmRaster.covarianceDilation, item, "covarianceDilation");
  LOAD1(prmRaster.msAntialiasing, item, "msAntialiasing");
  LOAD1(prmRaster.quantizeNormals, item, "quantizeNormals");
  LOAD1(prmRaster.ftbSyncMode, item, "ftbSyncMode");
  LOAD1(prmRaster.extentProjection, item, "extentProjection");

  // Color format (VkFormat enum, need explicit cast)
  // Must trigger GBuffer reinit when the format differs from the current one
  if(item.contains("colorFormat"))
  {
    VkFormat newFormat = static_cast<VkFormat>(item["colorFormat"].get<int>());
    if(newFormat != prmRender.colorFormat)
    {
      prmRender.colorFormat      = newFormat;
      ui->m_requestGBufferReinit = true;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Load splat global options (data storage, compression, etc.)
//
void VkgsProjectReader::loadSplatGlobalOptions(const json& data)
{
  if(!data.contains("splatsGlobals"))
    return;

  const auto& item = data["splatsGlobals"];
  //LOAD1(prmData.dataStorage, item, "dataStorage");
  LOAD1(prmData.shFormat, item, "shFormat");
  LOAD1(prmData.rgbaFormat, item, "rgbaFormat");
  LOAD1(prmRtxData.compressBlas, item, "compressBlas");
  LOAD1(prmRtxData.useAABBs, item, "useAABBs");
  LOAD1(prmRtxData.useSpheres, item, "useSpheres");
  LOAD1(prmRtxData.useTlasInstances, item, "useTlasInstances");
  if(item.contains("billboardBoundingMode"))
    prmRtxData.billboardBoundingMode = (BillboardBoundingMode)item["billboardBoundingMode"].get<int>();
  // Note: No manager requests needed here.
  // The scene will be loaded with these settings during initAll(),
  // which reads prmData/prmRtxData. Requesting an update would trigger
  // unnecessary buffer recreation.
}

//--------------------------------------------------------------------------------------------------
// Load splat assets (splat set paths, storage, format)
// Creates pre-configured SplatSetVk objects that will be filled by the loader
//
void VkgsProjectReader::loadSplatAssets(const json& data, int fileVersion, std::map<int, std::shared_ptr<SplatSetVk>>& splatSetIdToAsset)
{
  if(!data.contains("splatSets") || !data["splatSets"].is_array())
    return;

  for(const auto& item : data["splatSets"])
  {
    int         id      = item["id"].get<int>();
    std::string relPath = item["path"].get<std::string>();

    // Create pre-configured splat set
    auto splatSet  = std::make_shared<SplatSetVk>();
    splatSet->path = makeAbsolutePath(prmScene.projectToLoadFilename.parent_path(), relPath).string();

    // Version 5+: per-splat-set storage, format, and name
    if(fileVersion >= 5)
    {
      if(item.contains("storage"))
        splatSet->dataStorage = item["storage"].get<uint32_t>();
      if(item.contains("shFormat"))
        splatSet->shFormat = item["shFormat"].get<uint32_t>();
      if(item.contains("rgbaFormat"))
        splatSet->rgbaFormat = item["rgbaFormat"].get<uint32_t>();
      else if(item.contains("format"))  // backward compat with old v5 files
        splatSet->shFormat = item["format"].get<uint32_t>();
      // Note: 'name' was removed from SplatSet (now only in instances)
      // Path is loaded above and used to identify the asset
    }
    // Older versions: use defaults (STORAGE_BUFFERS=0, FORMAT_FLOAT32=0)

    splatSetIdToAsset[id] = splatSet;
  }
}

//--------------------------------------------------------------------------------------------------
// Load splat instances and create load requests
//
void VkgsProjectReader::loadSplatInstances(const json& data, int fileVersion, std::map<int, std::shared_ptr<SplatSetVk>>& splatSetIdToAsset)
{
  if(!data.contains("splats") || !data["splats"].is_array())
    return;

  // Parse instances and group them by splatSetId
  std::map<int, std::vector<std::shared_ptr<SplatSetInstanceVk>>> instancesBySplatSetId;

  for(const auto& item : data["splats"])
  {
    int splatSetId = item["splatSetId"].get<int>();

    // Pre-create and configure instance with all project settings
    auto instance = std::make_shared<SplatSetInstanceVk>();

    // Load name if present, otherwise will be generated at registration
    if(item.contains("name"))
    {
      instance->displayName = item["name"].get<std::string>();
    }
    // else: Name will be generated by registerInstance() if empty

    if(item.contains("show"))
      instance->show = item["show"].get<bool>();

    // Parse and set transform
    if(item.contains("position") && item.contains("rotation") && item.contains("scale"))
    {
      LOAD3(instance->translation, item, "position");
      LOAD3(instance->rotation, item, "rotation");
      LOAD3(instance->scale, item, "scale");

      // Compute transform matrices
      computeTransform(instance->scale, instance->rotation, instance->translation, instance->transform,
                       instance->transformInverse, instance->transformRotScaleInverse);
    }

    // Set splat-specific defaults before loading (struct defaults are for glTF meshes)
    instance->splatMaterial.baseColor           = glm::vec3(0.0f);
    instance->splatMaterial.specularFactor      = 0.0f;
    instance->splatMaterial.specularColorFactor = glm::vec3(0.0f);
    instance->splatMaterial.emissive            = glm::vec3(1.0f);
    instance->splatMaterial.maxBounces          = 0;

    // Parse and set material (overrides defaults with saved values if present)
    if(item.contains("material"))
    {
      loadMaterialFromJson(instance->splatMaterial, item["material"], fileVersion);
    }

    instancesBySplatSetId[splatSetId].push_back(instance);
  }

  // Create load requests: one per splat set, with all instances for that set
  for(const auto& [splatSetId, instances] : instancesBySplatSetId)
  {
    auto assetIt = splatSetIdToAsset.find(splatSetId);
    if(assetIt == splatSetIdToAsset.end())
      continue;  // Invalid splatSetId reference

    auto splatSet = assetIt->second;

    // First instance triggers the load
    SceneLoadRequest request;
    request.path      = splatSet->path;  // Path stored in splatSet->path
    request.porcelain = true;
    request.splatSet  = splatSet;      // Pass pre-configured splat set (with storage/format already set)
    request.instance  = instances[0];  // First instance

    // Store additional instances to create after load completes
    for(size_t i = 1; i < instances.size(); ++i)
    {
      request.additionalInstances.push_back(instances[i]);
    }

    prmScene.pushLoadRequest(request);
  }
}

//--------------------------------------------------------------------------------------------------
// Load splat sets and instances (with version-dependent format)
//
void VkgsProjectReader::loadSplatSetsAndInstances(const json& data, int fileVersion, GaussianSplattingUI* ui)
{
  if(fileVersion >= 1 && data.contains("splatSets") && data["splatSets"].is_array())
  {
    // Version 1+ format: separate splatSets and splats arrays
    std::map<int, std::shared_ptr<SplatSetVk>> splatSetIdToAsset;
    loadSplatAssets(data, fileVersion, splatSetIdToAsset);
    loadSplatInstances(data, fileVersion, splatSetIdToAsset);
  }
  else if(data.contains("splats") && data["splats"].is_array())
  {
    // Legacy format (version 0): each splat entry has its own path
    // This loads the same file multiple times if instances share a splat set
    for(const auto& item : data["splats"])
    {
      // Pre-create and configure instance with all project settings
      auto instance = std::make_shared<SplatSetInstanceVk>();

      // Load name if present, otherwise will be generated at registration
      if(item.contains("name"))
      {
        instance->displayName = item["name"].get<std::string>();
      }
      // else: Name will be generated by registerInstance() if empty

      if(item.contains("show"))
        instance->show = item["show"].get<bool>();

      // Parse and set transform
      if(item.contains("position") && item.contains("rotation") && item.contains("scale"))
      {
        LOAD3(instance->translation, item, "position");
        LOAD3(instance->rotation, item, "rotation");
        LOAD3(instance->scale, item, "scale");

        // Compute transform matrices
        computeTransform(instance->scale, instance->rotation, instance->translation, instance->transform,
                         instance->transformInverse, instance->transformRotScaleInverse);
      }

      // Set splat-specific defaults before loading (struct defaults are for glTF meshes)
      instance->splatMaterial.baseColor           = glm::vec3(0.0f);
      instance->splatMaterial.specularFactor      = 0.0f;
      instance->splatMaterial.specularColorFactor = glm::vec3(0.0f);
      instance->splatMaterial.emissive            = glm::vec3(1.0f);
      instance->splatMaterial.maxBounces          = 0;

      // Parse and set material (overrides defaults with saved values if present)
      if(item.contains("material"))
      {
        loadMaterialFromJson(instance->splatMaterial, item["material"], fileVersion);
      }

      // Create request with pre-configured instance
      SceneLoadRequest request;
      request.path      = makeAbsolutePath(prmScene.projectToLoadFilename.parent_path(), item["path"]);
      request.porcelain = true;      // We do not want UI questions
      request.instance  = instance;  // Pass pre-configured instance

      prmScene.pushLoadRequest(request);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Load mesh assets (mesh files)
//
void VkgsProjectReader::loadMeshAssets(const json&                             data,
                                       const std::string&                      projectPath,
                                       std::map<int, std::shared_ptr<MeshVk>>& assetIdToMesh,
                                       GaussianSplattingUI*                    ui)
{
  if(!data.contains("meshAssets") || !data["meshAssets"].is_array())
    return;

  for(const auto& assetItem : data["meshAssets"])
  {
    int         id      = assetItem["id"].get<int>();
    std::string relPath = assetItem["path"].get<std::string>();

    auto meshPath = makeAbsolutePath(std::filesystem::path(projectPath).parent_path(), relPath);
    auto mesh     = ui->m_assets.meshes.loadModel(meshPath.string());

    if(mesh)
    {
      // Remove the auto-created default instance (we'll create explicit ones from meshInstances)
      if(!ui->m_assets.meshes.instances.empty() && ui->m_assets.meshes.m_lastCreatedInstance)
      {
        ui->m_assets.meshes.instances.pop_back();
      }
      assetIdToMesh[id] = mesh;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Load mesh instances and apply transforms/materials
//
void VkgsProjectReader::loadMeshInstances(const json&                                   data,
                                          int                                           fileVersion,
                                          const std::map<int, std::shared_ptr<MeshVk>>& assetIdToMesh,
                                          GaussianSplattingUI*                          ui)
{
  // Get instances array (supports both object and array formats)
  const json* instancesArray = nullptr;
  if(data.contains("meshInstances"))
  {
    if(data["meshInstances"].is_object() && data["meshInstances"].contains("items"))
      instancesArray = &data["meshInstances"]["items"];
    else if(data["meshInstances"].is_array())
      instancesArray = &data["meshInstances"];
  }

  if(!instancesArray)
    return;

  for(const auto& instItem : *instancesArray)
  {
    int  meshAssetId = instItem["meshAssetId"].get<int>();
    auto meshIt      = assetIdToMesh.find(meshAssetId);
    if(meshIt == assetIdToMesh.end())
      continue;  // Invalid reference

    auto mesh = meshIt->second;

    // Create instance for this mesh
    auto instance = ui->m_assets.meshes.createInstance(mesh);
    if(!instance)
      continue;

    // Load name if present (overrides auto-generated name)
    if(instItem.contains("name"))
    {
      instance->name = instItem["name"].get<std::string>();
    }

    if(instItem.contains("show"))
      instance->show = instItem["show"].get<bool>();

    // Load transform
    LOAD3(instance->translation, instItem, "position");
    LOAD3(instance->rotation, instItem, "rotation");
    LOAD3(instance->scale, instItem, "scale");
    computeTransform(instance->scale, instance->rotation, instance->translation, instance->transform,
                     instance->transformInverse, instance->transformRotScaleInverse);

    // Load materials (per-instance)
    if(instItem.contains("materials"))
    {
      auto matId = 0;
      for(const auto& matItem : instItem["materials"])
      {
        if(matId >= mesh->materials.size())
          break;

        auto& mat = mesh->materials[matId];
        loadMaterialFromJson(mat, matItem, fileVersion);
        matId++;
      }

      // Use deferred API - materials will be uploaded in processVramUpdates()
      ui->m_assets.meshes.updateMeshMaterials(instance->mesh);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Load meshes (assets and instances, with version-dependent format)
//
void VkgsProjectReader::loadMeshes(const json& data, int fileVersion, const std::string& projectPath, GaussianSplattingUI* ui)
{
  if(fileVersion >= 2 && data.contains("meshAssets") && data.contains("meshInstances"))
  {
    // Version 2 format: separate meshAssets and meshInstances (like splat sets)

    // Load naming counter
    if(data["meshInstances"].is_object() && data["meshInstances"].contains("nextNamingNumber"))
      ui->m_assets.meshes.m_nextInstanceNumber = data["meshInstances"]["nextNamingNumber"].get<uint32_t>();

    // Load assets and instances using helper functions
    std::map<int, std::shared_ptr<MeshVk>> assetIdToMesh;
    loadMeshAssets(data, projectPath, assetIdToMesh, ui);
    loadMeshInstances(data, fileVersion, assetIdToMesh, ui);

    ui->m_requestUpdateShaders = true;
  }
  else if(data.contains("meshes"))
  {
    // Version 0/1 format: Backward compatibility
    // Load naming counter if present
    if(data["meshes"].is_object() && data["meshes"].contains("nextNamingNumber"))
      ui->m_assets.meshes.m_nextInstanceNumber = data["meshes"]["nextNamingNumber"].get<uint32_t>();

    // Get items array (Version 1 format) or use meshes directly (Version 0 format)
    const json& meshesArray =
        data["meshes"].is_object() && data["meshes"].contains("items") ? data["meshes"]["items"] : data["meshes"];

    auto meshId = 0;
    for(const auto& item : meshesArray)
    {
      std::string relPath;
      LOAD1(relPath, item, "path");
      if(relPath.empty())
        continue;

      auto meshPath = makeAbsolutePath(std::filesystem::path(projectPath).parent_path(), relPath);
      if(!ui->m_assets.meshes.loadModel(meshPath.string()))
      {
        meshId++;
        continue;
      }

      // Access to newly created mesh/instance via last created pointer
      auto instance = ui->m_assets.meshes.m_lastCreatedInstance;
      if(!instance || !instance->mesh)
      {
        meshId++;
        continue;  // Shouldn't happen
      }

      auto& mesh = *instance->mesh;

      // Load name if present (overrides auto-generated name)
      if(item.contains("name"))
      {
        instance->name = item["name"].get<std::string>();
      }
      // else: Keep auto-generated name from createInstance()

      // Transform
      LOAD3(instance->translation, item, "position");
      LOAD3(instance->rotation, item, "rotation");
      LOAD3(instance->scale, item, "scale");
      computeTransform(instance->scale, instance->rotation, instance->translation, instance->transform,
                       instance->transformInverse, instance->transformRotScaleInverse);

      // Materials
      if(item.contains("materials"))
      {
        auto matId = 0;
        for(const auto& matItem : item["materials"])
        {
          if(matId >= mesh.materials.size())
            break;

          auto& mat = mesh.materials[matId];
          loadMaterialFromJson(mat, matItem, fileVersion);
          matId++;
        }

        // Use deferred API - materials will be uploaded in processVramUpdates()
        ui->m_assets.meshes.updateMeshMaterials(instance->mesh);
      }

      meshId++;
    }

    // Mesh manager will set its own pendingRequests (RebuildBLAS, UpdateDescriptors, etc.)
    ui->m_requestUpdateShaders = true;
  }
}

//--------------------------------------------------------------------------------------------------
// Load cameras (active camera and presets)
//
void VkgsProjectReader::loadCameras(const json& data, GaussianSplattingUI* ui)
{
  // Clear presets created during reset() — they will be replaced by project data
  ui->m_assets.cameras.clearPresets();

  // Parse active camera
  if(data.contains("camera"))
  {
    auto&  item = data["camera"];
    Camera cam;
    LOAD1(cam.model, item, "model");
    LOAD3(cam.ctr, item, "ctr");
    LOAD3(cam.eye, item, "eye");
    LOAD3(cam.up, item, "up");
    LOAD1(cam.fov, item, "fov");
    LOAD2(cam.clip, item, "clip");
    // Backward compat: old files have "dofEnabled" (bool: false=0=DOF_DISABLED, true=1=DOF_FIXED_FOCUS)
    LOAD1(cam.dofMode, item, "dofEnabled");
    // New format overrides if present
    LOAD1(cam.dofMode, item, "dofMode");
    LOAD1(cam.focusDist, item, "focusDist");
    LOAD1(cam.aperture, item, "aperture");

    ui->m_assets.cameras.setCamera(cam);
  }

  // Parse camera presets
  if(data.contains("cameras"))
  {
    for(const auto& item : data["cameras"])
    {
      Camera cam;
      LOAD1(cam.model, item, "model");
      LOAD3(cam.ctr, item, "ctr");
      LOAD3(cam.eye, item, "eye");
      LOAD3(cam.up, item, "up");
      LOAD1(cam.fov, item, "fov");
      LOAD2(cam.clip, item, "clip");
      // Backward compat: old files have "dofEnabled" (bool: false=0=DOF_DISABLED, true=1=DOF_FIXED_FOCUS)
      LOAD1(cam.dofMode, item, "dofEnabled");
      // New format overrides if present
      LOAD1(cam.dofMode, item, "dofMode");
      LOAD1(cam.focusDist, item, "focusDist");
      LOAD1(cam.aperture, item, "aperture");

      ui->m_assets.cameras.createPreset(cam);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Load lights (assets and instances, with version-dependent format)
//
void VkgsProjectReader::loadLights(const json& data, int fileVersion, GaussianSplattingUI* ui)
{
  if(!data.contains("lights"))
    return;

  // Load naming counter if present
  if(data["lights"].is_object() && data["lights"].contains("nextNamingNumber"))
    ui->m_assets.lights.m_nextLightNumber = data["lights"]["nextNamingNumber"].get<uint32_t>();

  // Version 3+: asset/instance format
  if(fileVersion >= 3 && data["lights"].is_object() && data["lights"].contains("assets") && data["lights"].contains("instances"))
  {
    // Build map of asset ID to asset JSON data
    std::map<int, json> assetIdToJsonData;
    for(const auto& assetItem : data["lights"]["assets"])
    {
      int id                = assetItem["id"].get<int>();
      assetIdToJsonData[id] = assetItem;
    }

    // Create lights: first instance of each asset uses createLight(), rest use duplicateInstance()
    std::map<int, std::shared_ptr<LightSourceInstanceVk>> assetIdToFirstInstance;

    for(const auto& instanceItem : data["lights"]["instances"])
    {
      int assetId = instanceItem["assetId"].get<int>();

      std::shared_ptr<LightSourceInstanceVk> instance;

      // First instance of this asset - create new light
      if(assetIdToFirstInstance.find(assetId) == assetIdToFirstInstance.end())
      {
        // Create light (creates asset + instance + proxies)
        instance                        = ui->m_assets.lights.createLight();
        assetIdToFirstInstance[assetId] = instance;

        // Load asset data (shared across instances)
        const auto& assetItem = assetIdToJsonData[assetId];
        LOAD1(instance->lightSource->type, assetItem, "type");
        LOAD3(instance->lightSource->color, assetItem, "color");
        LOAD1(instance->lightSource->intensity, assetItem, "intensity");

        // Version 4+: new fields
        if(fileVersion >= 4)
        {
          LOAD1(instance->lightSource->range, assetItem, "range");
          LOAD1(instance->lightSource->innerConeAngle, assetItem, "innerConeAngle");
          LOAD1(instance->lightSource->outerConeAngle, assetItem, "outerConeAngle");
          LOAD1(instance->lightSource->attenuationMode, assetItem, "attenuationMode");
          if(assetItem.contains("radius"))
            LOAD1(instance->lightSource->radius, assetItem, "radius");
          else if(assetItem.contains("proxyScale"))
            LOAD1(instance->lightSource->radius, assetItem, "proxyScale");
          if(assetItem.contains("enabled"))
            LOAD1(instance->lightSource->enabled, assetItem, "enabled");
        }
        else  // Version 3: backward compatibility
        {
          // Map old "radius" to new "range"
          if(assetItem.contains("radius"))
            LOAD1(instance->lightSource->range, assetItem, "radius");
          else
            instance->lightSource->range = 10.0f;

          // Map old "scale" to radius
          if(assetItem.contains("scale"))
            LOAD1(instance->lightSource->radius, assetItem, "scale");
          else
            instance->lightSource->radius = 1.0f;

          // Set defaults for new fields
          instance->lightSource->innerConeAngle  = 30.0f;
          instance->lightSource->outerConeAngle  = 45.0f;
          instance->lightSource->attenuationMode = 2;  // Quadratic
        }
      }
      else
      {
        // Duplicate from first instance (shares asset)
        instance = ui->m_assets.lights.duplicateInstance(assetIdToFirstInstance[assetId]);
      }

      // Load instance-specific data
      if(instanceItem.contains("name"))
        instance->name = instanceItem["name"].get<std::string>();

      // Version 4+: translation and rotation
      if(fileVersion >= 4)
      {
        LOAD3(instance->translation, instanceItem, "translation");
        LOAD3(instance->rotation, instanceItem, "rotation");
      }
      else  // Version 3: backward compatibility
      {
        // Map old "position" to new "translation"
        if(instanceItem.contains("position"))
          LOAD3(instance->translation, instanceItem, "position");
        else
          instance->translation = glm::vec3(0.0f, 2.0f, 0.0f);

        // Set default rotation
        instance->rotation = glm::vec3(0.0f);
      }

      // Update light to reflect loaded position
      ui->m_assets.lights.updateLight(instance);
    }

    // Update asset colors/materials (affects all instances sharing each asset)
    for(auto& [assetId, firstInstance] : assetIdToFirstInstance)
    {
      ui->m_assets.lights.updateLightAsset(firstInstance->lightSource);
    }
  }
  else
  {
    // Version 0-2: flat format (backward compatibility)
    const json& lightsArray =
        data["lights"].is_object() && data["lights"].contains("items") ? data["lights"]["items"] : data["lights"];

    for(const auto& item : lightsArray)
    {
      // Create a new light (asset + instance)
      auto instance = ui->m_assets.lights.createLight();

      // Load name if present (overrides auto-generated name)
      if(item.contains("name"))
      {
        instance->name = item["name"].get<std::string>();
      }

      // Load light data (asset vs instance separation)
      auto& asset = instance->lightSource;
      LOAD1(asset->type, item, "type");
      LOAD3(instance->translation, item, "position");  // Old format: "position" maps to "translation"
      instance->rotation = glm::vec3(0.0f);            // Default rotation for old files
      LOAD3(asset->color, item, "color");
      LOAD1(asset->intensity, item, "intensity");

      // Old format: "radius" maps to "range"
      if(item.contains("radius"))
        LOAD1(asset->range, item, "radius");
      else
        asset->range = 10.0f;

      // Set defaults for Version 4 fields
      asset->innerConeAngle  = 30.0f;
      asset->outerConeAngle  = 45.0f;
      asset->attenuationMode = 2;  // Quadratic
      asset->radius          = 1.0f;

      if(!item.contains("color"))
      {
        asset->color = glm::vec3(1.0f);
      }

      // Mark this light as updated
      ui->m_assets.lights.updateLight(instance);
    }
    // pendingRequests set by createLight() and updateLight()
  }
}


void VkgsProjectReader::loadEnvironment(const json& data, const std::string& projectPath, GaussianSplattingUI* ui)
{
  if(!data.contains("environment"))
    return;

  const json& env = data["environment"];
  auto&       sky = ui->m_sky;

  if(env.contains("mode"))
    sky.setMode(static_cast<shaderio::EnvironmentMode>(env["mode"].get<int>()));
  if(env.contains("enabled"))
    sky.setEnabled(env["enabled"].get<bool>());
  if(env.contains("resolution") && env["resolution"].is_array() && env["resolution"].size() == 2)
    sky.setResolution(glm::ivec2(env["resolution"][0].get<int>(), env["resolution"][1].get<int>()));

  // Sky & Sun parameters
  if(env.contains("skyAndSun"))
  {
    const json&                      sun = env["skyAndSun"];
    shaderio::SkyPhysicalParameters& sp  = sky.skyParams();

    if(sun.contains("sunDirection") && sun["sunDirection"].is_array())
      sp.sunDirection = {sun["sunDirection"][0].get<float>(), sun["sunDirection"][1].get<float>(),
                         sun["sunDirection"][2].get<float>()};
    if(sun.contains("sunDiskScale"))
      sp.sunDiskScale = sun["sunDiskScale"].get<float>();
    if(sun.contains("sunDiskIntensity"))
      sp.sunDiskIntensity = sun["sunDiskIntensity"].get<float>();
    if(sun.contains("sunGlowIntensity"))
      sp.sunGlowIntensity = sun["sunGlowIntensity"].get<float>();
    if(sun.contains("haze"))
      sp.haze = sun["haze"].get<float>();
    if(sun.contains("redblueshift"))
      sp.redblueshift = sun["redblueshift"].get<float>();
    if(sun.contains("saturation"))
      sp.saturation = sun["saturation"].get<float>();
    if(sun.contains("horizonHeight"))
      sp.horizonHeight = sun["horizonHeight"].get<float>();
    if(sun.contains("groundColor") && sun["groundColor"].is_array())
      sp.groundColor = {sun["groundColor"][0].get<float>(), sun["groundColor"][1].get<float>(),
                        sun["groundColor"][2].get<float>()};
    if(sun.contains("horizonBlur"))
      sp.horizonBlur = sun["horizonBlur"].get<float>();
    if(sun.contains("nightColor") && sun["nightColor"].is_array())
      sp.nightColor = {sun["nightColor"][0].get<float>(), sun["nightColor"][1].get<float>(), sun["nightColor"][2].get<float>()};
  }

  // IBL parameters
  if(env.contains("ibl"))
  {
    const json& ibl = env["ibl"];

    if(ibl.contains("intensity"))
      sky.setIblIntensity(ibl["intensity"].get<float>());
    if(ibl.contains("rotation") && ibl["rotation"].is_array())
      sky.setIblRotation(
          glm::vec3(ibl["rotation"][0].get<float>(), ibl["rotation"][1].get<float>(), ibl["rotation"][2].get<float>()));

    if(ibl.contains("file") && !ibl["file"].get<std::string>().empty())
    {
      std::filesystem::path projDir = std::filesystem::path(projectPath).parent_path();
      std::filesystem::path hdrPath = makeAbsolutePath(projDir, ibl["file"].get<std::string>());

      if(std::filesystem::exists(hdrPath) && sky.mode() == shaderio::EnvironmentMode::eHDR)
      {
        VkCommandBuffer cmd = ui->m_app->createTempCmdBuffer();
        sky.loadHdrEnvironment(cmd, hdrPath);
        ui->m_app->submitAndWaitTempCmdBuffer(cmd);
      }
    }
  }

  sky.setDirty();
}

//--------------------------------------------------------------------------------------------------
// Load settings (navigation, visual helpers)
//
void VkgsProjectReader::loadSettings(const json& data, GaussianSplattingUI* ui)
{
  if(!data.contains("settings"))
    return;

  const auto& settings = data["settings"];

  // Navigation
  if(settings.contains("navigation"))
  {
    const auto& nav   = settings["navigation"];
    auto*       manip = ui->cameraManip.get();
    if(nav.contains("mode"))
      manip->setMode(static_cast<nvutils::CameraManipulator::Modes>(nav["mode"].get<int>()));
    if(nav.contains("speed"))
      manip->setSpeed(nav["speed"].get<float>());
    if(nav.contains("transition"))
      manip->setAnimationDuration(nav["transition"].get<float>());
    if(nav.contains("autoPlay"))
    {
      ui->m_autoPlayPresets = nav["autoPlay"].get<bool>();
      if(ui->m_autoPlayPresets)
        ui->m_playPresets = true;
    }
  }

  // Transform helpers
  if(settings.contains("transformHelpers"))
  {
    const auto& transform = settings["transformHelpers"];
    if(transform.contains("show"))
      ui->m_helpers.setEditingMode(transform["show"].get<bool>());
    if(transform.contains("snapEnabled"))
      ui->m_helpers.transform.setSnapEnabled(transform["snapEnabled"].get<bool>());
    if(transform.contains("snapTranslate") && transform.contains("snapRotate") && transform.contains("snapScale"))
    {
      ui->m_helpers.transform.setSnapValues(transform["snapTranslate"].get<float>(),
                                            transform["snapRotate"].get<float>(), transform["snapScale"].get<float>());
    }
  }

  // Grid
  if(settings.contains("grid"))
  {
    const auto& grid = settings["grid"];
    if(grid.contains("show"))
      ui->m_helpers.grid.setVisible(grid["show"].get<bool>());
  }

  // Light proxies
  if(settings.contains("lightProxies"))
  {
    const auto& lp = settings["lightProxies"];
    if(lp.contains("show"))
      ui->m_showLightProxies = lp["show"].get<bool>();
  }

  // Summary info overlay
  if(settings.contains("summaryOverlay"))
  {
    const auto& overlay = settings["summaryOverlay"];
    if(overlay.contains("show"))
      ui->m_showSummaryOverlay = overlay["show"].get<bool>();
  }
}

//--------------------------------------------------------------------------------------------------
// Load tonemapping parameters
//
void VkgsProjectReader::loadTonemapping(const json& data, GaussianSplattingUI* ui)
{
  if(!data.contains("tonemapping"))
    return;

  const auto& item = data["tonemapping"];
  auto&       tm   = ui->m_tonemapperData;

  LOAD1(tm.isActive, item, "isActive");
  LOAD1(tm.method, item, "method");
  LOAD1(tm.exposure, item, "exposure");
  LOAD1(tm.temperature, item, "temperature");
  LOAD1(tm.tint, item, "tint");

  LOAD1(tm.contrast, item, "contrast");
  LOAD1(tm.brightness, item, "brightness");
  LOAD1(tm.saturation, item, "saturation");
  LOAD1(tm.vignette, item, "vignette");

  LOAD1(tm.vibrance, item, "vibrance");
  LOAD1(tm.shadowBias, item, "shadowBias");
  LOAD1(tm.midtoneBias, item, "midtoneBias");
  LOAD1(tm.highlightBias, item, "highlightBias");
  LOAD3(tm.coolColor, item, "coolColor");
  LOAD3(tm.warmColor, item, "warmColor");
  LOAD1(tm.splitBalance, item, "splitBalance");

  LOAD1(tm.autoExposure, item, "autoExposure");
  LOAD1(tm.autoExposureSpeed, item, "autoExposureSpeed");
  LOAD1(tm.evMinValue, item, "evMinValue");
  LOAD1(tm.evMaxValue, item, "evMaxValue");
  LOAD1(tm.enableCenterMetering, item, "enableCenterMetering");
  LOAD1(tm.centerMeteringSize, item, "centerMeteringSize");
  LOAD1(tm.averageMode, item, "averageMode");

  LOAD1(tm.dither, item, "dither");
}

}  // namespace vk_gaussian_splatting
