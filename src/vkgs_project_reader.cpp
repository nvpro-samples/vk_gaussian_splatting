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

#include <filesystem>
#include <tinygltf/json.hpp>

using json   = nlohmann::json;
namespace fs = std::filesystem;

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// Helper macros to load JSON values only if they exist
//

#define LOAD1(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = (item)[name]

#define LOAD2(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1]}

#define LOAD3(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1], (item)[name][2]}

//--------------------------------------------------------------------------------------------------
// Helper function to convert relative path to absolute
//
static std::filesystem::path makeAbsolutePath(const std::filesystem::path& base, const std::string& relativePath)
{
  return std::filesystem::absolute(base / relativePath);
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

  // Lighting and shadows mode (enum class, need explicit cast from int)
  if(item.contains("lightingMode"))
    prmRender.lightingMode = (LightingMode)item["lightingMode"].get<int>();
  if(item.contains("shadowsMode"))
    prmRender.shadowsMode = (ShadowsMode)item["shadowsMode"].get<int>();

  // Backward compat: migrate old boolean fields to new enum modes
  if(!item.contains("lightingMode") && item.contains("lightingEnabled"))
    prmRender.lightingMode = item["lightingEnabled"].get<bool>() ? LightingMode::eLightingIndirect : LightingMode::eLightingDisabled;
  if(!item.contains("shadowsMode") && item.contains("shadowsEnabled"))
    prmRender.shadowsMode = item["shadowsEnabled"].get<bool>() ? ShadowsMode::eShadowsHard : ShadowsMode::eShadowsDisabled;
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
  LOAD1(prmRtxData.useTlasInstances, item, "useTlasInstances");
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

    // Parse and set material
    if(item.contains("material"))
    {
      LOAD3(instance->splatMaterial.ambient, item["material"], "ambient");
      LOAD3(instance->splatMaterial.diffuse, item["material"], "diffuse");
      LOAD3(instance->splatMaterial.specular, item["material"], "specular");
      LOAD3(instance->splatMaterial.emission, item["material"], "emission");
      LOAD1(instance->splatMaterial.shininess, item["material"], "shininess");
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

      // Parse and set material
      if(item.contains("material"))
      {
        LOAD3(instance->splatMaterial.ambient, item["material"], "ambient");
        LOAD3(instance->splatMaterial.diffuse, item["material"], "diffuse");
        LOAD3(instance->splatMaterial.specular, item["material"], "specular");
        LOAD3(instance->splatMaterial.emission, item["material"], "emission");
        LOAD1(instance->splatMaterial.shininess, item["material"], "shininess");
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
void VkgsProjectReader::loadMeshInstances(const json& data, const std::map<int, std::shared_ptr<MeshVk>>& assetIdToMesh, GaussianSplattingUI* ui)
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
        LOAD3(mat.ambient, matItem, "ambient");
        LOAD3(mat.diffuse, matItem, "diffuse");
        LOAD3(mat.emission, matItem, "emission");
        LOAD1(mat.illum, matItem, "illum");
        LOAD1(mat.ior, matItem, "ior");
        LOAD1(mat.shininess, matItem, "shininess");
        LOAD3(mat.specular, matItem, "specular");
        LOAD3(mat.transmittance, matItem, "transmittance");

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
    loadMeshInstances(data, assetIdToMesh, ui);

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
          LOAD3(mat.ambient, matItem, "ambient");
          LOAD3(mat.diffuse, matItem, "diffuse");
          LOAD3(mat.emission, matItem, "emission");
          LOAD1(mat.illum, matItem, "illum");
          LOAD1(mat.ior, matItem, "ior");
          LOAD1(mat.shininess, matItem, "shininess");
          LOAD3(mat.specular, matItem, "specular");
          LOAD3(mat.transmittance, matItem, "transmittance");

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
          LOAD1(instance->lightSource->proxyScale, assetItem, "proxyScale");
        }
        else  // Version 3: backward compatibility
        {
          // Map old "radius" to new "range"
          if(assetItem.contains("radius"))
            LOAD1(instance->lightSource->range, assetItem, "radius");
          else
            instance->lightSource->range = 10.0f;

          // Map old "scale" to new "proxyScale"
          if(assetItem.contains("scale"))
            LOAD1(instance->lightSource->proxyScale, assetItem, "scale");
          else
            instance->lightSource->proxyScale = 1.0f;

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
      asset->proxyScale      = 1.0f;

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


}  // namespace vk_gaussian_splatting
