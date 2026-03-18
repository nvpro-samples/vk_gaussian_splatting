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

#include "vkgs_project_writer.h"
#include "gaussian_splatting_ui.h"
#include "parameters.h"

#include <nvutils/file_operations.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <tinygltf/json.hpp>

using json   = nlohmann::json;
namespace fs = std::filesystem;

namespace vk_gaussian_splatting {

// Project file format version
constexpr int PROJECT_FILE_VERSION = 5;

//--------------------------------------------------------------------------------------------------
// Helper function to compute relative path from one directory to another
//
static fs::path getRelativePath(const fs::path& from, const fs::path& to)
{
  fs::path relativePath;

  auto fromIter = from.begin();
  auto toIter   = to.begin();

  // Find common point
  while(fromIter != from.end() && toIter != to.end() && (*fromIter) == (*toIter))
  {
    ++fromIter;
    ++toIter;
  }

  // Add ".." for each remaining part in `from` path
  for(; fromIter != from.end(); ++fromIter)
  {
    relativePath /= "..";
  }

  // Add remaining part of `to` path
  for(; toIter != to.end(); ++toIter)
  {
    relativePath /= *toIter;
  }

  return relativePath;
}

//--------------------------------------------------------------------------------------------------
// Save project to file
//
bool VkgsProjectWriter::saveProject(const std::string& path, const GaussianSplattingUI* ui)
{
  // Ensure the extension is always ".vkgs" (lowercase)
  std::filesystem::path savePath(path);
  if(!nvutils::extensionMatches(savePath, ".vkgs"))
  {
    savePath = savePath.replace_extension(".vkgs");
  }

  std::ofstream o(savePath);
  if(!o.is_open())
    return false;

  try
  {
    json data;

    // File format version (must be first for easy identification)
    data["version"] = PROJECT_FILE_VERSION;

    // Save all sections using helper functions
    saveRendererSettings(data, ui);
    saveActiveCamera(data, ui);
    saveCameraPresets(data, ui);
    saveLights(data, ui);
    saveSplatGlobalOptions(data);
    saveSplatSets(data, ui, path);
    saveSplatInstances(data, ui);
    saveMeshes(data, ui, path);

    // Write JSON to file
    o << std::setw(4) << data << std::endl;
    o.close();
    return true;
  }
  catch(...)
  {
    return false;
  }
}

//--------------------------------------------------------------------------------------------------
// Save renderer settings (vsync, pipeline, rendering parameters)
//
void VkgsProjectWriter::saveRendererSettings(json& data, const GaussianSplattingUI* ui)
{
  json item;
  item["vsync"]                   = ui->m_app->isVsync();
  item["pipeline"]                = prmSelectedPipeline;
  item["maxShDegree"]             = prmFrame.shDegree;
  item["opacityGaussianDisabled"] = prmRender.opacityGaussianDisabled;
  item["showShOnly"]              = prmRender.showShOnly;
  item["visualize"]               = prmRender.visualize;
  item["wireframe"]               = prmRender.wireframe;
  item["cpuLazySort"]             = prmRaster.cpuLazySort;
  item["distShaderWorkgroupSize"] = prmRaster.distShaderWorkgroupSize;
  item["fragmentBarycentric"]     = prmRaster.fragmentBarycentric;
  item["frustumCulling"]          = prmRaster.frustumCulling;
  item["sizeCulling"]             = prmRaster.sizeCulling;
  item["sizeCullingMinPixels"]    = prmFrame.sizeCullingMinPixels;
  item["meshShaderWorkgroupSize"] = prmRaster.meshShaderWorkgroupSize;
  item["pointCloudModeEnabled"]   = prmRaster.pointCloudModeEnabled;
  item["sortingMethod"]           = prmRaster.sortingMethod;
  item["temporalSampling"]        = prmRtx.temporalSampling;
  item["temporalSamplesCount"]    = prmFrame.frameSampleMax;
  item["kernelAdaptiveClamping"]  = prmRtx.kernelAdaptiveClamping;
  item["kernelDegree"]            = prmRtx.kernelDegree;
  item["kernelMinResponse"]       = prmRtx.kernelMinResponse;
  item["particleSamplesPerPass"]  = prmRtx.particleSamplesPerPass;
  item["rtxTraceStrategy"]        = prmRtx.rtxTraceStrategy;
  item["normalMethod"]            = (int)prmRender.normalMethod;
  item["thinParticleThreshold"]   = prmRender.thinParticleThreshold;
  item["lightingMode"]            = (int)prmRender.lightingMode;
  item["shadowsMode"]             = (int)prmRender.shadowsMode;

  data["renderer"] = item;
}

//--------------------------------------------------------------------------------------------------
// Save active camera state
//
void VkgsProjectWriter::saveActiveCamera(json& data, const GaussianSplattingUI* ui)
{
  auto cam = const_cast<GaussianSplattingUI*>(ui)->m_assets.cameras.getCamera();
  json item;
  item["model"]     = cam.model;
  item["ctr"]       = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
  item["eye"]       = {cam.eye.x, cam.eye.y, cam.eye.z};
  item["up"]        = {cam.up.x, cam.up.y, cam.up.z};
  item["fov"]       = cam.fov;
  item["clip"]      = {cam.clip.x, cam.clip.y};
  item["dofMode"]   = cam.dofMode;
  item["focusDist"] = cam.focusDist;
  item["aperture"]  = cam.aperture;

  data["camera"] = item;
}

//--------------------------------------------------------------------------------------------------
// Save camera presets
//
void VkgsProjectWriter::saveCameraPresets(json& data, const GaussianSplattingUI* ui)
{
  data["cameras"] = json::array();
  for(auto camId = 0; camId < ui->m_assets.cameras.size(); ++camId)
  {
    auto cam = ui->m_assets.cameras.getPreset(camId);
    json item;
    item["model"]     = cam.model;
    item["ctr"]       = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
    item["eye"]       = {cam.eye.x, cam.eye.y, cam.eye.z};
    item["up"]        = {cam.up.x, cam.up.y, cam.up.z};
    item["fov"]       = cam.fov;
    item["clip"]      = {cam.clip.x, cam.clip.y};
    item["dofMode"]   = cam.dofMode;
    item["focusDist"] = cam.focusDist;
    item["aperture"]  = cam.aperture;

    data["cameras"].push_back(item);
  }
}

//--------------------------------------------------------------------------------------------------
// Save lights (assets and instances)
//
void VkgsProjectWriter::saveLights(json& data, const GaussianSplattingUI* ui)
{
  data["lights"]                     = json::object();
  data["lights"]["nextNamingNumber"] = ui->m_assets.lights.m_nextLightNumber;

  // Collect unique light assets and assign IDs
  std::map<std::shared_ptr<LightSourceVk>, int> assetToId;
  int                                           nextAssetId = 0;

  json assetsArray    = json::array();
  json instancesArray = json::array();

  for(const auto& instance : ui->m_assets.lights.instances)
  {
    if(!instance || !instance->lightSource)
      continue;

    // Save asset if not already saved
    if(assetToId.find(instance->lightSource) == assetToId.end())
    {
      assetToId[instance->lightSource] = nextAssetId++;

      json assetItem;
      assetItem["id"]   = assetToId[instance->lightSource];
      assetItem["type"] = instance->lightSource->type;
      assetItem["color"] = {instance->lightSource->color.x, instance->lightSource->color.y, instance->lightSource->color.z};
      assetItem["intensity"]       = instance->lightSource->intensity;
      assetItem["range"]           = instance->lightSource->range;
      assetItem["innerConeAngle"]  = instance->lightSource->innerConeAngle;
      assetItem["outerConeAngle"]  = instance->lightSource->outerConeAngle;
      assetItem["attenuationMode"] = instance->lightSource->attenuationMode;
      assetItem["proxyScale"]      = instance->lightSource->proxyScale;

      assetsArray.push_back(assetItem);
    }

    // Save instance
    json instanceItem;
    instanceItem["assetId"]     = assetToId[instance->lightSource];
    instanceItem["name"]        = instance->name;
    instanceItem["translation"] = {instance->translation.x, instance->translation.y, instance->translation.z};
    instanceItem["rotation"]    = {instance->rotation.x, instance->rotation.y, instance->rotation.z};

    instancesArray.push_back(instanceItem);
  }

  data["lights"]["assets"]    = assetsArray;
  data["lights"]["instances"] = instancesArray;
}

//--------------------------------------------------------------------------------------------------
// Save splat global options (data storage, compression, etc.)
//
void VkgsProjectWriter::saveSplatGlobalOptions(json& data)
{
  json item;
  // item["dataStorage"]      = prmData.dataStorage;
  item["shFormat"]         = prmData.shFormat;
  item["rgbaFormat"]       = prmData.rgbaFormat;
  item["compressBlas"]     = prmRtxData.compressBlas;
  item["useAABBs"]         = prmRtxData.useAABBs;
  item["useTlasInstances"] = prmRtxData.useTlasInstances;

  data["splatsGlobals"] = item;
}

//--------------------------------------------------------------------------------------------------
// Save splat sets (assets)
//
void VkgsProjectWriter::saveSplatSets(json& data, const GaussianSplattingUI* ui, const std::string& projectPath)
{
  data["splatSets"] = json::array();

  for(const auto& splatSet : ui->m_assets.splatSets.getSplatSets())
  {
    if(!splatSet)
      continue;

    json item;
    item["id"] = static_cast<int>(splatSet->index);
    item["path"] = getRelativePath(std::filesystem::path(projectPath).parent_path(), std::filesystem::path(splatSet->path));
    // Note: 'name' removed from SplatSet - now only stored in instances
    item["storage"]    = splatSet->getStorage();
    item["shFormat"]   = splatSet->getShFormat();
    item["rgbaFormat"] = splatSet->getRgbaFormat();

    data["splatSets"].push_back(item);
  }
}

//--------------------------------------------------------------------------------------------------
// Save splat instances
//
void VkgsProjectWriter::saveSplatInstances(json& data, const GaussianSplattingUI* ui)
{
  data["splats"] = json::array();

  for(const auto& instance : ui->m_assets.splatSets.getInstances())
  {
    if(!instance || !instance->shouldShowInUI() || !instance->splatSet)
      continue;

    json item;
    item["splatSetId"] = static_cast<int>(instance->splatSet->index);
    item["name"]       = instance->displayName;
    item["position"]   = {instance->translation.x, instance->translation.y, instance->translation.z};
    item["rotation"]   = {instance->rotation.x, instance->rotation.y, instance->rotation.z};
    item["scale"]      = {instance->scale.x, instance->scale.y, instance->scale.z};

    // Save splat material
    item["material"]["ambient"]   = {instance->splatMaterial.ambient.x, instance->splatMaterial.ambient.y,
                                     instance->splatMaterial.ambient.z};
    item["material"]["diffuse"]   = {instance->splatMaterial.diffuse.x, instance->splatMaterial.diffuse.y,
                                     instance->splatMaterial.diffuse.z};
    item["material"]["specular"]  = {instance->splatMaterial.specular.x, instance->splatMaterial.specular.y,
                                     instance->splatMaterial.specular.z};
    item["material"]["emission"]  = {instance->splatMaterial.emission.x, instance->splatMaterial.emission.y,
                                     instance->splatMaterial.emission.z};
    item["material"]["shininess"] = instance->splatMaterial.shininess;

    data["splats"].push_back(item);
  }
}

//--------------------------------------------------------------------------------------------------
// Save meshes (assets and instances)
//
void VkgsProjectWriter::saveMeshes(json& data, const GaussianSplattingUI* ui, const std::string& projectPath)
{
  // Build map of unique mesh assets from instances
  std::map<MeshVk*, int> meshToId;
  std::vector<MeshVk*>   uniqueMeshes;
  int                    meshAssetId = 0;

  for(const auto& instance : ui->m_assets.meshes.instances)
  {
    if(!instance || !instance->mesh)
      continue;
    if(instance->type != shaderio::MeshType::eObject)
      continue;  // Skip internal meshes

    MeshVk* meshPtr = instance->mesh.get();
    if(meshToId.find(meshPtr) == meshToId.end())
    {
      meshToId[meshPtr] = meshAssetId++;
      uniqueMeshes.push_back(meshPtr);
    }
  }

  // Save unique mesh assets (just id and path)
  data["meshAssets"] = json::array();
  for(size_t i = 0; i < uniqueMeshes.size(); ++i)
  {
    const auto& mesh = *uniqueMeshes[i];
    json        item;
    item["id"]   = static_cast<int>(i);
    item["path"] = getRelativePath(std::filesystem::path(projectPath).parent_path(), mesh.path);

    data["meshAssets"].push_back(item);
  }

  // Save mesh instances (reference asset by id)
  json instancesArray = json::array();
  for(const auto& instance : ui->m_assets.meshes.instances)
  {
    if(!instance || !instance->mesh)
      continue;
    if(!instance->shouldShowInUI())
      continue;
    if(instance->type != shaderio::MeshType::eObject)
      continue;

    const auto& mesh = *instance->mesh;

    json item;
    item["meshAssetId"] = meshToId[instance->mesh.get()];
    item["name"]        = instance->name;

    // Transform
    item["position"] = {instance->translation.x, instance->translation.y, instance->translation.z};
    item["rotation"] = {instance->rotation.x, instance->rotation.y, instance->rotation.z};
    item["scale"]    = {instance->scale.x, instance->scale.y, instance->scale.z};

    // Materials (per-instance, may be base or overridden)
    item["materials"] = json::array();
    for(auto matId = 0; matId < mesh.matNames.size(); ++matId)
    {
      json        matItem;
      const auto& name = mesh.matNames[matId];
      const auto& mat  = mesh.materials[matId];

      matItem["name"]          = name;
      matItem["ambient"]       = {mat.ambient.x, mat.ambient.y, mat.ambient.z};
      matItem["diffuse"]       = {mat.diffuse.x, mat.diffuse.y, mat.diffuse.z};
      matItem["illum"]         = mat.illum;
      matItem["ior"]           = mat.ior;
      matItem["shininess"]     = mat.shininess;
      matItem["specular"]      = {mat.specular.x, mat.specular.y, mat.specular.z};
      matItem["transmittance"] = {mat.transmittance.x, mat.transmittance.y, mat.transmittance.z};

      item["materials"].push_back(matItem);
    }

    instancesArray.push_back(item);
  }

  data["meshInstances"]                     = json::object();
  data["meshInstances"]["nextNamingNumber"] = ui->m_assets.meshes.m_nextInstanceNumber;
  data["meshInstances"]["items"]            = instancesArray;
}

}  // namespace vk_gaussian_splatting
