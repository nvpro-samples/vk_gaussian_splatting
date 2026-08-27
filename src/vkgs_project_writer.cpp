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
// Version 6: PBR metallic-roughness material (baseColor, metallic, roughness, emissive, transmission, opacity)
// Version 7: Replace per-material illum with maxBounces; merge lighting modes (direct+indirect -> enabled)
constexpr int PROJECT_FILE_VERSION = 7;

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
    saveEnvironment(data, ui, path);
    saveSettings(data, ui);
    saveTonemapping(data, ui);

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
  item["fireflyClampThreshold"]   = prmRtx.fireflyClampThreshold;
#if defined(USE_DLSS)
  item["dlssMinRadianceThreshold"] = prmRtx.dlssMinRadianceThreshold;
  item["dlssEnabled"]              = ui->m_dlss.isEnabled();
  item["dlssSizeMode"]             = static_cast<int>(ui->m_dlss.getSizeMode());
#endif
  item["lightingEnabled"]                      = (prmRender.lightingEnabled != 0);
  item["shadowsMode"]                          = (int)prmRender.shadowsMode;
  item["gsShadowMask"]                         = prmRender.gsShadowMask;
  item["gsShadowMaskMin"]                      = prmRender.gsShadowMaskMin;
  item["gsShadowMaskFromParticles"]            = prmRender.gsShadowMaskFromParticles;
  item["colorFormat"]                          = (int)prmRender.colorFormat;
  item["rtxMaxBounces"]                        = prmFrame.rtxMaxBounces;
  item["rtxSecondaryRayOffset"]                = prmFrame.rtxSecondaryRayOffset;
  item["splatSetCompositeTransmittance"]       = prmFrame.minSplatSetCompositeTransmittance;
  item["temporalSamplingMode"]                 = prmRtx.temporalSamplingMode;
  item["alphaCullThreshold"]                   = prmFrame.alphaCullThreshold;
  item["splatScale"]                           = prmFrame.splatScale;
  item["alphaClamp"]                           = prmFrame.alphaClamp;
  item["minTransmittance"]                     = prmFrame.minTransmittance;
  item["maxPasses"]                            = prmFrame.maxPasses;
  item["particleDepth"]                        = prmRtx.particleDepth;
  item["billboardFrustumCulling"]              = prmRtx.billboardFrustumCulling;
  item["shortenRay"]                           = prmRtx.shortenRay;
  item["quantizeMeshPayload"]                  = prmRtx.quantizeMeshPayload;
  item["particleShadowOffset"]                 = prmRtx.particleShadowOffset;
  item["particleShadowTransmittanceThreshold"] = prmRtx.particleShadowTransmittanceThreshold;
  item["particleShadowColorStrength"]          = prmRtx.particleShadowColorStrength;
  item["particleEmissiveAoEnabled"]            = prmRtx.particleEmissiveAoEnabled;
  item["particleEmissiveAoRadius"]             = prmRtx.particleEmissiveAoRadius;
  item["particleEmissiveAoStrength"]           = prmRtx.particleEmissiveAoStrength;
  item["depthIsoThresholdRTX"]                 = prmRtx.depthIsoThresholdRTX;
  item["depthIsoThreshold"]                    = prmRaster.depthIsoThreshold;
  item["covarianceDilation"]                   = prmRaster.covarianceDilation;
  item["msAntialiasing"]                       = prmRaster.msAntialiasing;
  item["quantizeNormals"]                      = prmRaster.quantizeNormals;
  item["ftbSyncMode"]                          = prmRaster.ftbSyncMode;
  item["extentProjection"]                     = prmRaster.extentProjection;

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
      assetItem["radius"]          = instance->lightSource->radius;
      assetItem["enabled"]         = instance->lightSource->enabled;
      assetItem["shadowOnly"]      = instance->lightSource->shadowOnly;
      assetItem["castOnGs"]        = instance->lightSource->castOnGs;

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
  item["compressBlas"]          = prmRtxData.compressBlas;
  item["useAABBs"]              = prmRtxData.useAABBs;
  item["useSpheres"]            = prmRtxData.useSpheres;
  item["useTlasInstances"]      = prmRtxData.useTlasInstances;
  item["billboardBoundingMode"] = (int)prmRtxData.billboardBoundingMode;

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
    item["show"]       = instance->show;
    item["position"]   = {instance->translation.x, instance->translation.y, instance->translation.z};
    item["rotation"]   = {instance->rotation.x, instance->rotation.y, instance->rotation.z};
    item["scale"]      = {instance->scale.x, instance->scale.y, instance->scale.z};

    // Save splat material (PBR metallic-roughness)
    item["material"]["baseColor"]           = {instance->splatMaterial.baseColor.x, instance->splatMaterial.baseColor.y,
                                               instance->splatMaterial.baseColor.z};
    item["material"]["metallic"]            = instance->splatMaterial.metallic;
    item["material"]["roughness"]           = instance->splatMaterial.roughness;
    item["material"]["emissive"]            = {instance->splatMaterial.emissive.x, instance->splatMaterial.emissive.y,
                                               instance->splatMaterial.emissive.z};
    item["material"]["emissiveStrength"]    = instance->splatMaterial.emissiveStrength;
    item["material"]["ior"]                 = instance->splatMaterial.ior;
    item["material"]["transmission"]        = instance->splatMaterial.transmission;
    item["material"]["opacity"]             = instance->splatMaterial.opacity;
    item["material"]["maxBounces"]          = instance->splatMaterial.maxBounces;
    item["material"]["specularFactor"]      = instance->splatMaterial.specularFactor;
    item["material"]["specularColorFactor"] = {instance->splatMaterial.specularColorFactor.x,
                                               instance->splatMaterial.specularColorFactor.y,
                                               instance->splatMaterial.specularColorFactor.z};
    item["material"]["clearcoatFactor"]     = instance->splatMaterial.clearcoatFactor;
    item["material"]["clearcoatRoughness"]  = instance->splatMaterial.clearcoatRoughness;

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
    item["show"]        = instance->show;

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

      matItem["name"]             = name;
      matItem["baseColor"]        = {mat.baseColor.x, mat.baseColor.y, mat.baseColor.z};
      matItem["metallic"]         = mat.metallic;
      matItem["roughness"]        = mat.roughness;
      matItem["emissive"]         = {mat.emissive.x, mat.emissive.y, mat.emissive.z};
      matItem["emissiveStrength"] = mat.emissiveStrength;
      matItem["maxBounces"]       = mat.maxBounces;
      matItem["ior"]              = mat.ior;
      matItem["transmission"]     = mat.transmission;
      matItem["opacity"]          = mat.opacity;

      item["materials"].push_back(matItem);
    }

    instancesArray.push_back(item);
  }

  data["meshInstances"]                     = json::object();
  data["meshInstances"]["nextNamingNumber"] = ui->m_assets.meshes.m_nextInstanceNumber;
  data["meshInstances"]["items"]            = instancesArray;
}

void VkgsProjectWriter::saveEnvironment(json& data, const GaussianSplattingUI* ui, const std::string& projectPath)
{
  const auto& sky = const_cast<GaussianSplattingUI*>(ui)->m_sky;
  json        env;

  env["mode"]       = static_cast<int>(sky.mode());
  env["enabled"]    = sky.isEnabled();
  env["resolution"] = {sky.resolution().x, sky.resolution().y};

  // Sky & Sun parameters (always saved)
  const auto& sp = sky.skyParams();
  json        sun;
  sun["sunDirection"]     = {sp.sunDirection.x, sp.sunDirection.y, sp.sunDirection.z};
  sun["sunDiskScale"]     = sp.sunDiskScale;
  sun["sunDiskIntensity"] = sp.sunDiskIntensity;
  sun["sunGlowIntensity"] = sp.sunGlowIntensity;
  sun["haze"]             = sp.haze;
  sun["redblueshift"]     = sp.redblueshift;
  sun["saturation"]       = sp.saturation;
  sun["horizonHeight"]    = sp.horizonHeight;
  sun["groundColor"]      = {sp.groundColor.x, sp.groundColor.y, sp.groundColor.z};
  sun["horizonBlur"]      = sp.horizonBlur;
  sun["nightColor"]       = {sp.nightColor.x, sp.nightColor.y, sp.nightColor.z};
  env["skyAndSun"]        = sun;

  // IBL parameters (always saved)
  json ibl;
  if(!sky.iblFilePath().empty())
  {
    fs::path projDir = fs::path(projectPath).parent_path();
    ibl["file"]      = getRelativePath(projDir, sky.iblFilePath()).string();
  }
  else
  {
    ibl["file"] = "";
  }
  ibl["intensity"] = sky.iblIntensity();
  ibl["rotation"]  = {sky.iblRotation().x, sky.iblRotation().y, sky.iblRotation().z};
  env["ibl"]       = ibl;

  data["environment"] = env;
}

//--------------------------------------------------------------------------------------------------
// Save settings (navigation, visual helpers)
//
void VkgsProjectWriter::saveSettings(json& data, const GaussianSplattingUI* ui)
{
  json settings;

  // Navigation
  {
    json  nav;
    auto* manip            = const_cast<GaussianSplattingUI*>(ui)->cameraManip.get();
    nav["mode"]            = static_cast<int>(manip->getMode());
    nav["speed"]           = manip->getSpeed();
    nav["transition"]      = static_cast<float>(manip->getAnimationDuration());
    nav["autoPlay"]        = ui->m_autoPlayPresets;
    settings["navigation"] = nav;
  }

  // Transform helpers
  {
    json        transform;
    const auto& helpers          = const_cast<GaussianSplattingUI*>(ui)->m_helpers;
    transform["show"]            = helpers.isEditingMode();
    transform["snapEnabled"]     = helpers.transform.isSnapEnabled();
    transform["snapTranslate"]   = helpers.transform.getSnapTranslate();
    transform["snapRotate"]      = helpers.transform.getSnapRotate();
    transform["snapScale"]       = helpers.transform.getSnapScale();
    settings["transformHelpers"] = transform;
  }

  // Grid
  {
    json grid;
    grid["show"]     = const_cast<GaussianSplattingUI*>(ui)->m_helpers.grid.isVisible();
    settings["grid"] = grid;
  }

  // Light proxies
  {
    json lightProxies;
    lightProxies["show"]     = ui->m_showLightProxies;
    settings["lightProxies"] = lightProxies;
  }

  // Summary info overlay
  {
    json overlay;
    overlay["show"]            = ui->m_showSummaryOverlay;
    settings["summaryOverlay"] = overlay;
  }

  data["settings"] = settings;
}

//--------------------------------------------------------------------------------------------------
// Save tonemapping parameters
//
void VkgsProjectWriter::saveTonemapping(json& data, const GaussianSplattingUI* ui)
{
  const auto& tm = ui->m_tonemapperData;
  json        item;

  item["isActive"]    = tm.isActive;
  item["method"]      = tm.method;
  item["exposure"]    = tm.exposure;
  item["temperature"] = tm.temperature;
  item["tint"]        = tm.tint;

  item["contrast"]   = tm.contrast;
  item["brightness"] = tm.brightness;
  item["saturation"] = tm.saturation;
  item["vignette"]   = tm.vignette;

  item["vibrance"]      = tm.vibrance;
  item["shadowBias"]    = tm.shadowBias;
  item["midtoneBias"]   = tm.midtoneBias;
  item["highlightBias"] = tm.highlightBias;
  item["coolColor"]     = {tm.coolColor.x, tm.coolColor.y, tm.coolColor.z};
  item["warmColor"]     = {tm.warmColor.x, tm.warmColor.y, tm.warmColor.z};
  item["splitBalance"]  = tm.splitBalance;

  item["autoExposure"]         = tm.autoExposure;
  item["autoExposureSpeed"]    = tm.autoExposureSpeed;
  item["evMinValue"]           = tm.evMinValue;
  item["evMaxValue"]           = tm.evMaxValue;
  item["enableCenterMetering"] = tm.enableCenterMetering;
  item["centerMeteringSize"]   = tm.centerMeteringSize;
  item["averageMode"]          = tm.averageMode;

  item["dither"] = tm.dither;

  data["tonemapping"] = item;
}

}  // namespace vk_gaussian_splatting
