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

#pragma once

#include <string>
#include <map>
#include <memory>
#include <filesystem>
#include <tinygltf/json.hpp>

using json = nlohmann::json;

namespace vk_gaussian_splatting {

// Forward declarations
class GaussianSplattingUI;
struct MeshVk;
class SplatSetVk;

//--------------------------------------------------------------------------------------------------
// Project file reader - loads scene state from proprietary .vkgs file
//
class VkgsProjectReader
{
public:
  // Load project from JSON data
  // Returns true on success, false on failure
  static bool loadProject(const json& data, const std::string& path, GaussianSplattingUI* ui);

private:
  // Helper functions to load individual sections
  static void loadAssetNamingCounters(const json& data, GaussianSplattingUI* ui);
  static void loadRendererSettings(const json& data, GaussianSplattingUI* ui);
  static void loadSplatGlobalOptions(const json& data);
  static void loadSplatSetsAndInstances(const json& data, int fileVersion, GaussianSplattingUI* ui);
  static void loadMeshes(const json& data, int fileVersion, const std::string& projectPath, GaussianSplattingUI* ui);
  static void loadLights(const json& data, int fileVersion, GaussianSplattingUI* ui);
  static void loadCameras(const json& data, GaussianSplattingUI* ui);

  // Sub-helper functions for splat sets
  static void loadSplatAssets(const json& data, int fileVersion, std::map<int, std::shared_ptr<SplatSetVk>>& splatSetIdToAsset);
  static void loadSplatInstances(const json& data, int fileVersion, std::map<int, std::shared_ptr<SplatSetVk>>& splatSetIdToAsset);

  // Sub-helper functions for meshes
  static void loadMeshAssets(const json&                             data,
                             const std::string&                      projectPath,
                             std::map<int, std::shared_ptr<MeshVk>>& assetIdToMesh,
                             GaussianSplattingUI*                    ui);
  static void loadMeshInstances(const json& data, const std::map<int, std::shared_ptr<MeshVk>>& assetIdToMesh, GaussianSplattingUI* ui);
};

}  // namespace vk_gaussian_splatting
