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
#include <tinygltf/json.hpp>

namespace vk_gaussian_splatting {

using json = nlohmann::json;

// Forward declarations
class GaussianSplattingUI;

//--------------------------------------------------------------------------------------------------
// Project file writer - saves current scene state to proprietary .vkgs file
//
class VkgsProjectWriter
{
public:
  // Save current project state to file
  // Returns true on success, false on failure
  static bool saveProject(const std::string& path, const GaussianSplattingUI* ui);

private:
  // Helper functions to save individual sections
  static void saveRendererSettings(json& data, const GaussianSplattingUI* ui);
  static void saveActiveCamera(json& data, const GaussianSplattingUI* ui);
  static void saveCameraPresets(json& data, const GaussianSplattingUI* ui);
  static void saveLights(json& data, const GaussianSplattingUI* ui);
  static void saveSplatGlobalOptions(json& data);
  static void saveSplatSets(json& data, const GaussianSplattingUI* ui, const std::string& projectPath);
  static void saveSplatInstances(json& data, const GaussianSplattingUI* ui);
  static void saveMeshes(json& data, const GaussianSplattingUI* ui, const std::string& projectPath);
};

}  // namespace vk_gaussian_splatting
