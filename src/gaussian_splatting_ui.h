/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _GAUSSIAN_SPLATTING_UI_H_
#define _GAUSSIAN_SPLATTING_UI_H_

#include <iostream>
#include <string>
#include <array>
#include <chrono>
#include <filesystem>
#include <span>
// TODO: include Igmlui before Vulkan
// Or undef Status before including imgui
// need to solve this issue
#include <imgui/imgui.h>
//
#include <vulkan/vulkan_core.h>
// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
// GPU radix sort
#include <vk_radix_sort.h>
//
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/pipeline.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/alignment.hpp>

#include <nvvk/helpers.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/sbt_generator.hpp>

#include <nvvkglsl/glsl.hpp>

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_default_menu.hpp>
//
#include <nvgui/axis.hpp>
#include <nvgui/enum_registry.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/file_dialog.hpp>
//
#include <nvgpu_monitor/elem_gpu_monitor.hpp>

// Shared between host and device
#include "shaderio.h"

#include "utilities.h"
#include "splat_set.h"
#include "splat_set_vk.h"
#include "ply_loader_async.h"
#include "splat_sorter_async.h"
#include "mesh_set_vk.h"
#include "light_set_vk.h"
#include "camera_set.h"
#include "gaussian_splatting.h"

// Json
#include <tinygltf/json.hpp>
using nlohmann::json;

namespace vk_gaussian_splatting {

class GaussianSplattingUI : public GaussianSplatting, public nvapp::IAppElement
{
public:  // Methods specializing IAppElement
  GaussianSplattingUI(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry, bool* benchmarkEnabled);

  ~GaussianSplattingUI() override;

  void onAttach(nvapp::Application* app) override;

  void onDetach() override;

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;

  void onPreRender() override;

  void onRender(VkCommandBuffer cmd) override;

  void onUIRender() override;

  void onUIMenu() override;

  void onFileDrop(const std::filesystem::path& filename) override;

  // handle recent files save/load at imgui level
  void guiRegisterIniFileHandlers();

private:
  void guiDrawAssetsWindow(void);
  void guiDrawRendererTree();
  void guiDrawCameraTree();
  void guiDrawLightTree();
  void guiDrawRadianceFieldsTree();
  void guiDrawObjectTree();

  void guiDrawPropertiesWindow(void);
  void guiDrawRendererProperties();
  void guiDrawSplatSetProperties();
  void guiDrawMeshTransformProperties();
  void guiDrawMeshMaterialProperties();
  void guiDrawCameraProperties();
  void guiDrawNavigationProperties();
  void guiDrawLightProperties();

  void guiDrawRendererStatisticsWindow();

  void guiDrawMemoryStatisticsWindow(void);

  bool guiGetTransform(glm::vec3& scale, glm::vec3& rotation, glm::vec3& translation, glm::mat4& transform, glm::mat4& transformInv, bool disabled /*=false*/);

  // methods to handle recent files in file menu
  void guiAddToRecentFiles(std::filesystem::path filePath, int historySize = 20);
  void guiAddToRecentProjects(std::filesystem::path filePath, int historySize = 20);

  bool loadProjectIfNeeded();
  bool saveProject(std::string path);

private:
  // hide/show ui elements
  bool                                                  m_showUI = true;
  std::shared_ptr<nvapp::ElementProfiler::ViewSettings> m_profilerViewSettings;

  // benchmark mode (enabled by command line), loadings will be synchronous and vsync off
  bool* m_pBenchmarkEnabled = {};
  // screenshot file name (used by benchmark)
  std::filesystem::path m_screenshotFilename;

  // Recent files list
  std::vector<std::filesystem::path> m_recentFiles;

  // Recent projects list
  std::vector<std::filesystem::path> m_recentProjects;

  // for multiple choice selectors in the UI
  enum GuiEnums
  {
    GUI_STORAGE,              // model storage in VRAM (in texture or buffer)
    GUI_SORTING,              // the sorting method to use
    GUI_PIPELINE,             // the rendering pipeline to use
    GUI_FRUSTUM_CULLING,      // where to perform frustum culling (or disabled)
    GUI_SH_FORMAT,            // data format for storage of SH in VRAM
    GUI_KERNEL_DEGREE,        // Kernel degree for RTX
    GUI_VISUALIZE,            // visualization mode
    GUI_ILLUM_MODEL,          // TODO rename, "illumination" model is not the proper name
    GUI_DIST_SHADER_WG_SIZE,  // Distance shader workgroup size
    GUI_MESH_SHADER_WG_SIZE,  // Mesh shader workgroup size
    GUI_RAY_HIT_PER_PASS      // Max number of ray hits stored per pass (payload array size)
  };

  // UI utility for choice (a.k.a. "combo") menus
  nvgui::EnumRegistry m_ui;

  // Which asset is selected in the scene tree
  enum
  {
    NONE,
    RENDERER,
    CAMERAS,
    LIGHTS,
    SPLATSET,
    OBJECTS
  } m_selectedAsset = RENDERER;

  // which property to display in the property editor
  enum
  {
    GUI_NONE,
    GUI_RENDERER,
    GUI_CAMERA,
    GUI_LIGHT,
    GUI_SPLATSET,
    GUI_MATERIAL,
  } m_selectedProperty = GUI_RENDERER;

  bool        m_objJustImported = false;
  const float TREE_INDENT       = 16.0f;

  // Project loading
  bool loadingProject = false;
  json data;

  // Debuging
  void dumpSplat(uint32_t splatIdx);
};

}  // namespace vk_gaussian_splatting

#endif
