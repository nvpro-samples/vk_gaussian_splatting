/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
#include "mesh_manager_vk.h"
#include "light_manager_vk.h"
#include "camera_set.h"
#include "gaussian_splatting.h"
#include "image_compare_ui.h"
#include "shader_feedback_ui.h"
#include "memory_monitor_vk.h"

// Json
#include <tinygltf/json.hpp>
using nlohmann::json;

namespace vk_gaussian_splatting {

class GaussianSplattingUI : public GaussianSplatting, public nvapp::IAppElement
{
  friend class VkgsProjectReader;
  friend class VkgsProjectWriter;

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

  // Override reset to clear selections and detach helpers before base reset
  void reset() override;

  // handle recent files save/load at imgui level
  void guiRegisterIniFileHandlers();

  // Query if user is dragging comparison split divider (used to disable camera)
  bool isDraggingComparisonSlider() const { return m_imageCompareUI.isDraggingSplitDivider(); }
  bool isDraggingTransformHelper() const { return m_helpers.transform.isDragging(); }
  bool isDraggingCursorTarget() const { return m_cursorTargetDragging; }

private:
  // Selection helper methods
  void selectMeshInstance(std::shared_ptr<MeshInstanceVk> instance);
  void selectSplatSetInstance(std::shared_ptr<SplatSetInstanceVk> instance);
  void selectLightInstance(std::shared_ptr<LightSourceInstanceVk> instance);

  // Camera preset helper
  bool cameraPresetNeedsShaderRebuild(uint64_t presetIndex);

  void guiLoadSceneAndDrawProgressIfNeeded(void);
  void guiDrawViewport(void);
  void guiDrawAssetsWindow(void);
  void resetSelection();  // Clear any selection and detach transform gizmo
  void guiDrawRendererTree();
  void guiDrawCameraTree();
  void guiDrawLightTree();
  void guiDrawRadianceFieldsTree();
  void guiDrawObjectTree();

  void guiDrawPropertiesWindow(void);
  void guiDrawRendererProperties();
  void guiDrawCommonSplatSetProperties();
  void guiDrawSplatSetProperties();
  void guiDrawMeshTransformProperties();
  void guiDrawMeshMaterialProperties();
  void guiDrawCameraProperties();
  void guiDrawNavigationProperties();
  void guiDrawLightProperties();

  void guiDrawRendererStatisticsWindow();

  // UI utility functions for icon button styling
  void pushIconStyle(bool isActive);
  void popIconStyle();


  // Shader feedback window + footer bar (delegated to ShaderFeedbackUI)
  void guiDrawShaderFeedbackWindow(void);
  void guiDrawFooterBar(void);

  // Reusable selectors (used in both menu bar and property panels)
  void guiDrawSortingSelector(bool inMenuBar = false);
  void guiDrawLightingModeSelector(bool inMenuBar = false);
  void guiDrawShadowsModeSelector(bool inMenuBar = false);
  void guiDrawTracingStrategySelector(bool inMenuBar = false);

  bool guiGetTransform(glm::vec3& scale, glm::vec3& rotation, glm::vec3& translation, glm::mat4& transform, glm::mat4& transformInv, bool disabled /*=false*/);
  bool guiGetTransform(glm::vec3& scale,
                       glm::vec3& rotation,
                       glm::vec3& translation,
                       glm::mat4& transform,
                       glm::mat4& transformInv,
                       glm::mat3& transformRotScaleInv,
                       bool       disabled /*=false*/);

  // Helper method to toggle comparison mode
  void toggleComparisonMode(bool enable);

  // Summary info overlay (GPU name, FPS, VRAM)
  void guiDrawSummaryOverlay(ImVec2 imagePos, ImVec2 imageSize);

  // Helper method to save current visualization to image file
  void saveVisualizationImageToFile(const std::filesystem::path& filename);

  // Helper method to get settings string for comparison display
  std::string getSettingsString(int pipeline, int visualize);

  // methods to handle recent files in file menu
  void guiAddToRecentFiles(std::filesystem::path filePath, int historySize = 20);
  void guiAddToRecentProjects(std::filesystem::path filePath, int historySize = 20);

  bool loadProjectIfNeeded();
  bool saveProject(std::string path);

private:
  // hide/show ui elements
  bool m_showRendererStatistics = true;
  bool m_showMemoryStatistics   = true;
  bool m_showShaderFeedback     = false;

  // Persistent cursor target overlay (locks shader feedback cursor)
  bool   m_showCursorTargetOverlay = false;
  bool   m_cursorTargetDragging    = false;
  ImVec2 m_cursorTargetPos         = ImVec2(-1.0f, -1.0f);  // in viewport image pixels (top-left origin)

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
    GUI_CAMERA_TYPE,          // type of camera
    GUI_FRUSTUM_CULLING,      // where to perform frustum culling (or disabled)
    GUI_SH_FORMAT,            // data format for storage of SH in VRAM
    GUI_RGBA_FORMAT,          // data format for storage of RGBA colors in VRAM
    GUI_PARTICLE_FORMAT,      // Particle tracing mode for RTX
    GUI_KERNEL_DEGREE,        // Kernel degree for RTX
    GUI_VISUALIZE,            // visualization mode
    GUI_VISUALIZE_DLSS_ON,    // visualization mode with DLSS enabled
    GUI_ILLUM_MODEL,          // TODO rename, "illumination" model is not the proper name
    GUI_DIST_SHADER_WG_SIZE,  // Distance shader workgroup size
    GUI_MESH_SHADER_WG_SIZE,  // Mesh shader workgroup size
    GUI_RAY_HIT_PER_PASS,     // Particle samples per pass (controls PARTICLES_SPP)
    GUI_RTX_TRACE_STRATEGY,   // Ray tracing trace strategy (full any hit vs monte carlo)
    GUI_TEMPORAL_SAMPLING,    // Temporal sampling mode
    GUI_LIGHT_TYPE,           // Type of light
    GUI_ATTENUATION_MODE,     // Light attenuation mode
    GUI_EXTENT_METHOD,        // extent projection method
    GUI_COMPARISON_DISPLAY,   // comparison display mode (reference, current, difference)
    GUI_DLSS_MODE,            // DLSS quality mode (Disabled, Optimal, Minimal, Maximal)
    GUI_FTB_SYNC_MODE,        // FTB depth buffer synchronization mode (interlock vs disabled)
    GUI_COLOR_FORMAT,         // Color buffer format (precision/memory tradeoff)
    GUI_NORMAL_METHOD,        // Normal vector computation method (max density plane, iso surface)
    GUI_LIGHTING_MODE,        // Lighting mode (disabled, direct, indirect)
    GUI_SHADOWS_MODE,         // Shadows mode (disabled, hard, soft)
    GUI_DOF_MODE,             // Depth of Field mode (disabled, fixed focus, auto focus)
    GUI_DOF_MODE_NO_AUTO      // Depth of Field mode (disabled, fixed focus)
  };

  // UI utility for choice (a.k.a. "combo") menus
  nvgui::EnumRegistry m_ui;
  ImageCompareUI      m_imageCompareUI;    // UI overlay for image comparison
  ShaderFeedbackUI    m_shaderFeedbackUI;  // Shader feedback window + footer bar

  // Comparison mode: captured settings (for display in UI overlay)
  int m_referenceCapturePipeline      = 0;  // Pipeline used when reference was captured
  int m_referenceCaptureVisualization = 0;  // Visualization mode when reference was captured

  // which property to display in the property editor
  enum
  {
    GUI_NONE,
    GUI_RENDERER,
    GUI_CAMERA,
    GUI_LIGHT,
    GUI_SPLATSET,
    GUI_MESH,
  } m_selectedAsset = GUI_RENDERER;

  bool        m_objListUpdated = false;
  const float TREE_INDENT      = 16.0f;

  // Summary info overlay
  bool m_showSummaryOverlay = false;  // Toggle for the summary overlay
  // Last known screen-space rect of the summary overlay window.
  // Used to block viewport interactions (e.g., image-compare click handling) when the overlay is on top.
  ImVec2                                m_summaryOverlayRectMin{0.0f, 0.0f};
  ImVec2                                m_summaryOverlayRectMax{0.0f, 0.0f};
  bool                                  m_summaryOverlayRectValid = false;
  std::string                           m_cachedGpuName;                      // GPU device name (cached once at init)
  VRAMSummary                           m_cachedVRAM;                         // Cached VRAM usage/budget
  double                                m_cachedFps       = 0.0;              // Cached FPS value
  double                                m_cachedFrameTime = 0.0;              // Cached frame time in ms
  std::chrono::steady_clock::time_point m_lastOverlayRefreshTime{};           // Last time overlay data was refreshed
  static constexpr double               OVERLAY_REFRESH_INTERVAL_SEC = 0.25;  // Refresh overlay data every N seconds

  void  updateTitleIfNeeded();
  float m_titleUpdateTimer = 0.0f;

  // Project loading
  bool loadingProject = false;
  json data;

  // Debuging
  void dumpSplat();
};

}  // namespace vk_gaussian_splatting

#endif
