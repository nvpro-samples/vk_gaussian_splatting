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

#include "nvutils/file_operations.hpp"

#include "nvgui/fonts.hpp"
#include "nvgui/tooltip.hpp"

#include <nvvk/helpers.hpp>  // For imageToLinear and saveImageToFile

#include "shaderio.h"  // For MeshType enum

using shaderio::MeshType;  // Import MeshType enum for convenience

#include <glm/vec2.hpp>
// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <chrono>
#include <thread>
#include <filesystem>
#include <algorithm>  // for std::clamp
#include <cmath>      // for std::round

#include "gaussian_splatting_ui.h"
#include "memory_statistics.h"
#include "memory_monitor_vk.h"
#include "vkgs_project_reader.h"
#include "vkgs_project_writer.h"
#include <GLFW/glfw3.h>
#undef APIENTRY
#include <fmt/format.h>

namespace vk_gaussian_splatting {

GaussianSplattingUI::GaussianSplattingUI(nvutils::ProfilerManager*   profilerManager,
                                         nvutils::ParameterRegistry* parameterRegistry,
                                         bool*                       benchmarkEnabled)
    : GaussianSplatting(profilerManager, parameterRegistry)
    , m_pBenchmarkEnabled(benchmarkEnabled)
    , m_imageCompareUI(&m_imageCompare)
{

  // Register some very sepcific command line parameters, related to benchmarking, other parameters are registered in main or in registerCommandLineParameters

  // Add updateData parameter with callback to trigger splat set regeneration
  bool updateDataTrigger = false;
  parameterRegistry->add({.name = "updateData",
                          .help = "Use only in benchmark script. 1=triggers an update of data buffers or textures after a parameter change.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                m_assets.splatSets.markAllSplatSetsForRegeneration();
                                m_requestUpdateShaders = true;
                              }},
                         &updateDataTrigger, true);  // Trigger value = true

  parameterRegistry->add({.name = "screenshot",
                          .help = "Use only in benchmark script. Takes a screenshot.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                if(m_app)
                                {
                                  m_app->saveScreenShot(m_screenshotFilename);
                                }
                              }},
                         {".png"}, &m_screenshotFilename);
};

GaussianSplattingUI::~GaussianSplattingUI(){
    // Nothing to do here
};

void GaussianSplattingUI::onAttach(nvapp::Application* app)
{
  // Initializes the core

  GaussianSplatting::onAttach(app);

  // Cache GPU device name (static for the lifetime of the app)
  {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &properties);
    m_cachedGpuName = properties.deviceName;
  }

  // Init combo selectors used in UI

  m_ui.enumAdd(GUI_STORAGE, STORAGE_BUFFERS, "Buffers");
  m_ui.enumAdd(GUI_STORAGE, STORAGE_TEXTURES, "Textures");

  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_VERT, "Raster vertex shader 3DGS");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH, "Raster mesh shader 3DGS");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH_3DGUT, "Raster mesh shader 3DGUT");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_RTX, "Ray tracing 3DGRT");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_HYBRID, "Hybrid 3DGS+3DGRT");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_HYBRID_3DGUT, "Hybrid 3DGUT+3DGRT");

  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_EIGEN, "Eigen");
  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_CONIC, "Conic");

  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_FINAL, "Final render");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLOCK, "Clock cycles");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_RAYHITS, "Ray Hit Count");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DEPTH_INTEGRATED, "Depth (iso thres)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DEPTH, "Depth (Closest hit)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DEPTH_FOR_DLSS, "Depth (for DLSS)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_NORMAL_INTEGRATED, "Normal (Integrated)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_NORMAL, "Normal (closest hit)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_NORMAL_FOR_DLSS, "Normal (For DLSS)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_SPLAT_ID, "Splat ID (Harlequin)");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DLSS_INPUT, "DLSS Input", true);  // <- true means disabled
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DLSS_ALBEDO, "DLSS Guide: Albedo", true);
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DLSS_SPECULAR, "DLSS Guide: Specular", true);
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DLSS_NORMAL, "DLSS Guide: Normal", true);
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DLSS_MOTION, "DLSS Guide: Motion", true);
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DLSS_DEPTH, "DLSS Guide: Depth", true);

  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_FINAL, "Final render");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_CLOCK, "Clock cycles");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_RAYHITS, "Ray Hit Count");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DEPTH_INTEGRATED, "Depth (iso thres)");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DEPTH, "Depth (Closest hit)");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DEPTH_FOR_DLSS, "Depth (for DLSS)");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_NORMAL_INTEGRATED, "Normal (Integrated)");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_NORMAL, "Normal (closest hit)");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_NORMAL_FOR_DLSS, "Normal (For DLSS)");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_SPLAT_ID, "Splat ID (Harlequin)");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DLSS_INPUT, "DLSS Input");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DLSS_ALBEDO, "DLSS Guide: Albedo");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DLSS_SPECULAR, "DLSS Guide: Specular");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DLSS_NORMAL, "DLSS Guide: Normal");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DLSS_MOTION, "DLSS Guide: Motion");
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_DLSS_DEPTH, "DLSS Guide: Depth");

  m_ui.enumAdd(GUI_SORTING, SORTING_GPU_SYNC_RADIX, "GPU radix sort");
  m_ui.enumAdd(GUI_SORTING, SORTING_CPU_ASYNC_MULTI, "CPU async std multi");
  m_ui.enumAdd(GUI_SORTING, SORTING_STOCHASTIC_SPLAT, "Stochastic splat");

  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT32, "Float 32");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT16, "Float 16");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_UINT8, "Uint8");

  m_ui.enumAdd(GUI_RGBA_FORMAT, FORMAT_FLOAT32, "Float 32");
  m_ui.enumAdd(GUI_RGBA_FORMAT, FORMAT_FLOAT16, "Float 16");
  m_ui.enumAdd(GUI_RGBA_FORMAT, FORMAT_UINT8, "Uint8");

  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_ICOSAHEDRON, "Icosahedron");
  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_PARAMETRIC, "AABB + parametric");

  m_ui.enumAdd(GUI_CAMERA_TYPE, CAMERA_PINHOLE, "Pinhole");
  m_ui.enumAdd(GUI_CAMERA_TYPE, CAMERA_FISHEYE, "Fisheye");

  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_AUTO, "Automatic");
  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_ENABLED, "Force enabled");
  m_ui.enumAdd(GUI_TEMPORAL_SAMPLING, TEMPORAL_SAMPLING_DISABLED, "Force disabled");

  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_QUINTIC, "5 (Quintic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_TESSERACTIC, "4 (Tesseractic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_CUBIC, "3 (Cubic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_QUADRATIC, "2 (Quadratic)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_LAPLACIAN, "1 (Laplacian)");
  m_ui.enumAdd(GUI_KERNEL_DEGREE, KERNEL_DEGREE_LINEAR, "0 (Linear)");

  m_ui.enumAdd(GUI_LIGHT_TYPE, shaderio::LightType::ePointLight, "Point");
  m_ui.enumAdd(GUI_LIGHT_TYPE, shaderio::LightType::eDirectionalLight, "Directional");
  m_ui.enumAdd(GUI_LIGHT_TYPE, shaderio::LightType::eSpotLight, "Spot");

  m_ui.enumAdd(GUI_ATTENUATION_MODE, 0, "None");
  m_ui.enumAdd(GUI_ATTENUATION_MODE, 1, "Linear");
  m_ui.enumAdd(GUI_ATTENUATION_MODE, 2, "Quadratic");
  m_ui.enumAdd(GUI_ATTENUATION_MODE, 3, "Physical");

  m_ui.enumAdd(GUI_ILLUM_MODEL, 0, "No indirect");
  m_ui.enumAdd(GUI_ILLUM_MODEL, 1, "Reflective");
  m_ui.enumAdd(GUI_ILLUM_MODEL, 2, "Refractive");

  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 512, "512");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 256, "256");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 128, "128");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 64, "64");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 32, "32");
  m_ui.enumAdd(GUI_DIST_SHADER_WG_SIZE, 16, "16");

  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 128, "128");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 64, "64");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 32, "32");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 16, "16");
  m_ui.enumAdd(GUI_MESH_SHADER_WG_SIZE, 8, "8");

  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 128, "128");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 64, "64");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 32, "32");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 20, "20");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 18, "18");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 16, "16");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 12, "12");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 8, "8");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 4, "4");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 2, "2");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 1, "1");

  m_ui.enumAdd(GUI_RTX_TRACE_STRATEGY, RTX_TRACE_STRATEGY_FULL_ANYHIT, "All pass");
  m_ui.enumAdd(GUI_RTX_TRACE_STRATEGY, RTX_TRACE_STRATEGY_PASS_STOCHASTIC, "Stochastic pass");
  m_ui.enumAdd(GUI_RTX_TRACE_STRATEGY, RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT, "Stochastic any-hit");

  m_ui.enumAdd(GUI_DLSS_MODE, -1, "DLSS Disabled");
  m_ui.enumAdd(GUI_DLSS_MODE, 0, "DLSS Min");
  m_ui.enumAdd(GUI_DLSS_MODE, 1, "DLSS Optimal");
  m_ui.enumAdd(GUI_DLSS_MODE, 2, "DLSS Max");

  m_ui.enumAdd(GUI_FTB_SYNC_MODE, FTB_SYNC_DISABLED, "Disabled (fast)");
  m_ui.enumAdd(GUI_FTB_SYNC_MODE, FTB_SYNC_INTERLOCK, "Interlock (correct)");

  m_ui.enumAdd(GUI_COLOR_FORMAT, VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8 UNORM");
  m_ui.enumAdd(GUI_COLOR_FORMAT, VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16 SFLOAT");
  m_ui.enumAdd(GUI_COLOR_FORMAT, VK_FORMAT_R32G32B32A32_SFLOAT, "R32G32B32A32 SFLOAT");

  m_ui.enumAdd(GUI_COMPARISON_DISPLAY, (int)ImageCompare::Mode::eCapture, "Reference");
  m_ui.enumAdd(GUI_COMPARISON_DISPLAY, (int)ImageCompare::Mode::eCurrent, "Current render");
  m_ui.enumAdd(GUI_COMPARISON_DISPLAY, (int)ImageCompare::Mode::eDifferenceRaw, "Difference (Raw)");
  m_ui.enumAdd(GUI_COMPARISON_DISPLAY, (int)ImageCompare::Mode::eDifferenceRedGray, "Difference (Red on Gray)");
  m_ui.enumAdd(GUI_COMPARISON_DISPLAY, (int)ImageCompare::Mode::eDifferenceRedOnly, "Difference (Red only)");

  m_ui.enumAdd(GUI_NORMAL_METHOD, (int)NormalMethod::eMaxDensityPlane, "Max density plane");
  m_ui.enumAdd(GUI_NORMAL_METHOD, (int)NormalMethod::eIsoSurface, "Kernel elipsoid");

  m_ui.enumAdd(GUI_LIGHTING_MODE, (int)LightingMode::eLightingDisabled, "Lighting off");
  m_ui.enumAdd(GUI_LIGHTING_MODE, (int)LightingMode::eLightingDirect, "Direct lighting");
  m_ui.enumAdd(GUI_LIGHTING_MODE, (int)LightingMode::eLightingIndirect, "Indirect lighting");

  m_ui.enumAdd(GUI_SHADOWS_MODE, (int)ShadowsMode::eShadowsDisabled, "Shadows off");
  m_ui.enumAdd(GUI_SHADOWS_MODE, (int)ShadowsMode::eShadowsHard, "Hard shadows");
  m_ui.enumAdd(GUI_SHADOWS_MODE, (int)ShadowsMode::eShadowsSoft, "Soft shadows");

  m_ui.enumAdd(GUI_DOF_MODE, (int)DofMode::eDofDisabled, "Disabled");
  m_ui.enumAdd(GUI_DOF_MODE, (int)DofMode::eDofFixedFocus, "Fixed focus");
  m_ui.enumAdd(GUI_DOF_MODE, (int)DofMode::eDofAutoFocus, "Auto focus");

  m_ui.enumAdd(GUI_DOF_MODE_NO_AUTO, (int)DofMode::eDofDisabled, "Disabled");
  m_ui.enumAdd(GUI_DOF_MODE_NO_AUTO, (int)DofMode::eDofFixedFocus, "Fixed focus");
}

void GaussianSplattingUI::onDetach()
{
  GaussianSplatting::onDetach();
}

void GaussianSplattingUI::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  GaussianSplatting::onResize(cmd, size);
}

void GaussianSplattingUI::onPreRender()
{
  GaussianSplatting::onPreRender();
}

void GaussianSplattingUI::onRender(VkCommandBuffer cmd)
{
  // Hide all 3D visual helpers in benchmark mode (grid, transform gizmo, light proxies)
  if(*m_pBenchmarkEnabled)
  {
    m_helpers.grid.setVisible(false);
    m_helpers.setEditingMode(false);
    m_showLightProxies = false;
  }

  GaussianSplatting::onRender(cmd);
}

//--------------------------------------------------------------------------------------------------
// UI utility functions for icon button styling
//--------------------------------------------------------------------------------------------------
void GaussianSplattingUI::pushIconStyle(bool isActive)
{
  if(isActive)
  {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));         // Active green
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));  // Lighter green
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.5f, 0.1f, 1.0f));   // Darker green
  }
  else
  {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));         // Inactive gray
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.4f, 0.4f, 1.0f));  // Lighter gray
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));   // Darker gray
  }
}

void GaussianSplattingUI::popIconStyle()
{
  ImGui::PopStyleColor(3);
}

#define ICON_BLANK "     "

//--------------------------------------------------------------------------------------------------
// Toggle comparison mode on/off with proper state management
//
void GaussianSplattingUI::toggleComparisonMode(bool enable)
{
  bool prevState        = prmComparison.enabled;
  prmComparison.enabled = enable;

  if(prmComparison.enabled && !prevState)
  {
    // Enabling comparison mode: store current settings and request capture
    m_referenceCapturePipeline      = prmSelectedPipeline;
    m_referenceCaptureVisualization = prmRender.visualize;
    m_requestCaptureComparison      = true;
  }
  else if(!prmComparison.enabled && prevState)
  {
    // Disabling comparison mode: release reference
    m_imageCompare.releaseCaptureImage();
    m_imageCompare.setMetricsHistorySize(1);  // Reset to no-graph mode
  }
}

//--------------------------------------------------------------------------------------------------
// Draw summary info overlay in the top-left of the viewport
// Shows GPU name, FPS/frame time, and VRAM usage
// Uses large yellow text on transparent background for maximum visibility
//
void GaussianSplattingUI::guiDrawSummaryOverlay(ImVec2 imagePos, ImVec2 imageSize)
{
  if(!m_showSummaryOverlay)
    return;

  // --- Refresh cached data at throttled interval (FPS + VRAM together) ---
  auto   now     = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - m_lastOverlayRefreshTime).count();
  if(elapsed >= OVERLAY_REFRESH_INTERVAL_SEC || m_lastOverlayRefreshTime.time_since_epoch().count() == 0)
  {
    // Refresh VRAM
    m_cachedVRAM = queryVRAMSummary(m_app->getPhysicalDevice());

    // Refresh frame time from profiler
    nvutils::ProfilerTimeline::TimerInfo info{};
    std::string                          apiName;
    if(m_profilerTimeline->getFrameTimerInfo("Frame", info, apiName) && info.numAveraged > 0)
    {
      double gpuTimeMs  = info.gpu.average / 1000.0;  // microseconds -> milliseconds
      double cpuTimeMs  = info.cpu.average / 1000.0;
      m_cachedFrameTime = std::max(gpuTimeMs, cpuTimeMs);
      m_cachedFps       = (m_cachedFrameTime > 0.0) ? (1000.0 / m_cachedFrameTime) : 0.0;
    }

    m_lastOverlayRefreshTime = now;
  }

  // --- Format VRAM strings ---
  double vramUsedGB  = static_cast<double>(m_cachedVRAM.usedBytes) / (1024.0 * 1024.0 * 1024.0);
  double vramTotalGB = static_cast<double>(m_cachedVRAM.budgetBytes) / (1024.0 * 1024.0 * 1024.0);

  // --- Viewport resolution ---
  int viewportW = static_cast<int>(imageSize.x);
  int viewportH = static_cast<int>(imageSize.y);

  // --- Draw ImGui overlay window ---
  const float margin = 10.0f;
  ImVec2      overlayPos(imagePos.x + margin, imagePos.y + margin);

  // Default near the top-left, but allow the user to move it afterwards.
  ImGui::SetNextWindowPos(overlayPos, ImGuiCond_FirstUseEver);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.2f, 0.6f, 0.2f, 0.85f));

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;

  ImGui::Begin("##SummaryOverlay", nullptr, flags);

  // Large yellow text
  ImGui::SetWindowFontScale(1.8f);
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));

  ImGui::Text("vk_gaussian_splatting");

  // GPU name
  ImGui::Text("%s", m_cachedGpuName.c_str());

  // Viewport resolution | FPS and frame time
  ImGui::Text("%d x %d | %.1f FPS (%.2f ms)", viewportW, viewportH, m_cachedFps, m_cachedFrameTime);

  // SPP progress bar (same logic as the footer status bar)
  {
    float       progress = 0.0f;
    std::string buf      = "1/1";
    if(!m_dlss.isEnabled() && prmRtx.temporalSampling)
    {
      int displayFrame = std::max(1, prmFrame.frameSampleId + 1);
      progress         = (float)displayFrame / (float)prmFrame.frameSampleMax;
      buf              = fmt::format("{}/{}", displayFrame, prmFrame.frameSampleMax);
    }
    ImGui::Text("SPP");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.4f, 0.7f, 0.0f, 1.0f));
    ImGui::ProgressBar(progress, ImVec2(ImGui::GetContentRegionAvail().x * 0.75f, ImGui::GetTextLineHeight()), buf.c_str());
    ImGui::PopStyleColor();
  }

  // Current rendering pipeline (match by ivalue, same as combo selector)
  const char* pipelineName = "Unknown";
  for(const auto& e : m_ui.getEnums(GUI_PIPELINE))
  {
    if(e.ivalue == prmSelectedPipeline)
    {
      pipelineName = e.name.c_str();
      break;
    }
  }
  ImGui::Text("%s", pipelineName);

  // Particle count
  int32_t totalSplatCount = m_assets.splatSets.getTotalGlobalSplatCount();
  if(prmSelectedPipeline == PIPELINE_RTX)
  {
    // Pure ray tracing: show total only
    ImGui::Text("Particles %s", formatSize(totalSplatCount).c_str());
  }
  else
  {
    // Raster and hybrid modes: show rasterized / total
    const bool usesDistShader =
        (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
    int32_t rasterSplatCount = usesDistShader ? m_indirectReadback.instanceCount : totalSplatCount;
    ImGui::Text("Particles %s / %s", formatSize(rasterSplatCount).c_str(), formatSize(totalSplatCount).c_str());
  }

  // VRAM usage
  ImGui::Text("VRAM: %.1f / %.1f GB", vramUsedGB, vramTotalGB);

  ImGui::PopStyleColor();
  ImGui::SetWindowFontScale(1.0f);

  // Constrain overlay to remain fully inside the viewport rect.
  // (Allows dragging, but clamps the final position.)
  {
    const ImVec2 winPos  = ImGui::GetWindowPos();
    const ImVec2 winSize = ImGui::GetWindowSize();

    const ImVec2 boundsMin = imagePos;
    const ImVec2 boundsMax(imagePos.x + imageSize.x, imagePos.y + imageSize.y);

    float maxX = boundsMax.x - winSize.x;
    float maxY = boundsMax.y - winSize.y;
    if(maxX < boundsMin.x)
      maxX = boundsMin.x;
    if(maxY < boundsMin.y)
      maxY = boundsMin.y;

    const float clampedX = std::min(std::max(winPos.x, boundsMin.x), maxX);
    const float clampedY = std::min(std::max(winPos.y, boundsMin.y), maxY);
    if(clampedX != winPos.x || clampedY != winPos.y)
    {
      ImGui::SetWindowPos(ImVec2(clampedX, clampedY), ImGuiCond_Always);
    }
  }

  // Cache final rect for next frame's input gating.
  {
    const ImVec2 finalPos     = ImGui::GetWindowPos();
    const ImVec2 finalSize    = ImGui::GetWindowSize();
    m_summaryOverlayRectMin   = finalPos;
    m_summaryOverlayRectMax   = ImVec2(finalPos.x + finalSize.x, finalPos.y + finalSize.y);
    m_summaryOverlayRectValid = true;
  }

  // If something underneath set a resize cursor (e.g. image-compare splitter),
  // override it while hovering the summary overlay.
  {
    const ImVec2 mousePos = ImGui::GetIO().MousePos;
    const bool   inside   = mousePos.x >= m_summaryOverlayRectMin.x && mousePos.x <= m_summaryOverlayRectMax.x
                        && mousePos.y >= m_summaryOverlayRectMin.y && mousePos.y <= m_summaryOverlayRectMax.y;
    if(inside)
    {
      ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
      ImGui::GetIO().WantCaptureMouse = true;
    }
  }

  ImGui::End();
  ImGui::PopStyleColor();
}

//--------------------------------------------------------------------------------------------------
// Save current visualization image to file
// Captures the current viewport/visualization mode (including DLSS, helpers, etc.) to an image file
// Supports PNG, JPEG, BMP (LDR) and HDR formats
//
void GaussianSplattingUI::saveVisualizationImageToFile(const std::filesystem::path& filename)
{
  // Get current viewport image info (handles DLSS modes, helpers, etc.)
  ImageCompare::ImageInfo srcImageInfo = getCurrentVisualizationImageInfo();

  // Create temporary command buffer
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // Create linear image for readback
  VkImage        dstImage       = {};
  VkDeviceMemory dstImageMemory = {};

  // Determine output format based on file extension
  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;  // Default for PNG/JPG/BMP
  if(filename.extension() == ".hdr")
  {
    format = VK_FORMAT_R32G32B32A32_SFLOAT;  // HDR float format
  }

  // Convert to linear tiled image (handles format conversion via GPU blit)
  nvvk::imageToLinear(cmd, m_device, m_app->getPhysicalDevice(), srcImageInfo.image, srcImageInfo.size, dstImage,
                      dstImageMemory, format);

  // Submit and wait for completion (synchronous)
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Save to file (quality 90 for JPEG)
  nvvk::saveImageToFile(m_device, dstImage, dstImageMemory, srcImageInfo.size, filename, 90);

  // Clean up temporary resources
  vkFreeMemory(m_device, dstImageMemory, nullptr);
  vkDestroyImage(m_device, dstImage, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Get settings string for display (pipeline + visualization)
//
std::string GaussianSplattingUI::getSettingsString(int pipeline, int visualize)
{
  const auto& pipelineEnums  = m_ui.getEnums(GUI_PIPELINE);
  const auto& visualizeEnums = m_ui.getEnums(GUI_VISUALIZE);

  std::string pipelineName, visualizeName;
  for(const auto& e : pipelineEnums)
  {
    if(e.ivalue == pipeline)
    {
      pipelineName = e.name;
      break;
    }
  }
  for(const auto& e : visualizeEnums)
  {
    if(e.ivalue == visualize)
    {
      visualizeName = e.name;
      break;
    }
  }
  return pipelineName + " - " + visualizeName;
}

void GaussianSplattingUI::onUIMenu()
{
  static bool close_app{false};
  bool        v_sync = m_app->isVsync();
#ifndef NDEBUG
  static bool s_showDemo{false};
  static bool s_showDemoPlot{false};
  static bool s_showDemoIcons{false};
#endif
  if(ImGui::BeginMenu("File"))
  {
    // Project
    if(ImGui::MenuItem(ICON_MS_SCAN_DELETE " New project", ""))
    {
      reset();
    }
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open project", ""))
    {
      prmScene.projectToLoadFilename =
          nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load project file", "VKGS Files|*.vkgs");
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent projects"))
    {
      for(const auto& file : m_recentProjects)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.projectToLoadFilename = file;
        }
      }
      ImGui::EndMenu();
    }
    if(ImGui::MenuItem(ICON_MS_FILE_SAVE " Save project", ""))
    {
      auto path = nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save project file", "VKGS Files|*.vkgs");
      if(!path.empty())
      {
        // Ensure the extension is always ".vkgs" before saving and adding to recent projects
        if(!nvutils::extensionMatches(path, ".vkgs"))
        {
          path = path.replace_extension(".vkgs");
        }

        if(saveProject(path.string()))
        {
          guiAddToRecentProjects(path);
        }
      }
    }

    // Splat sets import
    ImGui::Separator();

    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open Splat Set", ""))
    {
      auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Splat Set",
                                              "All Files|*.ply;*.spz;*.splat|PLY Files|*.ply|SPZ files|*.spz|SPLAT files|*.splat");
      if(!path.empty())
      {
        prmScene.pushLoadRequest(path, false);  // Don't auto-reset, user can choose in dialog
      }
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent Splat Sets"))
    {
      for(const auto& file : m_recentFiles)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.pushLoadRequest(file, false);
        }
      }
      ImGui::EndMenu();
    }

    // Meshes import
    ImGui::Separator();

    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open Mesh", ""))
    {
      auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Mesh", "All Files|*.obj|OBJ Files|*.obj");
      if(!path.empty())
      {
        prmScene.meshToImportFilename = path;
      }
    }

    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_EXIT_TO_APP " Exit", "Ctrl+Q"))
    {
      close_app = true;
    }
    ImGui::EndMenu();
  }
  if(ImGui::BeginMenu("View"))
  {
    ImGui::MenuItem(ICON_MS_BOTTOM_PANEL_OPEN " V-Sync", "Ctrl+Shift+V", &v_sync);
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Renderer Statistics", nullptr, &m_showRendererStatistics);
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Memory Statistics", nullptr, &m_showMemoryStatistics);
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Shader Feedback", nullptr, &m_showShaderFeedback);
    ImGui::EndMenu();
  }
#ifndef NDEBUG
  if(ImGui::BeginMenu("Debug"))
  {
    ImGui::MenuItem("Show ImGui Demo", nullptr, &s_showDemo);
    ImGui::MenuItem("Show ImPlot Demo", nullptr, &s_showDemoPlot);
    ImGui::MenuItem("Show Icons Demo", nullptr, &s_showDemoIcons);
    ImGui::EndMenu();
  }
#endif  // !NDEBUG

  // V-Sync, Screenshot, and Image Comparison Toggle Buttons (centered as group on viewport)
  {
    // Store the position after all menus
    float postMenuPosX = ImGui::GetCursorPosX();

    // Total width of button group is measured from the previous frame's actual layout.
    // Using a static so the centering self-corrects after the first frame.
    float        buttonSpacing   = ImGui::GetStyle().ItemSpacing.x;
    static float totalGroupWidth = 0.0f;

    // Find the viewport window and calculate its horizontal center
    ImGuiWindow* viewportWindow = ImGui::FindWindowByName("Viewport");
    float        centerPosX     = 0.0f;

    if(viewportWindow)
    {
      // Get viewport's position and size
      ImVec2 viewportPos  = viewportWindow->Pos;
      ImVec2 viewportSize = viewportWindow->Size;

      // Calculate viewport's horizontal center in screen space
      float viewportCenterX = viewportPos.x + viewportSize.x * 0.5f;

      // Convert to menu bar's local space and center the button GROUP on viewport center
      ImVec2 menuBarPos = ImGui::GetWindowPos();
      centerPosX        = viewportCenterX - menuBarPos.x - totalGroupWidth * 0.5f;

      // Check if this position would overlap with menus
      // If so, position it right after the last menu entry instead
      if(centerPosX < postMenuPosX)
      {
        centerPosX = postMenuPosX;
      }
    }
    else
    {
      // Fallback: position after menus if viewport not found
      centerPosX = postMenuPosX;
    }

    ImGui::SetCursorPosX(centerPosX);
    float groupStartScreenX = ImGui::GetCursorScreenPos().x;

    // V-Sync button
    pushIconStyle(v_sync);

    if(ImGui::Button(ICON_MS_BOTTOM_PANEL_OPEN))
    {
      v_sync = !v_sync;
    }

    popIconStyle();

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Toggle V-Sync");
    }

    // Screenshot button (on same line)
    ImGui::SameLine();

    if(ImGui::Button(ICON_MS_CAMERA))
    {
      // Open save file dialog
      std::filesystem::path filename =
          nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Viewport Capture",
                                      "All Files|*.png;*.jpg;*.bmp;*.hdr|PNG Image|*.png|JPEG Image|*.jpg|BMP Image|*.bmp|HDR Image|*.hdr");

      if(!filename.empty())
      {
        saveVisualizationImageToFile(filename);
      }
    }

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Capture viewport to image file");
    }

    // Cursor target overlay toggle (on same line, between capture and comparison)
    ImGui::SameLine();
    pushIconStyle(m_showCursorTargetOverlay);
    if(ImGui::Button(ICON_MS_CENTER_FOCUS_WEAK))
    {
      m_showCursorTargetOverlay = !m_showCursorTargetOverlay;
      m_cursorTargetDragging    = false;
    }
    popIconStyle();
    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Toggle target overlay (lock shader feedback cursor)");
    }

    // Comparison button (on same line)
    ImGui::SameLine();

    pushIconStyle(prmComparison.enabled);

    if(ImGui::Button(ICON_MS_COMPARE))
    {
      toggleComparisonMode(!prmComparison.enabled);
    }

    popIconStyle();

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Toggle image comparison mode");
    }

    // Summary info overlay button (on same line)
    ImGui::SameLine();

    pushIconStyle(m_showSummaryOverlay);

    if(ImGui::Button(ICON_MS_MONITORING))
    {
      m_showSummaryOverlay = !m_showSummaryOverlay;
    }

    popIconStyle();

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Toggle summary info overlay");
    }

    // Editing button (on same line)
    ImGui::SameLine();

    bool editingMode = m_helpers.isEditingMode();
    pushIconStyle(editingMode);

    if(ImGui::Button(ICON_MS_EDIT))
    {
      m_helpers.setEditingMode(!editingMode);
    }

    popIconStyle();

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Toggle to editing mode (translate/rotate/scale)");
    }

    // Grid toggle button
    ImGui::SameLine();

    bool gridVisible = m_helpers.grid.isVisible();
    pushIconStyle(gridVisible);

    if(ImGui::Button(ICON_MS_GRID_ON))
    {
      m_helpers.grid.toggleVisible();
    }

    popIconStyle();

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Toggle infinite grid (G)");
    }

    // Light Proxies toggle button
    ImGui::SameLine();

    pushIconStyle(m_showLightProxies);

    if(ImGui::Button(ICON_MS_LIGHT_MODE))
    {
      m_showLightProxies = !m_showLightProxies;
      resetFrameCounter();
    }

    popIconStyle();

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Toggle light proxy visibility");
    }

    // Vertical separator before sorting/RTX settings
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Sorting method selector (disabled only for pure RTX - hybrid modes still use rasterization)
    ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_RTX);
    guiDrawSortingSelector(true);
    ImGui::EndDisabled();

    ImGui::SameLine();

    // Ray tracing Strategy selector (only for ray tracing pipelines)
    guiDrawTracingStrategySelector(true);

    // Vertical separator before shading/shadows
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Lighting mode combo
    guiDrawLightingModeSelector(true);

    // Shadows mode combo (disabled when lighting is off or not RTX pipeline)
    ImGui::SameLine();
    guiDrawShadowsModeSelector(true);

    // Vertical separator after shading/shadows
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);

#if defined(USE_DLSS)
    ImGui::SameLine();

    // DLSS Mode selector (only if supported)
    ImGui::BeginDisabled(!isDlssSupportedPipeline());

    // Convert DLSS state to combined mode: -1=Disabled, 0=Min, 1=Optimal, 2=Max
    int dlssMode = m_dlss.isEnabled() ? static_cast<int>(m_dlss.getSizeMode()) : -1;

    ImGui::SetNextItemWidth(150.0f);
    if(m_ui.enumCombobox(GUI_DLSS_MODE, "##DlssMode", &dlssMode))
    {
      if(dlssMode == -1)
      {
        // Disable DLSS
        if(m_dlss.isEnabled())
        {
          m_dlss.setEnabled(false);
          m_requestUpdateShaders = true;
        }
      }
      else
      {
        // Enable DLSS and set size mode
        if(!m_dlss.isEnabled())
        {
          m_dlss.setEnabled(true);
          m_requestUpdateShaders = true;
        }
        m_dlss.setSizeMode(static_cast<DlssDenoiser::SizeMode>(dlssMode));
      }
    }
    ImGui::EndDisabled();
#endif

    ImGui::SameLine();

    // Visualization selector available for all ray tracing pipelines (pure RTX and hybrids)
    ImGui::BeginDisabled(!isRtxPipelineActive());
    // visualization mode selector
    auto visuMenu = GUI_VISUALIZE;
    if(m_dlss.isEnabled())
      visuMenu = GUI_VISUALIZE_DLSS_ON;

    ImGui::SetNextItemWidth(150.0f);
    if(m_ui.enumCombobox(visuMenu, "##ID", &prmRender.visualize))
    {
      m_requestUpdateShaders = true;
    }

    ImGui::EndDisabled();

    // Measure actual group width for next frame's centering calculation
    totalGroupWidth = ImGui::GetItemRectMax().x - groupStartScreenX;
  }

  // Shortcuts

  // Helper: check if an alphanumeric key OR its numpad equivalent was pressed
  auto isNumberKeyPressed = [](int n) -> bool {
    constexpr ImGuiKey alphaKeys[] = {ImGuiKey_0, ImGuiKey_1, ImGuiKey_2, ImGuiKey_3, ImGuiKey_4,
                                      ImGuiKey_5, ImGuiKey_6, ImGuiKey_7, ImGuiKey_8, ImGuiKey_9};
    constexpr ImGuiKey padKeys[]   = {ImGuiKey_Keypad0, ImGuiKey_Keypad1, ImGuiKey_Keypad2, ImGuiKey_Keypad3,
                                      ImGuiKey_Keypad4, ImGuiKey_Keypad5, ImGuiKey_Keypad6, ImGuiKey_Keypad7,
                                      ImGuiKey_Keypad8, ImGuiKey_Keypad9};
    return ImGui::IsKeyPressed(alphaKeys[n]) || ImGui::IsKeyPressed(padKeys[n]);
  };

  const bool wantTextInput = ImGui::GetIO().WantTextInput;

  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space))
  {
    m_lastLoadedCamera = (m_lastLoadedCamera + 1) % m_assets.cameras.size();

    // Check if shader rebuild is needed
    // Defer shader rebuild until animation completes if needed
    m_requestUpdateShadersAfterCameraAnim = cameraPresetNeedsShaderRebuild(m_lastLoadedCamera);

    m_assets.cameras.loadPreset(m_lastLoadedCamera, false);
    m_selectedCameraPresetIndex = -1;
  }
  if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
  {
    close_app = true;
  }

  if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
  {
    v_sync = !v_sync;
  }
  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_G))
  {
    m_helpers.grid.toggleVisible();
  }
  if(ImGui::IsKeyPressed(ImGuiKey_F5))
  {
    if(!m_recentFiles.empty())
      prmScene.pushLoadRequest(m_recentFiles[0], false);
  }
  if(ImGui::IsKeyPressed(ImGuiKey_F1))
  {
    std::string statsFrame;
    std::string statsSingle;
    m_profilerManager->appendPrint(statsFrame, statsSingle, true);
    // print old stats
    nvutils::Logger::getInstance().log(nvutils::Logger::eSTATS, "ParameterSequence %d \"%s\" = {\n%s\n%s}\n", 0,
                                       "F1 pressed ", statsFrame.c_str(), statsSingle.c_str());
  }
  // Debug state dump (F6) — writes SplatSetManagerVk internal state to a timestamped file
  if(ImGui::IsKeyPressed(ImGuiKey_F6))
  {
    m_assets.splatSets.dumpDebugState("manual_F6");
  }

  // Pipeline selection: supports both alphanumeric and numpad keys, skipped when typing in a text field.
  // Known issue: alphanumeric keys '4' and '5' don't trigger IsKeyPressed() — likely consumed by ImGui's
  // key ownership system (LockThisFrame). Numpad keys work as a workaround for all pipelines.
  if(!wantTextInput)
  {
    if(isNumberKeyPressed(1))
      prmSelectedPipeline = PIPELINE_VERT;
    if(isNumberKeyPressed(2))
      prmSelectedPipeline = PIPELINE_MESH;
    if(isNumberKeyPressed(3))
      prmSelectedPipeline = PIPELINE_RTX;
    if(isNumberKeyPressed(4))
      prmSelectedPipeline = PIPELINE_HYBRID;
    if(isNumberKeyPressed(5))
      prmSelectedPipeline = PIPELINE_MESH_3DGUT;
    if(isNumberKeyPressed(6))
      prmSelectedPipeline = PIPELINE_HYBRID_3DGUT;
  }

  // hot rebuild of shaders only if scene exist
  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_R))
  {
    if(!m_loadedSceneFilename.empty())
      m_requestUpdateShaders = true;
    else
      LOGW("No scene loaded, skipping shader rebuild\n");
  }
  if(close_app)
  {
    m_app->close();
  }
#ifndef NDEBUG
  if(s_showDemo)
  {
    ImGui::ShowDemoWindow(&s_showDemo);
  }
  if(s_showDemoPlot)
  {
    //ImPlot::ShowDemoWindow(&s_showDemoPlot);
  }
  if(s_showDemoIcons)
  {
    //nvgui::showDemoIcons();
  }
#endif  // !NDEBUG

  if(m_app->isVsync() != v_sync)
  {
    m_app->setVsync(v_sync);
  }

  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_P))
    dumpSplat();

  // Query VRAM memory information with 'M' key
  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_M))
  {
    queryVRAMInfo(m_app->getPhysicalDevice());
  }

  // Transform gizmo shortcuts (when a mesh is selected)
  if(m_helpers.transform.isAttached())
  {
    // Toggle between World and Local space with T key
    /* Not supported yet 
    if(ImGui::IsKeyPressed(ImGuiKey_T))
    {
      auto currentSpace = m_helpers.transform.getSpace();
      m_helpers.transform.setSpace(currentSpace == TransformHelperVk::TransformSpace::eWorld ?
                                             TransformHelperVk::TransformSpace::eLocal :
                                             TransformHelperVk::TransformSpace::eWorld);
    }
    */
  }
}

void GaussianSplattingUI::onFileDrop(const std::filesystem::path& filename)
{
  // extension To lower case
  std::string extension = filename.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  // Add to queue - user can drop multiple files at once!
  if(extension == ".ply" || extension == ".spz" || extension == ".splat")
  {
    prmScene.pushLoadRequest(filename);
  }
  else if(extension == ".vkgs")
  {
    prmScene.projectToLoadFilename = filename;
  }
  else if(extension == ".obj")
  {
    prmScene.meshToImportFilename = filename;
  }
  else
    LOGE("Error: unsupported file extension %s\n", extension.c_str());
}

void GaussianSplattingUI::updateTitleIfNeeded()
{
  if(!m_app)
    return;
  GLFWwindow* window = m_app->getWindowHandle();
  if(!window)
    return;
  m_titleUpdateTimer += ImGui::GetIO().DeltaTime;
  if(m_titleUpdateTimer < 1.0f)
    return;
  m_titleUpdateTimer = 0.0f;
  const auto& size   = m_app->getViewportSize();
  std::string title  = "vk_gaussian_splatting " VKGS_VERSION
#ifndef NDEBUG
                      " | debug"
#endif
                      " | "
                      + fmt::format("{}x{} | {:.0f} FPS / {:.3f}ms", size.width, size.height, ImGui::GetIO().Framerate,
                                    1000.F / ImGui::GetIO().Framerate);
  glfwSetWindowTitle(window, title.c_str());
}

void GaussianSplattingUI::onUIRender()
{
  updateTitleIfNeeded();

  // Rendering Viewport display the GBuffer
  guiDrawViewport();

  // Handle project loading (synchronous), may trigger a scene loading
  loadProjectIfNeeded();

  // synchronous with no progress bar if benchmarkEnabled
  // asynchronous and multi-frame progress bar update if not benchmarking.
  guiLoadSceneAndDrawProgressIfNeeded();

  // we never show the UI elements in benchmark mode
  if(*m_pBenchmarkEnabled)
    return;

  // Draw the UI parts

  guiDrawAssetsWindow();

  guiDrawPropertiesWindow();

  guiDrawRendererStatisticsWindow();

  vk_gaussian_splatting::guiDrawMemoryStatisticsWindow(&m_showMemoryStatistics);

  guiDrawShaderFeedbackWindow();

  guiDrawFooterBar();
}

void GaussianSplattingUI::guiLoadSceneAndDrawProgressIfNeeded(void)
{
#ifdef WITH_DEFAULT_SCENE_FEATURE
  // load a default scene if none was provided by command line
  if(prmScene.enableDefaultScene && m_loadedSceneFilename.empty() && prmScene.sceneLoadQueue.empty()
     && prmScene.projectToLoadFilename.empty() && m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
  {
    const std::vector<std::filesystem::path> defaultSearchPaths = getResourcesDirs();
    auto defaultPath = nvutils::findFile("flowers_1/flowers_1.ply", defaultSearchPaths);
    prmScene.pushLoadRequest(defaultPath, true);
    prmScene.enableDefaultScene = false;
  }
#endif

  // Process next item in queue
  if(!prmScene.sceneLoadQueue.empty() && m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
  {
    static bool             firstBatchRequest = true;
    const SceneLoadRequest& request           = prmScene.sceneLoadQueue.front();
    bool                    doLoad            = true;

    // Show confirmation dialog only if:
    // Not loading from project
    if(firstBatchRequest && !request.porcelain)
    {
      ImGui::OpenPopup("Load .ply file ?");
      firstBatchRequest = false;
    }

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    // this block is executed only if OpenPopup was executed
    if(ImGui::BeginPopupModal("Load .ply file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      doLoad = false;

      ImGui::Text("Load additional splat set or reset project?");
      if(prmScene.sceneLoadQueue.size() > 1)
      {
        ImGui::Text("Queue has %zu file(s) pending.", prmScene.sceneLoadQueue.size());
      }
      ImGui::Separator();

      if(ImGui::Button("Import", ImVec2(120, 0)))
      {
        // Import - continue without reset
        doLoad = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if(ImGui::Button("Reset", ImVec2(120, 0)))
      {
        // Reset and continue
        reset();
        doLoad = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if(ImGui::Button("Cancel All", ImVec2(120, 0)))
      {
        // Cancel entire queue
        prmScene.sceneLoadQueue.clear();
        prmScene.projectToLoadFilename.clear();
        prmScene.projectLoadPorcelain = false;
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doLoad)
    {
      m_loadedSceneFilename = request.path;
      vkDeviceWaitIdle(m_device);

      if(prmScene.sceneLoadQueue.size() > 1)
      {
        LOGI("Start loading file %s (%zu more in queue)\n", request.path.string().c_str(), prmScene.sceneLoadQueue.size() - 1);
      }
      else
      {
        LOGI("Start loading file %s\n", request.path.string().c_str());
        // We process last request of the queue
        // reset flag for next batch
        firstBatchRequest = true;
      }

      // Use pre-configured splat set if provided (project loading)
      // Otherwise create a new one
      std::shared_ptr<SplatSetVk> splatSetToLoad = request.splatSet ? request.splatSet : std::make_shared<SplatSetVk>();

      if(!m_plyLoader.loadScene(request.path, splatSetToLoad))
      {
        LOGE("Error: cannot start scene load while loader is not ready status=%d\n", m_plyLoader.getStatus());
        // Remove failed request
        prmScene.sceneLoadQueue.pop_front();
      }
      else
      {
        // Store request for later (needed to access pre-configured instance)
        m_currentLoadRequest = request;

        // Remove from queue (will process in completion handler)
        prmScene.sceneLoadQueue.pop_front();

        // open the modal window that will collect results
        ImGui::OpenPopup("Loading");
      }
    }
  }

  // display loading jauge modal window
  // Always center this window when appearing
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if(ImGui::BeginPopupModal("Loading", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    // specific wait for benchmarking mode
    // prevent display of loading jauge and frame advancing while loading
    // ensure scene is loaded before moving to next frame
    if(*m_pBenchmarkEnabled)
    {
      while(m_plyLoader.getStatus() == PlyLoaderAsync::State::E_LOADING)
      {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
      }
    }
    // managment of async load
    switch(m_plyLoader.getStatus())
    {
      case PlyLoaderAsync::State::E_LOADING: {
        ImGui::Text("%s", m_plyLoader.getFilename().string().c_str());
        if(!prmScene.sceneLoadQueue.empty())
        {
          ImGui::Text("(%zu more file(s) queued)", prmScene.sceneLoadQueue.size());
        }
        ImGui::ProgressBar(m_plyLoader.getProgress(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f));
      }
      break;
      case PlyLoaderAsync::State::E_FAILURE: {
        ImGui::Text("Error: invalid ply file");
        if(ImGui::Button("Ok", ImVec2(120, 0)))
        {
          m_loadedSceneFilename = "";
          // destroy scene just in case it was
          // loaded but not properly since in error
          deinitScene();
          // set ready for next load
          m_plyLoader.reset();
          ImGui::CloseCurrentPopup();
        }
      }
      break;
      case PlyLoaderAsync::State::E_LOADED: {

        // Create splat set asset (data is already in the SplatSetVk object from loader)
        auto loadedSplatSet = m_assets.splatSets.createSplatSet(m_loadedSceneFilename.string(), m_plyLoader.getLoadedSplatSet());

        // If request provided a pre-configured instance, use it
        // Otherwise create a new one with identity transform
        if(m_currentLoadRequest.instance)
        {
          // Pre-configured instance (from project loading)
          // Instance already has transform, material, etc. set by project loader
          // Just register it with the manager and associate with the loaded splat set
          m_currentLoadRequest.instance->splatSet = loadedSplatSet;

          // Register the pre-configured instance
          m_selectedSplatInstance = m_assets.splatSets.registerInstance(loadedSplatSet, m_currentLoadRequest.instance);

          LOGD("Loaded with pre-configured instance (project mode)\n");

          // Handle additional instances sharing the same splat set (Version 1+ project files)
          for(auto& additionalInstance : m_currentLoadRequest.additionalInstances)
          {
            additionalInstance->splatSet = loadedSplatSet;
            m_assets.splatSets.registerInstance(loadedSplatSet, additionalInstance);
            LOGD("  Created additional instance sharing same splat set\n");
          }
        }
        else
        {
          // Standard path: create new instance with identity transform
          m_selectedSplatInstance = m_assets.splatSets.createInstance(loadedSplatSet);

          LOGD("Loaded with new default instance\n");
        }

        // createSplatSet and createInstance/registerInstance already set appropriate manager requests
        // No shader rebuild needed - bindless system handles descriptor updates at runtime

        // add only if not loaded by project or command
        if(!m_currentLoadRequest.porcelain)
          guiAddToRecentFiles(m_loadedSceneFilename);

        // set ready for next load
        m_plyLoader.reset();

        // Close modal only if queue is empty
        // Otherwise, next file will start loading automatically
        if(prmScene.sceneLoadQueue.empty())
        {
          ImGui::CloseCurrentPopup();
        }
      }
      break;
      default: {
        // nothing to do for READY or SHUTDOWN
      }
    }
    ImGui::EndPopup();
  }
}

void GaussianSplattingUI::guiDrawSortingSelector(bool inMenuBar)
{
  namespace PE = nvgui::PropertyEditor;

  static constexpr const char* tooltip =
      "Sorting method for pipelines using rasterization:\n"
      "- GPU radix: GPU-based radix sort (fast).\n"
      "- CPU async: Multi-threaded CPU sorting (slow to ultra slow).\n"
      "- Stochastic splat: Probabilistic accept/reject (no sorting needed, ultra fast, noisy).";

  bool changed = false;
  if(inMenuBar)
  {
    // Menu bar style: simple combo without property editor wrapper
    ImGui::SetNextItemWidth(150.0f);
    changed = m_ui.enumCombobox(GUI_SORTING, "##SortingMethod", &prmRaster.sortingMethod);
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("%s", tooltip);
  }
  else
  {
    // Property editor style: with label and tooltip
    changed = PE::entry(
        "Sorting method", [&]() { return m_ui.enumCombobox(GUI_SORTING, "##ID", &prmRaster.sortingMethod); }, tooltip);
  }

  if(changed)
  {
    m_requestUpdateShaders = true;

    // Handle frustum culling mode changes
    // GPU radix sort and stochastic splat both use the distance compute shader, so they support FRUSTUM_CULLING_AT_DIST
    const bool usesDistShader =
        (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
    if(!usesDistShader && prmRaster.frustumCulling == FRUSTUM_CULLING_AT_DIST)
    {
      prmRaster.frustumCulling = FRUSTUM_CULLING_AT_RASTER;
      m_requestUpdateShaders   = true;
    }
    if(usesDistShader && prmRaster.frustumCulling != FRUSTUM_CULLING_AT_DIST)
    {
      prmRaster.frustumCulling = FRUSTUM_CULLING_AT_DIST;
      m_requestUpdateShaders   = true;
    }

    // Handle size culling mode changes
    // Size culling only works with the distance compute shader
    if(!usesDistShader && prmRaster.sizeCulling == SIZE_CULLING_ENABLED)
    {
      prmRaster.sizeCulling  = SIZE_CULLING_DISABLED;
      m_requestUpdateShaders = true;
    }
  }
}

void GaussianSplattingUI::guiDrawLightingModeSelector(bool inMenuBar)
{
  namespace PE = nvgui::PropertyEditor;

  static constexpr const char* tooltip =
      "- Lighting off: no lighting computed.\n"
      "- Direct lighting: single-bounce shading from lights (no reflections/refractions).\n"
      "- Indirect lighting: full path tracing with bounces, reflections and refractions (ray tracing only).";

  bool changed = false;
  if(inMenuBar)
  {
    ImGui::SetNextItemWidth(150.0f);
    changed = m_ui.enumCombobox(GUI_LIGHTING_MODE, "##LightingMode", (int*)&prmRender.lightingMode);
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("%s", tooltip);
  }
  else
  {
    changed = PE::entry(
        "Lighting mode", [&]() { return m_ui.enumCombobox(GUI_LIGHTING_MODE, "##ID", (int*)&prmRender.lightingMode); }, tooltip);
  }

  if(changed)
  {
    m_requestUpdateShaders = true;
  }
}

void GaussianSplattingUI::guiDrawShadowsModeSelector(bool inMenuBar)
{
  namespace PE = nvgui::PropertyEditor;

  static constexpr const char* tooltip =
      "For pipelines using ray tracing:\n"
      "- Shadows off: no shadow rays traced.\n"
      "- Hard shadows: sharp point-sampled shadows.\n"
      "- Soft shadows: stochastic disk-sampled shadows around lights.";

  bool disabled = (prmRender.lightingMode == LightingMode::eLightingDisabled || !isRtxPipelineActive());

  bool changed = false;
  if(inMenuBar)
  {
    ImGui::BeginDisabled(disabled);
    ImGui::SetNextItemWidth(150.0f);
    changed = m_ui.enumCombobox(GUI_SHADOWS_MODE, "##ShadowsMode", (int*)&prmRender.shadowsMode);
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("%s", tooltip);
    ImGui::EndDisabled();
  }
  else
  {
    ImGui::BeginDisabled(disabled);
    changed = PE::entry(
        "Shadows mode", [&]() { return m_ui.enumCombobox(GUI_SHADOWS_MODE, "##ID", (int*)&prmRender.shadowsMode); }, tooltip);
    ImGui::EndDisabled();
  }

  if(changed)
  {
    m_requestUpdateShaders = true;
  }
}

void GaussianSplattingUI::guiDrawTracingStrategySelector(bool inMenuBar)
{
  namespace PE = nvgui::PropertyEditor;

  static constexpr const char* tooltip =
      "Sorting method for pipelines using ray tracing:\n"
      "- All pass: process all gaussians along each ray.\n"
      "- Stochastic pass: per pass stochastic transparency.\n"
      "- Stochastic any hit: per hit stochastic transparency.";

  bool disabled = !isRtxPipelineActive();

  bool changed = false;
  if(inMenuBar)
  {
    ImGui::BeginDisabled(disabled);
    ImGui::SetNextItemWidth(150.0f);
    changed = m_ui.enumCombobox(GUI_RTX_TRACE_STRATEGY, "##TraceStrategy", &prmRtx.rtxTraceStrategy);
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("%s", tooltip);
    ImGui::EndDisabled();
  }
  else
  {
    ImGui::BeginDisabled(disabled);
    changed = PE::entry(
        "Trace strategy", [&]() { return m_ui.enumCombobox(GUI_RTX_TRACE_STRATEGY, "##ID", &prmRtx.rtxTraceStrategy); }, tooltip);
    ImGui::EndDisabled();
  }

  if(changed)
  {
    m_requestUpdateShaders = true;
  }
}

void GaussianSplattingUI::guiDrawViewport()
{
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Display the appropriate buffer (skip during GBuffer reinit to avoid stale descriptors)
    ImVec2 imageSize = ImGui::GetContentRegionAvail();
    ImVec2 imagePos  = ImGui::GetCursorScreenPos();
    if(!m_requestGBufferReinit)
    {
      VkDescriptorSet displayDescriptor = getPresentationImageDescriptorSet();
      ImGui::Image((ImTextureID)displayDescriptor, imageSize);
    }
    else
    {
      ImGui::Dummy(imageSize);
    }

    // Cache image hover state now (later overlays may create other items)
    const bool   imageHovered = ImGui::IsItemHovered();
    ImGuiIO&     io           = ImGui::GetIO();
    const ImVec2 mp           = io.MousePos;                                   // Mouse position in screen space
    const ImVec2 mouseInImage = ImVec2(mp.x - imagePos.x, mp.y - imagePos.y);  // (0,0) top-left of image
    const bool   mouseInBounds =
        (mouseInImage.x >= 0.0f && mouseInImage.y >= 0.0f && mouseInImage.x < imageSize.x && mouseInImage.y < imageSize.y);

    // Check if user interacted with the viewport (for disabling comparison mode)
    // Detect any mouse button click or mouse wheel scroll
    bool viewportClicked = ImGui::IsItemClicked(ImGuiMouseButton_Left) || ImGui::IsItemClicked(ImGuiMouseButton_Right)
                           || ImGui::IsItemClicked(ImGuiMouseButton_Middle);

    // Check for mouse wheel scroll over the viewport
    bool viewportScrolled = false;
    if(imageHovered && ImGui::GetIO().MouseWheel != 0.0f)
    {
      viewportScrolled = true;
    }

    // ------------------------------------------------------------------------
    // Persistent cursor target overlay: interaction, drawing, and cursor override.
    // If dragging the target, block ALL other viewport interactions (simple modal behavior).
    // ------------------------------------------------------------------------
    bool cursorTargetLocksViewport = false;
    if(m_showCursorTargetOverlay)
    {
      constexpr float kTargetIconScale = 2.4f;

      // Initialize target position when enabling it (use mouse if valid, else center)
      if(m_cursorTargetPos.x < 0.0f || m_cursorTargetPos.y < 0.0f)
      {
        m_cursorTargetPos = mouseInBounds ? mouseInImage : ImVec2(imageSize.x * 0.5f, imageSize.y * 0.5f);
      }

      // Clamp target to image bounds
      m_cursorTargetPos.x = std::clamp(m_cursorTargetPos.x, 0.0f, std::max(0.0f, imageSize.x - 1.0f));
      m_cursorTargetPos.y = std::clamp(m_cursorTargetPos.y, 0.0f, std::max(0.0f, imageSize.y - 1.0f));

      const float  hitR         = (ImGui::GetFontSize() * kTargetIconScale) * 0.60f;
      const ImVec2 targetCenter = ImVec2(imagePos.x + m_cursorTargetPos.x, imagePos.y + m_cursorTargetPos.y);
      const float  dx           = mp.x - targetCenter.x;
      const float  dy           = mp.y - targetCenter.y;
      const bool   overTarget   = (dx * dx + dy * dy) <= (hitR * hitR);

      const bool lmbPressed = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
      const bool lmbDown    = io.MouseDown[ImGuiMouseButton_Left];

      if(imageHovered && lmbPressed && overTarget)
        m_cursorTargetDragging = true;
      if(!lmbDown)
        m_cursorTargetDragging = false;

      if(m_cursorTargetDragging && imageHovered)
      {
        m_cursorTargetPos         = mouseInImage;
        cursorTargetLocksViewport = true;
        io.WantCaptureMouse       = true;
      }

      // Draw reticle glyph (shadow + main)
      const float  dpi      = ImGui::GetWindowDpiScale();
      ImDrawList*  dl       = ImGui::GetWindowDrawList();
      ImFont*      font     = ImGui::GetFont();
      const float  fontSize = ImGui::GetFontSize() * kTargetIconScale;
      const char*  icon     = ICON_MS_CENTER_FOCUS_WEAK;
      const ImVec2 ts       = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, icon);
      const ImVec2 p        = ImVec2(targetCenter.x - ts.x * 0.5f, targetCenter.y - ts.y * 0.5f);
      dl->AddText(font, fontSize, ImVec2(p.x + 1.5f * dpi, p.y + 1.5f * dpi), IM_COL32(0, 0, 0, 220), icon);
      dl->AddText(font, fontSize, p, IM_COL32(255, 255, 255, 210), icon);

      // Provide cursor to shader from target position
      prmFrame.cursor = {int(std::round(m_cursorTargetPos.x)), int(std::round(m_cursorTargetPos.y))};
    }
    else
    {
      // Provide cursor to shader from mouse (only when over the image)
      if(!mouseInBounds)
        prmFrame.cursor.x = prmFrame.cursor.y = -1;
      else
        prmFrame.cursor = {int(std::round(mouseInImage.x)), int(std::round(mouseInImage.y))};
    }

    // When dragging the cursor target, disable viewport interactions for this frame
    if(cursorTargetLocksViewport)
    {
      viewportClicked  = false;
      viewportScrolled = false;
    }

    // If the summary overlay is on top of the viewport, block viewport-based interactions underneath
    // (notably image-compare overlay handling that keys off viewport clicks/scrolls).
    if(m_showSummaryOverlay && m_summaryOverlayRectValid)
    {
      const ImVec2 mousePos = ImGui::GetIO().MousePos;
      if(mousePos.x >= m_summaryOverlayRectMin.x && mousePos.x <= m_summaryOverlayRectMax.x
         && mousePos.y >= m_summaryOverlayRectMin.y && mousePos.y <= m_summaryOverlayRectMax.y)
      {
        viewportClicked                 = false;
        viewportScrolled                = false;
        ImGui::GetIO().WantCaptureMouse = true;
      }
    }

    // Process transform gizmo input if a mesh is selected AND editing mode is active
    // Note: Always process when attached (not just when hovering) to ensure hover feedback works
    // Must also check editing mode to prevent interaction when gizmo is visually hidden
    if(!cursorTargetLocksViewport && m_helpers.isEditingMode() && m_helpers.transform.isAttached())
    {
      ImVec2 wp = ImGui::GetWindowPos();
      ImVec2 ws = ImGui::GetWindowSize();

      // Convert to viewport coordinates
      glm::vec2 mousePos(mp.x - wp.x, mp.y - wp.y);

      // Track mouse delta (persistent across frames)
      static glm::vec2 lastMousePos = mousePos;
      glm::vec2        mouseDelta   = mousePos - lastMousePos;
      lastMousePos                  = mousePos;

      // Mouse button states
      bool mouseDown     = io.MouseDown[ImGuiMouseButton_Left];
      bool mousePressed  = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
      bool mouseReleased = ImGui::IsMouseReleased(ImGuiMouseButton_Left);

      // Process gizmo input
      const VkExtent2D& viewportSize = m_app->getViewportSize();
      bool              gizmoHandledInput =
          m_helpers.transform.processInput(mousePos, mouseDelta, mouseDown, mousePressed, mouseReleased,
                                           cameraManip->getViewMatrix(), cameraManip->getPerspectiveMatrix(),
                                           glm::vec2(viewportSize.width, viewportSize.height));

      // If gizmo handled input (clicked or dragging), prevent camera manipulation
      if(gizmoHandledInput || m_helpers.transform.isDragging())
      {
        io.WantCaptureMouse = true;
      }
    }

    // Keep image-compare UI informed about temporal sampling status (for its metrics header display).
    m_imageCompareUI.setTemporalSamplingState(prmRtx.temporalSampling, prmFrame.frameSampleId, prmFrame.frameSampleMax);

    // Draw comparison overlay if enabled
    if(!cursorTargetLocksViewport && prmComparison.enabled && m_imageCompare.hasValidCaptureImage())
    {
      // Update titles before rendering overlay
      std::string refTitle = getSettingsString(m_referenceCapturePipeline, m_referenceCaptureVisualization);
      std::string curTitle = getSettingsString(prmSelectedPipeline, prmRender.visualize);
      m_imageCompareUI.setCaptureViewTitle(refTitle);
      m_imageCompareUI.setCurrentViewTitle(curTitle);

      // Render overlay and check if capture was requested
      bool captureRequested = m_imageCompareUI.renderOverlay(imagePos, imageSize, viewportClicked, viewportScrolled);

      if(captureRequested)
      {
        // Store current settings and request capture
        m_referenceCapturePipeline      = prmSelectedPipeline;
        m_referenceCaptureVisualization = prmRender.visualize;
        m_requestCaptureComparison      = true;
      }
    }

    // Draw summary info overlay if enabled (mutually exclusive with comparison overlay)
    guiDrawSummaryOverlay(imagePos, imageSize);

    ImVec2 wp = ImGui::GetWindowPos();
    ImVec2 ws = ImGui::GetWindowSize();

    // display the basis widget at bottom left
    float  size   = 25.F;
    ImVec2 offset = ImVec2(size * 1.1F, -size * 1.1F) * ImGui::GetWindowDpiScale();
    ImVec2 pos    = ImVec2(wp.x, wp.y + ws.y) + offset;
    nvgui::Axis(pos, cameraManip->getViewMatrix(), size);

    ImGui::End();
    ImGui::PopStyleVar();
  }
}

void GaussianSplattingUI::guiDrawAssetsWindow()
{
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ChildBg]);

  if(ImGui::Begin("Assets"))
  {
    guiDrawRendererTree();

    guiDrawCameraTree();

    guiDrawLightTree();

    guiDrawRadianceFieldsTree();

    guiDrawObjectTree();
  }
  ImGui::End();

  ImGui::PopStyleColor();
}

void GaussianSplattingUI::resetSelection()
{
  m_selectedAsset             = GUI_NONE;
  m_selectedCameraPresetIndex = -1;
  m_selectedMeshInstance      = nullptr;
  m_selectedSplatInstance     = nullptr;
  m_selectedLightInstance     = nullptr;
  m_helpers.transform.clearAttachment();
}

void GaussianSplattingUI::reset()
{
  // Clear all selections and detach transform helper BEFORE clearing assets
  // This prevents dangling pointers when instances are deleted
  resetSelection();

  // Reset UI settings to their defaults
  m_helpers.setEditingMode(true);   // Editing mode on by default
  m_showLightProxies = true;        // Light proxies visible by default
  m_helpers.grid.setVisible(true);  // Grid visible by default

  // Clear shader feedback cached data (stale from previous scene)
  m_shaderFeedbackUI.reset();

  // Call base class reset to perform the actual scene reset
  GaussianSplatting::reset();
}

void GaussianSplattingUI::selectMeshInstance(std::shared_ptr<MeshInstanceVk> instance)
{
  if(!instance)
  {
    resetSelection();
    return;
  }

  resetSelection();  // Clear any previous selection

  m_selectedAsset        = GUI_MESH;
  m_selectedMeshInstance = instance;

  // Attach transform helper to this mesh's transform components
  m_helpers.transform.attachTransform(&instance->translation, &instance->rotation, &instance->scale, TransformHelperVk::ShowAll);

  // Set callback specific to mesh instances
  m_helpers.transform.setOnTransformChange([this, meshInstance = instance]() {
    // Rebuild transform matrix from components using utility function
    computeTransform(meshInstance->scale, meshInstance->rotation, meshInstance->translation, meshInstance->transform,
                     meshInstance->transformInverse, meshInstance->transformRotScaleInverse);

    // Mark for GPU update
    m_assets.meshes.updateInstanceTransform(meshInstance);
  });
}

void GaussianSplattingUI::selectSplatSetInstance(std::shared_ptr<SplatSetInstanceVk> instance)
{
  if(!instance || !instance->splatSet)
  {
    resetSelection();
    return;
  }

  resetSelection();  // Clear any previous selection

  m_selectedAsset         = GUI_SPLATSET;
  m_selectedSplatInstance = instance;

  // Attach transform helper to this splat set instance's transform components
  m_helpers.transform.attachTransform(&instance->translation, &instance->rotation, &instance->scale, TransformHelperVk::ShowAll);

  // Set callback specific to splat set instances
  m_helpers.transform.setOnTransformChange([this, splatInstance = instance]() {
    // Rebuild transform matrix from components using utility function
    computeTransform(splatInstance->scale, splatInstance->rotation, splatInstance->translation,
                     splatInstance->transform, splatInstance->transformInverse, splatInstance->transformRotScaleInverse);

    // Mark for GPU update
    m_assets.splatSets.updateInstanceTransform(splatInstance);
  });
}

void GaussianSplattingUI::selectLightInstance(std::shared_ptr<LightSourceInstanceVk> instance)
{
  if(!instance || !instance->lightSource)
  {
    resetSelection();
    return;
  }

  resetSelection();  // Clear any previous selection

  m_selectedAsset         = GUI_LIGHT;
  m_selectedLightInstance = instance;

  // Create static dummy rotation and scale (not used for lights)
  static glm::vec3 dummyRotation(0.0f);
  static glm::vec3 dummyScale(1.0f);

  // Attach transform helper to light INSTANCE translation and rotation
  m_helpers.transform.attachTransform(&instance->translation, &instance->rotation, &dummyScale,
                                      TransformHelperVk::ShowTranslation | TransformHelperVk::ShowRotation);

  // Set callback to update light instance position
  m_helpers.transform.setOnTransformChange([this, lightInstance = instance]() {
    m_assets.lights.updateLight(lightInstance);  // Update instance position/proxy
  });
}

void GaussianSplattingUI::guiDrawRendererTree()
{
  static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  bool node_open = false;

  ImGuiTreeNodeFlags node_flags;

  // Renderer
  std::string pipelineName = m_ui.getEnums(GUI_PIPELINE)[prmSelectedPipeline].name;
  node_flags               = base_flags;
  if(m_selectedAsset == GUI_RENDERER)
    node_flags |= ImGuiTreeNodeFlags_Selected;
  ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  node_open = ImGui::TreeNodeEx(ICON_MS_CAMERA " Renderer", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    resetSelection();
    m_selectedAsset             = GUI_RENDERER;
    m_selectedCameraPresetIndex = -1;
  }
  if(node_open)
  {
    // display the pipeline selector
    int i = 0;
    {
      ImGui::Indent(30);
      ImGui::Text(ICON_MS_SUBDIRECTORY_ARROW_RIGHT);
      ImGui::SameLine();
      if(m_ui.enumCombobox(GUI_PIPELINE, "##ID", &prmSelectedPipeline))
      {
        m_requestUpdateShaders = true;
      }
      ImGui::Unindent(30);
    }
    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawCameraTree()
{

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  if(m_selectedAsset == GUI_CAMERA && m_selectedCameraPresetIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  bool node_open = ImGui::TreeNodeEx(ICON_MS_PHOTO_CAMERA " Camera", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    resetSelection();
    m_selectedAsset             = GUI_CAMERA;
    m_selectedCameraPresetIndex = -1;
  }
  ImGui::PushID(-1);
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
  if(ImGui::SmallButton(ICON_MS_ADD_A_PHOTO))
  {
    m_assets.cameras.storeCurrentCamera();
  }
  nvgui::tooltip("Store current camera settings in presets");
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
  if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
  {
    auto name = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Import INRIA Camera file", "INRIA Camera file|*.json");
    if(!name.empty())
    {
      importCamerasINRIA(name.string(), m_assets.cameras);
    }
  }
  nvgui::tooltip("Import INRIA Camera file");
  ImGui::PopID();

  if(node_open)
  {
    // display the camera tree
    for(int i = 0; i < m_assets.cameras.size(); ++i)
    {
      ImGui::PushID(i);
      node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_CAMERA && m_selectedCameraPresetIndex == i)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      const auto name = fmt::format(ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Camera Preset ({})", i + 1);

      bool node_open = ImGui::TreeNodeEx(name.c_str(), node_flags);
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        resetSelection();
        m_selectedAsset             = GUI_CAMERA;
        m_selectedCameraPresetIndex = i;
      }
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 110);
      if(ImGui::SmallButton(ICON_MS_LOCAL_SEE))
      {
        // Check if shader rebuild is needed (camera model or depth of field flag change)
        m_requestUpdateShadersAfterCameraAnim = cameraPresetNeedsShaderRebuild(i);

        // Load preset with animation (instantSet=false)
        m_assets.cameras.loadPreset(i, false);
        m_lastLoadedCamera          = i;
        m_selectedCameraPresetIndex = -1;  // Will select current camera
      }
      nvgui::tooltip("Load camera preset");
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
      if(ImGui::SmallButton(ICON_MS_ADD_A_PHOTO))
      {
        m_assets.cameras.setPreset(i, m_assets.cameras.getCamera());
        m_lastLoadedCamera          = i;
        m_selectedCameraPresetIndex = -1;  // Will select current camera
        m_requestUpdateShaders      = true;
      }
      nvgui::tooltip("Overwrite preset with current camera settings");
      if(m_assets.cameras.size() > 1)
      {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
        if(ImGui::SmallButton(ICON_MS_DELETE))
        {
          m_assets.cameras.erasePreset(i);
        }
        nvgui::tooltip("Delete preset");
      }
      ImGui::PopID();
    }
    //
    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawLightTree()
{
  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;
  if(m_selectedAsset == GUI_LIGHT && !m_selectedLightInstance)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::SetNextItemOpen(true, ImGuiCond_Once);

  bool node_open = ImGui::TreeNodeEx(ICON_MS_LIGHT_MODE " Lights", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    resetSelection();
  }
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
  if(ImGui::SmallButton(ICON_MS_ADD_CIRCLE))
  {
    auto newInstance = m_assets.lights.createLight();
    selectLightInstance(newInstance);
    // pendingRequests set by createLight()
  }
  nvgui::tooltip("Create light");

  if(node_open)
  {
    // display the lights tree
    for(size_t i = 0; i < m_assets.lights.size(); ++i)
    {
      ImGui::PushID((int)i);

      auto instance = m_assets.lights.getInstance(i);
      if(!instance)
      {
        ImGui::PopID();
        continue;
      }

      ImGuiTreeNodeFlags node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_LIGHT && m_selectedLightInstance == instance)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      const std::string& lightName = instance->name;
      bool               node_open =
          ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, ICON_MS_SUBDIRECTORY_ARROW_RIGHT "%s", lightName.c_str());
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        selectLightInstance(instance);
      }

      // Copy button
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
      if(ImGui::SmallButton(ICON_MS_CONTENT_COPY))
      {
        auto newInstance = m_assets.lights.duplicateInstance(instance);
        selectLightInstance(newInstance);
      }
      nvgui::tooltip("Duplicate light");

      // Delete button
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
      if(ImGui::SmallButton(ICON_MS_DELETE))
      {
        m_assets.lights.deleteInstance(instance);
        // pendingRequests set by deleteInstance()
        resetSelection();
      }
      nvgui::tooltip("Delete light");

      ImGui::PopID();
    }
    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawRadianceFieldsTree()
{
  // Count instances and check for RTX errors
  const auto& instances     = m_assets.splatSets.getInstances();
  size_t      instanceCount = instances.size();

  std::string rtxError    = " ";
  bool        hasRtxError = false;

  // Check if any splat set has RTX errors
  // Show error even if we're in raster mode (since we may have fallen back due to the error)
  if(instanceCount > 0)
  {
    // Check RTX for actual errors (not just "not yet initialized")
    if(m_assets.splatSets.isRtxError())
    {
      rtxError    = " (RTX allocation failed)";
      hasRtxError = true;
    }
  }

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  if(m_selectedAsset == GUI_SPLATSET && !m_selectedSplatInstance)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::SetNextItemOpen(true, ImGuiCond_Once);

  if(hasRtxError)
  {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
  }

  bool node_open =
      ImGui::TreeNodeEx(fmt::format(ICON_MS_GRAIN " Radiance Fields ({}){}", instanceCount, rtxError).c_str(), node_flags);

  if(hasRtxError)
    ImGui::PopStyleColor();

  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    resetSelection();
    m_selectedAsset = GUI_SPLATSET;
  }

  // Import button
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
  if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
  {
    auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Splat Set",
                                            "All Files|*.ply;*.spz;*.splat|PLY Files|*.ply|SPZ files|*.spz|SPLAT files|*.splat");
    if(!path.empty())
    {
      prmScene.pushLoadRequest(path, false);  // Don't auto-reset, user can choose in dialog
    }
  }

  if(node_open)
  {
    // Note: Iterate by index (not reference) to avoid iterator invalidation when duplicating
    size_t instanceCount = instances.size();
    for(size_t i = 0; i < instanceCount; ++i)
    {
      const auto& instance = instances[i];
      if(!instance || !instance->splatSet)
        continue;  // Skip invalid instances

      ImGui::PushID(static_cast<int>(instance->index));

      ImGuiTreeNodeFlags instanceFlags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_SPLATSET && m_selectedSplatInstance == instance)
        instanceFlags |= ImGuiTreeNodeFlags_Selected;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)instance->index, instanceFlags,
                                         ICON_MS_SUBDIRECTORY_ARROW_RIGHT "%s", instance->displayName.c_str());

      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        selectSplatSetInstance(instance);
      }

      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
      if(ImGui::SmallButton(ICON_MS_CONTENT_COPY))
      {
        // Duplicate the instance (manager handles vector reallocation safely)
        auto newInstance = m_assets.splatSets.duplicateInstance(instance);
        if(newInstance)
        {
          selectSplatSetInstance(newInstance);
          // No shader rebuild needed - bindless system handles new instance at runtime
        }
      }
      nvgui::tooltip("Duplicate instance");
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
      if(ImGui::SmallButton(ICON_MS_DELETE))
      {
        // Mark for deletion (delete after loop to avoid iterator invalidation)
        m_assets.splatSets.deleteInstance(instance);
      }
      nvgui::tooltip("Delete instance");

      ImGui::PopID();
    }

    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawObjectTree()
{

  namespace PE = nvgui::PropertyEditor;

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  // Only select parent if no specific mesh instance is selected
  if(m_selectedAsset == GUI_MESH && !m_selectedMeshInstance)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::SetNextItemOpen(true, ImGuiCond_Once);

  if(m_objListUpdated)
  {
    ImGui::SetNextItemOpen(true);
    m_objListUpdated = false;
  }
  // Count only user objects (exclude light proxies and other internal meshes)
  size_t userObjectCount = 0;
  for(const auto& inst : m_assets.meshes.instances)
  {
    if(inst && inst->shouldShowInUI() && inst->type == MeshType::eObject)
    {
      userObjectCount++;
    }
  }

  bool node_open = ImGui::TreeNodeEx(fmt::format(ICON_MS_DEPLOYED_CODE " Mesh Models ({})", userObjectCount).c_str(), node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    resetSelection();
  }
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
  if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
  {
    prmScene.meshToImportFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load obj file", "OBJ|*.obj");
  }
  // Handle the request form file open or from drag and drop
  if(!prmScene.meshToImportFilename.empty())
  {
    bool valid = true;

    const auto name               = prmScene.meshToImportFilename;
    prmScene.meshToImportFilename = "";  // reset the request
    if(!name.empty())
    {
      // synchronous load
      auto meshPtr = m_assets.meshes.loadModel(name);
      valid        = (meshPtr != nullptr);
    }

    if(!valid)
    {
      ImGui::OpenPopup("Obj Loading");
    }
    else
    {
      // Clear any existing gizmo attachment
      m_helpers.transform.clearAttachment();

      // Mesh manager will set its own pendingRequests (RebuildBLAS, UpdateDescriptors, etc.)
      // No shader rebuild needed - bindless system handles new mesh at runtime
      m_selectedAsset        = GUI_MESH;
      m_selectedMeshInstance = m_assets.meshes.m_lastCreatedInstance;  // Store pointer directly
      //
      m_objListUpdated = true;  // so that next loop will force the Object open if selected

      // Note: Gizmo attachment happens when user clicks the instance in the tree
    }

    // definition of the obj error popup
    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if(ImGui::BeginPopupModal("Obj Loading", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      ImGui::Text("Error: invalid obj file");
      if(ImGui::Button("Ok", ImVec2(120, 0)))
      {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
  }
  if(node_open)
  {
    // display the objects tree (excluding internal meshes like light proxies)
    // Note: Iterate by index (not reference) to avoid iterator invalidation when duplicating
    int    idx           = 0;
    size_t instanceCount = m_assets.meshes.instances.size();
    for(size_t i = 0; i < instanceCount; ++i)
    {
      const auto& instance = m_assets.meshes.instances[i];
      if(!instance || !instance->mesh)
        continue;  // Skip invalid

      // Skip instances marked for deletion
      if(!instance->shouldShowInUI())
        continue;

      // Skip internal mesh types (light proxies, etc.) - only show user objects
      if(instance->type != MeshType::eObject)
        continue;

      ImGui::PushID(idx++);
      ImGuiTreeNodeFlags node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_MESH && m_selectedMeshInstance == instance)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)instance.get(), node_flags,
                                         ICON_MS_SUBDIRECTORY_ARROW_RIGHT "%s", instance->name.c_str());
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        selectMeshInstance(instance);
      }
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
      if(ImGui::SmallButton(ICON_MS_CONTENT_COPY))
      {
        // Duplicate the mesh instance (manager handles vector reallocation safely)
        auto newInstance = m_assets.meshes.duplicateInstance(instance);
        if(newInstance)
        {
          selectMeshInstance(newInstance);
          // No shader rebuild needed - bindless system handles new instance at runtime
          m_objListUpdated = true;
        }
      }
      nvgui::tooltip("Duplicate instance");
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
      if(ImGui::SmallButton(ICON_MS_DELETE))
      {
        // Delete instance immediately (deferred VRAM cleanup)
        m_assets.meshes.deleteInstance(instance);
        if(m_selectedMeshInstance == instance)
        {
          m_selectedMeshInstance = nullptr;
          m_helpers.transform.clearAttachment();  // Clear gizmo when deleted
        }
        m_objListUpdated = true;
      }
      nvgui::tooltip("Delete instance");
      ImGui::PopID();
    }
    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawPropertiesWindow()
{
  if(ImGui::Begin("Properties"))
  {
    switch(m_selectedAsset)
    {
      case GUI_RENDERER:
        if(ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen))
        {
          guiDrawRendererProperties();
        }
        break;
      case GUI_SPLATSET:
        if(m_selectedSplatInstance)
        {
          guiDrawSplatSetProperties();
        }
        else
        {
          guiDrawCommonSplatSetProperties();
        }
        break;
      case GUI_MESH:
        // Validate pointer - if instance no longer in set, clear selection
        if(m_selectedMeshInstance)
        {
          auto it = std::find(m_assets.meshes.instances.begin(), m_assets.meshes.instances.end(), m_selectedMeshInstance);
          if(it == m_assets.meshes.instances.end())
          {
            m_selectedMeshInstance = nullptr;       // Clear selection if instance was deleted
            m_helpers.transform.clearAttachment();  // Clear gizmo
          }
        }
        if(m_selectedMeshInstance)
        {
          if(ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawMeshTransformProperties();
          }
          if(ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawMeshMaterialProperties();
          }
        }
        break;
      case GUI_CAMERA:
        m_selectedCameraPresetIndex = std::clamp<int64_t>(m_selectedCameraPresetIndex, -1, m_assets.cameras.size() - 1);
        //if(ImGui::CollapsingHeader("Camera Intrinsics", ImGuiTreeNodeFlags_DefaultOpen))
        {
          guiDrawCameraProperties();
        }
        if(m_selectedCameraPresetIndex == -1)
        {
          if(ImGui::CollapsingHeader("Navigation", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawNavigationProperties();
          }
        }
        break;
      case GUI_LIGHT:
        if(m_selectedLightInstance)
        {
          if(ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawLightProperties();
          }
        }
        break;
      default:
        // display nothing
        break;
    };
  }
  ImGui::End();
}

void GaussianSplattingUI::guiDrawRendererProperties()
{

  namespace PE = nvgui::PropertyEditor;

  PE::begin("## Global settings ");
  bool vsync = m_app->isVsync();
  if(PE::Checkbox("V-Sync", &vsync))
    m_app->setVsync(vsync);

  if(PE::entry(
         "Pipeline", [&]() { return m_ui.enumCombobox(GUI_PIPELINE, "##ID", &prmSelectedPipeline); }, "Selects the rendering method"))
  {
    m_requestUpdateShaders = true;
  }

  if(PE::entry(
         "Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
  {
    resetRenderSettings();
    m_requestUpdateShaders = true;
    // resetRenderSettings() may change global parameters - regenerate all splat sets
    m_assets.splatSets.markAllSplatSetsForRegeneration();
  }

  int colorFormatInt = static_cast<int>(m_colorFormat);
  if(PE::entry(
         "Color Format", [&]() { return m_ui.enumCombobox(GUI_COLOR_FORMAT, "##ColorFormat", &colorFormatInt); },
         "Color buffer format.\n"
         "Higher precision improves temporal accumulation quality but uses more memory.\n"
         "R8G8B8A8 UNORM: 32-bit (lowest memory, fastest rendering)\n"
         "R16G16B16A16 SFLOAT: 64-bit (default, good balance)\n"
         "R32G32B32A32 SFLOAT: 128-bit (highest precision)"))
  {
    m_colorFormat          = static_cast<VkFormat>(colorFormatInt);
    m_requestGBufferReinit = true;
    resetFrameCounter();
  }

  PE::end();

  PE::begin("## Visualization");

  auto visuMenu = GUI_VISUALIZE;
  if(m_dlss.isEnabled())
    visuMenu = GUI_VISUALIZE_DLSS_ON;

  // Visualization available for all ray tracing pipelines (pure RTX and hybrids)
  ImGui::BeginDisabled(!isRtxPipelineActive());

  if(PE::entry(
         "Visualize", [&]() { return m_ui.enumCombobox(visuMenu, "##ID", &prmRender.visualize); },
         "Selects the visualization mode.\nDLSS guide modes display the G-buffers used for DLSS (only when DLSS is enabled)."))
  {
    m_requestUpdateShaders = true;
  }
  switch(prmRender.visualize)
  {
    case VISUALIZE_CLOCK: {
      bool changed = PE::DragFloat2("Min/max", glm::value_ptr(prmRender.clockVisuMinMax), 0.01f);
      changed |= PE::SliderFloat("Shift", &prmRender.clockVisuShift, -1.0f, 1.0f);
      prmFrame.visuMinMax = prmRender.clockVisuMinMax;
      prmFrame.visuShift  = prmRender.clockVisuShift;
      if(changed)
        resetFrameCounter();
      break;
    }
    case VISUALIZE_DEPTH:
    case VISUALIZE_DEPTH_INTEGRATED:
    case VISUALIZE_DEPTH_FOR_DLSS: {
      bool changed = PE::DragFloat2("Min/max", glm::value_ptr(prmRender.depthVisuMinMax), 0.01f);
      changed |= PE::SliderFloat("Shift", &prmRender.depthVisuShift, -1.0f, 1.0f);
      prmFrame.visuMinMax = prmRender.depthVisuMinMax;
      prmFrame.visuShift  = prmRender.depthVisuShift;
      if(changed)
        resetFrameCounter();
      break;
    }
    case VISUALIZE_RAYHITS: {
      bool changed = PE::DragFloat2("Min/max", glm::value_ptr(prmRender.hitsVisuMinMax), 1.0f);
      changed |= PE::SliderFloat("Shift", &prmRender.hitsVisuShift, -1.0f, 1.0f);
      prmFrame.visuMinMax = prmRender.hitsVisuMinMax;
      prmFrame.visuShift  = prmRender.hitsVisuShift;
      if(changed)
        resetFrameCounter();
      break;
    }
  }
  ImGui::EndDisabled();

  PE::end();

  PE::begin("## Common settings");
  if(PE::Checkbox("Wireframe", &prmRender.wireframe, "Show particle bounds in wireframe "))
    m_requestUpdateShaders = true;

  int alphaThres = int(255.0 * prmFrame.alphaCullThreshold);
  if(PE::SliderInt("Alpha culling threshold", &alphaThres, 0, 255, "%d", 0, "Discard splats with low opacity (with low contribution)."))
  {
    prmFrame.alphaCullThreshold = (float)alphaThres / 255.0f;
  }

  if(PE::SliderInt("Maximum SH degree", (int*)&prmFrame.shDegree, 0, 3, "%d", 0,
                   "Sets the highest degree of Spherical Harmonics (SH) used for view-dependent effects."))
  {
    // not needed anymore
    // m_requestUpdateShaders = true;
  }

  if(PE::Checkbox("Show SH deg > 0 only", &prmRender.showShOnly,
                  "Removes the base color from SH degree 0, applying only color deduced from \n"
                  "higher-degree SH to a neutral gray. This helps visualize their contribution."))
    m_requestUpdateShaders = true;

  if(PE::Checkbox("Disable opacity gaussian ", &prmRender.opacityGaussianDisabled,
                  "Disables the alpha component of the Gaussians, making their full range visible.\n"
                  "This helps analyze splat distribution and scales, especially when combined with Splat Scale adjustments."))
    m_requestUpdateShaders = true;

  if(PE::entry(
         "Normal vectors", [&]() { return m_ui.enumCombobox(GUI_NORMAL_METHOD, "##ID", (int*)&prmRender.normalMethod); },
         "Select the method used to compute normal vectors for Gaussian particles.\n"
         "Max density plane: approximates the iso-density surface with a tangent plane at the\n"
         "  Gaussian center (StochasticSplats approach). Fast and good quality.\n"
         "Iso-surface ellipsoid: computes ray-ellipsoid surface intersection in canonical space.\n"
         "  More geometrically accurate for individual particles."))
    m_requestUpdateShaders = true;

  PE::DragFloat("Thin particle threshold", &prmRender.thinParticleThreshold, 0.0001f, 0.0f, 1.0f, "%.6f", 0,
                "Scale below which a particle axis is considered degenerate for normal computation.\n"
                "Particles with an axis thinner than this threshold are treated as flat disks\n"
                "(normal along the thin axis) instead of using the full ellipsoid computation.");

  guiDrawLightingModeSelector(false);
  guiDrawShadowsModeSelector(false);

  if(PE::entry(
         "Temporal sampling",
         [&]() { return m_ui.enumCombobox(GUI_TEMPORAL_SAMPLING, "##ID", &prmRtx.temporalSamplingMode); },
         "Enable accumulation of frame results over time.\n"
         "Automatic will activate sampling depending on other effects such as DoF or Pass Monte Carlo trace strategy.\n"
         "If enabled, the specified number of temporal samples will be accumulated over \"Temporal samples count\" frames,\n"
         "and the last accumulated frame will be presented without additional rendering.\n"
         "Note that rendering converges faster if v-sync is off.\n"
         "If disabled, the system renders in free run mode."))
  {
    resetFrameCounter();
    m_requestUpdateShaders = true;

    // Update metrics history size if comparison is active
    if(prmComparison.enabled && m_imageCompare.hasValidCaptureImage())
    {
      int historySize = prmRtx.temporalSampling ? prmFrame.frameSampleMax : 25;
      m_imageCompare.setMetricsHistorySize(historySize);
    }
  }

  if(PE::InputInt("Temporal samples count", &prmFrame.frameSampleMax, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue,
                  "Number of frames after which temporal sampling is stopped. \n"
                  "A value of 0 disables temporal sampling."))
  {
    prmFrame.frameSampleMax = std::clamp(prmFrame.frameSampleMax, 1, 100000);
    resetFrameCounter();

    // Update metrics history size if comparison is active and temporal sampling is enabled
    if(prmComparison.enabled && m_imageCompare.hasValidCaptureImage() && prmRtx.temporalSampling)
    {
      m_imageCompare.setMetricsHistorySize(prmFrame.frameSampleMax);
    }
  }

  PE::end();

  ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
  if(ImGui::BeginTabBar("##SpecificsBar", tab_bar_flags))
  {
    if(prmSelectedPipeline != PIPELINE_RTX)
    {
      if(ImGui::BeginTabItem("Rasterization specifics"))
      {
        PE::begin("## Raster settings");

        guiDrawSortingSelector(false);

        // CPU sorting options disabled for GPU radix and stochastic modes
        const bool cpuSortingDisabled =
            (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX || prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
        ImGui::BeginDisabled(cpuSortingDisabled);
        PE::Checkbox("Lazy CPU sorting", &prmRaster.cpuLazySort, "Perform sorting only if viewpoint changes");

        PE::Text("CPU sorting state", m_assets.splatSets.getCpuSorterStatus() == SplatSorterAsync::E_SORTING ? "Sorting" : "Idled");
        ImGui::EndDisabled();

        // Radio buttons for exclusive selection
        PE::entry(
            "Frustum culling",
            [&]() {
              if(ImGui::RadioButton("Disabled", prmRaster.frustumCulling == FRUSTUM_CULLING_NONE))
              {
                prmRaster.frustumCulling = FRUSTUM_CULLING_NONE;
                m_requestUpdateShaders   = true;
              }

              // GPU radix sort and stochastic splat both use the distance compute shader
              const bool usesDistShader = (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
                                          || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
              ImGui::BeginDisabled(!usesDistShader);
              if(ImGui::RadioButton("At distance stage", prmRaster.frustumCulling == FRUSTUM_CULLING_AT_DIST))
              {
                prmRaster.frustumCulling = FRUSTUM_CULLING_AT_DIST;
                m_requestUpdateShaders   = true;
              }
              ImGui::EndDisabled();

              if(ImGui::RadioButton("At raster stage", prmRaster.frustumCulling == FRUSTUM_CULLING_AT_RASTER))
              {
                prmRaster.frustumCulling = FRUSTUM_CULLING_AT_RASTER;
                m_requestUpdateShaders   = true;
              }
              return true;
            },
            "Defines where frustum culling is performed: in the distance compute shader or \n"
            "at rasterization (in vertex or mesh shader). Culling can also be disabled for performance comparisons.");

        PE::SliderFloat("Frustum dilation", &prmFrame.frustumDilation, 0.0f, 1.0f, "%.1f", 0,
                        "Adjusts the frustum culling bounds to account for the fact that visibility is tested \n"
                        "only at the center of each splat, rather than its full elliptical shape. A positive \n"
                        "value expands the frustum by the given percentage, reducing the risk of prematurely \n"
                        "discarding splats near the frustum boundaries.");

        // Size culling: cull splats whose projected bounding sphere is smaller than a threshold
        {
          // GPU radix sort and stochastic splat both use the distance compute shader
          const bool usesDistShader = (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
                                      || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
          ImGui::BeginDisabled(!usesDistShader);
          if(PE::Checkbox("Screen size culling", (bool*)&prmRaster.sizeCulling,
                          "Cull splats whose projected bounding sphere is smaller than the specified pixel coverage.\n"
                          "Only available when using the distance compute shader (GPU radix sort or stochastic splat)."))
          {
            m_requestUpdateShaders = true;
          }
          ImGui::EndDisabled();

          ImGui::BeginDisabled(!prmRaster.sizeCulling || !usesDistShader);
          PE::entry(
              "Min pixel coverage",
              [&]() {
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                return ImGui::DragFloat("##sizeCullingMinPixels", &prmFrame.sizeCullingMinPixels, 0.1f, 0.1f, 20.0f, "%.2f");
              },
              "Minimum projected pixel coverage for a splat to be visible.\n"
              "Splats with a projected bounding sphere diameter smaller than this value will be culled.");
          ImGui::EndDisabled();
        }

        if(PE::entry(
               "Dist WG size",
               [&]() { return m_ui.enumCombobox(GUI_DIST_SHADER_WG_SIZE, "##ID", &prmRaster.distShaderWorkgroupSize); },
               "Distance shader workgroup size"))
        {
          m_requestUpdateShaders = true;
        }

        if(PE::entry(
               "Mesh WG size",
               [&]() { return m_ui.enumCombobox(GUI_MESH_SHADER_WG_SIZE, "##ID", &prmRaster.meshShaderWorkgroupSize); },
               "Mesh shader workgroup size"))
        {
          m_requestUpdateShaders = true;
        }

        bool forceExtentProjection = prmSelectedPipeline == PIPELINE_VERT || prmSelectedPipeline == PIPELINE_MESH
                                     || prmSelectedPipeline == PIPELINE_HYBRID;

        ImGui::BeginDisabled(forceExtentProjection);
        if(PE::entry(
               "Projection Method",
               [&]() { return m_ui.enumCombobox(GUI_EXTENT_METHOD, "##ID", &prmRaster.extentProjection); },
               "Available for 3DGUT pipelines only, 3DGS allways uses Eigen.\n"
               "Method used to compute the 2D extent projection from the 3D covariance:\n"
               "- Eigen method leads to basis aligned rectangular extent, more performant\n"
               "- Conic method leads to axis aligned rectangular extent as in 3DGS and 3DGUT papers"))
        {
          m_requestUpdateShaders = true;
        }
        ImGui::EndDisabled();

        if(PE::Checkbox("Mip splatting antialiasing", &prmRaster.msAntialiasing,
                        "Indicates if Gaussians were trained (and should be rendered) with mip-splatting antialiasing method."))
          m_requestUpdateShaders = true;

        // Shading sub-options (disabled when shading is off)
        ImGui::BeginDisabled(prmRender.lightingMode == LightingMode::eLightingDisabled);
        {
          if(PE::Checkbox("Quantize Normals", &prmRaster.quantizeNormals,
                          "Use octahedral encoding for normals (Meyer et al. 2010).\n"
                          "Reduces mesh-to-fragment bandwidth from 96 bits to 32 bits per normal."))
          {
            m_requestUpdateShaders = true;
          }

          // FTB options only apply when shading is ON and not using stochastic splat
          const bool ftbDisabled = (prmRender.lightingMode == LightingMode::eLightingDisabled)
                                   || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
          ImGui::BeginDisabled(ftbDisabled);
          if(PE::entry(
                 "FTB Sync Mode", [&]() { return m_ui.enumCombobox(GUI_FTB_SYNC_MODE, "##ID", &prmRaster.ftbSyncMode); },
                 "Synchronization mode for depth buffer storage image access.\n"
                 "Interlock: Correct ordering via fragment shader interlock (slower).\n"
                 "Disabled: No synchronization (faster, may have rare artifacts)."))
          {
            m_requestUpdateShaders = true;
          }

          PE::DragFloat("Depth Iso Threshold", &prmRaster.depthIsoThreshold, 0.01f, 0.0f, 1.0f, "%.2f", 0,
                        "Transmittance threshold for depth picking.\n"
                        "Depth is captured when transmittance drops below this value.\n"
                        "Lower values pick depth later (more accumulated opacity).");
          ImGui::EndDisabled();
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);

        if(PE::Checkbox("Fragment shader barycentric", &prmRaster.fragmentBarycentric,
                        "Enables fragment shader barycentric to reduce vertex and mesh shaders outputs."))
          m_requestUpdateShaders = true;

        // we set a different size range for point and splat rendering
        PE::entry(
            "Splat scale",
            [&]() {
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return ImGui::DragFloat("##splatScale", &prmFrame.splatScale, 0.01f, 0.1f,
                                      prmRaster.pointCloudModeEnabled != 0 ? 10.0f : 2.0f, "%.3f");
            },
            "Adjusts the size of the splats for visualization purposes.");

        if(PE::Checkbox("Disable splatting", &prmRaster.pointCloudModeEnabled,
                        "Switches to point cloud mode, displaying only the splat centers. \n"
                        "Other parameters such as Splat Scale still apply in this mode."))
          m_requestUpdateShaders = true;

        ImGui::EndDisabled();

        PE::end();

        ImGui::EndTabItem();
      }
    }

    if(prmSelectedPipeline == PIPELINE_RTX || prmSelectedPipeline == PIPELINE_HYBRID
       || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT || prmSelectedPipeline == PIPELINE_MESH_3DGUT)
    {
      bool updated = false;

      if(ImGui::BeginTabItem("Ray tracing and 3DGUT specifics"))
      {
        PE::begin("## Raytrace sampling and bounces");

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);
        updated |= PE::SliderInt("Max bounces", &prmFrame.rtxMaxBounces, 1, 16);
        ImGui::EndDisabled();

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_HYBRID);
        {
          // #DLSS - DLSS UI section
#if defined(USE_DLSS)
          ImGui::BeginDisabled(!isDlssSupportedPipeline());
          {
            // Check if DLSS state changed
            if(m_dlss.onUi())
            {
              // Descriptors need to be updated to bind DLSS G-buffers
              m_requestUpdateShaders = true;
            }
          }
          ImGui::EndDisabled();
#endif
        }
        ImGui::EndDisabled();

        PE::end();

        PE::begin("## Raytrace gaussians settings");

        if(PE::entry("Kernel degree",
                     [&]() { return m_ui.enumCombobox(GUI_KERNEL_DEGREE, "##ID", &prmRtx.kernelDegree); }))
        {
          // Kernel degree affects particle shape - rebuild BLAS
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          m_requestUpdateShaders = true;
          updated                = true;
        }

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);

        int parametric = prmRtxData.useAABBs ? PARTICLE_FORMAT_PARAMETRIC : PARTICLE_FORMAT_ICOSAHEDRON;

        if(PE::entry(
               "Particles format", [&]() { return m_ui.enumCombobox(GUI_PARTICLE_FORMAT, "##ID", &parametric); },
               "This is a convenience shortcut to switch the Radiance Field use AABB property.\n"
               "Note that activating parametric will force the use of TLAS instance.\n"))
        {
          if(parametric == PARTICLE_FORMAT_ICOSAHEDRON)
          {
            prmRtxData.useAABBs = false;
          }
          if(parametric == PARTICLE_FORMAT_PARAMETRIC)
          {
            prmRtxData.useAABBs         = true;
            prmRtxData.useTlasInstances = true;
          }
          // Particle format affects BLAS geometry - rebuild BLAS
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          m_requestUpdateShaders = true;
          updated                = true;
        }

        if(PE::Checkbox("Adaptive clamp", &prmRtx.kernelAdaptiveClamping))
        {
          // Adaptive clamping affects particle shape - rebuild BLAS
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          m_requestUpdateShaders = true;
          updated                = true;
        }

        updated |= PE::InputFloat("Alpha clamp", &prmFrame.alphaClamp, 0.0, 3.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                                  "Maximum alpha value per particle hit.\n"
                                  "Clamps the opacity computed from the kernel response,\n"
                                  "preventing any single splat from being fully opaque.\n"
                                  "Default 0.99 (from the original 3DGS paper).\n"
                                  "Avoid numerical instabilities (see paper appendix).\n"
                                  "Not really needed in our visualization context.");

        updated |= PE::InputFloat("Minimum transmittance", &prmFrame.minTransmittance, 0.0, 1.0, "%.2f",
                                  ImGuiInputTextFlags_EnterReturnsTrue,
                                  "Transmittance threshold below which particle ray marching stops.");

        guiDrawTracingStrategySelector(false);

        {
          const bool stochasticAnyhit = (prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT);
          ImGui::BeginDisabled(stochasticAnyhit);
          int displaySpp = stochasticAnyhit ? 1 : prmRtx.particleSamplesPerPass;
          if(PE::entry(
                 "Particle samples per pass",
                 [&]() { return m_ui.enumCombobox(GUI_RAY_HIT_PER_PASS, "##ID", &displaySpp); },
                 "Number of particle ray hits stored per pass (PARTICLES_SPP).\n"
                 "Payload array size is max(this, mesh minimum)."))
          {
            prmRtx.particleSamplesPerPass = displaySpp;
            m_requestUpdateShaders        = true;
            updated                       = true;
          }
          ImGui::EndDisabled();
        }

        if(PE::InputInt("Maximum pass count", &prmFrame.maxPasses, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue))
        {
          prmFrame.maxPasses = std::clamp(prmFrame.maxPasses, 1, 1000);
          updated            = true;
        }

        {
          const int effectiveSpp = (prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT) ? 1 : prmRtx.particleSamplesPerPass;
          PE::Text("Maximum anyhit/pixel", std::to_string(effectiveSpp * prmFrame.maxPasses));
        }

        ImGui::EndDisabled();

        updated |= PE::InputFloat("Particle shadow offset", &prmRtx.particleShadowOffset, 0.0, 1.0, "%.2f",
                                  ImGuiInputTextFlags_EnterReturnsTrue,
                                  "Shadow ray origin offset for particles.\n"
                                  "Larger values prevent self-shadowing artifacts due to the volumetric nature of splats.");
        updated |= PE::DragFloat("Particle shadow threshold", &prmRtx.particleShadowTransmittanceThreshold, 0.01f, 0.0f,
                                 0.99f, "%.2f", 0,
                                 "Transmittance threshold for particle shadow termination.\n"
                                 "Higher values = earlier termination = harder shadows.\n"
                                 "Lower values = more gradual shadow falloff.");
        updated |= PE::DragFloat("Colored shadow strength", &prmRtx.particleShadowColorStrength, 0.01f, 0.0f, 5.0f, "%.2f", 0,
                                 "Per-channel color tinting of particle shadows (stained-glass effect).\n"
                                 "Below the transmittance threshold, shadows are fully black.\n"
                                 "Above it, particle color modulates per-channel light transmission.\n"
                                 "0 = monochrome shadows. Higher values = stronger color bleeding.");

        updated |= PE::DragFloat("Mesh composite threshold", &prmFrame.minMeshCompositeTransmittance, 0.01f, 0.0f, 1.0f, "%.2f", 0,
                                 "Minimum transmittance required for meshes to be composited with splats.\n"
                                 "Below this threshold, splats fully occlude meshes behind them.");

        updated |= PE::DragFloat("Depth Iso Threshold", &prmRtx.depthIsoThresholdRTX, 0.01f, 0.0f, 1.0f, "%.2f", 0,
                                 "Transmittance threshold for depth picking in ray tracing.\n"
                                 "Depth is captured when transmittance drops below this value.\n"
                                 "Lower values pick depth later (more accumulated opacity).");

        PE::end();

        ImGui::EndTabItem();
      }
      if(updated)
        resetFrameCounter();
    }
  }
  ImGui::EndTabBar();
}

void GaussianSplattingUI::guiDrawCommonSplatSetProperties()
{
  namespace PE = nvgui::PropertyEditor;

  ImGui::Text("Changes impact all the splat sets.");

  if(ImGui::CollapsingHeader("Splat Set Format in VRAM", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if(PE::begin("##VRAM format"))
    {
      if(PE::entry(
             "Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
      {
        resetDataParameters();
        m_assets.splatSets.markAllSplatSetsForRegeneration();
        m_requestUpdateShaders = true;
      }
      if(PE::entry(
             "SH format", [&]() { return m_ui.enumCombobox(GUI_SH_FORMAT, "##ID", &prmData.shFormat); },
             "Selects storage format for SH coefficient, balancing precision and memory usage"))
      {
        m_assets.splatSets.markAllSplatSetsForRegeneration();
        m_requestUpdateShaders = true;
      }
      if(PE::entry(
             "RGBA format", [&]() { return m_ui.enumCombobox(GUI_RGBA_FORMAT, "##RGBAID", &prmData.rgbaFormat); },
             "Selects storage format for RGBA color+alpha data, balancing precision and memory usage.\n"
             "Float 32: highest precision (16 bytes/splat)\n"
             "Float 16: good balance (8 bytes/splat)\n"
             "Uint8: lowest memory (4 bytes/splat)"))
      {
        m_assets.splatSets.markAllSplatSetsForRegeneration();
        m_requestUpdateShaders = true;
      }
      PE::end();
    }
  }

  if(ImGui::CollapsingHeader("RTX acceleration structures", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if(PE::begin("##VRAM format RTX"))
    {
      if(PE::entry(
             "Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
      {
        resetRtxDataParameters();
        m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
      }
      if(PE::Checkbox("Use AABBs", &prmRtxData.useAABBs,
                      "If on, uses AABBs for splats in BLAS instead of ICOSAHEDRON meshes."
                      "In this case the renderer will use the collision shader instead of "
                      "the ray/triangle intersection specialized hardware."))
      {
        m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
        m_requestUpdateShaders = true;  // CRITICAL: Shader recompile needed for RTX_USE_AABBS macro
      }

      // We do not allow useAABBs without instances (prevent bvh with very bad properties leading to very low frame rate and device lost error)
      if(prmRtxData.useAABBs)
        prmRtxData.useTlasInstances = true;

      ImGui::BeginDisabled(prmRtxData.useAABBs);
      if(PE::Checkbox("Use TLAS instances", &prmRtxData.useTlasInstances,
                      "If on, uses one TLAS entry per splat and a small BLAS "
                      "with a unit Icosahedron. \nOtherwise use a single TLAS "
                      "entry and a huge BLAS containing all the transformed Icosahedrons."))
      {
        m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
        m_requestUpdateShaders = true;  // CRITICAL: Shader recompile needed for RTX_USE_INSTANCES macro
      }
      ImGui::EndDisabled();

      if(PE::Checkbox("BLAS Compaction", &prmRtxData.compressBlas, "Bottom Level Acceleration structure compression."))
        m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;

      PE::end();
    }
  }
}

void GaussianSplattingUI::guiDrawSplatSetProperties()
{
  namespace PE = nvgui::PropertyEditor;

  // Get selected splat set and instance
  if(!m_selectedSplatInstance || !m_selectedSplatInstance->splatSet)
    return;  // No active splat set/instance

  auto splatSet = m_selectedSplatInstance->splatSet;
  auto instance = m_selectedSplatInstance;

  // Splat Set Info section
  size_t      instanceCount = splatSet->instanceRefCount;
  std::string infoHeader =
      "Splat Set Info (" + std::to_string(instanceCount) + " instance" + (instanceCount != 1 ? "s" : "") + ")";

  if(ImGui::CollapsingHeader(infoHeader.c_str()))
  {
    PE::begin("##SplatSetInfo");

    // Total number of splats
    PE::Text("Total Splats", std::to_string(splatSet->splatCount));

    // SH degree
    PE::Text("SH Degree", std::to_string(splatSet->shDegree));

    // Full path (read-only, selectable for copying)
    if(PE::entry(
           "Path",
           [&]() {
             ImGui::InputText("##Path", const_cast<char*>(splatSet->path.c_str()), splatSet->path.length() + 1,
                              ImGuiInputTextFlags_ReadOnly);
             return false;
           },
           "Full path to the source .ply file"))
    {
      // Read-only, no action needed
    }

    PE::end();
  }

  if(ImGui::CollapsingHeader("Model Transform", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Transform");
    if(guiGetTransform(instance->scale, instance->rotation, instance->translation, instance->transform, instance->transformInverse, false))
    {
      // guiGetTransform already updated transform matrices in RAM
      // Signal update needed (deferred to processVramUpdates)
      if(m_selectedSplatInstance)
        m_assets.splatSets.updateInstanceTransform(m_selectedSplatInstance);

      // Defer RTX Acceleration Structure rebuild if currently in raster mode
      // This avoids expensive RTX updates when not actively using ray tracing
      if(!isRtxPipelineActive())
      {
        m_deferredRtxRebuildPending = true;
      }
    }
    PE::end();
  }

  // Material properties for splat set
  if(ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##SplatMaterial");

    bool materialChanged = false;

    materialChanged |= PE::ColorEdit3("ambient", glm::value_ptr(instance->splatMaterial.ambient));
    materialChanged |= PE::ColorEdit3("diffuse", glm::value_ptr(instance->splatMaterial.diffuse));
    materialChanged |= PE::ColorEdit3("specular", glm::value_ptr(instance->splatMaterial.specular));
    materialChanged |= PE::ColorEdit3("emission", glm::value_ptr(instance->splatMaterial.emission));
    materialChanged |= PE::SliderFloat("shininess", &instance->splatMaterial.shininess, 0.0f, 2000.0f);

    if(materialChanged)
    {
      // Material already modified in RAM by ImGui widgets
      // Signal update needed (deferred to processVramUpdates)
      if(m_selectedSplatInstance)
        m_assets.splatSets.updateInstanceMaterial(m_selectedSplatInstance);
    }

    PE::end();
  }

  if(ImGui::CollapsingHeader("Splat Set Storage in VRAM", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::Text("Changes impact all instances of this splat set.");

    if(PE::begin("##VRAM format"))
    {
      if(PE::entry(
             "Storage", [&] { return m_ui.enumCombobox(GUI_STORAGE, "##ID", &splatSet->dataStorage); },
             "Selects between Data Buffers and Textures for storing model attributes, including:\n"
             "Position, Color and Opacity, Covariance Matrix\n"
             "and Spherical Harmonics (SH) Coefficients (for degrees higher than 0)"))
      {
        m_assets.splatSets.markSplatSetsForRegeneration(splatSet);
      }
      PE::end();
    }
  }
}

void GaussianSplattingUI::guiDrawMeshTransformProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(!m_selectedMeshInstance)
    return;  // No selection

  PE::begin("##Transform");
  if(guiGetTransform(m_selectedMeshInstance->scale, m_selectedMeshInstance->rotation,
                     m_selectedMeshInstance->translation, m_selectedMeshInstance->transform,
                     m_selectedMeshInstance->transformInverse, m_selectedMeshInstance->transformRotScaleInverse, false))
  {
    // guiGetTransform already updated transform matrices in RAM
    // Just signal update needed (deferred to processVramUpdates)
    m_assets.meshes.updateInstanceTransform(m_selectedMeshInstance);
  }
  PE::end();
}

void GaussianSplattingUI::guiDrawMeshMaterialProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(!m_selectedMeshInstance || !m_selectedMeshInstance->mesh)
    return;  // No selection

  auto& materials          = m_selectedMeshInstance->mesh->materials;
  bool  needMaterialUpdate = false;

  for(auto i = 0; i < materials.size(); ++i)
  {
    PE::begin("##Material");
    auto& material = materials[i];
    ImGui::PushID(i);
    PE::Text("Name", m_selectedMeshInstance->mesh->matNames[i]);
    needMaterialUpdate |= PE::entry(
        "Model", [&]() { return m_ui.enumCombobox(GUI_ILLUM_MODEL, "##ID", &material.illum); }, "TODO");
    needMaterialUpdate |= PE::ColorEdit3("ambient", glm::value_ptr(material.ambient));
    needMaterialUpdate |= PE::ColorEdit3("diffuse", glm::value_ptr(material.diffuse));
    needMaterialUpdate |= PE::ColorEdit3("specular", glm::value_ptr(material.specular));
    needMaterialUpdate |= PE::ColorEdit3("transmittance", glm::value_ptr(material.transmittance));
    needMaterialUpdate |= PE::ColorEdit3("emission", glm::value_ptr(material.emission));
    needMaterialUpdate |= PE::SliderFloat("shininess", &material.shininess, 0.0f, 2000.0f);
    needMaterialUpdate |= PE::SliderFloat("ior", &material.ior, 1.0f, 3.0f);
    //needMaterialUpdate |= PE::DragFloat("dissolve", &material.dissolve);
    ImGui::PopID();
    PE::end();
  }
  if(needMaterialUpdate)
  {
    // Use deferred API - materials will be uploaded in processVramUpdates()
    m_assets.meshes.updateMeshMaterials(m_selectedMeshInstance->mesh);
  }
}

void GaussianSplattingUI::guiDrawCameraProperties()
{
  namespace PE = nvgui::PropertyEditor;

  Camera camera = m_assets.cameras.getCamera();
  if(m_selectedCameraPresetIndex > -1)  // we show a preset - "read only"
  {
    camera = m_assets.cameras.getPreset(m_selectedCameraPresetIndex);
    ImGui::Text("To modify a preset:");
    ImGui::Text("  1. Load the preset");
    ImGui::Text("  2. Modify the current camera");
    ImGui::Text("  3. Overwrite the preset with active camera");
  }

  bool changed = false;

  if(ImGui::CollapsingHeader("Camera Intrinsics", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::BeginDisabled(m_selectedCameraPresetIndex != -1 || cameraManip->isAnimated());
    if(PE::begin())
    {
      if(PE::entry(
             "Camera type", [&] { return m_ui.enumCombobox(GUI_CAMERA_TYPE, "##ID", &camera.model); },
             "Fisheye type may not be supported by all the Pipelines.\n"
             "The Camera type is not stored per camera for the time beeing."))
      {
        m_requestUpdateShaders = true;
        changed                = true;
      }

      PE::InputFloat2("Clip planes", glm::value_ptr(camera.clip));
      changed |= ImGui::IsItemDeactivatedAfterEdit();

      if(PE::SliderFloat("FOV", &camera.fov, 1.F, 179.F, "%.1f deg", ImGuiSliderFlags_Logarithmic, "Field of view in degrees"))
      {
        changed = true;
      }

      ImGui::BeginDisabled(prmSelectedPipeline != PIPELINE_RTX && prmSelectedPipeline != PIPELINE_HYBRID_3DGUT
                           && prmSelectedPipeline != PIPELINE_MESH_3DGUT);

      const bool autoFocusSupported = supportsAutoFocus();
      if(!autoFocusSupported && camera.dofMode == DOF_AUTO_FOCUS)
      {
        camera.dofMode = DOF_FIXED_FOCUS;
        changed        = true;
      }

      {
        const int  prevDofMode = camera.dofMode;
        const auto dofModeMenu = autoFocusSupported ? GUI_DOF_MODE : GUI_DOF_MODE_NO_AUTO;
        if(PE::entry(
               "Depth of Field", [&]() { return m_ui.enumCombobox(dofModeMenu, "##DofMode", &camera.dofMode); },
               "Depth of Field mode. Only works with 3DGRT, 3DGUT and hybrid 3DGUT/3DGRT.\n"
               "- Fixed focus: manual focus distance\n"
               "- Auto focus: uses surface distance at cursor position (requires 3DGRT)\n"
               "Triggers \"Temporal sampling\" if set to automatic."))
        {
          // Only rebuild shaders when crossing the disabled/enabled boundary
          // (switching between Fixed Focus and Auto Focus doesn't change shader code)
          if((prevDofMode == DOF_DISABLED) != (camera.dofMode == DOF_DISABLED))
          {
            m_requestUpdateShaders = true;
          }
          resetFrameCounter();
          changed = true;
        }
      }
      ImGui::BeginDisabled(camera.dofMode == DOF_DISABLED);
      ImGui::BeginDisabled(camera.dofMode == DOF_AUTO_FOCUS);
      if(PE::DragFloat("Focus distance", &camera.focusDist, 0.1F, 0.1F, 15.0F, "%.3f"))
      {
        resetFrameCounter();
        changed = true;
      }
      ImGui::EndDisabled();  // Auto focus (focus distance read-only)
      if(PE::SliderFloat("Aperture", &camera.aperture, 0.0F, 0.01F, "%.6f"))
      {
        resetFrameCounter();
        changed = true;
      }
      ImGui::EndDisabled();  // DoF disabled

      ImGui::EndDisabled();  // Modifiable

      PE::end();
    }

    ImGui::EndDisabled();
  }
  if(ImGui::CollapsingHeader("Camera Extrinsics", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::BeginDisabled(m_selectedCameraPresetIndex != -1 || cameraManip->isAnimated());
    if(PE::begin())
    {

      PE::InputFloat3("Eye", &camera.eye.x, "%.5f", 0, "Position of the Camera");
      changed |= ImGui::IsItemDeactivatedAfterEdit();
      PE::InputFloat3("Center", &camera.ctr.x, "%.5f", 0, "Center of camera interest");
      changed |= ImGui::IsItemDeactivatedAfterEdit();
      PE::InputFloat3("Up", &camera.up.x, "%.5f", 0, "Up vector interest");
      changed |= ImGui::IsItemDeactivatedAfterEdit();

      PE::end();
    }
    ImGui::EndDisabled();
  }

  // if changed it is necessarly the active camera
  if(changed)
    m_assets.cameras.setCamera(camera);
}

//--------------------------------------------------------------------------------------------------
// Helper: Check if loading a camera preset requires shader rebuild
// Returns true if camera model or depth of field mode changes
//
bool GaussianSplattingUI::cameraPresetNeedsShaderRebuild(uint64_t presetIndex)
{
  const Camera& preset  = m_assets.cameras.getPreset(presetIndex);
  const Camera  current = m_assets.cameras.getCamera();

  // Check if camera model changes (e.g., pinhole to fisheye)
  if(preset.model != current.model)
    return true;

  // Check if depth of field mode changes (affects NEED_SURFACE_INFO)
  if((preset.dofMode != DOF_DISABLED) != (current.dofMode != DOF_DISABLED))
    return true;

  return false;
}

void GaussianSplattingUI::guiDrawNavigationProperties()
{

  namespace PE = nvgui::PropertyEditor;

  bool changed = false;

  ImGui::BeginDisabled(cameraManip->isAnimated());

  // Navigation Mode
  if(PE::begin())
  {
    auto mode     = cameraManip->getMode();
    auto speed    = cameraManip->getSpeed();
    auto duration = static_cast<float>(cameraManip->getAnimationDuration());

    changed |= PE::entry(
        "Navigation and Animation",
        [&] {
          int rmode = static_cast<int>(mode);
          changed |= ImGui::RadioButton("Examine", &rmode, nvutils::CameraManipulator::Examine);
          nvgui::tooltip("The camera orbit around a point of interest");
          changed |= ImGui::RadioButton("Fly", &rmode, nvutils::CameraManipulator::Fly);
          nvgui::tooltip("The camera is free and move toward the looking direction");
          changed |= ImGui::RadioButton("Walk", &rmode, nvutils::CameraManipulator::Walk);
          nvgui::tooltip("The camera is free but stay on a plane");
          cameraManip->setMode(static_cast<nvutils::CameraManipulator::Modes>(rmode));
          return changed;
        },
        "Camera Navigation Mode");

    changed |= PE::SliderFloat("Speed", &speed, 0.01F, 10.0F, "%.3f", 0, "Changing the default movement speed");
    changed |= PE::SliderFloat("Transition", &duration, 0.0F, 2.0F, "%.3f", 0,
                               "Nb seconds to move to new position when loading a camera preset");

    cameraManip->setSpeed(speed);
    cameraManip->setAnimationDuration(duration);

    PE::end();
  }

  ImGui::EndDisabled();
}

void GaussianSplattingUI::guiDrawLightProperties()
{
  if(!m_selectedLightInstance || !m_selectedLightInstance->lightSource)
  {
    ImGui::Text("No light selected");
    return;
  }

  namespace PE = nvgui::PropertyEditor;

  bool needInstanceUpdate = false;  // Position/rotation changed
  bool needAssetUpdate    = false;  // Color/intensity/range changed
  bool needProxyRecreate  = false;  // Type changed (requires new proxy mesh)

  auto& instance = m_selectedLightInstance;
  auto& asset    = instance->lightSource;

  shaderio::LightType previousType = asset->type;

  PE::begin("##Light");

  // Asset properties (shared across all instances)
  if(PE::entry(
         "Type", [&]() { return m_ui.enumCombobox(GUI_LIGHT_TYPE, "##ID", (int*)&asset->type); }, "Type of light."))
  {
    if(asset->type != previousType)
    {
      needProxyRecreate = true;  // Type changed - need new proxy mesh

      // Set appropriate defaults for the new type
      if(asset->type == shaderio::LightType::eDirectionalLight)
      {
        asset->attenuationMode = 0;  // Force None for directional
      }
      else if(asset->type == shaderio::LightType::ePointLight)
      {
        asset->attenuationMode = 2;  // Quadratic for point
      }
      else if(asset->type == shaderio::LightType::eSpotLight)
      {
        asset->attenuationMode = 2;  // Quadratic for spot
      }
    }
    needAssetUpdate = true;
  }

  // Instance properties (per-instance)
  needInstanceUpdate |= PE::DragFloat3("Translation", glm::value_ptr(instance->translation), 0.01f);

  // Rotation (for directional and spot lights only)
  if(asset->type == shaderio::LightType::eDirectionalLight || asset->type == shaderio::LightType::eSpotLight)
  {
    needInstanceUpdate |= PE::DragFloat3("Rotation", glm::value_ptr(instance->rotation), 0.1f, -180.0f, 180.0f, "%.1f°");
  }

  // Asset properties (shared)
  needAssetUpdate |= PE::DragFloat("Intensity", &asset->intensity, 0.01f, 0.0f, 10000000.0f);
  asset->intensity = std::clamp(asset->intensity, 0.0f, 10000000.0f);

  // Range control (for point and spot lights)
  if(asset->type == shaderio::LightType::ePointLight || asset->type == shaderio::LightType::eSpotLight)
  {
    needAssetUpdate |= PE::DragFloat("Range", &asset->range, 0.1f, 0.1f, 10000000.0f, "%.2f", 0,
                                     "Effective range of the light in world units.\n"
                                     "Light smoothly fades to zero at this distance.");
    asset->range = std::clamp(asset->range, 0.1f, 10000000.0f);
  }

  needAssetUpdate |= PE::ColorEdit3("Color", glm::value_ptr(asset->color));

  // Attenuation mode (for point and spot lights, forced to None for directional)
  if(asset->type == shaderio::LightType::eDirectionalLight)
  {
    // Directional lights always have no attenuation
    asset->attenuationMode = 0;  // eNone
    ImGui::BeginDisabled();
    int attenMode = 0;
    if(PE::entry(
           "Attenuation", [&]() { return m_ui.enumCombobox(GUI_ATTENUATION_MODE, "##ID", &attenMode); },
           "Directional lights have no attenuation (forced)"))
    {
      // No-op (disabled)
    }
    ImGui::EndDisabled();
  }
  else  // Point or Spot
  {
    if(PE::entry(
           "Attenuation", [&]() { return m_ui.enumCombobox(GUI_ATTENUATION_MODE, "##ID", &asset->attenuationMode); },
           "How light intensity falls off with distance:\n"
           "- None: No falloff (constant)\n"
           "- Linear: 1.0 - (distance/range)\n"
           "- Quadratic: 1.0 / (1.0 + distance²)\n"
           "- Physical: 1.0 / distance²"))
    {
      needAssetUpdate = true;
    }
  }

  // Spot light cone angles
  if(asset->type == shaderio::LightType::eSpotLight)
  {
    needAssetUpdate |= PE::DragFloat("Inner Cone Angle", &asset->innerConeAngle, 0.5f, 0.0f, 90.0f, "%.1f°", 0,
                                     "Full intensity within this angle");
    asset->innerConeAngle = std::clamp(asset->innerConeAngle, 0.0f, 90.0f);

    needAssetUpdate |= PE::DragFloat("Outer Cone Angle", &asset->outerConeAngle, 0.5f, 0.0f, 90.0f, "%.1f°", 0,
                                     "Light fades to zero between inner and outer angles");
    asset->outerConeAngle = std::clamp(asset->outerConeAngle, asset->innerConeAngle, 90.0f);  // Must be >= inner
  }

  needAssetUpdate |= PE::DragFloat("Proxy Scale", &asset->proxyScale, 0.01f, 0.01f, 100.0f, "%.2f", 0,
                                   "Visual scale of the light proxy in the viewport");
  PE::end();

  // Update appropriately based on what changed
  if(needProxyRecreate)
  {
    m_assets.lights.recreateProxyForAsset(asset);  // Type changed → recreate proxy mesh
  }
  if(needInstanceUpdate)
  {
    m_assets.lights.updateLight(instance);  // Position/rotation changed → update this instance only
  }
  if(needAssetUpdate)
  {
    m_assets.lights.updateLightAsset(asset);  // Asset changed → update ALL instances using this asset
  }
}

bool GaussianSplattingUI::guiGetTransform(glm::vec3& scale,
                                          glm::vec3& rotation,
                                          glm::vec3& translation,
                                          glm::mat4& transform,
                                          glm::mat4& transformInv,
                                          bool       disabled /*=false*/)
{
  namespace PE = nvgui::PropertyEditor;

  bool updated = false;
  ImGui::BeginDisabled(disabled);
  updated |= PE::DragFloat3("Translate", glm::value_ptr(translation), 0.05f);
  updated |= PE::DragFloat3("Rotate", glm::value_ptr(rotation), 0.5f);
  updated |= PE::DragFloat3("Scale", glm::value_ptr(scale), 0.01f);
  ImGui::EndDisabled();

  if(updated)
  {
    computeTransform(scale, rotation, translation, transform, transformInv);
  }

  return updated;
}

bool GaussianSplattingUI::guiGetTransform(glm::vec3& scale,
                                          glm::vec3& rotation,
                                          glm::vec3& translation,
                                          glm::mat4& transform,
                                          glm::mat4& transformInv,
                                          glm::mat3& transformRotScaleInv,
                                          bool       disabled /*=false*/)
{
  namespace PE = nvgui::PropertyEditor;

  bool updated = false;
  ImGui::BeginDisabled(disabled);
  updated |= PE::DragFloat3("Translate", glm::value_ptr(translation), 0.05f);
  updated |= PE::DragFloat3("Rotate", glm::value_ptr(rotation), 0.5f);
  updated |= PE::DragFloat3("Scale", glm::value_ptr(scale), 0.01f);
  ImGui::EndDisabled();

  if(updated)
  {
    computeTransform(scale, rotation, translation, transform, transformInv, transformRotScaleInv);
  }

  return updated;
}

void GaussianSplattingUI::guiDrawRendererStatisticsWindow()
{
  if(!m_showRendererStatistics)
    return;

  if(ImGui::Begin("Rendering Statistics", &m_showRendererStatistics))
  {
    namespace PE = nvgui::PropertyEditor;

    // ===== Splat sets overview =====
    {
      const uint32_t splatSetCount   = static_cast<uint32_t>(m_assets.splatSets.getSplatSetCount());
      const uint32_t splatInstCount  = static_cast<uint32_t>(m_assets.splatSets.getInstanceCount());
      const int32_t  totalSplatCount = m_assets.splatSets.getTotalGlobalSplatCount();

      if(PE::begin("## Scene"))
      {
        PE::Text("Splat sets", fmt::format("{} ({} instances)", splatSetCount, splatInstCount));
        PE::Text("Total particles", fmt::format("{} ({})", formatSize(totalSplatCount), totalSplatCount));
        PE::end();
      }
    }

    // ===== Rasterization =====
    {
      const int32_t totalSplatCount = m_assets.splatSets.getTotalGlobalSplatCount();
      // GPU radix sort and stochastic splat both use the distance shader which populates the indirect buffer
      const bool usesDistShader =
          (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
      const int32_t  rasterSplatCount = usesDistShader ? m_indirectReadback.instanceCount : totalSplatCount;
      const uint32_t wgCount =
          (prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_MESH_3DGUT) ?
              (usesDistShader ? m_indirectReadback.groupCountX :
                                (prmFrame.splatCount + prmRaster.meshShaderWorkgroupSize - 1) / prmRaster.meshShaderWorkgroupSize) :
              0;

      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_RTX);
      if(PE::begin("## Rasterization"))
      {
        PE::Text("Rasterized splats", fmt::format("{} ({})", formatSize(rasterSplatCount), rasterSplatCount));
        PE::Text("Mesh shader work groups", fmt::format("{} ({})", formatSize(wgCount), wgCount));
        PE::end();
      }
      ImGui::EndDisabled();
    }

    // ===== Ray Tracing =====
    {
      const auto& splatSets = m_assets.splatSets;

      const uint32_t tlasCount   = splatSets.getRtxTlasCount();
      const uint32_t tlasEntries = splatSets.getRtxTlasEntryCount();
      const uint32_t blasCount   = splatSets.getRtxBlasCount();
      const bool     blasChunked = splatSets.isUsingBlasChunks();
      const bool     multiTlas   = splatSets.isUsingMultiTlas();
      const bool     rtxValid    = splatSets.isRtxValid();

      ImGui::BeginDisabled(!isRtxPipelineActive());
      if(PE::begin("## Ray Tracing"))
      {
        // Mode name based on (instanced/per-splat-set) x (AABB/icosa)
        const char* modeName = prmRtxData.useTlasInstances ?
                                   (prmRtxData.useAABBs ? "Per-particle instanced AABB" : "Per-particle instanced icosa") :
                                   (prmRtxData.useAABBs ? "Per-splat-set AABB" : "Per-splat-set icosa soup");
        PE::Text("Mode", modeName);

        // TLAS info (append "multi-TLAS" if >1)
        if(multiTlas)
          PE::Text("TLAS count", fmt::format("{} (multi-TLAS)", tlasCount));
        else
          PE::Text("TLAS count", fmt::format("{}", tlasCount));

        PE::Text("TLAS entries", fmt::format("{}", formatSize(tlasEntries)));

        // BLAS info (append "chunked" if using BLAS chunks)
        if(blasChunked)
          PE::Text("BLAS count", fmt::format("{} (chunked)", blasCount));
        else
          PE::Text("BLAS count", fmt::format("{}", blasCount));

        PE::end();
      }
      ImGui::EndDisabled();
    }
  }
  ImGui::End();
}


void GaussianSplattingUI::guiDrawShaderFeedbackWindow()
{
  m_shaderFeedbackUI.drawWindow(m_showShaderFeedback, m_showCursorTargetOverlay, m_indirectReadback, m_requestUpdateShaders);
}

void GaussianSplattingUI::guiDrawFooterBar()
{
  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;
  float height = ImGui::GetFrameHeight();

  if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL, ImGuiDir_Down, height, window_flags))
  {
    if(ImGui::BeginMenuBar())
    {
      ImGui::Text("%s ", m_showCursorTargetOverlay ? "Target" : "Mouse");
      ImGui::Text("%s", fmt::format("{} {}", prmFrame.cursor.x, prmFrame.cursor.y).c_str());
      ImGui::Text(" | Global ");
      ImGui::Text("%s", fmt::format("{}", m_indirectReadback.particleGlobalId).c_str());
      ImGui::Text(" | Set ");
      ImGui::Text("%s", fmt::format("{}", m_indirectReadback.splatSetId).c_str());
      ImGui::Text(" | Local ");
      ImGui::Text("%s", fmt::format("{}", m_indirectReadback.particleId).c_str());
      ImGui::Text(" | Dist ");
      ImGui::Text("%s", formatFloatInf(m_indirectReadback.particleDist).c_str());

      // temporal sampling progress bar (1-based display: "1/200" to "200/200")
      {
        float       progress = 0.0f;
        std::string buf      = "1/1";
        if(!m_dlss.isEnabled() && prmRtx.temporalSampling)
        {
          int displayFrame = std::max(1, prmFrame.frameSampleId + 1);  // 1-based for display
          progress         = (float)displayFrame / (float)prmFrame.frameSampleMax;
          buf              = fmt::format("{}/{}", displayFrame, prmFrame.frameSampleMax);
        }
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 255);
        ImGui::Text("%s", "SPP");
        nvgui::tooltip("Samples Per Pixel");
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.4f, 0.7f, 0.0f, 1.0f));
        ImGui::ProgressBar(progress, ImVec2(200.f, 0.f), buf.c_str());
        ImGui::PopStyleColor();
      }

      ImGui::EndMenuBar();
    }
    ImGui::End();
  }
}

void GaussianSplattingUI::guiAddToRecentFiles(std::filesystem::path filePath, int historySize)
{
  // first check if filePath is absolute
  if(filePath.is_relative())
  {
    filePath = std::filesystem::absolute(filePath);
  }
  //
  auto it = std::find(m_recentFiles.begin(), m_recentFiles.end(), filePath);
  if(it != m_recentFiles.end())
  {
    m_recentFiles.erase(it);
  }
  m_recentFiles.insert(m_recentFiles.begin(), filePath);
  if(m_recentFiles.size() > historySize)
  {
    m_recentFiles.pop_back();
  }
}

void GaussianSplattingUI::guiAddToRecentProjects(std::filesystem::path filePath, int historySize)
{
  // first check if filePath is absolute
  if(filePath.is_relative())
  {
    filePath = std::filesystem::absolute(filePath);
  }
  //
  auto it = std::find(m_recentProjects.begin(), m_recentProjects.end(), filePath);
  if(it != m_recentProjects.end())
  {
    m_recentProjects.erase(it);
  }
  m_recentProjects.insert(m_recentProjects.begin(), filePath);
  if(m_recentProjects.size() > historySize)
  {
    m_recentProjects.pop_back();
  }
}

void GaussianSplattingUI::guiRegisterIniFileHandlers()
{
  // mandatory to work, see ImGui::DockContextInitialize as an example
  auto readOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void* {
    if(strcmp(name, "Data") != 0)
      return NULL;
    // Make sure we clear out our current recent vectors so we don't just keep adding to the list every time we load
    // This is if the .ini file is loaded twice, which happens in nvpro_core2
    auto* ui = static_cast<GaussianSplattingUI*>(handler->UserData);
    if(strcmp(handler->TypeName, "RecentFiles") == 0)
    {
      ui->m_recentFiles.clear();
    }
    else if(strcmp(handler->TypeName, "RecentProjects") == 0)
    {
      ui->m_recentProjects.clear();
    }
    return (void*)1;
  };

  {
    // Save settings handler, not using capture so can be used as a function pointer
    auto saveRecentFilesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentFiles)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentFilesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentFiles.push_back(filePath);
      }
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "RecentFiles";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpen;
    iniHandler.WriteAllFn = saveRecentFilesToIni;
    iniHandler.ReadLineFn = loadRecentFilesFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
  {
    // Save settings handler, not using capture so can be used as a function pointer
    auto saveRecentProjectsToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentProjects)
      {
        buf->appendf("File=%s\n", file.string().c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentProjectsFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentProjects.push_back(filePath);
      }
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "RecentProjects";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpen;
    iniHandler.WriteAllFn = saveRecentProjectsToIni;
    iniHandler.ReadLineFn = loadRecentProjectsFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
  {
    // Save window visibility settings handler
    auto saveWindowStatesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      buf->appendf("ShaderDebugging=%d\n", self->m_showShaderFeedback ? 1 : 0);
      buf->appendf("MemoryStatistics=%d\n", self->m_showMemoryStatistics ? 1 : 0);
      buf->appendf("RendererStatistics=%d\n", self->m_showRendererStatistics ? 1 : 0);
      buf->append("\n");
    };

    // Load window visibility settings handler
    auto loadWindowStatesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      int   value;
#ifdef _MSC_VER
      if(sscanf_s(line, "ShaderDebugging=%d", &value) == 1)
#else
      if(sscanf(line, "ShaderDebugging=%d", &value) == 1)
#endif
      {
        self->m_showShaderFeedback = (value == 1);
      }
#ifdef _MSC_VER
      else if(sscanf_s(line, "MemoryStatistics=%d", &value) == 1)
#else
      else if(sscanf(line, "MemoryStatistics=%d", &value) == 1)
#endif
      {
        self->m_showMemoryStatistics = (value == 1);
      }
#ifdef _MSC_VER
      else if(sscanf_s(line, "RendererStatistics=%d", &value) == 1)
#else
      else if(sscanf(line, "RendererStatistics=%d", &value) == 1)
#endif
      {
        self->m_showRendererStatistics = (value == 1);
      }
    };

    // Custom readOpen for WindowStates that checks for "Data" section
    auto readOpenWindowStates = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void* {
      if(strcmp(name, "Data") != 0)
        return NULL;
      return (void*)1;
    };

    //
    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "WindowStates";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpenWindowStates;
    iniHandler.WriteAllFn = saveWindowStatesToIni;
    iniHandler.ReadLineFn = loadWindowStatesFromIni;
    iniHandler.UserData   = this;  // Pass the current instance to the handler
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
}

///////////////////////////////////
// Loading and Saving Projects

namespace fs = std::filesystem;

fs::path getRelativePath(const fs::path& from, const fs::path& to)
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

std::filesystem::path makeAbsolutePath(const std::filesystem::path& base, const std::string& relativePath)
{
  return std::filesystem::absolute(base / relativePath);
}

// This method is multi pass
bool GaussianSplattingUI::loadProjectIfNeeded()
{
  // Nothing to load
  if(prmScene.projectToLoadFilename.empty())
    return true;

  auto path = prmScene.projectToLoadFilename.string();

  // load the json and set loading status
  if(!loadingProject)
  {
    bool doReset = prmScene.projectLoadPorcelain;

    if(!doReset)
    {
      ImGui::OpenPopup("Load .vkg project file ?");

      // Always center this window when appearing
      ImVec2 center = ImGui::GetMainViewport()->GetCenter();
      ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

      if(ImGui::BeginPopupModal("Load .vkg project file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
      {
        ImGui::Text("The current project will be entirely replaced.\nThis operation cannot be undone!");
        ImGui::Separator();

        if(ImGui::Button("OK", ImVec2(120, 0)))
        {
          doReset = true;
          ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if(ImGui::Button("Cancel", ImVec2(120, 0)))
        {
          // cancel any request leading to a reset
          prmScene.sceneLoadQueue.clear();
          prmScene.projectToLoadFilename  = "";
          prmScene.projectLoadPorcelain = false;
          ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
      }
    }

    if(doReset)
    {
      LOGI("Opening project file %s\n", path.c_str());

      std::ifstream i(path);
      if(!i.is_open())
      {
        LOGE("Error: unable to open project file %s\n", path.c_str());
        prmScene.projectToLoadFilename  = "";
        prmScene.projectLoadPorcelain = false;
        return false;
      }

      try
      {
        i >> data;
      }
      catch(...)
      {
        LOGE("Error: invalid project file %s\n", path.c_str());
        prmScene.projectToLoadFilename  = "";
        prmScene.projectLoadPorcelain = false;
        return false;
      }
      i.close();

      // IMPORTANT: Reset the scene FIRST (just like when loading a single splat set)
      // This ensures everything is properly deinitialized before loading new data
      // Pipelines are re-initialized in reset()
      reset();

      loadingProject = true;
    }

    // Will do the rest of the work on next call when splatset is loaded
    return true;
  }

  // we skip until the splat set is being loaded
  if(m_plyLoader.getStatus() != PlyLoaderAsync::State::E_READY)
    return true;

  // we finalize
  guiAddToRecentProjects(prmScene.projectToLoadFilename);
  loadingProject = false;

  // Load project data using VkgsProjectReader
  bool success = VkgsProjectReader::loadProject(data, path, this);

  if(!success)
  {
    prmScene.projectToLoadFilename  = "";
    prmScene.projectLoadPorcelain = false;
    return false;
  }

  prmScene.projectToLoadFilename  = "";
  prmScene.projectLoadPorcelain = false;
  return true;
}

// Note: PROJECT_FILE_VERSION moved to vkgs_project_writer.cpp
// Note: Helper functions (getRelativePath, makeAbsolutePath, LOAD macros) moved to vkgs_project_reader.cpp

bool GaussianSplattingUI::saveProject(std::string path)
{
  return VkgsProjectWriter::saveProject(path, this);
}

void GaussianSplattingUI::dumpSplat()
{
  // Use readback data to get the correct splat set and local splat index
  int32_t globalSplatId   = m_indirectReadback.particleGlobalId;
  int32_t splatSetIndex   = m_indirectReadback.splatSetId;
  int32_t localSplatIndex = m_indirectReadback.particleId;

  if(splatSetIndex < 0 || localSplatIndex < 0)
  {
    LOGE("Error: no valid splat to dump (splatSetIndex=%d, localSplatIndex=%d)\n", splatSetIndex, localSplatIndex);
    return;
  }

  // Get the splat set instances
  const auto& instances = m_assets.splatSets.getInstances();
  if(splatSetIndex >= static_cast<int32_t>(instances.size()) || !instances[splatSetIndex])
  {
    LOGE("Error: invalid splat set index %d\n", splatSetIndex);
    return;
  }

  auto instance        = instances[splatSetIndex];
  auto currentSplatSet = instance->splatSet;
  if(!currentSplatSet || localSplatIndex >= static_cast<int32_t>(currentSplatSet->size()))
  {
    LOGE("Error: invalid local splat index %d for splat set size %zu\n", localSplatIndex, currentSplatSet->size());
    return;
  }

  uint32_t splatIdx = static_cast<uint32_t>(localSplatIndex);

  std::ofstream out("c:\\Temp\\debug_splat.ply");
  if(!out)
  {
    LOGE("Error: could not open file c:\\Temp\\debug_splat.ply\n");
    return;
  }

  // prints the header
  out << "ply" << std::endl;
  out << "format ascii 1.0" << std::endl;
  out << "element vertex 1" << std::endl;
  out << "property float x" << std::endl;
  out << "property float y" << std::endl;
  out << "property float z" << std::endl;
  out << "property float nx" << std::endl;
  out << "property float ny" << std::endl;
  out << "property float nz" << std::endl;
  for(auto i = 0; i < 3; ++i)
    out << "property float f_dc_" << i << std::endl;
  for(auto i = 0; i < 45; ++i)
    out << "property float f_rest_" << i << std::endl;
  out << "property float opacity" << std::endl;
  for(auto i = 0; i < 3; ++i)
    out << "property float scale_" << i << std::endl;
  for(auto i = 0; i < 4; ++i)
    out << "property float rot_" << i << std::endl;
  out << "end_header" << std::endl;

  // prints the splat values
  for(auto i = 0; i < 3; ++i)
    out << currentSplatSet->positions[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 3; ++i)
    out << "0 ";  // no normals
  for(auto i = 0; i < 3; ++i)
    out << currentSplatSet->f_dc[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 45; ++i)
    out << currentSplatSet->f_rest[splatIdx * 45 + i] << " ";
  out << currentSplatSet->opacity[splatIdx] << " ";
  for(auto i = 0; i < 3; ++i)
    out << currentSplatSet->scale[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 4; ++i)
    out << currentSplatSet->rotation[splatIdx * 4 + i] << " ";

  //
  out.close();

  //
  LOGI("Splat dumped: Global ID=%d, SplatSet Index=%d, Local Splat Index=%d -> c:\\Temp\\debug_splat.ply\n",
       globalSplatId, splatSetIndex, splatIdx);
}

}  // namespace vk_gaussian_splatting
