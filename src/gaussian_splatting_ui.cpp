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
#include "nvgui/azimuth_sliders.hpp"
#include "nvgui/tonemapper.hpp"

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
#include "utilities_ui.h"
#include "memory_statistics.h"
#include "memory_monitor_vk.h"
#include "vkgs_project_reader.h"
#include "vkgs_project_writer.h"
#include "hardware_support.h"
#include <GLFW/glfw3.h>
#undef APIENTRY
#include <fmt/format.h>

namespace vk_gaussian_splatting {

static VkResult saveRawImageToFile(VkDevice                     device,
                                   VkImage                      dstImage,
                                   VkDeviceMemory               dstImageMemory,
                                   VkExtent2D                   size,
                                   const std::filesystem::path& filename);

GaussianSplattingUI::GaussianSplattingUI(nvutils::ProfilerManager*   profilerManager,
                                         nvutils::ParameterRegistry* parameterRegistry,
                                         bool*                       benchmarkEnabled)
    : GaussianSplatting(profilerManager, parameterRegistry)
    , m_pBenchmarkEnabled(benchmarkEnabled)
    , m_imageCompareUI(&m_imageCompare)
{
  registerParameters(parameterRegistry);
};

void GaussianSplattingUI::registerParameters(nvutils::ParameterRegistry* parameterRegistry)
{
  // Parameters requiring callbacks with access to UI-level state (m_assets, m_app, etc.).
  // Simple value-assignment parameters are registered in registerCommandLineParameters() in parameters.cpp.

  parameterRegistry->add({.name = "updateData",
                          .help = "Use only in benchmark script. 1=triggers an update of data buffers or textures after a parameter change.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                m_assets.splatSets.markAllSplatSetsForRegeneration();
                                m_requestUpdateShaders = true;
                              }},
                         &m_updateDataTrigger, true);

  parameterRegistry->add({.name = "screenshot",
                          .help = "Use only in benchmark script. Takes a screenshot from the swapchain (not available in headless mode; use --saveImage instead).",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                if(m_app)
                                {
                                  m_app->saveScreenShot(m_screenshotFilename);
                                }
                              }},
                         {".png"}, &m_screenshotFilename);

  parameterRegistry->add({"saveImageBuffer",
                          "Buffer index for --saveImage. -1=all(default), 0=main, 1=aux1, "
                          "2=comparison(if active), 3=normal, 4=depth, 5=ldr(if tonemapping), "
                          "6+=dlss buffers(if enabled). Use --saveImage after setting this."},
                         &m_saveImageBufferIndex, int32_t(-1), int32_t(20));

  parameterRegistry->add(
      {.name = "saveImage",
       .help = "Save internal render buffer(s) to file. Supports .png/.jpg/.hdr extensions.\n"
               "Use --saveImageBuffer to select which buffer (-1=all).\n"
               "Buffer indices:\n"
               "  0: main (HDR color buffer)\n"
               "  1: aux1 (temporal intermediate)\n"
               "  2: comparison (if image compare active)\n"
               "  3: normal\n"
               "  4: depth\n"
               "  5: ldr (if tonemapping active)\n"
               "  6+: DLSS buffers (if DLSS enabled):\n"
               "    6: dlss_input, 7: dlss_albedo, 8: dlss_specular,\n"
               "    9: dlss_normal, 10: dlss_motion, 11: dlss_depth\n"
               "  -1: all available buffers (default)",
       .callbackSuccess =
           [&](const nvutils::ParameterBase* const) { saveBufferToFile(m_saveImageFilename, m_saveImageBufferIndex); }},
      {".png"}, &m_saveImageFilename);

  parameterRegistry->add({.name = "loadCameraPresets",
                          .help = "Load camera presets from an INRIA-format JSON file. Presets are appended to the existing list.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                if(!importCamerasINRIA(m_cameraPresetsFilename.string(), m_assets.cameras))
                                {
                                  LOGW("Failed to load camera presets from '%s'\n", m_cameraPresetsFilename.string().c_str());
                                }
                              }},
                         {".json"}, &m_cameraPresetsFilename);

  parameterRegistry->add({.name = "activateCameraPreset",
                          .help = "Instantly activate a camera preset by index (0-based). Use after --loadCameraPresets.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                if(m_activateCameraPresetIndex < 0
                                   || static_cast<uint64_t>(m_activateCameraPresetIndex) >= m_assets.cameras.size())
                                {
                                  LOGW("Camera preset index %d is out of range [0, %zu)\n", m_activateCameraPresetIndex,
                                       m_assets.cameras.size());
                                  return;
                                }
                                uint64_t index = static_cast<uint64_t>(m_activateCameraPresetIndex);
                                if(cameraPresetNeedsShaderRebuild(index))
                                {
                                  m_requestUpdateShaders = true;
                                }
                                m_assets.cameras.loadPreset(index, true);
                                m_lastLoadedCamera = index;
                              }},
                         &m_activateCameraPresetIndex);

  parameterRegistry->add({.name = "colorBufferFormat",
                          .help = "Color buffer format: 0=R8G8B8A8_UNORM (32-bit), 1=R16G16B16A16_SFLOAT (64-bit, default), "
                                  "2=R32G32B32A32_SFLOAT (128-bit). Use in benchmark .cfg only (requires initialized app).",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                constexpr VkFormat formats[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                                VK_FORMAT_R32G32B32A32_SFLOAT};
                                if(m_colorBufferFormatIndex < 0 || m_colorBufferFormatIndex > 2)
                                {
                                  LOGW("colorBufferFormat %d is out of range [0, 2]\n", m_colorBufferFormatIndex);
                                  return;
                                }
                                prmRender.colorFormat = formats[m_colorBufferFormatIndex];
                                if(!m_app)
                                {
                                  LOGW("colorBufferFormat: app not initialized, format will be applied at startup\n");
                                  return;
                                }
                                m_requestGBufferReinit = true;
                                resetFrameCounter();
                              }},
                         &m_colorBufferFormatIndex, 0, 2);
}

GaussianSplattingUI::~GaussianSplattingUI(){
    // Nothing to do here
};

void GaussianSplattingUI::onAttach(nvapp::Application* app)
{
  // Initializes the core

  GaussianSplatting::onAttach(app);

  // Override global tree node / selectable header colors to match
  // our collapsible group header background (WindowBg + small offset)
  {
    ImGuiStyle& style                    = ImGui::GetStyle();
    ImVec4      bg                       = style.Colors[ImGuiCol_WindowBg];
    ImVec4      headerBg                 = ImVec4(bg.x + 0.06f, bg.y + 0.06f, bg.z + 0.06f, 1.0f);
    ImVec4      headerHovered            = ImVec4(bg.x + 0.10f, bg.y + 0.10f, bg.z + 0.10f, 1.0f);
    style.Colors[ImGuiCol_Header]        = headerBg;
    style.Colors[ImGuiCol_HeaderHovered] = headerHovered;
    style.Colors[ImGuiCol_HeaderActive]  = headerBg;
  }

  // Cache GPU device name (static for the lifetime of the app)
  {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &properties);
    m_cachedGpuName = properties.deviceName;
  }

  // Create and register NVML monitor and profiler elements (skip in benchmark mode)
  if(!*m_pBenchmarkEnabled)
  {
    // NVML GPU monitor — suppress its own View menu entry
    struct GpuMonitorNoMenu : nvgpu_monitor::ElementGpuMonitor
    {
      void onUIMenu() override {}
    };
    auto gpuMonitor = std::make_shared<GpuMonitorNoMenu>();
    m_gpuMonitor    = gpuMonitor.get();
    m_app->addElement(gpuMonitor);

    // Profiler — suppress its own View menu entry
    m_profilerViewSettings = std::make_shared<nvapp::ElementProfiler::ViewSettings>(
        nvapp::ElementProfiler::ViewSettings{.name       = "Profiler",
                                             .defaultTab = nvapp::ElementProfiler::TABLE,
                                             .pieChart   = {.cpuTotal = false, .levels = true},
                                             .lineChart  = {.cpuLine = false}});
    struct ProfilerNoMenu : nvapp::ElementProfiler
    {
      using ElementProfiler::ElementProfiler;
      void onUIMenu() override {}
    };
    m_app->addElement(std::make_shared<ProfilerNoMenu>(m_profilerManager, m_profilerViewSettings));
  }

  // Init combo selectors used in UI

  m_ui.enumAdd(GUI_STORAGE, STORAGE_BUFFERS, "Buffers");
  m_ui.enumAdd(GUI_STORAGE, STORAGE_TEXTURES, "Textures");

  // Pipeline entries are greyed out at creation time when their required
  // extensions are missing. Hardware support is fixed for the lifetime of the
  // process so a single registration suffices. Coercion of any out-of-range
  // value (project file / CLI / hotkey) lives in
  // GaussianSplatting::onPreRender().
  //   - PIPELINE_MESH / PIPELINE_MESH_3DGUT need VK_EXT_mesh_shader.
  //   - PIPELINE_RTX needs the KHR ray tracing trio (raytracing flag).
  //   - PIPELINE_HYBRID and PIPELINE_HYBRID_3DGUT need both, since they
  //     dispatch via vkCmdDrawMeshTasksEXT for primary rays and ray trace
  //     secondary rays.
  const bool meshUnavailable = !isSupported.meshShader;
  const bool rtxUnavailable  = !isSupported.raytracing;
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_VERT, "Raster 3DGS vertex shader");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH, "Raster 3DGS mesh shader", meshUnavailable);
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH_3DGUT, "Raster 3DGUT mesh shader", meshUnavailable);
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_RTX, "Ray tracing 3DGRT", rtxUnavailable);
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_HYBRID, "Hybrid 3DGS+3DGRT", meshUnavailable || rtxUnavailable);
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_HYBRID_3DGUT, "Hybrid 3DGUT+3DGRT", meshUnavailable || rtxUnavailable);

  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_EIGEN, "Eigen");
  m_ui.enumAdd(GUI_EXTENT_METHOD, EXTENT_CONIC, "Conic");

  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_FINAL, "Final render");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLAY, "Clay mode");
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
  m_ui.enumAdd(GUI_VISUALIZE_DLSS_ON, VISUALIZE_CLAY, "Clay");
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

  m_ui.enumAdd(GUI_FRUSTUM_CULLING, FRUSTUM_CULLING_NONE, "Disabled");
  m_ui.enumAdd(GUI_FRUSTUM_CULLING, FRUSTUM_CULLING_AT_DIST, "At distance stage");
  m_ui.enumAdd(GUI_FRUSTUM_CULLING, FRUSTUM_CULLING_AT_RASTER, "At raster stage");

  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT32, "Float 32");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT16, "Float 16");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_UINT8, "Uint8");

  m_ui.enumAdd(GUI_RGBA_FORMAT, FORMAT_FLOAT32, "Float 32");
  m_ui.enumAdd(GUI_RGBA_FORMAT, FORMAT_FLOAT16, "Float 16");
  m_ui.enumAdd(GUI_RGBA_FORMAT, FORMAT_UINT8, "Uint8");

  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_ICOSAHEDRON, "Icosahedron");
  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_PARAMETRIC, "AABB + parametric");
  // Sphere (NV) is greyed out when the device does not expose
  // VK_NV_ray_tracing_linear_swept_spheres. Hardware support is fixed for
  // the lifetime of the process so a single menu suffices. Coercion of any
  // out-of-range value (e.g. a project file or CLI requesting sphere mode
  // on hardware without LSS) lives in GaussianSplatting::onPreRender().
  m_ui.enumAdd(GUI_PARTICLE_FORMAT, PARTICLE_FORMAT_SPHERE, "Sphere (NV)", !isSupported.rayTracingLinearSweptSpheres);

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
  m_ui.enumAdd(GUI_NORMAL_METHOD, (int)NormalMethod::eIsoSurface, "Kernel ellipsoid");

  m_ui.enumAdd(GUI_LIGHTING_MODE, LIGHTING_DISABLED, "Lighting off");
  m_ui.enumAdd(GUI_LIGHTING_MODE, LIGHTING_ENABLED, "Lighting on");

  m_ui.enumAdd(GUI_SHADOWS_MODE, (int)ShadowsMode::eShadowsDisabled, "Shadows off");
  m_ui.enumAdd(GUI_SHADOWS_MODE, (int)ShadowsMode::eShadowsHard, "Hard shadows");
  m_ui.enumAdd(GUI_SHADOWS_MODE, (int)ShadowsMode::eShadowsSoft, "Soft shadows");

  m_ui.enumAdd(GUI_DOF_MODE, (int)DofMode::eDofDisabled, "Disabled");
  m_ui.enumAdd(GUI_DOF_MODE, (int)DofMode::eDofFixedFocus, "Fixed focus");
  m_ui.enumAdd(GUI_DOF_MODE, (int)DofMode::eDofAutoFocus, "Auto focus");

  m_ui.enumAdd(GUI_DOF_MODE_NO_AUTO, (int)DofMode::eDofDisabled, "Disabled");
  m_ui.enumAdd(GUI_DOF_MODE_NO_AUTO, (int)DofMode::eDofFixedFocus, "Fixed focus");

  m_ui.enumAdd(GUI_PARTICLE_DEPTH, PARTICLE_DEPTH_BILLBOARD, "Billboard (3DGS/3DGUT)");
  m_ui.enumAdd(GUI_PARTICLE_DEPTH, PARTICLE_DEPTH_ELLIPSOID, "Ellipsoid (3DGRT)");
  // m_ui.enumAdd(GUI_PARTICLE_DEPTH, PARTICLE_DEPTH_MAX_DENSITY_PLANE, "Max Density Plane (StochasticSplat)");

  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingFitted, "Fitted");
  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingUniform, "Uniform");
  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingUniform3_4, "Uniform 3/4");
  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingUniform2_3, "Uniform 2/3");
  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingUniform1_2, "Uniform 1/2");
  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingUniform1_3, "Uniform 1/3");
  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingUniform1_4, "Uniform 1/4");
  m_ui.enumAdd(GUI_BILLBOARD_BOUNDING_MODE, (int)BillboardBoundingMode::eBillboardBoundingOptimal, "Optimal");

  m_ui.enumAdd(GUI_COVARIANCE_DILATION, 0, "0.0");
  m_ui.enumAdd(GUI_COVARIANCE_DILATION, 1, "0.1");
  m_ui.enumAdd(GUI_COVARIANCE_DILATION, 2, "0.2");
  m_ui.enumAdd(GUI_COVARIANCE_DILATION, 3, "0.3");
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

  // Particle count (uint32_t to avoid overflow beyond INT32_MAX ~2.1B splats)
  uint32_t totalSplatCount = m_assets.splatSets.getTotalGlobalSplatCount();
  if(isRtxPipelineOnly())
  {
    // Pure ray tracing: show total only
    ImGui::Text("Particles %s", formatSize(totalSplatCount).c_str());
  }
  else
  {
    // Raster and hybrid modes: show rasterized / total
    const bool usesDistShader =
        (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
    uint32_t rasterSplatCount = usesDistShader ? m_indirectReadback.instanceCount : totalSplatCount;
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
  const bool isFloat = (filename.extension() == ".hdr" || filename.extension() == ".raw");
  VkFormat   format  = isFloat ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;

  // Convert to linear tiled image (handles format conversion via GPU blit)
  nvvk::imageToLinear(cmd, m_device, m_app->getPhysicalDevice(), srcImageInfo.image, srcImageInfo.size, dstImage,
                      dstImageMemory, format);

  // Submit and wait for completion (synchronous)
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Save to file
  if(isFloat && filename.extension() == ".raw")
    saveRawImageToFile(m_device, dstImage, dstImageMemory, srcImageInfo.size, filename);
  else
    nvvk::saveImageToFile(m_device, dstImage, dstImageMemory, srcImageInfo.size, filename, 90);

  // Clean up temporary resources
  vkFreeMemory(m_device, dstImageMemory, nullptr);
  vkDestroyImage(m_device, dstImage, nullptr);

  // Multi-buffer dump: save all available G-buffers with postfixed filenames
  const std::filesystem::path stem = filename.parent_path() / filename.stem();
  const std::filesystem::path ext  = filename.extension();

  for(const auto& buf : getAllDumpableBuffers())
  {
    if(buf.image == VK_NULL_HANDLE)
      continue;

    const std::filesystem::path postfixedPath = std::filesystem::path(stem.string() + buf.postfix + ext.string());

    VkFormat bufDstFormat = isFloat ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;

    VkCommandBuffer bufCmd       = m_app->createTempCmdBuffer();
    VkImage         bufDstImage  = {};
    VkDeviceMemory  bufDstMemory = {};

    nvvk::imageToLinear(bufCmd, m_device, m_app->getPhysicalDevice(), buf.image, buf.size, bufDstImage, bufDstMemory, bufDstFormat);
    m_app->submitAndWaitTempCmdBuffer(bufCmd);

    if(isFloat && ext == ".raw")
      saveRawImageToFile(m_device, bufDstImage, bufDstMemory, buf.size, postfixedPath);
    else
      nvvk::saveImageToFile(m_device, bufDstImage, bufDstMemory, buf.size, postfixedPath, 90);

    vkFreeMemory(m_device, bufDstMemory, nullptr);
    vkDestroyImage(m_device, bufDstImage, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Save a specific buffer by index, or all buffers if bufferIndex == -1.
//
void GaussianSplattingUI::saveBufferToFile(const std::filesystem::path& filename, int32_t bufferIndex)
{
  const auto buffers = getAllDumpableBuffers();

  if(buffers.empty())
  {
    LOGW("saveBufferToFile: No buffers available for saving.\n");
    return;
  }

  const bool isFloat   = (filename.extension() == ".hdr" || filename.extension() == ".raw");
  VkFormat   dstFormat = isFloat ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;

  const std::filesystem::path stem = filename.parent_path() / filename.stem();
  const std::filesystem::path ext  = filename.extension();

  auto saveOneBuffer = [&](const BufferDumpInfo& buf) {
    if(buf.image == VK_NULL_HANDLE)
      return;

    const std::filesystem::path outPath = std::filesystem::path(stem.string() + buf.postfix + ext.string());

    VkCommandBuffer cmd       = m_app->createTempCmdBuffer();
    VkImage         dstImage  = {};
    VkDeviceMemory  dstMemory = {};

    nvvk::imageToLinear(cmd, m_device, m_app->getPhysicalDevice(), buf.image, buf.size, dstImage, dstMemory, dstFormat);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    if(isFloat && ext == ".raw")
      saveRawImageToFile(m_device, dstImage, dstMemory, buf.size, outPath);
    else
      nvvk::saveImageToFile(m_device, dstImage, dstMemory, buf.size, outPath, 90);

    vkFreeMemory(m_device, dstMemory, nullptr);
    vkDestroyImage(m_device, dstImage, nullptr);
  };

  if(bufferIndex == -1)
  {
    for(const auto& buf : buffers)
      saveOneBuffer(buf);
  }
  else if(bufferIndex >= 0 && bufferIndex < static_cast<int32_t>(buffers.size()))
  {
    saveOneBuffer(buffers[bufferIndex]);
  }
  else
  {
    LOGW("saveBufferToFile: Buffer index %d out of range [0, %d]. Use -1 for all.\n", bufferIndex,
         static_cast<int32_t>(buffers.size()) - 1);
  }
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
  static bool save_overwrite_requested{false};
  bool        v_sync = m_app->isVsync();
#ifndef NDEBUG
  static bool s_showDemo{false};
  static bool s_showDemoPlot{false};
  static bool s_showDemoIcons{false};
#endif

  // Ctrl+S shortcut (outside menu so it works even when menu is closed)
  if(ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S) && !m_projectPath.empty())
  {
    save_overwrite_requested = true;
  }

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
    if(ImGui::MenuItem(ICON_MS_FILE_SAVE " Save project", "Ctrl+S", false, !m_projectPath.empty()))
    {
      save_overwrite_requested = true;
    }
    if(ImGui::MenuItem(ICON_MS_FILE_SAVE " Save project as...", ""))
    {
      auto path = nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save project file", "VKGS Files|*.vkgs");
      if(!path.empty())
      {
        if(!nvutils::extensionMatches(path, ".vkgs"))
        {
          path = path.replace_extension(".vkgs");
        }

        if(saveProject(path.string()))
        {
          m_projectPath = std::filesystem::absolute(path);
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
      auto path = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Mesh",
                                              "All Mesh Files|*.obj;*.gltf;*.glb|OBJ Files|*.obj|glTF Files|*.gltf;*.glb");
      if(!path.empty())
      {
        prmScene.meshToImportFilename = path;
      }
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent Meshes"))
    {
      for(const auto& file : m_recentMeshes)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.meshToImportFilename = file;
        }
      }
      ImGui::EndMenu();
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
    if(ImGui::MenuItem(ICON_MS_FULLSCREEN " Full Screen", "F11", m_fullScreen))
    {
      m_fullScreen = !m_fullScreen;
    }
    ImGui::Separator();
    ImGui::MenuItem(ICON_MS_FOLDER_OPEN " Assets Browser", nullptr, &m_showAssetsWindow);
    ImGui::MenuItem(ICON_MS_TUNE " Assets Properties", nullptr, &m_showPropertiesWindow);
    ImGui::Separator();
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Renderer Statistics", nullptr, &m_showRendererStatistics);
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Memory Statistics", nullptr, &m_showMemoryStatistics);
    ImGui::MenuItem(ICON_MS_DATA_TABLE " Shader Feedback", nullptr, &m_showShaderFeedback);
    ImGui::MenuItem(ICON_MS_DOCK_TO_BOTTOM " Footer Bar", nullptr, &m_showFooterBar);
    if(m_profilerViewSettings)
      ImGui::MenuItem(ICON_MS_BLOOD_PRESSURE " Profiler", nullptr, &m_profilerViewSettings->show);
    if(m_gpuMonitor)
      ImGui::MenuItem(ICON_MS_BROWSE_ACTIVITY " NVML Monitor", nullptr, &m_gpuMonitor->showWindow);
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_VISIBILITY " Show all"))
    {
      m_showAssetsWindow = m_showPropertiesWindow = m_showRendererStatistics = m_showMemoryStatistics =
          m_showShaderFeedback = m_showFooterBar = true;
      if(m_profilerViewSettings)
        m_profilerViewSettings->show = true;
    }
    if(ImGui::MenuItem(ICON_MS_VISIBILITY_OFF " Hide all"))
    {
      m_showAssetsWindow = m_showPropertiesWindow = m_showRendererStatistics = m_showMemoryStatistics =
          m_showShaderFeedback = m_showFooterBar = false;
      if(m_profilerViewSettings)
        m_profilerViewSettings->show = false;
      if(m_gpuMonitor)
        m_gpuMonitor->showWindow = false;
    }
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

  // Save overwrite confirmation modal
  if(save_overwrite_requested)
  {
    ImGui::OpenPopup("Overwrite project file?");
    save_overwrite_requested = false;
  }
  {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if(ImGui::BeginPopupModal("Overwrite project file?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      ImGui::Text("Overwrite %s?", m_projectPath.filename().string().c_str());
      ImGui::Separator();

      if(ImGui::Button("OK", ImVec2(120, 0)))
      {
        if(saveProject(m_projectPath.string()))
        {
          guiAddToRecentProjects(m_projectPath);
        }
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if(ImGui::Button("Cancel", ImVec2(120, 0)))
      {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
  }

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

    // Full Screen button (on same line)
    ImGui::SameLine();

    pushIconStyle(m_fullScreen);

    if(ImGui::Button(m_fullScreen ? ICON_MS_FULLSCREEN_EXIT : ICON_MS_FULLSCREEN))
    {
      m_fullScreen = !m_fullScreen;
    }

    popIconStyle();

    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip(m_fullScreen ? "Exit full screen (F11)" : "Enter full screen (F11)");
    }

    // Screenshot button (on same line)
    ImGui::SameLine();

    if(ImGui::Button(ICON_MS_ADD_PHOTO_ALTERNATE))
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

    if(ImGui::Button(ICON_MS_INFO))
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
      ImGui::SetTooltip("Toggle to editing mode (E)");
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
      ImGui::SetTooltip("Toggle light proxy visibility (L)");
    }

    // Vertical separator before navigation mode
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Navigation mode buttons (exclusive: Examine / Fly / Walk)
    {
      auto navMode = cameraManip->getMode();

      pushIconStyle(navMode == nvutils::CameraManipulator::Examine);
      if(ImGui::Button(ICON_MS_3D_ROTATION))
      {
        cameraManip->setMode(nvutils::CameraManipulator::Examine);
      }
      popIconStyle();
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("Examine - orbit around point of interest");
      }

      ImGui::SameLine();

      pushIconStyle(navMode == nvutils::CameraManipulator::Fly);
      if(ImGui::Button(ICON_MS_FLIGHT))
      {
        cameraManip->setMode(nvutils::CameraManipulator::Fly);
      }
      popIconStyle();
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("Fly - free camera movement - WASD keys plus mouse");
      }

      ImGui::SameLine();

      pushIconStyle(navMode == nvutils::CameraManipulator::Walk);
      if(ImGui::Button(ICON_MS_DIRECTIONS_WALK))
      {
        cameraManip->setMode(nvutils::CameraManipulator::Walk);
      }
      popIconStyle();
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("Walk - move on XZ plane - WASD keys plus mouse");
      }

      ImGui::SameLine();

      pushIconStyle(m_playPresets);
      if(ImGui::Button(ICON_MS_LAPS))
      {
        m_playPresets = !m_playPresets;
        if(m_playPresets && m_assets.cameras.size() >= 2)
        {
          m_lastLoadedCamera                    = 0;
          m_requestUpdateShadersAfterCameraAnim = cameraPresetNeedsShaderRebuild(0);
          m_assets.cameras.loadPreset(0, false);
          m_selectedCameraPresetIndex = -1;
        }
        else if(!m_playPresets && cameraManip->isAnimated())
        {
          // Stop mid-flight: snap to the current interpolated position (instantSet=true
          // cancels the animation and keeps the camera where it is right now)
          cameraManip->setCamera(cameraManip->getCamera(), true);
        }
      }
      popIconStyle();
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("Play/pause camera preset cycling");
      }
    }

    // Vertical separator before pipeline/sorting/RTX settings
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Pipeline selector
    {
      ImGui::SetNextItemWidth(200.0f);
      if(m_ui.enumCombobox(GUI_PIPELINE, "##PipelineBar", &prmSelectedPipeline))
      {
        m_requestUpdateShaders = true;
      }
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("Rendering pipeline");
    }

    ImGui::SameLine();

    // Sorting method selector (disabled only for pure RTX - hybrid modes still use rasterization)
    ImGui::BeginDisabled(isRtxPipelineOnly());
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
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip(
          "DLSS Ray Reconstruction denoising mode.\n"
          "Available with ray tracing and hybrid pipelines.\n\n"
          "- Disabled: DLSS is off, rendering at native resolution.\n"
          "- Min: smallest internal resolution, fastest but lowest quality.\n"
          "- Optimal: balanced upscaling, recommended for most use cases.\n"
          "- Max: largest internal resolution, denoising and anti-aliasing only.");
    ImGui::EndDisabled();
#endif

    ImGui::SameLine();

    // Visualization selector available for all ray tracing pipelines (pure RTX and hybrids)
    ImGui::BeginDisabled(!isRtxPipelineActive());
    // visualization mode selector
    auto visuMenu = GUI_VISUALIZE;
    if(m_dlss.isEnabled())
      visuMenu = GUI_VISUALIZE_DLSS_ON;

    static constexpr const char* visualizeTooltip =
        "Selects the visualization mode.\n"
        "Available only with ray tracing and hybrid pipelines.\n\n"
        "Final render: standard rendered output.\n"
        "Clay mode: renders all surfaces with a uniform clay color.\n"
        "Clock cycles: heat-map of GPU clock cycles per pixel.\n"
        "Ray Hit Count: heat-map of ray intersection tests per pixel.\n"
        "Depth (iso thres): depth from integrated iso-threshold.\n"
        "Depth (Closest hit): depth of the closest ray-particle hit.\n"
        "Depth (for DLSS): depth buffer as fed to the DLSS denoiser.\n"
        "Normal (Integrated): surface normal from integrated contributions.\n"
        "Normal (closest hit): surface normal at the closest ray-particle hit.\n"
        "Normal (For DLSS): normal buffer as fed to the DLSS denoiser.\n"
        "Splat ID (Harlequin): unique color per splat for identification.\n\n"
        "DLSS guide modes (enabled only when DLSS is active):\n"
        "DLSS Input: raw radiance input before denoising.\n"
        "DLSS Guide Albedo/Specular/Normal/Motion/Depth: individual G-buffers\n"
        "  used by the DLSS denoiser.";

    ImGui::SetNextItemWidth(150.0f);
    if(m_ui.enumCombobox(visuMenu, "##ID", &prmRender.visualize))
    {
      m_requestUpdateShaders = true;
    }
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("%s", visualizeTooltip);

    ImGui::EndDisabled();

    // Measure actual group width for next frame's centering calculation
    totalGroupWidth = ImGui::GetItemRectMax().x - groupStartScreenX;
  }

  // Shortcuts


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
  if(m_playPresets && !cameraManip->isAnimated() && m_assets.cameras.size() >= 2)
  {
    m_lastLoadedCamera                    = (m_lastLoadedCamera + 1) % m_assets.cameras.size();
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
  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_E))
  {
    m_helpers.setEditingMode(!m_helpers.isEditingMode());
  }
  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_G))
  {
    m_helpers.grid.toggleVisible();
  }
  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_L))
  {
    m_showLightProxies = !m_showLightProxies;
    resetFrameCounter();
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


  // hot rebuild of shaders
  if(!wantTextInput && ImGui::IsKeyPressed(ImGuiKey_R))
  {
    m_requestUpdateShaders = true;
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

  // F11 fullscreen toggle
  if(ImGui::IsKeyPressed(ImGuiKey_F11))
  {
    m_fullScreen = !m_fullScreen;
  }

  // Apply fullscreen state change
  {
    GLFWwindow* window       = m_app->getWindowHandle();
    bool        isFullScreen = glfwGetWindowMonitor(window) != nullptr;
    if(m_fullScreen != isFullScreen)
    {
      if(m_fullScreen)
      {
        glfwGetWindowPos(window, &m_windowedPos[0], &m_windowedPos[1]);
        glfwGetWindowSize(window, &m_windowedSize[0], &m_windowedSize[1]);
        GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
      }
      else
      {
        glfwSetWindowMonitor(window, nullptr, m_windowedPos[0], m_windowedPos[1], m_windowedSize[0], m_windowedSize[1], 0);
      }
    }
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
  else if(extension == ".obj" || extension == ".gltf" || extension == ".glb")
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
      ;
  if(!m_projectPath.empty())
  {
    title += " | " + m_projectPath.stem().string();
  }
  title += " | "
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

  // Handle mesh import requests (modal + synchronous load)
  guiImportMeshIfNeeded();

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

  if(m_showFooterBar)
    guiDrawFooterBar();
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
    changed =
        PE::entry("Sorting method", [&]() { return m_ui.enumCombobox(GUI_SORTING, "##ID", &prmRaster.sortingMethod); }, tooltip);
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
      "- Lighting on: enables shading from lights (direct in raster, full path tracing in ray tracing).";

  bool changed = false;
  if(inMenuBar)
  {
    ImGui::SetNextItemWidth(150.0f);
    changed = m_ui.enumCombobox(GUI_LIGHTING_MODE, "##LightingMode", &prmRender.lightingEnabled);
    if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
      ImGui::SetTooltip("%s", tooltip);
  }
  else
  {
    changed =
        PE::entry("Lighting", [&]() { return m_ui.enumCombobox(GUI_LIGHTING_MODE, "##ID", &prmRender.lightingEnabled); }, tooltip);
  }

  if(changed)
  {
    m_requestUpdateShaders = true;
    m_tonemapperData.isActive = (prmRender.lightingEnabled == LIGHTING_ENABLED) ? 1 : 0;
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

  bool disabled = (!prmRender.lightingEnabled || !isRtxPipelineActive());

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
  if(!m_showAssetsWindow)
    return;

  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ChildBg]);

  if(ImGui::Begin("Assets", &m_showAssetsWindow))
  {
    // Settings (leaf node)
    {
      ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_SETTINGS)
        flags |= ImGuiTreeNodeFlags_Selected;
      ImGui::TreeNodeEx(ICON_MS_SETTINGS " Settings", flags);
      if(ImGui::IsItemClicked())
      {
        resetSelection();
        m_selectedAsset = GUI_SETTINGS;
      }
    }

    guiDrawRendererTree();

    guiDrawCameraTree();

    guiDrawLightTree();

    guiDrawSkyTree();

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

  // Renderer node with inline pipeline selector
  node_flags = base_flags | ImGuiTreeNodeFlags_SpanTextWidth;
  if(m_selectedAsset == GUI_RENDERER)
    node_flags |= ImGuiTreeNodeFlags_Selected;
  ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  node_open = ImGui::TreeNodeEx(ICON_MS_CAMERA " Renderer -##Renderer", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    resetSelection();
    m_selectedAsset             = GUI_RENDERER;
    m_selectedCameraPresetIndex = -1;
  }

  // Pipeline selector on the same line, compact height, filling remaining width
  ImGui::SameLine();
  float availWidth = ImGui::GetContentRegionAvail().x;
  ImGui::SetNextItemWidth(availWidth);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1.0f));
  bool pipelineChanged = m_ui.enumCombobox(GUI_PIPELINE, "##PipelineSelector", &prmSelectedPipeline);
  ImGui::PopStyleVar();
  if(pipelineChanged)
  {
    m_requestUpdateShaders = true;
  }

  if(node_open)
  {
    guiDrawRasterizationTree();
    guiDrawRaytracingTree();
    guiDrawDenoisingTree();
    guiDrawTonemappingTree();
    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawRasterizationTree()
{
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  if(m_selectedAsset == GUI_RASTERIZATION)
    flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::BeginDisabled(!isRasterPipelineActive());
  ImGui::TreeNodeEx(ICON_MS_GRID_ON " Rasterization", flags);
  if(ImGui::IsItemClicked())
  {
    resetSelection();
    m_selectedAsset = GUI_RASTERIZATION;
  }
  ImGui::EndDisabled();
}

void GaussianSplattingUI::guiDrawRaytracingTree()
{
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  if(m_selectedAsset == GUI_RAYTRACING)
    flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::BeginDisabled(!isRtxPipelineActive());
  ImGui::TreeNodeEx(ICON_MS_CALL_MISSED_OUTGOING " Raytracing", flags);
  if(ImGui::IsItemClicked())
  {
    resetSelection();
    m_selectedAsset = GUI_RAYTRACING;
  }
  ImGui::EndDisabled();
}

void GaussianSplattingUI::guiDrawDenoisingTree()
{
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  if(m_selectedAsset == GUI_DENOISING)
    flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::BeginDisabled(!isDlssSupportedPipeline());
  ImGui::TreeNodeEx(ICON_MS_BLUR_ON " Denoising", flags);
  if(ImGui::IsItemClicked())
  {
    resetSelection();
    m_selectedAsset = GUI_DENOISING;
  }
  ImGui::EndDisabled();
}

void GaussianSplattingUI::guiDrawTonemappingTree()
{
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  if(m_selectedAsset == GUI_TONEMAPPING)
    flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::TreeNodeEx(ICON_MS_TONALITY " Tone mapping", flags);
  if(ImGui::IsItemClicked())
  {
    resetSelection();
    m_selectedAsset = GUI_TONEMAPPING;
  }
}

void GaussianSplattingUI::guiDrawDenoisingProperties()
{
  bool denoisingDisabled = !isDlssSupportedPipeline();
  ImGui::BeginDisabled(denoisingDisabled);

  namespace PE = nvgui::PropertyEditor;

  bool open = beginCollapsibleGroup("DLSS-RR", true);
  if(open)
  {
    PE::begin("## DLSS", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

#if defined(USE_DLSS)
    bool dlssAvailable = m_dlss.isRuntimeSupported() && m_dlss.isInitialized();

    if(!dlssAvailable && m_dlss.isInitialized())
    {
      PE::Text("DLSS", "Not available on this device");
    }

    ImGui::BeginDisabled(!dlssAvailable);

    bool dlssEnabled = m_dlss.isEnabled();
    if(PE::Checkbox("Enable", &dlssEnabled,
                    "Enable DLSS Ray Reconstruction denoising.\n"
                    "Requires a supported NVIDIA GPU and driver.\n"
                    "Available with ray tracing and hybrid pipelines."))
    {
      m_dlss.setEnabled(dlssEnabled);
      m_requestUpdateShaders = true;
    }

    ImGui::BeginDisabled(!m_dlss.isEnabled());

    {
      const char* sizeModes[]     = {"Min", "Optimal", "Max"};
      int         currentSizeMode = static_cast<int>(m_dlss.getSizeMode());
      if(PE::entry(
             "Size Mode",
             [&]() {
               bool changed = ImGui::Combo("##DLSSSizeMode", &currentSizeMode, sizeModes, IM_ARRAYSIZE(sizeModes));
               if(changed)
                 m_dlss.setSizeMode(static_cast<DlssDenoiser::SizeMode>(currentSizeMode));
               return changed;
             },
             "DLSS internal rendering resolution mode.\n"
             "- Min: smallest internal resolution, fastest but lowest quality.\n"
             "- Optimal: balanced upscaling, recommended for most use cases.\n"
             "- Max: largest internal resolution, denoising and anti-aliasing only."))
      {
      }
    }

    if(m_dlss.isInitialized())
    {
      VkExtent2D nativeSize = m_gBuffers.getSize();
      PE::entry(
          "Current Resolution",
          [&]() {
            if(m_dlss.isEnabled())
            {
              VkExtent2D dlssSize = m_dlss.getRenderSize();
              ImGui::Text("%d x %d (%d x %d)", dlssSize.width, dlssSize.height, nativeSize.width, nativeSize.height);
            }
            else
            {
              ImGui::Text("%d x %d", nativeSize.width, nativeSize.height);
            }
            return false;
          },
          "Internal rendering resolution.\n"
          "When DLSS is enabled, shows the reduced resolution followed\n"
          "by the native viewport resolution in parentheses.\n"
          "When DLSS is disabled, shows the native viewport resolution.");
    }

    PE::DragFloat("Minimum Radiance", &prmRtx.dlssMinRadianceThreshold, 0.001f, 0.0f, 0.1f, "%.3f", 0,
                  "Minimum radiance threshold for DLSS input.\n"
                  "Clamps pixel radiance to this floor before feeding to the DLSS denoiser.\n"
                  "Prevents negative values (e.g. from amplified AO) from causing\n"
                  "noisy artifacts in DLSS-denoised regions.");

    ImGui::EndDisabled();  // !m_dlss.isEnabled()
    ImGui::EndDisabled();  // !dlssAvailable
#else
    PE::Text("DLSS", "Not available (compile without USE_DLSS)");
#endif

    PE::end();
  }
  endCollapsibleGroup(open);

  ImGui::EndDisabled();
}

void GaussianSplattingUI::guiDrawTonemappingProperties()
{
  namespace PE = nvgui::PropertyEditor;

  bool changed = false;

  const char* methods[] = {"Filmic", "Uncharted 2", "Clip", "ACES", "AgX", "Khronos PBR"};

  {
    bool open = beginCollapsibleGroup("General", true);
    if(open)
    {
      PE::begin("##ToneMapMain", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      changed |= PE::Checkbox("Enable", reinterpret_cast<bool*>(&m_tonemapperData.isActive),
                              "Enable/disable tone mapping post-processing");
      ImGui::BeginDisabled(!m_tonemapperData.isActive);
      changed |= PE::Combo("Method", &m_tonemapperData.method, methods, IM_ARRAYSIZE(methods), 0,
                           "Tone mapping algorithm to compress high dynamic range (HDR) to standard dynamic range (SDR)");
      changed |= PE::SliderFloat("Exposure", &m_tonemapperData.exposure, 0.1F, 200.0F, "%.3f", ImGuiSliderFlags_Logarithmic,
                                 "Multiplier for input colors (0.1 = very dark, 1 = neutral, 200 = very bright)");
      changed |= PE::SliderFloat("Contrast", &m_tonemapperData.contrast, 0.0F, 2.0F, "%.2f", 0,
                                 "Scales colors away from gray (0 = no contrast, 1 = neutral, 2 = high contrast)");
      changed |= PE::SliderFloat("Brightness", &m_tonemapperData.brightness, 0.0F, 2.0F, "%.2f", 0,
                                 "Gamma curve for output colors (1 = neutral, higher values make midtones brighter)");
      changed |= PE::SliderFloat("Saturation", &m_tonemapperData.saturation, 0.0F, 2.0F, "%.2f", 0,
                                 "Controls color intensity (0 = grayscale, 1 = neutral, 2 = high saturation)");
      changed |= PE::SliderFloat("Vignette", &m_tonemapperData.vignette, -1.0F, 1.0F, "%.2f", 0,
                                 "Darkens image edges (-1 = very bright, 0 = none, 1 = very dark)");
      ImGui::EndDisabled();
      PE::end();
    }
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Auto Exposure", true);
    if(open)
    {
      ImGui::BeginDisabled(!m_tonemapperData.isActive);
      PE::begin("##AutoExposure", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      changed |= PE::Checkbox("Enable", reinterpret_cast<bool*>(&m_tonemapperData.autoExposure),
                              "Enable automatic exposure adjustment based on scene brightness");
      ImGui::BeginDisabled(!m_tonemapperData.autoExposure);
      changed |= PE::Combo("Average Mode", (int*)&m_tonemapperData.averageMode, "Mean\0Median", 0,
                           "Method for calculating scene brightness (Mean = average, Median = value where 50% of pixels are darker and 50% of pixels are brighter)");
      changed |= PE::DragFloat("Adaptation Speed", &m_tonemapperData.autoExposureSpeed, 0.001f, 0.f, 100.f, "%.3f",
                               ImGuiSliderFlags_AlwaysClamp,
                               "How quickly auto exposure adapts to lighting changes (higher = faster adaptation)");
      changed |= PE::DragFloat("Min (EV100)", &m_tonemapperData.evMinValue, 0.01f, -24.f, 24.f, "%.2f", 0,
                               "Minimum histogram luminance in logarithmic stops (-24 = very dark, +24 = very bright)");
      changed |= PE::DragFloat("Max (EV100)", &m_tonemapperData.evMaxValue, 0.01f, -24.f, 24.f, "%.2f", 0,
                               "Maximum histogram luminance in logarithmic stops (-24 = very dark, +24 = very bright)");
      changed |= PE::Checkbox("Center Weighted Metering", (bool*)&m_tonemapperData.enableCenterMetering,
                              "Use center area for exposure calculation instead of full frame");
      ImGui::BeginDisabled(!m_tonemapperData.enableCenterMetering);
      changed |= PE::DragFloat("Center Metering Size", &m_tonemapperData.centerMeteringSize, 0.01f, 0.01f, 1.0f, "%.2f",
                               0, "Size of center area for exposure calculation (0.01 = small spot, 1.0 = full frame)");
      ImGui::EndDisabled();
      ImGui::EndDisabled();
      ImGui::EndDisabled();
      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Advanced Color Grading ---
  {
    bool open = beginCollapsibleGroup("Advanced Color Grading");
    if(open)
    {
      ImGui::BeginDisabled(!m_tonemapperData.isActive);
      PE::begin("##ColorGrading", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      changed |= PE::SliderFloat("Vibrance", &m_tonemapperData.vibrance, -1.0F, 1.0F, "%.2f", 0,
                                 "Selective saturation boost for desaturated colors (0 = neutral, positive values boost muted colors)");
      changed |= PE::SliderFloat("Shadow Bias", &m_tonemapperData.shadowBias, -1.0F, 1.0F, "%.2f", 0,
                                 "Adjust shadow tones (-1 = darker, 0 = neutral, 1 = brighter)");
      changed |= PE::SliderFloat("Midtone Bias", &m_tonemapperData.midtoneBias, -1.0F, 1.0F, "%.2f", 0,
                                 "Adjust midtone brightness (-1 = darker, 0 = neutral, 1 = brighter)");
      changed |= PE::SliderFloat("Highlight Bias", &m_tonemapperData.highlightBias, -1.0F, 1.0F, "%.2f", 0,
                                 "Adjust highlight tones (-1 = darker, 0 = neutral, 1 = brighter)");

      if(PE::treeNode("Split Toning"))
      {
        changed |= PE::ColorEdit3("Cool Shadows", &m_tonemapperData.coolColor.x, ImGuiColorEditFlags_Float,
                                  "Color tint applied to shadow regions (default: white = no tint)");
        changed |= PE::ColorEdit3("Warm Highlights", &m_tonemapperData.warmColor.x, ImGuiColorEditFlags_Float,
                                  "Color tint applied to highlight regions (default: white = no tint)");
        changed |= PE::SliderFloat("Split Balance", &m_tonemapperData.splitBalance, -0.5F, 0.5F, "%.2f", 0,
                                   "Balance between cool and warm tones (-0.5 = more shadows cool, 0 = neutral, 0.5 = more highlights warm)");
        PE::treePop();
      }
      PE::end();
      ImGui::EndDisabled();
    }
    endCollapsibleGroup(open);
  }

  // --- White Balance ---
  {
    bool open = beginCollapsibleGroup("White Balance");
    if(open)
    {
      ImGui::BeginDisabled(!m_tonemapperData.isActive);
      PE::begin("##WhiteBalance", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      const float itemSpacing = 4.F;
      const float resetButtonWidth = ImGui::CalcTextSize(ICON_MS_RESET_WHITE_BALANCE).x + ImGui::GetStyle().FramePadding.x * 2.F;
      const float whiteBalanceSliderWidth = ImGui::GetContentRegionAvail().x - resetButtonWidth - itemSpacing;
      PE::entry(
          "Temperature",
          [&]() {
            ImGui::SetNextItemWidth(whiteBalanceSliderWidth);
            changed |= ImGui::SliderFloat("##Temperature", &m_tonemapperData.temperature, 2000.0F, 15000.0F, "%.0f K");
            ImGui::SameLine(0, itemSpacing);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if(ImGui::Button(ICON_MS_RESET_WHITE_BALANCE))
            {
              m_tonemapperData.temperature = shaderio::TonemapperData().temperature;
              changed                      = true;
            }
            return changed;
          },
          "Scene lighting temperature to correct for in degrees Kelvin "
          "(6506K = D65 neutral, higher values make the image more orange because they're correcting for cooler lighting)");

      PE::entry(
          "Tint",
          [&]() {
            ImGui::SetNextItemWidth(whiteBalanceSliderWidth);
            changed |= ImGui::SliderFloat("##Tint", &m_tonemapperData.tint, -.03F, .03F, "%.5f");
            ImGui::SameLine(0, itemSpacing);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if(ImGui::Button(ICON_MS_RESET_WHITE_BALANCE))
            {
              m_tonemapperData.tint = shaderio::TonemapperData().tint;
              changed               = true;
            }
            return changed;
          },
          "Green/magenta lighting tint to correct for in ANSI C78.377-2008 Duv units "
          "(-.03 = very green, 0 = blackbody, .00326 = D65 neutral, .03 = very magenta)");
      PE::end();
      ImGui::EndDisabled();
    }
    endCollapsibleGroup(open);
  }

  // --- Dithering ---
  {
    bool open = beginCollapsibleGroup("Dithering");
    if(open)
    {
      ImGui::BeginDisabled(!m_tonemapperData.isActive);
      PE::begin("##Dithering", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      changed |= PE::Checkbox("Enable", reinterpret_cast<bool*>(&m_tonemapperData.dither),
                              "Apply dithering to reduce color banding in smooth gradients");
      PE::end();
      ImGui::EndDisabled();
    }
    endCollapsibleGroup(open);
  }

  // --- Default Properties ---
  {
    bool open = beginCollapsibleGroup("Default Properties");
    if(open)
    {
      ImGui::BeginDisabled(!m_tonemapperData.isActive);
      if(ImGui::SmallButton("reset"))
      {
        m_tonemapperData = {};
        changed          = true;
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("Reset all tonemapper settings to default values");
      }
      ImGui::EndDisabled();
    }
    endCollapsibleGroup(open);
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

  bool node_open = ImGui::TreeNodeEx(ICON_MS_LIGHTBULB_2 " Lights", node_flags);
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

void GaussianSplattingUI::guiDrawSkyTree()
{
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  if(m_selectedAsset == GUI_SKY)
    flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::TreeNodeEx(ICON_MS_PARTLY_CLOUDY_DAY " Environment", flags);
  if(ImGui::IsItemClicked())
  {
    resetSelection();
    m_selectedAsset = GUI_SKY;
  }
}

void GaussianSplattingUI::guiDrawSkyProperties()
{
  namespace PE = nvgui::PropertyEditor;

  bool changed = false;

  // Common settings
  if(PE::begin("##env_common", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
  {
    int envMode = static_cast<int>(m_sky.mode());
    if(PE::Combo("Mode", &envMode, "None\0Sky\0HDR\0"))
    {
      m_sky.setMode(static_cast<shaderio::EnvironmentMode>(envMode));
      changed = true;
    }

    if(m_sky.mode() != shaderio::EnvironmentMode::eNone)
    {
      bool enabled = m_sky.isEnabled();
      if(PE::Checkbox("Lighting", &enabled))
      {
        m_sky.setEnabled(enabled);
        changed = true;
      }

      glm::ivec2 res         = m_sky.resolution();
      bool       resReadOnly = (m_sky.mode() == shaderio::EnvironmentMode::eHDR);
      if(resReadOnly)
        ImGui::BeginDisabled();
      if(PE::InputInt2("Resolution", &res[0]))
      {
        m_sky.setResolution(res);
        changed = true;
      }
      if(resReadOnly)
        ImGui::EndDisabled();
    }

    PE::end();
  }

  // Sky parameters (eSky mode only)
  if(m_sky.mode() == shaderio::EnvironmentMode::eSky)
  {
    auto& params = m_sky.skyParams();

    if(PE::begin("##sky_sun", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
    {
      if(PE::entry("", [&] { return ImGui::SmallButton("reset"); }, "Default values"))
      {
        params  = shaderio::SkyPhysicalParameters();
        changed = true;
      }
      changed |= nvgui::azimuthElevationSliders(params.sunDirection, false, params.yIsUp == 1);
      changed |= PE::SliderFloat("Sun Disk Scale", &params.sunDiskScale, 0.F, 10.F);
      changed |= PE::SliderFloat("Sun Disk Intensity", &params.sunDiskIntensity, 0.F, 5.F);
      changed |= PE::SliderFloat("Sun Glow Intensity", &params.sunGlowIntensity, 0.F, 5.F);
      PE::end();
    }

    if(PE::begin("##sky_extra", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
    {
      changed |= PE::SliderFloat("Haze", &params.haze, 0.F, 15.F);
      changed |= PE::SliderFloat("Red Blue Shift", &params.redblueshift, -1.F, 1.F);
      changed |= PE::SliderFloat("Saturation", &params.saturation, 0.F, 1.F);
      changed |= PE::SliderFloat("Horizon Height", &params.horizonHeight, -1.F, 1.F);
      changed |= PE::ColorEdit3("Ground Color", &params.groundColor.x, ImGuiColorEditFlags_Float);
      changed |= PE::SliderFloat("Horizon Blur", &params.horizonBlur, 0.F, 5.F);
      changed |= PE::ColorEdit3("Night Color", &params.nightColor.x, ImGuiColorEditFlags_Float);
      PE::end();
    }
  }

  // HDR IBL parameters (eHDR mode only)
  if(m_sky.mode() == shaderio::EnvironmentMode::eHDR)
  {
    if(PE::begin("##ibl_settings", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
    {
      PE::entry("File", [&] {
        ImGui::TextUnformatted(m_sky.iblFilePath().empty() ? "(none)" : m_sky.iblFilePath().filename().string().c_str());
        ImGui::SameLine();
        if(ImGui::SmallButton("Load..."))
        {
          std::filesystem::path filename =
              nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load HDR Environment", "HDR Image|*.hdr");
          if(!filename.empty())
          {
            VkCommandBuffer cmd = m_app->createTempCmdBuffer();
            m_sky.loadHdrEnvironment(cmd, filename);
            m_app->submitAndWaitTempCmdBuffer(cmd);
            changed = true;
          }
        }
        return false;
      });

      float intensity = m_sky.iblIntensity();
      if(PE::SliderFloat("Intensity", &intensity, 0.0f, 10.0f))
      {
        m_sky.setIblIntensity(intensity);
        changed = true;
      }

      glm::vec3 rotation = m_sky.iblRotation();
      if(PE::DragFloat3("Rotation", &rotation[0], 0.5f))
      {
        m_sky.setIblRotation(rotation);
        changed = true;
      }

      PE::end();
    }
  }

  if(changed)
  {
    m_sky.setDirty();
    resetFrameCounter();
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

      if(!instance->show)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 0.5f));

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)instance->index, instanceFlags,
                                         ICON_MS_SUBDIRECTORY_ARROW_RIGHT "%s", instance->displayName.c_str());

      if(!instance->show)
        ImGui::PopStyleColor();

      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        selectSplatSetInstance(instance);
      }

      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 110);
      if(ImGui::SmallButton(instance->show ? ICON_MS_VISIBILITY : ICON_MS_VISIBILITY_OFF))
      {
        instance->show = !instance->show;
        m_assets.splatSets.setVisibilityDirty();
        resetFrameCounter();
      }
      nvgui::tooltip(instance->show ? "Hide" : "Show");
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
    prmScene.meshToImportFilename =
        nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Mesh",
                                    "All Mesh Files|*.obj;*.gltf;*.glb|OBJ Files|*.obj|glTF Files|*.gltf;*.glb");
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

      if(!instance->show)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 0.5f));

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)instance.get(), node_flags,
                                         ICON_MS_SUBDIRECTORY_ARROW_RIGHT "%s", instance->name.c_str());

      if(!instance->show)
        ImGui::PopStyleColor();

      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        selectMeshInstance(instance);
      }
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 110);
      if(ImGui::SmallButton(instance->show ? ICON_MS_VISIBILITY : ICON_MS_VISIBILITY_OFF))
      {
        instance->show = !instance->show;
        m_assets.meshes.setVisibilityDirty();
        m_objListUpdated = true;
        resetFrameCounter();
      }
      nvgui::tooltip(instance->show ? "Hide" : "Show");
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
  if(!m_showPropertiesWindow)
    return;

  if(ImGui::Begin("Properties", &m_showPropertiesWindow))
  {
    switch(m_selectedAsset)
    {
      case GUI_RENDERER:
      case GUI_RASTERIZATION:
      case GUI_RAYTRACING:
      case GUI_DENOISING:
      case GUI_TONEMAPPING: {
        // Force-select a tab only when the tree selection actually changes
        static int prevSelectedAsset = GUI_NONE;
        bool       selectionChanged  = (m_selectedAsset != prevSelectedAsset);

        ImGuiTabItemFlags selectRenderer =
            (selectionChanged && m_selectedAsset == GUI_RENDERER) ? ImGuiTabItemFlags_SetSelected : 0;
        ImGuiTabItemFlags selectRaster =
            (selectionChanged && m_selectedAsset == GUI_RASTERIZATION) ? ImGuiTabItemFlags_SetSelected : 0;
        ImGuiTabItemFlags selectRaytrace =
            (selectionChanged && m_selectedAsset == GUI_RAYTRACING) ? ImGuiTabItemFlags_SetSelected : 0;
        ImGuiTabItemFlags selectDenoise =
            (selectionChanged && m_selectedAsset == GUI_DENOISING) ? ImGuiTabItemFlags_SetSelected : 0;
        ImGuiTabItemFlags selectTonemap =
            (selectionChanged && m_selectedAsset == GUI_TONEMAPPING) ? ImGuiTabItemFlags_SetSelected : 0;

        if(ImGui::BeginTabBar("##RendererTabs"))
        {
          if(ImGui::BeginTabItem(ICON_MS_CAMERA " Renderer", nullptr, selectRenderer))
          {
            if(!selectionChanged)
              m_selectedAsset = GUI_RENDERER;
            guiDrawRendererProperties();
            ImGui::EndTabItem();
          }

          ImGui::BeginDisabled(!isRasterPipelineActive());
          if(ImGui::BeginTabItem(ICON_MS_GRID_ON " Raster", nullptr, selectRaster))
          {
            if(!selectionChanged)
              m_selectedAsset = GUI_RASTERIZATION;
            guiDrawRasterizationProperties();
            ImGui::EndTabItem();
          }
          ImGui::EndDisabled();

          ImGui::BeginDisabled(!isRtxPipelineActive());
          if(ImGui::BeginTabItem(ICON_MS_CALL_MISSED_OUTGOING " Raytrace", nullptr, selectRaytrace))
          {
            if(!selectionChanged)
              m_selectedAsset = GUI_RAYTRACING;
            guiDrawRaytracingProperties();
            ImGui::EndTabItem();
          }
          ImGui::EndDisabled();

          ImGui::BeginDisabled(!isDlssSupportedPipeline());
          if(ImGui::BeginTabItem(ICON_MS_BLUR_ON " Denoise", nullptr, selectDenoise))
          {
            if(!selectionChanged)
              m_selectedAsset = GUI_DENOISING;
            guiDrawDenoisingProperties();
            ImGui::EndTabItem();
          }
          ImGui::EndDisabled();

          if(ImGui::BeginTabItem(ICON_MS_TONALITY " Tone map", nullptr, selectTonemap))
          {
            if(!selectionChanged)
              m_selectedAsset = GUI_TONEMAPPING;
            guiDrawTonemappingProperties();
            ImGui::EndTabItem();
          }
          ImGui::EndTabBar();
        }

        prevSelectedAsset = m_selectedAsset;
        break;
      }
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
          {
            bool open = beginCollapsibleGroup("Transform", true);
            if(open)
              guiDrawMeshTransformProperties();
            endCollapsibleGroup(open);
          }
          {
            bool open = beginCollapsibleGroup("Materials", true);
            if(open)
              guiDrawMeshMaterialProperties();
            endCollapsibleGroup(open);
          }
        }
        break;
      case GUI_CAMERA:
        m_selectedCameraPresetIndex = std::clamp<int64_t>(m_selectedCameraPresetIndex, -1, m_assets.cameras.size() - 1);
        //if(ImGui::CollapsingHeader("Camera Intrinsics", ImGuiTreeNodeFlags_DefaultOpen))
        {
          guiDrawCameraProperties();
        }
        break;
      case GUI_LIGHT:
        if(m_selectedLightInstance)
        {
          {
            bool open = beginCollapsibleGroup("Light", true);
            if(open)
              guiDrawLightProperties();
            endCollapsibleGroup(open);
          }
        }
        break;
      case GUI_SKY: {
        bool open = beginCollapsibleGroup("Sky", true);
        if(open)
          guiDrawSkyProperties();
        endCollapsibleGroup(open);
      }
      break;
      case GUI_SETTINGS:
        guiDrawSettingsProperties();
        break;
      default:
        // display nothing
        break;
    };
  }
  ImGui::End();
}

void GaussianSplattingUI::guiDrawSettingsProperties()
{
  {
    bool open = beginCollapsibleGroup("Navigation", true);
    if(open)
      guiDrawNavigationProperties();
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Transform Helpers", true);
    if(open)
    {
      namespace PE        = nvgui::PropertyEditor;
      bool  editingMode   = m_helpers.isEditingMode();
      bool  snapEnabled   = m_helpers.transform.isSnapEnabled();
      float snapTranslate = m_helpers.transform.getSnapTranslate();
      float snapRotate    = m_helpers.transform.getSnapRotate();
      float snapScale     = m_helpers.transform.getSnapScale();

      if(PE::begin("##TransformHelpers", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        if(PE::Checkbox("Show", &editingMode, "Show/hide transform helpers (E)"))
          m_helpers.setEditingMode(editingMode);
        if(PE::Checkbox("Snapping", &snapEnabled, "Enable grid snapping for transform operations"))
          m_helpers.transform.setSnapEnabled(snapEnabled);
        PE::end();
      }

      ImGui::BeginDisabled(!snapEnabled);
      if(PE::begin("## Snap Values", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        if(PE::DragFloat("Translate", &snapTranslate, 0.01f, 0.001f, 100.0f, "%.3f", 0, "Translation snap increment"))
          m_helpers.transform.setSnapValues(snapTranslate, snapRotate, snapScale);
        if(PE::DragFloat("Rotate", &snapRotate, 0.5f, 0.1f, 180.0f, "%.1f", 0, "Rotation snap increment (degrees)"))
          m_helpers.transform.setSnapValues(snapTranslate, snapRotate, snapScale);
        if(PE::DragFloat("Scale", &snapScale, 0.01f, 0.001f, 10.0f, "%.3f", 0, "Scale snap increment"))
          m_helpers.transform.setSnapValues(snapTranslate, snapRotate, snapScale);
        PE::end();
      }
      ImGui::EndDisabled();
    }
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Infinite Grid", true);
    if(open)
    {
      namespace PE     = nvgui::PropertyEditor;
      bool gridVisible = m_helpers.grid.isVisible();
      if(PE::begin("##InfiniteGrid", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        if(PE::Checkbox("Show", &gridVisible, "Show/hide infinite grid (G)"))
          m_helpers.grid.setVisible(gridVisible);
        PE::end();
      }
    }
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Light Proxies", true);
    if(open)
    {
      namespace PE = nvgui::PropertyEditor;
      if(PE::begin("##LightProxies", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        if(PE::Checkbox("Show", &m_showLightProxies, "Show/hide light proxy meshes (L)"))
          resetFrameCounter();
        PE::end();
      }
    }
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Summary Info Overlay", true);
    if(open)
    {
      namespace PE = nvgui::PropertyEditor;
      if(PE::begin("##SummaryOverlay", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        PE::Checkbox("Show", &m_showSummaryOverlay, "Show/hide summary info overlay in viewport");
        PE::end();
      }
    }
    endCollapsibleGroup(open);
  }
}

void GaussianSplattingUI::guiDrawRendererProperties()
{
  namespace PE = nvgui::PropertyEditor;

  // --- Global settings ---
  {
    bool open = beginCollapsibleGroup("Global Settings", true);
    if(open)
    {
      PE::begin("## Global settings ", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      bool vsync = m_app->isVsync();
      if(PE::Checkbox("V-Sync", &vsync))
        m_app->setVsync(vsync);

      if(PE::entry("Pipeline", [&]() { return m_ui.enumCombobox(GUI_PIPELINE, "##ID", &prmSelectedPipeline); }, "Selects the rendering method"))
      {
        m_requestUpdateShaders = true;
      }

      int colorFormatInt = static_cast<int>(prmRender.colorFormat);
      if(PE::entry(
             "Color Format", [&]() { return m_ui.enumCombobox(GUI_COLOR_FORMAT, "##ColorFormat", &colorFormatInt); },
             "Color buffer format.\n"
             "Higher precision improves temporal accumulation quality but uses more memory.\n"
             "R8G8B8A8 UNORM: 32-bit (lowest memory, fastest rendering)\n"
             "R16G16B16A16 SFLOAT: 64-bit (default, good balance)\n"
             "R32G32B32A32 SFLOAT: 128-bit (highest precision)"))
      {
        prmRender.colorFormat  = static_cast<VkFormat>(colorFormatInt);
        m_requestGBufferReinit = true;
        resetFrameCounter();
      }

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Lighting and temporal ---
  {
    bool open = beginCollapsibleGroup("Lighting and Temporal", true);
    if(open)
    {
      PE::begin("## Lighting and temporal", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
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

        if(prmComparison.enabled && m_imageCompare.hasValidCaptureImage() && prmRtx.temporalSampling)
        {
          m_imageCompare.setMetricsHistorySize(prmFrame.frameSampleMax);
        }
      }
      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Visualization ---
  {
    bool open = beginCollapsibleGroup("Visualization", true);
    if(open)
    {
      PE::begin("## Visualization", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      auto visuMenu = GUI_VISUALIZE;
      if(m_dlss.isEnabled())
        visuMenu = GUI_VISUALIZE_DLSS_ON;

      ImGui::BeginDisabled(!isRtxPipelineActive());

      static constexpr const char* visualizeTooltip =
          "Selects the visualization mode.\n"
          "Available only with ray tracing and hybrid pipelines.\n\n"
          "Final render: standard rendered output.\n"
          "Clay mode: renders all surfaces with a uniform clay color.\n"
          "Clock cycles: heat-map of GPU clock cycles per pixel.\n"
          "Ray Hit Count: heat-map of ray intersection tests per pixel.\n"
          "Depth (iso thres): depth from integrated iso-threshold.\n"
          "Depth (Closest hit): depth of the closest ray-particle hit.\n"
          "Depth (for DLSS): depth buffer as fed to the DLSS denoiser.\n"
          "Normal (Integrated): surface normal from integrated contributions.\n"
          "Normal (closest hit): surface normal at the closest ray-particle hit.\n"
          "Normal (For DLSS): normal buffer as fed to the DLSS denoiser.\n"
          "Splat ID (Harlequin): unique color per splat for identification.\n\n"
          "DLSS guide modes (enabled only when DLSS is active):\n"
          "DLSS Input: raw radiance input before denoising.\n"
          "DLSS Guide Albedo/Specular/Normal/Motion/Depth: individual G-buffers\n"
          "  used by the DLSS denoiser.";

      if(PE::entry("Visualize Mode", [&]() { return m_ui.enumCombobox(visuMenu, "##ID", &prmRender.visualize); }, visualizeTooltip))
      {
        m_requestUpdateShaders = true;
      }
      switch(prmRender.visualize)
      {
        case VISUALIZE_CLAY: {
          if(PE::ColorEdit3("Clay Color", glm::value_ptr(prmRender.clayColor)))
            resetFrameCounter();
          break;
        }
        case VISUALIZE_CLOCK: {
          bool changed = PE::DragFloat2("Min/max", glm::value_ptr(prmRender.clockVisuMinMax), 0.01f);
          changed |= PE::SliderFloat("Shift", &prmRender.clockVisuShift, -1.0f, 1.0f);
          if(changed)
            resetFrameCounter();
          break;
        }
        case VISUALIZE_DEPTH:
        case VISUALIZE_DEPTH_INTEGRATED:
        case VISUALIZE_DEPTH_FOR_DLSS: {
          bool changed = PE::DragFloat2("Min/max", glm::value_ptr(prmRender.depthVisuMinMax), 0.01f);
          changed |= PE::SliderFloat("Shift", &prmRender.depthVisuShift, -1.0f, 1.0f);
          if(changed)
            resetFrameCounter();
          break;
        }
        case VISUALIZE_RAYHITS: {
          bool changed = PE::DragFloat2("Min/max", glm::value_ptr(prmRender.hitsVisuMinMax), 1.0f);
          changed |= PE::SliderFloat("Shift", &prmRender.hitsVisuShift, -1.0f, 1.0f);
          if(changed)
            resetFrameCounter();
          break;
        }
      }
      ImGui::EndDisabled();

      {
        const bool wireframeHwSupported = isSupported.shaderFloat64 && isSupported.fragmentShaderBarycentric;
        // Mirrors the WIREFRAME macro coercion in updateSlangMacros(): wireframe is
        // unsupported with the stochastic trace strategies in the pure RTX pipeline.
        const bool wireframeStrategyOk =
            !(prmSelectedPipeline == PIPELINE_RTX && prmRtx.rtxTraceStrategy != RTX_TRACE_STRATEGY_FULL_ANYHIT);
        ImGui::BeginDisabled(!wireframeHwSupported || !wireframeStrategyOk);
        if(PE::Checkbox("Wireframe", &prmRender.wireframe,
                        !wireframeHwSupported ? "Wireframe requires VK_KHR_fragment_shader_barycentric and shaderFloat64, neither of which are supported by this device." :
                        !wireframeStrategyOk ? "Wireframe is not supported with the stochastic trace strategies in the RTX pipeline." :
                                               "Show particle bounds in wireframe."))
          m_requestUpdateShaders = true;
        ImGui::EndDisabled();
      }

      int alphaThres = int(255.0 * prmFrame.alphaCullThreshold);
      if(PE::SliderInt("Alpha culling threshold", &alphaThres, 0, 255, "%d", 0, "Discard splats with low opacity (with low contribution)."))
      {
        prmFrame.alphaCullThreshold = (float)alphaThres / 255.0f;
      }

      if(PE::SliderInt("Maximum SH degree", (int*)&prmFrame.shDegree, 0, 3, "%d", 0,
                       "Sets the highest degree of Spherical Harmonics (SH) used for view-dependent effects."))
      {
      }

      if(PE::Checkbox("Show SH deg > 0 only", &prmRender.showShOnly,
                      "Removes the base color from SH degree 0, applying only color deduced from \n"
                      "higher-degree SH to a neutral gray. This helps visualize their contribution."))
        m_requestUpdateShaders = true;

      if(PE::Checkbox("Disable opacity gaussian ", &prmRender.opacityGaussianDisabled,
                      "Disables the alpha component of the Gaussians, making their full range visible.\n"
                      "This helps analyze splat distribution and scales, especially when combined with Splat Scale adjustments."))
        m_requestUpdateShaders = true;

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Particle Filtering ---
  {
    bool open = beginCollapsibleGroup("Particle Filtering", true);
    if(open)
    {
      PE::begin("## Particle Filtering", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      {
        static const float kernelSizeValues[] = {0.0f, 0.1f, 0.2f, 0.3f};
        int                kernelIdx          = 0;
        for(int i = 0; i < IM_ARRAYSIZE(kernelSizeValues); i++)
        {
          if(prmRaster.covarianceDilation == kernelSizeValues[i])
            kernelIdx = i;
        }

        if(PE::entry(
               "Low pass Kernel Size",
               [&]() { return m_ui.enumCombobox(GUI_COVARIANCE_DILATION, "##AAKernelSize", &kernelIdx); },
               "2D covariance dilation, low-pass kernel size (0.0 = no filtering).\n"
               "Larger values produce more smoothing but may reduce sharpness.\n"
               "0.0: disabled\n"
               "0.1: MipSplatting default\n"
               "0.2\n"
               "0.3: 3DGS, 3DGUT, and StochasticSplat default"))
        {
          prmRaster.covarianceDilation = kernelSizeValues[kernelIdx];
          m_requestUpdateShaders       = true;
        }
      }

      ImGui::BeginDisabled(prmRaster.covarianceDilation == 0.0);
      if(PE::Checkbox("Mip splatting antialiasing", &prmRaster.msAntialiasing,
                      "Indicates if Gaussians were trained (and should be rendered) with mip-splatting antialiasing method.\n"
                      "Compensates the particle opacity with respect to the selected Low pass Kernel Size\n"
                      "Active only if covariance dilation is > 0.0"))
      {
        m_requestUpdateShaders = true;
      }
      ImGui::EndDisabled();

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Particle Normal Vectors ---
  {
    bool open = beginCollapsibleGroup("Particle Normal Vectors", true);
    if(open)
    {
      PE::begin("## Normal computation", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      if(PE::entry(
             "Normal vectors",
             [&]() { return m_ui.enumCombobox(GUI_NORMAL_METHOD, "##ID", (int*)&prmRender.normalMethod); },
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

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Default Settings ---
  {
    bool open = beginCollapsibleGroup("Default Settings", true);
    if(open)
    {
      PE::begin("##ResetSettings", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      if(PE::entry("Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
      {
        resetRenderSettings();
        m_requestUpdateShaders = true;
        m_assets.splatSets.markAllSplatSetsForRegeneration();
      }
      PE::end();
    }
    endCollapsibleGroup(open);
  }
}

void GaussianSplattingUI::guiDrawRasterizationProperties()
{
  bool rasterDisabled = !isRasterPipelineActive();
  ImGui::BeginDisabled(rasterDisabled);

  namespace PE = nvgui::PropertyEditor;

  // --- Shape ---
  {
    bool open = beginCollapsibleGroup("Shape", true);
    if(open)
    {
      PE::begin("## Shape", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      {
        bool is3dgut = (prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
        ImGui::BeginDisabled(!is3dgut);
        if(PE::entry(
               "Kernel degree",
               [&]() { return m_ui.enumCombobox(GUI_KERNEL_DEGREE, "##RasterKD", &prmRtx.kernelDegree); },
               "Degree of the kernel function used for Gaussian evaluation.\n"
               "Must match the degree used during model training/generation.\n"
               "Only available for 3DGUT and Hybrid 3DGUT pipelines.\n"
               "Changing this triggers a BLAS rebuild."))
        {
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          m_requestUpdateShaders = true;
        }
        ImGui::EndDisabled();
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

      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
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
    }
    endCollapsibleGroup(open);
  }

  // --- Sorting ---
  {
    bool open = beginCollapsibleGroup("Sorting", true);
    if(open)
    {
      PE::begin("## Sorting", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      guiDrawSortingSelector(false);

      const bool cpuSortingDisabled =
          (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX || prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
      ImGui::BeginDisabled(cpuSortingDisabled);
      PE::Checkbox("Lazy CPU sorting", &prmRaster.cpuLazySort, "Perform sorting only if viewpoint changes");
      PE::Text("CPU sorting state", m_assets.splatSets.getCpuSorterStatus() == SplatSorterAsync::E_SORTING ? "Sorting" : "Idled");
      nvgui::tooltip(
          "Current state of the CPU-based asynchronous sorting.\n"
          "'Sorting' = sort in progress, 'Idled' = sort complete or not needed.");
      ImGui::EndDisabled();

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Culling ---
  {
    bool open = beginCollapsibleGroup("Culling", true);
    if(open)
    {
      PE::begin("## Culling", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      if(PE::entry(
             "Frustum culling",
             [&]() { return m_ui.enumCombobox(GUI_FRUSTUM_CULLING, "##FrustumCulling", &prmRaster.frustumCulling); },
             "Defines where frustum culling is performed:\n"
             "- Disabled: no frustum culling (for performance comparisons).\n"
             "- At distance stage: culling in the distance compute shader\n"
             "  (only available with GPU radix sort or stochastic splat).\n"
             "- At raster stage: culling in the vertex or mesh shader."))
      {
        const bool usesDistShader =
            (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
        if(!usesDistShader && prmRaster.frustumCulling == FRUSTUM_CULLING_AT_DIST)
          prmRaster.frustumCulling = FRUSTUM_CULLING_AT_RASTER;
        m_requestUpdateShaders = true;
      }

      PE::SliderFloat("Frustum dilation", &prmFrame.frustumDilation, 0.0f, 1.0f, "%.1f", 0,
                      "Adjusts the frustum culling bounds to account for the fact that visibility is tested \n"
                      "only at the center of each splat, rather than its full elliptical shape. A positive \n"
                      "value expands the frustum by the given percentage, reducing the risk of prematurely \n"
                      "discarding splats near the frustum boundaries.");

      {
        const bool usesDistShader =
            (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
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

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Shading ---
  {
    bool open = beginCollapsibleGroup("Shading", true);
    if(open)
    {
      PE::begin("## Shading", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      ImGui::BeginDisabled(!prmRender.lightingEnabled);
      {
        if(PE::Checkbox("Quantize Normals", &prmRaster.quantizeNormals,
                        "Use octahedral encoding for normals (Meyer et al. 2010).\n"
                        "Reduces mesh-to-fragment bandwidth from 96 bits to 32 bits per normal."))
        {
          m_requestUpdateShaders = true;
        }

        const bool ftbDisabled = !prmRender.lightingEnabled || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
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

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Advanced ---
  {
    bool open = beginCollapsibleGroup("Advanced", true);
    if(open)
    {
      PE::begin("## Advanced", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      if(PE::entry(
             "Dist WG size",
             [&]() { return m_ui.enumCombobox(GUI_DIST_SHADER_WG_SIZE, "##ID", &prmRaster.distShaderWorkgroupSize); },
             "Workgroup size for the distance compute shader.\n"
             "Affects GPU occupancy and performance.\n"
             "Best value depends on the GPU architecture."))
      {
        m_requestUpdateShaders = true;
      }

      if(PE::entry(
             "Mesh WG size",
             [&]() { return m_ui.enumCombobox(GUI_MESH_SHADER_WG_SIZE, "##ID", &prmRaster.meshShaderWorkgroupSize); },
             "Workgroup size for the mesh shader.\n"
             "Affects GPU occupancy and performance.\n"
             "Best value depends on the GPU architecture."))
      {
        m_requestUpdateShaders = true;
      }

      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
      ImGui::BeginDisabled(!isSupported.fragmentShaderBarycentric);
      if(PE::Checkbox("Fragment shader barycentric", &prmRaster.fragmentBarycentric,
                      isSupported.fragmentShaderBarycentric ?
                          "Enables fragment shader barycentric to reduce vertex and mesh shaders outputs." :
                          "VK_KHR_fragment_shader_barycentric is not supported by this device."))
        m_requestUpdateShaders = true;
      ImGui::EndDisabled();
      ImGui::EndDisabled();

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  ImGui::EndDisabled();
}

void GaussianSplattingUI::guiDrawRaytracingProperties()
{
  bool raytraceDisabled = !isRtxPipelineActive();
  ImGui::BeginDisabled(raytraceDisabled);

  namespace PE = nvgui::PropertyEditor;

  bool updated = false;

  // --- Path Tracing ---
  {
    bool open = beginCollapsibleGroup("Path Tracing", true);
    if(open)
    {
      PE::begin("## Path Tracing", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);
      updated |= PE::SliderInt("Max bounces", &prmFrame.rtxMaxBounces, 0, 16, "%d", 0,
                               "Maximum number of light bounces for path tracing.\n"
                               "Higher values produce more realistic global illumination but are slower.\n"
                               "0 = direct lighting only.");
      updated |= PE::DragFloat("Secondary ray offset", &prmFrame.rtxSecondaryRayOffset, 0.0001f, 0.0f, 1.0f, "%.4f", 0,
                               "TMin offset for all secondary rays (bounce and mesh shadow).\n"
                               "Prevents self-intersection artifacts.\n"
                               "Primary rays use the camera near clip plane distance as TMin.\n"
                               "Particle shadow rays use the dedicated 'Particle shadow offset' instead.");
      ImGui::EndDisabled();

      updated |= PE::DragFloat("Firefly clamp", &prmRtx.fireflyClampThreshold, 0.1f, 0.0f, 100.0f, "%.1f", 0,
                               "Luminance threshold to suppress stochastic bright outliers (fireflies).\n"
                               "Radiance above this threshold is scaled down proportionally.\n"
                               "Reduces temporal streaks during fast camera movement, especially with DLSS.\n"
                               "Set to 0 to disable.");

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Particle Shape ---
  {
    bool open = beginCollapsibleGroup("Particle Shape", true);
    if(open)
    {
      PE::begin("## Particle Shape", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      if(PE::entry(
             "Kernel degree", [&]() { return m_ui.enumCombobox(GUI_KERNEL_DEGREE, "##ID", &prmRtx.kernelDegree); },
             "Degree of the kernel function used for Gaussian evaluation.\n"
             "Must match the degree used during model training/generation.\n"
             "Affects particle shape and rendering quality.\n"
             "Changing this triggers a BLAS rebuild."))
      {
        m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
        m_requestUpdateShaders = true;
        updated                = true;
      }

      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);

      int parametric = prmRtxData.useSpheres ? PARTICLE_FORMAT_SPHERE :
                       prmRtxData.useAABBs   ? PARTICLE_FORMAT_PARAMETRIC :
                                               PARTICLE_FORMAT_ICOSAHEDRON;

      if(PE::entry(
             "Particle format", [&]() { return m_ui.enumCombobox(GUI_PARTICLE_FORMAT, "##ID", &parametric); },
             "Particle primitive type for RT acceleration structures.\n"
             "Parametric and Sphere modes force the use of TLAS instances.\n"
             "Sphere mode requires VK_NV_ray_tracing_linear_swept_spheres \n"
             "(greyed out when the device does not expose this extension).\n"))
      {
        prmRtxData.useAABBs   = false;
        prmRtxData.useSpheres = false;
        if(parametric == PARTICLE_FORMAT_PARAMETRIC)
        {
          prmRtxData.useAABBs         = true;
          prmRtxData.useTlasInstances = true;
          prmRtx.particleDepth        = PARTICLE_DEPTH_BILLBOARD;
        }
        else if(parametric == PARTICLE_FORMAT_SPHERE)
        {
          prmRtxData.useSpheres       = true;
          prmRtxData.useTlasInstances = true;
        }
        m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
        m_requestUpdateShaders = true;
        updated                = true;
      }

      if(PE::Checkbox("Adaptive clamp", &prmRtx.kernelAdaptiveClamping,
                      "Adapt particle bounding volume based on its opacity (density).\n"
                      "Low-opacity particles get tighter bounds, improving ray tracing performance.\n"
                      "When disabled, all particles use the same bounding scale regardless of opacity.\n"
                      "Triggers a BLAS rebuild when toggled."))
      {
        m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
        m_requestUpdateShaders = true;
        updated                = true;
      }

      {
        const bool billboardAllowed = prmRtxData.useAABBs || prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT;
        ImGui::BeginDisabled(!billboardAllowed);
        if(PE::entry(
               "Particle depth",
               [&]() { return m_ui.enumCombobox(GUI_PARTICLE_DEPTH, "##ParticleDepth", &prmRtx.particleDepth); },
               "Method used to compute depth for particle hits:\n"
               "- Billboard (3DGS/3DGUT): depth is the ray-to-billboard-plane intersection.\n"
               "  Requires AABB geometry or Stochastic any-hit trace strategy.\n"
               "- Ellipsoid (3DGRT): depth depends on geometry mode.\n"
               "  - AABB (custom intersection): depth is the point of maximum density along the ray\n"
               "  (closest approach to the ellipsoid center in canonical space).\n"
               "  - Icosahedron/Sphere (hardware intersection): depth is the actual ellipsoid surface hit."))
        {
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          m_requestUpdateShaders = true;
          updated                = true;
        }
        ImGui::EndDisabled();
      }

      ImGui::BeginDisabled(prmRtx.particleDepth != PARTICLE_DEPTH_BILLBOARD
                           || (!prmRtxData.useAABBs && prmRtx.rtxTraceStrategy != RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT));
      {
        int bbMode = (int)prmRtxData.billboardBoundingMode;
        if(PE::entry(
               "Billboard bounding mode",
               [&]() { return m_ui.enumCombobox(GUI_BILLBOARD_BOUNDING_MODE, "##BillboardBoundingModeRT", &bbMode); },
               "How TLAS instance bounding boxes are scaled in billboard mode:\n"
               "- Fitted: per-axis scales (default, fastest traversal; may miss anisotropic billboards).\n"
               "- Uniform / fractions: clamp each axis toward max scale (trade speed vs. correctness).\n"
               "- Optimal: per-axis (max + s_i) / 2 — recommended for raster-matching quality."))
        {
          prmRtxData.billboardBoundingMode = (BillboardBoundingMode)bbMode;
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          updated = true;
        }
      }
      if(PE::Checkbox("Billboard frustum culling", &prmRtx.billboardFrustumCulling,
                      "Cull particles whose center is outside the camera frustum in the any-hit shader.\n"
                      "Reduces border artifacts from particles that are visible in ray tracing\n"
                      "but would be culled in rasterization."))
      {
        m_requestUpdateShaders = true;
        updated                = true;
      }
      ImGui::EndDisabled();

      ImGui::BeginDisabled(prmRtx.particleDepth != PARTICLE_DEPTH_BILLBOARD
                           || prmRtx.rtxTraceStrategy != RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT);
      if(PE::Checkbox("Shorten ray", &prmRtx.shortenRay,
                      "Use payload max distance to early-terminate ray traversal in stochastic any-hit.\n"
                      "When enabled, rays are shortened based on the farthest stored hit,\n"
                      "allowing the hardware to skip particles beyond that distance."))
      {
        m_requestUpdateShaders = true;
        updated                = true;
      }
      ImGui::EndDisabled();

      ImGui::EndDisabled();

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Particle Tracing ---
  {
    bool open = beginCollapsibleGroup("Particle Tracing", true);
    if(open)
    {
      PE::begin("## Particle Tracing", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      guiDrawTracingStrategySelector(false);

      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);
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

        if(PE::InputInt("Maximum pass count", &prmFrame.maxPasses, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue,
                        "Maximum number of ray marching passes per pixel.\n"
                        "Each pass processes up to 'Particle samples per pass' hits.\n"
                        "More passes allow rendering denser scenes at the cost of performance."))
        {
          prmFrame.maxPasses = std::clamp(prmFrame.maxPasses, 1, 1000);
          updated            = true;
        }

        {
          const int effectiveSpp = (prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT) ? 1 : prmRtx.particleSamplesPerPass;
          PE::Text("Maximum anyhit/pixel", std::to_string(effectiveSpp * prmFrame.maxPasses));
          nvgui::tooltip(
              "Total maximum any-hit invocations per pixel (samples per pass x pass count).\n"
              "Read-only computed value.");
        }
      }
      ImGui::EndDisabled();

      updated |= PE::InputFloat("Minimum transmittance", &prmFrame.minTransmittance, 0.0, 1.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                                "Transmittance threshold below which particle ray marching stops.");

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Shading ---
  {
    bool open = beginCollapsibleGroup("Shading", true);
    if(open)
    {
      PE::begin("## RT Shading", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      updated |= PE::DragFloat("Depth Iso Threshold", &prmRtx.depthIsoThresholdRTX, 0.01f, 0.0f, 1.0f, "%.2f", 0,
                               "Transmittance threshold for depth picking in ray tracing.\n"
                               "Depth is captured when transmittance drops below this value.\n"
                               "Lower values pick depth later (more accumulated opacity).");

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Shadows ---
  {
    bool open = beginCollapsibleGroup("Shadows", true);
    if(open)
    {
      PE::begin("## Shadows", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

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

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Ambient Occlusion ---
  {
    bool open = beginCollapsibleGroup("Ambient Occlusion", true);
    if(open)
    {
      PE::begin("## Ambient Occlusion", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      if(PE::Checkbox("Particle emissive AO", &prmRtx.particleEmissiveAoEnabled,
                      "Enable ambient occlusion for emissive splat sets.\n"
                      "Attenuates emissive radiance when nearby meshes occlude the hemisphere.\n"
                      "Traces against meshes only (not other splat sets)."))
      {
        m_requestUpdateShaders = true;
        updated                = true;
      }
      ImGui::BeginDisabled(!prmRtx.particleEmissiveAoEnabled);
      updated |= PE::DragFloat("Particle emissive AO radius", &prmRtx.particleEmissiveAoRadius, 0.01f, 0.001f, FLT_MAX, "%.3f", 0,
                               "Hemisphere sampling radius for emissive AO.\n"
                               "Controls how far AO rays are traced to find occluding meshes.");
      updated |= PE::DragFloat("Particle emissive AO strength", &prmRtx.particleEmissiveAoStrength, 0.01f, 0.0f, 5.0f, "%.2f", 0,
                               "Intensity of AO darkening for emissive splat sets.\n"
                               "0 = no darkening. 1 = full occlusion. >1 = exaggerated darkening.");
      ImGui::EndDisabled();

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Compositing ---
  {
    bool open = beginCollapsibleGroup("Compositing", true);
    if(open)
    {
      PE::begin("## Compositing", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      updated |= PE::DragFloat("Splat set composite threshold", &prmFrame.minSplatSetCompositeTransmittance, 0.01f,
                               0.0f, 1.0f, "%.2f", 0,
                               "Minimum transmittance required for meshes and environment to be composited behind splats.\n"
                               "Below this threshold, splats fully occlude everything behind them.");

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  // --- Advanced ---
  {
    bool open = beginCollapsibleGroup("Advanced", true);
    if(open)
    {
      PE::begin("## RT Advanced", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      updated |= PE::InputFloat("Alpha clamp", &prmFrame.alphaClamp, 0.0, 3.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue,
                                "Maximum alpha value per particle hit.\n"
                                "Clamps the opacity computed from the kernel response,\n"
                                "preventing any single splat from being fully opaque.\n"
                                "Default 0.99 (from the original 3DGS paper).\n"
                                "Avoid numerical instabilities (see paper appendix).\n"
                                "Not really needed in our visualization context.");

      if(PE::Checkbox("Quantize mesh payload", &prmRtx.quantizeMeshPayload,
                      "Pack mesh hit data using octahedral normal encoding and fp16 UVs/tangent.\n"
                      "Reduces payload from 15 to 9 float slots, lowering register pressure\n"
                      "and local memory spilling in the any-hit shader."))
      {
        m_requestUpdateShaders = true;
        updated                = true;
      }

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  if(updated)
    resetFrameCounter();

  ImGui::EndDisabled();
}

void GaussianSplattingUI::guiDrawCommonSplatSetProperties()
{
  namespace PE = nvgui::PropertyEditor;

  ImGui::Text("Changes impact all the splat sets.");

  {
    bool open = beginCollapsibleGroup("Splat Set Format in VRAM", true);
    if(open)
    {
      if(PE::begin("##VRAM format", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        if(PE::entry("Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
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
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("RTX Acceleration Structures", true);
    if(open)
    {
      if(PE::begin("##VRAM format RTX", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        if(PE::entry("Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
        {
          resetRtxDataParameters();
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
        }
        if(PE::Checkbox("Use AABBs", &prmRtxData.useAABBs,
                        "If on, uses AABBs for splats in BLAS instead of ICOSAHEDRON meshes."
                        "In this case the renderer will use the collision shader instead of "
                        "the ray/triangle intersection specialized hardware."))
        {
          if(prmRtxData.useAABBs)
            prmRtxData.useSpheres = false;
          m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          m_requestUpdateShaders = true;
        }

        // We do not allow useAABBs/useSpheres without instances (prevent bvh with very bad properties)
        if(prmRtxData.useAABBs || prmRtxData.useSpheres)
          prmRtxData.useTlasInstances = true;

        ImGui::BeginDisabled(prmRtxData.useAABBs || prmRtxData.useSpheres);
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

        {
          int bbMode = (int)prmRtxData.billboardBoundingMode;
          if(PE::entry(
                 "Billboard bounding mode",
                 [&]() { return m_ui.enumCombobox(GUI_BILLBOARD_BOUNDING_MODE, "##BillboardBoundingMode", &bbMode); },
                 "How TLAS instance bounding boxes are scaled in billboard mode:\n"
                 "- Fitted: uses per-axis particle scales (faster traversal, tighter bounds).\n"
                 "- Uniform: uses max(sx,sy,sz) homogeneous scale (correct billboard rendering)."))
          {
            prmRtxData.billboardBoundingMode = (BillboardBoundingMode)bbMode;
            m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
          }
        }

        PE::end();
      }
    }
    endCollapsibleGroup(open);
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

  {
    bool open = beginCollapsibleGroup(infoHeader.c_str());
    if(open)
    {
      PE::begin("##SplatSetInfo", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      PE::Text("Total Splats", std::to_string(splatSet->splatCount));
      PE::Text("SH Degree", std::to_string(splatSet->shDegree));

      if(PE::entry(
             "Path",
             [&]() {
               ImGui::InputText("##Path", const_cast<char*>(splatSet->path.c_str()), splatSet->path.length() + 1,
                                ImGuiInputTextFlags_ReadOnly);
               return false;
             },
             "Full path to the source .ply file"))
      {
      }

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Model Transform", true);
    if(open)
    {
      PE::begin("##Transform", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
      if(guiGetTransform(instance->scale, instance->rotation, instance->translation, instance->transform,
                         instance->transformInverse, false))
      {
        if(m_selectedSplatInstance)
          m_assets.splatSets.updateInstanceTransform(m_selectedSplatInstance);

        if(!isRtxPipelineActive())
        {
          m_deferredRtxRebuildPending = true;
        }
      }
      PE::end();
    }
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Material", true);
    if(open)
    {
      PE::begin("##SplatMaterial", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

      bool materialChanged = false;

      materialChanged |= PE::SliderInt("Max Bounces", &instance->splatMaterial.maxBounces, 0, 16);
      materialChanged |= PE::ColorEdit3("Base Color", glm::value_ptr(instance->splatMaterial.baseColor));
      materialChanged |= PE::SliderFloat("Metallic", &instance->splatMaterial.metallic, 0.0f, 1.0f);
      materialChanged |= PE::SliderFloat("Roughness", &instance->splatMaterial.roughness, 0.0f, 1.0f);
      materialChanged |= PE::ColorEdit3("Emissive", glm::value_ptr(instance->splatMaterial.emissive));
      materialChanged |=
          PE::DragFloat("Emissive Strength", &instance->splatMaterial.emissiveStrength, 0.1f, 0.0f, FLT_MAX, "%.1f");
      materialChanged |= PE::SliderFloat("IOR", &instance->splatMaterial.ior, 1.0f, 3.0f);
      materialChanged |= PE::SliderFloat("Transmission", &instance->splatMaterial.transmission, 0.0f, 1.0f);
      materialChanged |= PE::SliderFloat("Opacity", &instance->splatMaterial.opacity, 0.0f, 1.0f);

      if(materialChanged)
      {
        if(m_selectedSplatInstance)
          m_assets.splatSets.updateInstanceMaterial(m_selectedSplatInstance);
      }

      PE::end();
    }
    endCollapsibleGroup(open);
  }

  {
    bool open = beginCollapsibleGroup("Splat Set Storage in VRAM", true);
    if(open)
    {
      ImGui::Text("Changes impact all instances of this splat set.");

      if(PE::begin("##VRAM format", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
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
    endCollapsibleGroup(open);
  }
}

void GaussianSplattingUI::guiDrawMeshTransformProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(!m_selectedMeshInstance)
    return;  // No selection

  PE::begin("##Transform", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
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
    PE::begin("##Material", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);
    auto& material = materials[i];
    ImGui::PushID(i);
    PE::Text("Name", m_selectedMeshInstance->mesh->matNames[i]);
    needMaterialUpdate |= PE::SliderInt("Max Bounces", &material.maxBounces, 0, 16);
    needMaterialUpdate |= PE::ColorEdit3("Base Color", glm::value_ptr(material.baseColor));
    needMaterialUpdate |= PE::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
    needMaterialUpdate |= PE::SliderFloat("Roughness", &material.roughness, 0.0f, 1.0f);
    needMaterialUpdate |= PE::ColorEdit3("Emissive", glm::value_ptr(material.emissive));
    needMaterialUpdate |= PE::DragFloat("Emissive Strength", &material.emissiveStrength, 0.1f, 0.0f, FLT_MAX, "%.1f");
    needMaterialUpdate |= PE::SliderFloat("IOR", &material.ior, 1.0f, 3.0f);
    needMaterialUpdate |= PE::SliderFloat("Transmission", &material.transmission, 0.0f, 1.0f);
    needMaterialUpdate |= PE::SliderFloat("Opacity", &material.opacity, 0.0f, 1.0f);
    needMaterialUpdate |= PE::SliderFloat("Specular Factor", &material.specularFactor, 0.0f, 1.0f);
    needMaterialUpdate |= PE::ColorEdit3("Specular Color", glm::value_ptr(material.specularColorFactor));
    needMaterialUpdate |= PE::SliderFloat("Clearcoat Factor", &material.clearcoatFactor, 0.0f, 1.0f);
    needMaterialUpdate |= PE::SliderFloat("Clearcoat Roughness", &material.clearcoatRoughness, 0.0f, 1.0f);
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

  {
    bool open = beginCollapsibleGroup("Camera Intrinsics", true);
    if(open)
    {
      ImGui::BeginDisabled(m_selectedCameraPresetIndex != -1 || cameraManip->isAnimated());
      if(PE::begin("##CameraIntrinsics", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
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
    endCollapsibleGroup(open);
  }
  {
    bool open = beginCollapsibleGroup("Camera Extrinsics", true);
    if(open)
    {
      ImGui::BeginDisabled(m_selectedCameraPresetIndex != -1 || cameraManip->isAnimated());
      if(PE::begin("##CameraExtrinsics", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
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
    endCollapsibleGroup(open);
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
  if(PE::begin("##Navigation", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
  {
    auto mode     = cameraManip->getMode();
    float speed    = static_cast<float>(cameraManip->getSpeed());
    float duration = static_cast<float>(cameraManip->getAnimationDuration());

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

    changed |= PE::DragFloat("Speed", &speed, 0.01F, 0.0F, 1000.0F, "%.3f", 0, "Changing the default movement speed");
    changed |= PE::DragFloat("Transition", &duration, 0.01F, 0.0F, 10.0F, "%.3f", 0,
                             "Nb seconds to move to new position when loading a camera preset");
    duration = std::min(duration, 10.0f);
    cameraManip->setSpeed(speed);
    cameraManip->setAnimationDuration(duration);

    PE::end();
  }

  ImGui::EndDisabled();

  if(PE::begin("## Playback", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
  {
    bool wasPlaying = m_playPresets;
    PE::Checkbox("Play", &m_playPresets, "Cycle through camera presets");
    if(m_playPresets && !wasPlaying && m_assets.cameras.size() >= 2)
    {
      m_lastLoadedCamera                    = 0;
      m_requestUpdateShadersAfterCameraAnim = cameraPresetNeedsShaderRebuild(0);
      m_assets.cameras.loadPreset(0, false);
      m_selectedCameraPresetIndex = -1;
    }
    else if(!m_playPresets && wasPlaying && cameraManip->isAnimated())
    {
      // Stop mid-flight: snap to the current interpolated position (instantSet=true
      // cancels the animation and keeps the camera where it is right now)
      cameraManip->setCamera(cameraManip->getCamera(), true);
    }
    PE::Checkbox("Auto-play", &m_autoPlayPresets, "Automatically start playback when project is loaded");
    PE::end();
  }
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

  PE::begin("##Light", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame);

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

  {
    bool enabled = (asset->enabled != 0);
    if(PE::Checkbox("Enabled", &enabled))
    {
      asset->enabled  = enabled ? 1 : 0;
      needAssetUpdate = true;
    }
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

  needAssetUpdate |= PE::DragFloat("Radius", &asset->radius, 0.01f, 0.01f, 100.0f, "%.2f", 0,
                                   "Light source radius (soft shadows + proxy visualization)");
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
      const uint32_t totalSplatCount = m_assets.splatSets.getTotalGlobalSplatCount();

      if(PE::begin("## Scene", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        PE::Text("Splat sets", fmt::format("{} ({} instances)", splatSetCount, splatInstCount));
        PE::Text("Total particles", fmt::format("{} ({})", formatSize(totalSplatCount), totalSplatCount));
        PE::end();
      }
    }

    // ===== Rasterization =====
    {
      const uint32_t totalSplatCount = m_assets.splatSets.getTotalGlobalSplatCount();
      // GPU radix sort and stochastic splat both use the distance shader which populates the indirect buffer
      const bool usesDistShader =
          (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT);
      const uint32_t rasterSplatCount = usesDistShader ? m_indirectReadback.instanceCount : totalSplatCount;
      // The mesh shader is dispatched as a 2D grid (groupCountX is capped at the device
      // maxMeshWorkGroupCount[0]), so report the true total workgroup count derived from the
      // rasterized splat count rather than the X extent alone.
      const bool isMeshPipeline = (prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_MESH_3DGUT);
      const uint32_t meshRasterSplatCount = usesDistShader ? m_indirectReadback.instanceCount : prmFrame.splatCount;
      const uint32_t wgCount              = isMeshPipeline ? getMeshTaskWorkgroupCount(meshRasterSplatCount) : 0;
      // Too many particles to dispatch as mesh tasks (exceeds the device Y / total workgroup limits):
      // splats beyond the limit are dropped. Flag the affected stats in red.
      const bool   meshOverflow = isMeshPipeline && isMeshTaskDispatchOverflow(meshRasterSplatCount);
      const ImVec4 kRed(1.0f, 0.3f, 0.3f, 1.0f);

      ImGui::BeginDisabled(isRtxPipelineOnly());
      if(PE::begin("## Rasterization", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        if(meshOverflow)
          ImGui::PushStyleColor(ImGuiCol_Text, kRed);
        PE::Text("Rasterized splats", fmt::format("{} ({})", formatSize(rasterSplatCount), rasterSplatCount));
        PE::Text("Mesh shader work groups", fmt::format("{} ({})", formatSize(wgCount), wgCount));
        if(meshOverflow)
        {
          PE::Text("Mesh dispatch overflow", fmt::format("exceeds device limit ({} max work groups) - splats dropped",
                                                         isSupported.maxMeshWorkGroupTotalCount));
          ImGui::PopStyleColor();
        }
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
      if(PE::begin("## Ray Tracing", ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame))
      {
        // Mode name based on (instanced/per-splat-set) x (AABB/icosa)
        const char* modeName = prmRtxData.useTlasInstances ? (prmRtxData.useSpheres ? "Per-particle instanced sphere" :
                                                              prmRtxData.useAABBs   ? "Per-particle instanced AABB" :
                                                                                      "Per-particle instanced icosa") :
                                                             (prmRtxData.useSpheres ? "Per-splat-set sphere" :
                                                              prmRtxData.useAABBs   ? "Per-splat-set AABB" :
                                                                                      "Per-splat-set icosa soup");
        PE::Text("Mode", modeName);

        // TLAS info (append "multi-TLAS" if >1)
        if(multiTlas)
          PE::Text("TLAS count", fmt::format("{} (multi-TLAS)", tlasCount));
        else
          PE::Text("TLAS count", fmt::format("{}", tlasCount));

        PE::Text("TLAS entries", fmt::format("{}", formatSize(tlasEntries)));

        const uint32_t rtxDescCount  = splatSets.getRtxDescriptorCount();
        const size_t   instanceCount = splatSets.getInstanceCount();
        if(rtxDescCount > 0 && instanceCount > 0)
          PE::Text("RTX descriptors",
                   fmt::format("{} ({} inst x {} chunks)", rtxDescCount, instanceCount, rtxDescCount / instanceCount));

        // BLAS info (append "chunked" if using BLAS chunks)
        if(blasChunked)
          PE::Text("BLAS count", fmt::format("{} (chunked)", blasCount));
        else
          PE::Text("BLAS count", fmt::format("{}", blasCount));

        bool stochastic = prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT;
        bool hasMeshes  = !m_assets.meshes.instances.empty();
        bool singleSet  = getRtxParticleSetMode() == 1;  // RTX_HAS_PARTICLES == 1
        int  spp        = stochastic ? 1 : prmRtx.particleSamplesPerPass;
        int  distSize   = getPayloadArraySize();             // max(spp, meshSlots)
        int  idSize     = (hasMeshes && spp < 2) ? 2 : spp;  // mesh needs id[0]=objId, id[1]=matId
        int  setIdxSize = singleSet ? 0 : spp;               // single set: splatSetIdx[] removed from payload
        // id[idSize] + splatSetIdx[setIdxSize] + dist[distSize] + currentSplatSetInstance (multi-set) + rayBounce
        int totalDwords = idSize + setIdxSize + distSize + (singleSet ? 1 : 2);
        if(stochastic)
          totalDwords += 1;  // rngSeed (accepted-sample radiance/normal are re-evaluated in raygen)
        PE::Text("Payload array sizes", fmt::format("dist={}, id={}, splatSetIdx={}", distSize, idSize, setIdxSize));
        PE::Text("Payload total size", fmt::format("{} dwords ({} bytes)", totalDwords, totalDwords * 4));

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

void GaussianSplattingUI::guiAddToRecentMeshes(std::filesystem::path filePath, int historySize)
{
  if(filePath.is_relative())
  {
    filePath = std::filesystem::absolute(filePath);
  }
  auto it = std::find(m_recentMeshes.begin(), m_recentMeshes.end(), filePath);
  if(it != m_recentMeshes.end())
  {
    m_recentMeshes.erase(it);
  }
  m_recentMeshes.insert(m_recentMeshes.begin(), filePath);
  if(m_recentMeshes.size() > historySize)
  {
    m_recentMeshes.pop_back();
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
    else if(strcmp(handler->TypeName, "RecentMeshes") == 0)
    {
      ui->m_recentMeshes.clear();
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
        // UTF-8, native separators: the .ini is machine-local and never shared across systems
        buf->appendf("File=%s\n", nvutils::utf8FromPath(file).c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentFilesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentFiles.push_back(nvutils::pathFromUtf8(filePath));
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
    auto saveRecentMeshesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentMeshes)
      {
        // UTF-8, native separators: the .ini is machine-local and never shared across systems
        buf->appendf("File=%s\n", nvutils::utf8FromPath(file).c_str());
      }
      buf->append("\n");
    };

    auto loadRecentMeshesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentMeshes.push_back(nvutils::pathFromUtf8(filePath));
      }
    };

    ImGuiSettingsHandler iniHandler;
    iniHandler.TypeName   = "RecentMeshes";
    iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
    iniHandler.ReadOpenFn = readOpen;
    iniHandler.WriteAllFn = saveRecentMeshesToIni;
    iniHandler.ReadLineFn = loadRecentMeshesFromIni;
    iniHandler.UserData   = this;
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
  }
  {
    // Save settings handler, not using capture so can be used as a function pointer
    auto saveRecentProjectsToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      buf->appendf("[%s][Data]\n", handler->TypeName);
      for(const auto& file : self->m_recentProjects)
      {
        // UTF-8, native separators: the .ini is machine-local and never shared across systems
        buf->appendf("File=%s\n", nvutils::utf8FromPath(file).c_str());
      }
      buf->append("\n");
    };

    // Load settings handler, not using capture so can be used as a function pointer
    auto loadRecentProjectsFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      if(strncmp(line, "File=", 5) == 0)
      {
        const char* filePath = line + 5;
        self->m_recentProjects.push_back(nvutils::pathFromUtf8(filePath));
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
      buf->appendf("AssetsWindow=%d\n", self->m_showAssetsWindow ? 1 : 0);
      buf->appendf("PropertiesWindow=%d\n", self->m_showPropertiesWindow ? 1 : 0);
      buf->appendf("ShaderDebugging=%d\n", self->m_showShaderFeedback ? 1 : 0);
      buf->appendf("MemoryStatistics=%d\n", self->m_showMemoryStatistics ? 1 : 0);
      buf->appendf("RendererStatistics=%d\n", self->m_showRendererStatistics ? 1 : 0);
      buf->appendf("FooterBar=%d\n", self->m_showFooterBar ? 1 : 0);
      buf->append("\n");
    };

    // Load window visibility settings handler
    auto loadWindowStatesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      auto* self = static_cast<GaussianSplattingUI*>(handler->UserData);
      int   value;
#ifdef _MSC_VER
      if(sscanf_s(line, "AssetsWindow=%d", &value) == 1)
#else
      if(sscanf(line, "AssetsWindow=%d", &value) == 1)
#endif
      {
        self->m_showAssetsWindow = (value == 1);
      }
#ifdef _MSC_VER
      else if(sscanf_s(line, "PropertiesWindow=%d", &value) == 1)
#else
      else if(sscanf(line, "PropertiesWindow=%d", &value) == 1)
#endif
      {
        self->m_showPropertiesWindow = (value == 1);
      }
#ifdef _MSC_VER
      else if(sscanf_s(line, "ShaderDebugging=%d", &value) == 1)
#else
      else if(sscanf(line, "ShaderDebugging=%d", &value) == 1)
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
#ifdef _MSC_VER
      else if(sscanf_s(line, "FooterBar=%d", &value) == 1)
#else
      else if(sscanf(line, "FooterBar=%d", &value) == 1)
#endif
      {
        self->m_showFooterBar = (value == 1);
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
// Loading splt sets and meshes

void GaussianSplattingUI::guiLoadSceneAndDrawProgressIfNeeded(void)
{
#ifdef WITH_DEFAULT_SCENE_FEATURE
  // load a default scene if none was provided by command line
  if(prmScene.enableDefaultScene && m_currentLoadingSplatSetFilename.empty() && prmScene.sceneLoadQueue.empty()
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
      if(ImGui::Button("Cancel", ImVec2(120, 0)))
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
      m_currentLoadingSplatSetFilename = request.path;
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
          m_currentLoadingSplatSetFilename = "";
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
        auto loadedSplatSet =
            m_assets.splatSets.createSplatSet(m_currentLoadingSplatSetFilename.string(), m_plyLoader.getLoadedSplatSet());

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
          guiAddToRecentFiles(m_currentLoadingSplatSetFilename);

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

void GaussianSplattingUI::guiImportMeshIfNeeded()
{
  static std::filesystem::path pendingMeshImport;
  if(!prmScene.meshToImportFilename.empty())
  {
    pendingMeshImport             = prmScene.meshToImportFilename;
    prmScene.meshToImportFilename = "";
    ImGui::OpenPopup("Import mesh file?");
  }

  {
    bool doImport = false;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if(ImGui::BeginPopupModal("Import mesh file?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      ImGui::Text("Import mesh or reset project?");
      ImGui::Separator();

      if(ImGui::Button("Import", ImVec2(120, 0)))
      {
        doImport = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::SameLine();
      if(ImGui::Button("Reset", ImVec2(120, 0)))
      {
        reset();
        doImport = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if(ImGui::Button("Cancel", ImVec2(120, 0)))
      {
        pendingMeshImport.clear();
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doImport && !pendingMeshImport.empty())
    {
      const auto name = pendingMeshImport;
      pendingMeshImport.clear();

      auto meshPtr = m_assets.meshes.loadModel(name);
      bool valid   = (meshPtr != nullptr);
      if(valid)
        guiAddToRecentMeshes(name);

      if(!valid)
      {
        ImGui::OpenPopup("Obj Loading");
      }
      else
      {
        m_helpers.transform.clearAttachment();
        m_selectedAsset        = GUI_MESH;
        m_selectedMeshInstance = m_assets.meshes.m_lastCreatedInstance;
        m_objListUpdated       = true;
      }
    }
  }

  {
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if(ImGui::BeginPopupModal("Obj Loading", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      ImGui::Text("Error: invalid mesh file");
      if(ImGui::Button("Ok", ImVec2(120, 0)))
      {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
  }
}

///////////////////////////////////
// Loading and Saving Projects

namespace fs = std::filesystem;

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
          prmScene.projectToLoadFilename = "";
          prmScene.projectLoadPorcelain  = false;
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
        prmScene.projectToLoadFilename = "";
        prmScene.projectLoadPorcelain  = false;
        return false;
      }

      try
      {
        i >> data;
      }
      catch(...)
      {
        LOGE("Error: invalid project file %s\n", path.c_str());
        prmScene.projectToLoadFilename = "";
        prmScene.projectLoadPorcelain  = false;
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
    prmScene.projectToLoadFilename = "";
    prmScene.projectLoadPorcelain  = false;
    return false;
  }

  m_projectPath                  = std::filesystem::absolute(path);
  prmScene.projectToLoadFilename = "";
  prmScene.projectLoadPorcelain  = false;
  return true;
}

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

// Local .raw file writer: 16-byte header (width, height, channels, bytesPerChannel) + float32 RGBA data.
// Bypasses nvpro_core so we don't need to modify the submodule.
static VkResult saveRawImageToFile(VkDevice                     device,
                                   VkImage                      dstImage,
                                   VkDeviceMemory               dstImageMemory,
                                   VkExtent2D                   size,
                                   const std::filesystem::path& filename)
{
  VkImageSubresource  subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  VkSubresourceLayout subResourceLayout{};
  vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

  const char* data   = nullptr;
  VkResult    result = vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
  if(result != VK_SUCCESS)
    return result;
  data += subResourceLayout.offset;

  std::string filenameUtf8 = nvutils::utf8FromPath(filename);
  FILE*       fp           = fopen(filenameUtf8.c_str(), "wb");
  if(fp)
  {
    uint32_t header[4] = {size.width, size.height, 4, sizeof(float)};
    fwrite(header, sizeof(uint32_t), 4, fp);
    for(uint32_t y = 0; y < size.height; y++)
    {
      fwrite(data + y * subResourceLayout.rowPitch, sizeof(float), size.width * 4, fp);
    }
    fclose(fp);
  }

  vkUnmapMemory(device, dstImageMemory);
  return VK_SUCCESS;
}

}  // namespace vk_gaussian_splatting
