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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvutils/file_operations.hpp"

#include "nvgui/fonts.hpp"
#include "nvgui/tooltip.hpp"

#include <glm/vec2.hpp>
// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <chrono>
#include <thread>
#include <filesystem>
#include <algorithm>  // for std::clamp

#include "gaussian_splatting_ui.h"

namespace vk_gaussian_splatting {

GaussianSplattingUI::GaussianSplattingUI(nvutils::ProfilerManager*   profilerManager,
                                         nvutils::ParameterRegistry* parameterRegistry,
                                         bool*                       benchmarkEnabled)
    : GaussianSplatting(profilerManager, parameterRegistry)
    , m_pBenchmarkEnabled(benchmarkEnabled)
{

  // Register some very sepcific command line parameters, related to benchmarking, other parameters are registered in main or in registerCommandLineParameters

  parameterRegistry->add({"updateData", "Use only in benchmark script. 1=triggers an update of data buffers or textures after a parameter change."},
                         &m_requestUpdateSplatData);

  parameterRegistry->add({.name = "screenshot",
                          .help = "Use only in benchmark script. Takes a screenshot.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                if(m_app)
                                {
                                  m_app->screenShot(m_screenshotFilename);
                                }
                              }},
                         {".png"}, &m_screenshotFilename);
};

GaussianSplattingUI::~GaussianSplattingUI(){
    // Nothing to do here
};

void GaussianSplattingUI::onAttach(nvapp::Application* app)
{
  // we hide the UI dy default in benchmark mode
  m_showUI = !(*m_pBenchmarkEnabled);

  // Initializes the core

  GaussianSplatting::onAttach(app);

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
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_DEPTH, "Splats depth");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_RAYHITS, "Ray Hit Count");

  m_ui.enumAdd(GUI_SORTING, SORTING_GPU_SYNC_RADIX, "GPU radix sort");
  m_ui.enumAdd(GUI_SORTING, SORTING_CPU_ASYNC_MULTI, "CPU async std multi");

  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT32, "Float 32");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_FLOAT16, "Float 16");
  m_ui.enumAdd(GUI_SH_FORMAT, FORMAT_UINT8, "Uint8");

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

  m_ui.enumAdd(GUI_LIGHT_TYPE, LIGHT_TYPE_POINT, "Point");
  m_ui.enumAdd(GUI_LIGHT_TYPE, LIGHT_TYPE_DIRECTIONAL, "Directional");

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
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 8, "8");
  m_ui.enumAdd(GUI_RAY_HIT_PER_PASS, 4, "4");
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
  GaussianSplatting::onRender(cmd);
}

#define ICON_BLANK "     "

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
    if(ImGui::MenuItem(ICON_MS_FILE_OPEN " Open file", ""))
    {
      prmScene.sceneToLoadFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load ply file",
                                                                 "All Files|*.ply;*.spz|PLY Files|*.ply|SPZ files|*.spz");
    }
    if(ImGui::MenuItem(ICON_MS_RESTORE_PAGE " Re Open", "F5", false, m_loadedSceneFilename != ""))
    {
      prmScene.sceneToLoadFilename = m_loadedSceneFilename;
    }
    if(ImGui::BeginMenu(ICON_MS_HISTORY " Recent Files"))
    {
      for(const auto& file : m_recentFiles)
      {
        if(ImGui::MenuItem(file.string().c_str()))
        {
          prmScene.sceneToLoadFilename = file;
        }
      }
      ImGui::EndMenu();
    }
    ImGui::Separator();
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
        saveProject(path.string());
      }
    }
    ImGui::Separator();
    if(ImGui::MenuItem(ICON_MS_SCAN_DELETE " Close", ""))
    {
      deinitAll();
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
    ImGui::MenuItem(ICON_MS_SPACE_DASHBOARD " ShowUI", "", &m_showUI);
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

  // Shortcuts
  if(ImGui::IsKeyPressed(ImGuiKey_Space))
  {
    m_lastLoadedCamera = (m_lastLoadedCamera + 1) % m_cameraSet.size();
    m_cameraSet.loadPreset(m_lastLoadedCamera, false);
    m_requestUpdateShaders = true;
  }
  if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
  {
    close_app = true;
  }

  if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
  {
    v_sync = !v_sync;
  }
  if(ImGui::IsKeyPressed(ImGuiKey_F5))
  {
    if(!m_recentFiles.empty())
      prmScene.sceneToLoadFilename = m_recentFiles[0];
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
  if(ImGui::IsKeyPressed(ImGuiKey_1))
    prmSelectedPipeline = PIPELINE_VERT;
  if(ImGui::IsKeyPressed(ImGuiKey_2))
    prmSelectedPipeline = PIPELINE_MESH;
  if(ImGui::IsKeyPressed(ImGuiKey_3))
    prmSelectedPipeline = PIPELINE_RTX;
  if(ImGui::IsKeyPressed(ImGuiKey_4))  // TODO find why the shortcut does not work
    prmSelectedPipeline = PIPELINE_HYBRID;

  // hot rebuild of shaders only if scene exist
  if(ImGui::IsKeyPressed(ImGuiKey_R))
  {
    if(!m_loadedSceneFilename.empty())
      m_requestUpdateShaders = true;
    else
      std::cout << "No scene loaded, cannot rebuild shader" << std::endl;
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

  if(ImGui::IsKeyPressed(ImGuiKey_P))
    dumpSplat(m_indirectReadback.particleID);
}

void GaussianSplattingUI::onFileDrop(const std::filesystem::path& filename)
{
  // extension To lower case
  std::string extension = filename.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  //
  if(extension == ".ply")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".spz")
    prmScene.sceneToLoadFilename = filename;
  else if(extension == ".vkgs")
    prmScene.projectToLoadFilename = filename;
  else if(extension == ".obj")
    prmScene.meshToImportFilename = filename;
  else
    std::cout << "Error: unsupported file extension " << extension << std::endl;
}

void GaussianSplattingUI::onUIRender()
{
  /////////////
  // Rendering Viewport display the GBuffer
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Display the G-Buffer image
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());

    ImVec2 wp = ImGui::GetWindowPos();
    ImVec2 ws = ImGui::GetWindowSize();

    // display the basis widget at bottom left
    float  size   = 25.F;
    ImVec2 offset = ImVec2(size * 1.1F, -size * 1.1F) * ImGui::GetWindowDpiScale();
    ImVec2 pos    = ImVec2(wp.x, wp.y + ws.y) + offset;
    nvgui::Axis(pos, cameraManip->getViewMatrix(), size);

    // store mouse cursor
    // will be available for next frame in frameInfo
    ImVec2 mp = ImGui::GetMousePos();  // Mouse position in screen space

    // Convert to viewport space (0,0 at bottom-left)
    ImVec2 mouseInViewport = ImVec2(mp.x - wp.x, mp.y - wp.y);

    if(mouseInViewport.x < 0 || mouseInViewport.y < 0 || mouseInViewport.x >= ws.x || mouseInViewport.y >= ws.y)
      prmFrame.cursor.x = prmFrame.cursor.y = -1;  // just so it is easy to test in shader if pos is valid
    else
      prmFrame.cursor = {mouseInViewport.x, mouseInViewport.y};

    ImGui::End();
    ImGui::PopStyleVar();
  }

  /////////////////
  // Handle project loading, may trigger a scene loading
  loadProjectIfNeeded();

  /////////////////
  // Handle scene loading

#ifdef WITH_DEFAULT_SCENE_FEATURE
  // load a default scene if none was provided by command line
  if(prmScene.enableDefaultScene && m_loadedSceneFilename.empty() && prmScene.sceneToLoadFilename.empty()
     && m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
  {
    const std::vector<std::filesystem::path> defaultSearchPaths = getResourcesDirs();
    prmScene.sceneToLoadFilename = nvutils::findFile("flowers_1/flowers_1.ply", defaultSearchPaths).string();
    prmScene.enableDefaultScene  = false;
  }
#endif

  // do we need to load a new scene ?
  if(!prmScene.sceneToLoadFilename.empty() && m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
  {

    if(!m_loadedSceneFilename.empty() && prmScene.projectToLoadFilename.empty())
      ImGui::OpenPopup("Load .ply file ?");

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    bool doReset = true;

    if(ImGui::BeginPopupModal("Load .ply file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      doReset = false;

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
        prmScene.sceneToLoadFilename   = "";
        prmScene.projectToLoadFilename = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doReset)
    {
      // reset if a scene already exists
      const auto splatCount = m_splatSet.positions.size() / 3;
      if(splatCount)
      {
        deinitAll();
      }

      m_loadedSceneFilename = prmScene.sceneToLoadFilename;
      //
      vkDeviceWaitIdle(m_device);

      std::cout << "Start loading file " << prmScene.sceneToLoadFilename << std::endl;
      if(!m_plyLoader.loadScene(prmScene.sceneToLoadFilename, m_splatSet))
      {
        // this should never occur since status is READY.
        std::cout << "Error: cannot start scene load while loader is not ready status=" << m_plyLoader.getStatus() << std::endl;
      }
      else
      {
        // open the modal window that will collect results
        ImGui::OpenPopup("Loading");
      }

      // reset request
      prmScene.sceneToLoadFilename.clear();
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
        // TODO add error modal or better continue on error since it is false only if shaders does not compile
        // Then print shader compilation error directly as a viewport overlay
        // Will allow for fix and hot reload
        if(!initAll())
        {
          // destroy scene
          deinitScene();
        }
        else
        {
          guiAddToRecentFiles(m_loadedSceneFilename);
        }
        // set ready for next load
        m_plyLoader.reset();
        ImGui::CloseCurrentPopup();
      }
      break;
      default: {
        // nothing to do for READY or SHUTDOWN
      }
    }
    ImGui::EndPopup();
  }

  if(!m_showUI)
    return;

  /////////////////
  // Draw the UI parts

  guiDrawAssetsWindow();

  guiDrawPropertiesWindow();

  guiDrawRendererStatisticsWindow();

  guiDrawMemoryStatisticsWindow();

  guiDrawFooterBar();
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
    m_selectedAsset     = GUI_RENDERER;
    m_selectedItemIndex = -1;
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

  if(m_selectedAsset == GUI_CAMERA && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  bool node_open = ImGui::TreeNodeEx(ICON_MS_PHOTO_CAMERA " Camera", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_CAMERA;
    m_selectedItemIndex = -1;
  }
  ImGui::PushID(-1);
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
  if(ImGui::SmallButton(ICON_MS_ADD_A_PHOTO))
  {
    m_cameraSet.storeCurrentCamera();
  }
  nvgui::tooltip("Store current camera settings in presets");
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
  if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
  {
    auto name = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Import INRIA Camera file", "INRIA Camera file|*.json");
    if(!name.empty())
    {
      importCamerasINRIA(name.string(), m_cameraSet);
    }
  }
  nvgui::tooltip("Import INRIA Camera file");
  ImGui::PopID();

  if(node_open)
  {
    // display the camera tree
    for(int i = 0; i < m_cameraSet.size(); ++i)
    {
      ImGui::PushID(i);
      node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_CAMERA && m_selectedItemIndex == i)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      const auto name = i == 0 ? fmt::format(ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Default Preset ", i) :
                                 fmt::format(ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Camera Preset ({})", i);

      bool node_open = ImGui::TreeNodeEx(name.c_str(), node_flags);
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_CAMERA;
        m_selectedItemIndex = i;
      }
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 110);
      if(ImGui::SmallButton(ICON_MS_LOCAL_SEE))
      {
        if(m_cameraSet.getPreset(i).model != m_cameraSet.getCamera().model)
        {
          m_requestUpdateShaders = true;
        }
        m_cameraSet.loadPreset(i, false);
        m_lastLoadedCamera     = i;
        m_selectedItemIndex    = -1;  // Will select current camera
        m_requestUpdateShaders = true;
      }
      nvgui::tooltip("Load camera preset");
      if(i > 0)
      {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
        if(ImGui::SmallButton(ICON_MS_ADD_A_PHOTO))
        {
          m_cameraSet.setPreset(i, m_cameraSet.getCamera());
          m_lastLoadedCamera     = i;
          m_selectedItemIndex    = -1;  // Will select current camera
          m_requestUpdateShaders = true;
        }
        nvgui::tooltip("Overwrite preset with current camera settings");
      }
      // Delete button only if not default
      if(i != 0)
      {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
        if(ImGui::SmallButton(ICON_MS_DELETE))
        {
          m_cameraSet.erasePreset(i);
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
  if(m_selectedAsset == GUI_LIGHT && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  bool node_open = ImGui::TreeNodeEx(ICON_MS_LIGHT_MODE " Lights", node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_NONE;
    m_selectedItemIndex = -1;
  }
  ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 70);
  if(ImGui::SmallButton(ICON_MS_ADD_CIRCLE))
  {
    m_selectedItemIndex         = m_lightSet.createLight();
    m_requestUpdateLightsBuffer = true;
  }
  nvgui::tooltip("Create light");

  if(node_open)
  {
    // display the lights tree
    for(int i = 0; i < m_lightSet.size(); ++i)
    {
      ImGui::PushID(i);
      ImGuiTreeNodeFlags node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_LIGHT && m_selectedItemIndex == i)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Light %d", i);
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_LIGHT;
        m_selectedItemIndex = i;
      }
      if(m_lightSet.size() > 1)
      {
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
        if(ImGui::SmallButton(ICON_MS_DELETE))
        {
          m_lightSet.eraseLight(i);
          m_requestUpdateLightsBuffer = true;
          // deselect all
          m_selectedAsset     = GUI_NONE;
          m_selectedItemIndex = -1;
        }
        nvgui::tooltip("Delete light");
      }
      ImGui::PopID();
    }
    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawRadianceFieldsTree()
{

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  if(m_selectedAsset == GUI_SPLATSET && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  ImGui::SetNextItemOpen(true, ImGuiCond_Once);
  std::string rtxError = " ";
  if(m_splatSet.size() != 0 && !m_splatSetVk.rtxValid)
  {
    rtxError = " Error: RTX allocation failed";
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
  }
  bool node_open = ImGui::TreeNodeEx(
      fmt::format(ICON_MS_GRAIN " Radiance Fields ({}){}", m_loadedSceneFilename.empty() ? 0 : 1, rtxError).c_str(), node_flags);
  if(m_splatSet.size() != 0 && !m_splatSetVk.rtxValid)
    ImGui::PopStyleColor();
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_NONE;
    m_selectedItemIndex = -1;
  }
  if(node_open)
  {
    // display the radiance fields tree
    for(int i = 0; i < 1; ++i)
    {
      ImGuiTreeNodeFlags node_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_SPLATSET && m_selectedItemIndex != -1)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Splat set %d - %s",
                                         i, m_loadedSceneFilename.filename().string().c_str());
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_SPLATSET;
        m_selectedItemIndex = i;
      }
    }

    ImGui::TreePop();
  }
}

void GaussianSplattingUI::guiDrawObjectTree()
{

  namespace PE = nvgui::PropertyEditor;

  const ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  ImGuiTreeNodeFlags node_flags = base_flags;

  if(m_selectedAsset == GUI_MESH && m_selectedItemIndex == -1)
    node_flags |= ImGuiTreeNodeFlags_Selected;

  if(m_objListUpdated)
  {
    ImGui::SetNextItemOpen(true);
    m_objListUpdated = false;
  }
  bool node_open =
      ImGui::TreeNodeEx(fmt::format(ICON_MS_DEPLOYED_CODE " Mesh Models ({})", m_meshSetVk.instances.size()).c_str(), node_flags);
  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    m_selectedAsset     = GUI_NONE;
    m_selectedItemIndex = -1;
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
      valid = m_meshSetVk.loadModel(name);
    }

    if(!valid)
    {
      ImGui::OpenPopup("Obj Loading");
    }
    else
    {
      //
      m_requestUpdateMeshData = true;
      m_requestUpdateShaders  = true;
      //
      m_selectedAsset     = GUI_MESH;
      m_selectedItemIndex = m_meshSetVk.instances.size() - 1;
      //
      m_objListUpdated = true;  // so that next loop will force the Object open if selected
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
    // display the objects tree
    for(int i = 0; i < m_meshSetVk.instances.size(); ++i)
    {
      ImGui::PushID(i);
      int                instanceIndex = i;
      int                objectIndex   = m_meshSetVk.instances[instanceIndex].objIndex;
      ImGuiTreeNodeFlags node_flags    = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
      if(m_selectedAsset == GUI_MESH && m_selectedItemIndex == instanceIndex)
        node_flags |= ImGuiTreeNodeFlags_Selected;

      bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, ICON_MS_SUBDIRECTORY_ARROW_RIGHT "Model %d - %s",
                                         i, m_meshSetVk.meshes[objectIndex].name.c_str());
      if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
      {
        m_selectedAsset     = GUI_MESH;
        m_selectedItemIndex = instanceIndex;
      }
      ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 30);
      if(ImGui::SmallButton(ICON_MS_DELETE))
      {
        m_requestDeleteSelectedMesh = true;
        m_selectedItemIndex         = instanceIndex;
        m_objListUpdated            = true;  // so that next loop will force the Object node open if selected
      }
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
        guiDrawSplatSetProperties();
        break;
      case GUI_MESH:
        m_selectedItemIndex = std::clamp<int64_t>(m_selectedItemIndex, -1, m_meshSetVk.instances.size() - 1);
        if(m_selectedItemIndex >= 0)
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
        m_selectedItemIndex = std::clamp<int64_t>(m_selectedItemIndex, -1, m_cameraSet.size() - 1);
        //if(ImGui::CollapsingHeader("Camera Intrinsics", ImGuiTreeNodeFlags_DefaultOpen))
        {
          guiDrawCameraProperties();
        }
        if(m_selectedItemIndex == -1)
        {
          if(ImGui::CollapsingHeader("Navigation", ImGuiTreeNodeFlags_DefaultOpen))
          {
            guiDrawNavigationProperties();
          }
        }
        break;
      case GUI_LIGHT:
        m_selectedItemIndex = std::clamp<int64_t>(m_selectedItemIndex, -1, m_lightSet.size() - 1);
        if(m_selectedItemIndex >= 0)
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
    m_requestUpdateShaders   = true;
    m_requestUpdateSplatData = true;
  }

  ImGui::BeginDisabled(prmSelectedPipeline != PIPELINE_RTX);
  if(PE::entry(
         "Visualize", [&]() { return m_ui.enumCombobox(GUI_VISUALIZE, "##ID", &prmRender.visualize); }, "Selects the visualization mode"))
  {
    m_requestUpdateShaders = true;
  }
  ImGui::BeginDisabled(prmRender.visualize == 0);
  if(PE::DragFloat("Multiplier", (float*)&prmFrame.multiplier, 1.0F, 0.0F, 1000.0F))
    resetFrameCounter();
  ImGui::EndDisabled();
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

  const int maxModelShDegree = m_splatSet.maxShDegree();
  prmRender.maxShDegree      = std::min(prmRender.maxShDegree, maxModelShDegree);

  if(PE::SliderInt("Maximum SH degree", (int*)&prmRender.maxShDegree, 0, maxModelShDegree, "%d", 0,
                   "Sets the highest degree of Spherical Harmonics (SH) used for view-dependent effects."))
    m_requestUpdateShaders = true;

  if(PE::Checkbox("Show SH deg > 0 only", &prmRender.showShOnly,
                  "Removes the base color from SH degree 0, applying only color deduced from \n"
                  "higher-degree SH to a neutral gray. This helps visualize their contribution."))
    m_requestUpdateShaders = true;

  if(PE::Checkbox("Disable opacity gaussian ", &prmRender.opacityGaussianDisabled,
                  "Disables the alpha component of the Gaussians, making their full range visible.\n"
                  "This helps analyze splat distribution and scales, especially when combined with Splat Scale adjustments."))
    m_requestUpdateShaders = true;

  PE::end();

  ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
  if(ImGui::BeginTabBar("##SpecificsBar", tab_bar_flags))
  {
    if(prmSelectedPipeline != PIPELINE_RTX)
    {
      if(ImGui::BeginTabItem("Rasterization specifics"))
      {
        PE::begin("## Raster settings");

        if(PE::entry("Sorting method", [&]() { return m_ui.enumCombobox(GUI_SORTING, "##ID", &prmRaster.sortingMethod); }))
        {
          if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX && prmRaster.frustumCulling == FRUSTUM_CULLING_AT_DIST)
          {
            prmRaster.frustumCulling = FRUSTUM_CULLING_AT_RASTER;
            m_requestUpdateShaders   = true;
          }
          if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX && prmRaster.frustumCulling != FRUSTUM_CULLING_AT_DIST)
          {
            prmRaster.frustumCulling = FRUSTUM_CULLING_AT_DIST;
            m_requestUpdateShaders   = true;
          }
        }

        ImGui::BeginDisabled(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX);
        PE::Checkbox("Lazy CPU sorting", &prmRaster.cpuLazySort, "Perform sorting only if viewpoint changes");

        PE::Text("CPU sorting state", m_cpuSorter.getStatus() == SplatSorterAsync::E_SORTING ? "Sorting" : "Idled");
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

              ImGui::BeginDisabled(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX);
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

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);

        if(PE::Checkbox("Fragment shader barycentric", &prmRaster.fragmentBarycentric,
                        "Enables fragment shader barycentric to reduce vertex and mesh shaders outputs."))
          m_requestUpdateShaders = true;

        // we set a different size range for point and splat rendering
        PE::SliderFloat("Splat scale", (float*)&prmFrame.splatScale, 0.1f, prmRaster.pointCloudModeEnabled != 0 ? 10.0f : 2.0f,
                        "%.3f", 0, "Adjusts the size of the splats for visualization purposes.");

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
      if(ImGui::BeginTabItem("Ray tracing and 3DGUT specifics"))
      {
        PE::begin("## Raytrace sampling and bounces");

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_MESH_3DGUT);
        PE::SliderInt("Max bounces", &prmFrame.rtxMaxBounces, 1, 16);
        ImGui::EndDisabled();

        ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_HYBRID);
        {
          if(PE::entry(
                 "Temporal sampling",
                 [&]() { return m_ui.enumCombobox(GUI_TEMPORAL_SAMPLING, "##ID", &prmRtx.temporalSamplingMode); },
                 "Enable accumulation of frame results over time.\n"
                 "Automatic will activate sampling depending on other effects such as DoF.\n"
                 "If enabled, the specified number of temporal samples will be accumulated over \"Temporal samples count\" frames,\n"
                 "and the last accumulated frame will be presented without additional rendering.\n"
                 "Note that rendering converges faster if v-sync is off.\n"
                 "If disabled, the system renders in free run mode."))
          {
            resetFrameCounter();
            m_requestUpdateShaders = true;
          }

          if(PE::InputInt("Temporal samples count", &prmFrame.frameSampleMax, 1, 100, 0,
                          "Number of frames after which temporal sampling is stopped. \n"
                          "A value of 0 disables temporal sampling."))
          {
            prmFrame.frameSampleMax = std::clamp(prmFrame.frameSampleMax, 1, 1000);
            resetFrameCounter();
          }
        }
        ImGui::EndDisabled();

        PE::end();

        PE::begin("## Raytrace gaussians settings");

        if(PE::entry("Kernel degree",
                     [&]() { return m_ui.enumCombobox(GUI_KERNEL_DEGREE, "##ID", &prmRtx.kernelDegree); }))
          m_requestUpdateSplatData = true;

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
          m_requestUpdateSplatData = true;
        }

        if(PE::Checkbox("Adaptive clamp", &prmRtx.kernelAdaptiveClamping))
          m_requestUpdateSplatData = true;

        PE::InputFloat("Alpha clamp", &prmFrame.alphaClamp, 0.0, 3.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);

        PE::InputFloat("Minimum transmittance", &prmFrame.minTransmittance, 0.0, 1.0, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);

        if(PE::entry(
               "Ray hits per pass",
               [&]() { return m_ui.enumCombobox(GUI_RAY_HIT_PER_PASS, "##ID", &prmRtx.payloadArraySize); },
               "Max number of ray hits stored per pass (i.e. payload array size)"))
        {
          m_requestUpdateShaders = true;
        }

        if(PE::InputInt("Maximum pass count", &prmFrame.maxPasses))
        {
          prmFrame.maxPasses = std::clamp(prmFrame.maxPasses, 1, 1000);
        }

        PE::Text("Maximum anyhit/pixel", std::to_string(prmRtx.payloadArraySize * prmFrame.maxPasses));

        ImGui::EndDisabled();

        PE::end();

        ImGui::EndTabItem();
      }
    }
  }
  ImGui::EndTabBar();
}

void GaussianSplattingUI::guiDrawSplatSetProperties()
{
  namespace PE = nvgui::PropertyEditor;

  if(ImGui::CollapsingHeader("Model Transform", ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Transform");
    if(guiGetTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation, m_splatSetVk.transform,
                       m_splatSetVk.transformInverse, false))
    {
      // delay update of Acceleration Structures if not using ray tracing
      m_requestDelayedUpdateSplatAs = true;
    }
    PE::end();
  }
  if(ImGui::CollapsingHeader("Splat Set Format in VRAM", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if(PE::begin("##VRAM format"))
    {
      if(PE::entry(
             "Default settings", [&] { return ImGui::Button("Reset"); }, "resets to default settings"))
      {
        resetDataParameters();
        m_requestUpdateSplatData = true;
      }
      if(PE::entry(
             "Storage", [&] { return m_ui.enumCombobox(GUI_STORAGE, "##ID", &prmData.dataStorage); },
             "Selects between Data Buffers and Textures for storing model attributes, including:\n"
             "Position, Color and Opacity, Covariance Matrix\n"
             "and Spherical Harmonics (SH) Coefficients (for degrees higher than 0)"))
      {
        m_requestUpdateSplatData = true;
      }
      ImGui::BeginDisabled(m_splatSet.maxShDegree() == 0);
      if(PE::entry(
             "SH format", [&]() { return m_ui.enumCombobox(GUI_SH_FORMAT, "##ID", &prmData.shFormat); },
             "Selects storage format for SH coefficient, balancing precision and memory usage"))
      {
        m_requestUpdateSplatData = true;
      }
      ImGui::EndDisabled();
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
        m_requestUpdateSplatAs = true;
      }
      if(PE::Checkbox("Use AABBs", &prmRtxData.useAABBs,
                      "If on, uses AABBs for splats in BLAS instead of ICOSAHEDRON meshes."
                      "In this case the renderer will use the collision shader instead of "
                      "the ray/triangle intersection specialized hardware."))
        m_requestUpdateSplatAs = true;

      // We do not allow useAABBs without instances (prevent bvh with very bad properties leading to very low frame rate and device lost error)
      if(prmRtxData.useAABBs)
        prmRtxData.useTlasInstances = true;

      ImGui::BeginDisabled(prmRtxData.useAABBs);
      if(PE::Checkbox("Use TLAS instances", &prmRtxData.useTlasInstances,
                      "If on, uses one TLAS entry per splat and a small BLAS "
                      "with a unit Icosahedron. \nOtherwise use a single TLAS "
                      "entry and a huge BLAS containing all the transformed Icosahedrons."))
        m_requestUpdateSplatAs = true;
      ImGui::EndDisabled();

      if(PE::Checkbox("BLAS Compaction", &prmRtxData.compressBlas, "Bottom Level Acceleration structure compression."))
        m_requestUpdateSplatAs = true;

      if(m_splatSet.size() != 0 && !m_splatSetVk.rtxValid)
      {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
        PE::Text("Error", "RTX allocation failed");
        ImGui::PopStyleColor();
      }

      PE::end();
    }
  }
}

void GaussianSplattingUI::guiDrawMeshTransformProperties()
{
  namespace PE = nvgui::PropertyEditor;

  Instance& inst = m_meshSetVk.instances[m_selectedItemIndex];
  PE::begin("##Transform");
  if(guiGetTransform(inst.scale, inst.rotation, inst.translation, inst.transform, inst.transformInverse, false))
  {
    m_meshSetVk.rtxUpdateTopLevelAccelerationStructure();
  }
  PE::end();
}

void GaussianSplattingUI::guiDrawMeshMaterialProperties()
{
  namespace PE = nvgui::PropertyEditor;

  const auto objIndex           = m_meshSetVk.instances[m_selectedItemIndex].objIndex;
  auto&      materials          = m_meshSetVk.meshes[objIndex].materials;
  bool       needMaterialUpdate = false;

  for(auto i = 0; i < materials.size(); ++i)
  {
    PE::begin("##Material");
    auto& material = materials[i];
    ImGui::PushID(i);
    PE::Text("Name", m_meshSetVk.meshes[objIndex].matNames[i]);
    needMaterialUpdate |= PE::entry(
        "Model", [&]() { return m_ui.enumCombobox(GUI_ILLUM_MODEL, "##ID", &material.illum); }, "TODO");
    needMaterialUpdate |= PE::ColorEdit3("ambient", glm::value_ptr(material.ambient));
    needMaterialUpdate |= PE::ColorEdit3("diffuse", glm::value_ptr(material.diffuse));
    needMaterialUpdate |= PE::ColorEdit3("specular", glm::value_ptr(material.specular));
    needMaterialUpdate |= PE::ColorEdit3("transmittance", glm::value_ptr(material.transmittance));
    // TODO implement in Shader
    //needMaterialUpdate |= PE::ColorEdit3("emission", glm::value_ptr(material.emission));
    needMaterialUpdate |= PE::SliderFloat("shininess", &material.shininess, 0.0f, 2000.0f);
    needMaterialUpdate |= PE::SliderFloat("ior", &material.ior, 1.0f, 3.0f);
    //needMaterialUpdate |= PE::DragFloat("dissolve", &material.dissolve);
    ImGui::PopID();
    PE::end();
  }
  if(needMaterialUpdate)
  {
    m_meshSetVk.updateObjMaterialsBuffer(objIndex);
  }
}

void GaussianSplattingUI::guiDrawCameraProperties()
{
  namespace PE = nvgui::PropertyEditor;

  Camera camera = m_cameraSet.getCamera();
  if(m_selectedItemIndex > -1)  // we show a preset - "read only"
  {
    camera = m_cameraSet.getPreset(m_selectedItemIndex);
    if(m_selectedItemIndex > 0)
    {
      ImGui::Text("To modify a preset.");
      ImGui::Text("  1. Load the preset");
      ImGui::Text("  2. Modify the current camera");
      ImGui::Text("  3. Overwrite the preset with active camera");
    }
    else
      ImGui::Text("Default Preset cannot be modified.");
  }

  bool changed = false;

  if(ImGui::CollapsingHeader("Camera Intrinsics", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::BeginDisabled(m_selectedItemIndex != -1 || cameraManip->isAnimated());
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

      if(PE::Checkbox("Depth of Field", &camera.dofEnabled,
                      "Activates Depth of Field effect (DoF). Only works with 3DGRT, 3DGUT and hybrid 3DGUT/3GDRT.\n"
                      "Activating \"Temporal sampling\" in addition to DoF leads to better visual results."))
      {
        m_requestUpdateShaders = true;
        changed                = true;
      }
      ImGui::BeginDisabled(!camera.dofEnabled);
      if(PE::DragFloat("Focus distance", &camera.focusDist, 0.1F, 0.1F, 15.0F, "%.3f"))
      {
        resetFrameCounter();
        changed = true;
      }
      if(PE::SliderFloat("Aperture", &camera.aperture, 0.0F, 0.01F, "%.6f"))
      {
        resetFrameCounter();
        changed = true;
      }
      ImGui::EndDisabled();  // DoF

      ImGui::EndDisabled();  // Modifiable
    }
    PE::end();
    ImGui::EndDisabled();
  }
  if(ImGui::CollapsingHeader("Camera Extrinsics", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::BeginDisabled(m_selectedItemIndex != -1 || cameraManip->isAnimated());
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
    m_cameraSet.setCamera(camera);
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
  namespace PE = nvgui::PropertyEditor;

  bool needUpdate = false;

  auto& light = m_lightSet.getLight(m_selectedItemIndex);
  ImGui::Text("Light sources only affect meshes");
  ImGui::Text("Point lights have quadratic attenuation");
  PE::begin("##Light");
  if(PE::entry("Type", [&]() { return m_ui.enumCombobox(GUI_LIGHT_TYPE, "##ID", &light.type); }, "Type of light."))
  {
    needUpdate = true;
  }
  needUpdate |= PE::DragFloat3("Position", glm::value_ptr(light.position));
  needUpdate |= PE::DragFloat("Intensity", &light.intensity);
  PE::end();

  m_requestUpdateLightsBuffer |= needUpdate;
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

void GaussianSplattingUI::guiDrawRendererStatisticsWindow()
{
  if(ImGui::Begin("Rendering Statistics"))
  {
    const int32_t totalSplatCount = (uint32_t)m_splatSet.size();
    const int32_t rasterSplatCount =
        (prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX) ? totalSplatCount : m_indirectReadback.instanceCount;
    const uint32_t wgCount =
        (prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_MESH_3DGUT) ?
            ((prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX) ?
                 m_indirectReadback.groupCountX :
                 (prmFrame.splatCount + prmRaster.meshShaderWorkgroupSize - 1) / prmRaster.meshShaderWorkgroupSize) :
            0;

    if(ImGui::BeginTable("Stats", 3, ImGuiTableFlags_BordersOuter))
    {
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 230.0f);
      ImGui::TableSetupColumn("Size short", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Size Fill", ImGuiTableColumnFlags_WidthStretch);
      // ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Total splats");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatSize(totalSplatCount).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", totalSplatCount);
      ImGui::TableNextRow();
      ImGui::BeginDisabled(prmSelectedPipeline == PIPELINE_RTX);
      ImGui::TableNextColumn();
      ImGui::Text("Sorted splats");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatSize(rasterSplatCount).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", rasterSplatCount);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Mesh shader work groups");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatSize(wgCount).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%d", wgCount);
      ImGui::TableNextRow();
      ImGui::EndDisabled();
      ImGui::EndTable();
    }
  }
  ImGui::End();
}


void GaussianSplattingUI::guiDrawMemoryStatisticsWindow()
{
  ImGuiTableFlags commonFlags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_SpanAllColumns;
  ImGuiTableFlags itemFlags   = commonFlags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
  ImGuiTableFlags totalFlags  = commonFlags | ImGuiTreeNodeFlags_DefaultOpen;

  if(ImGui::Begin("Memory Statistics"))
  {
    if(ImGui::BeginTable("Scene stats", 4, ImGuiTableFlags_RowBg))
    {
      // to draw horizontal line for specific rows.
      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Host used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device allocated", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      bool open = ImGui::TreeNodeEx("Model data", totalFlags);
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devAll).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("Centers", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcCenters).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevCenters).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devCenters).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Covariances", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcCov).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevCov).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devCov).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH degree 0", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcSh0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevSh0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devSh0).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH degree 1,2,3", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcShOther).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevShOther).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devShOther).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH total", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcShAll).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevShAll).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devShAll).c_str());
        // end if(open)
        ImGui::TreePop();
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      open = ImGui::TreeNodeEx("Rasterization", totalFlags);
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rasterHostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rasterDeviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rasterDeviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("UBO frame info", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Indirect params", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedIndirect).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedIndirect).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Distances", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.hostAllocDistances).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedDistances).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.allocDistances).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Indices", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.hostAllocIndices).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.usedIndices).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.allocIndices).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("GPU sort", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal)
                              .c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal)
                              .c_str());
        // end if(open)
        ImGui::TreePop();
      }
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      open = ImGui::TreeNodeEx("Ray tracing", totalFlags);
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxHostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxDeviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxDeviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("TLAS", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedTlas).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedTlas).c_str());
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("BLAS", itemFlags);
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedBlas).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatMemorySize(m_renderMemoryStats.rtxUsedBlas).c_str());
        // end if(open)
        ImGui::TreePop();
      }
      ImGui::EndTable();
    }
    ImGui::Separator();
    if(ImGui::BeginTable("Total", 4, ImGuiTableFlags_None))
    {
      ImGui::TableSetupColumn("Rendering", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Host used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device allocated", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableNextColumn();
      ImGui::Text("Total");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.srcAll + m_renderMemoryStats.hostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.odevAll + m_renderMemoryStats.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(m_splatSetVk.memoryStats.devAll + m_renderMemoryStats.deviceAllocTotal).c_str());
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

void GaussianSplattingUI::guiDrawFooterBar()
{
  //
  //ImGuiViewportP* viewport = (ImGuiViewportP*)(void*)ImGui::GetMainViewport();
  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;
  float height = ImGui::GetFrameHeight();

  if(ImGui::BeginViewportSideBar("##MainStatusBar", NULL, ImGuiDir_Down, height, window_flags))
  {
    if(ImGui::BeginMenuBar())
    {
      ImGui::Text("Mouse ");
      ImGui::Text("%s", fmt::format("{} {}", prmFrame.cursor.x, prmFrame.cursor.y).c_str());
      ImGui::Text(" | Splat Id ");
      ImGui::Text("%s", std::to_string(m_indirectReadback.particleID).c_str());
      ImGui::Text(" | Splat Dist ");
      ImGui::Text("%s", std::to_string(m_indirectReadback.particleDist).c_str());

      // DEBUG Feedback
      /*
      ImGui::Text(" %s", "debug: ");
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val1).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val2).c_str());
      ImGui::Text(" %s  ", std::to_string(m_indirectReadback.val3).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val4).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val5).c_str());
      ImGui::Text(" %s  ", std::to_string(m_indirectReadback.val6).c_str());
      ImGui::Text(" %s", std::to_string(m_indirectReadback.val7).c_str());
      */

      // temporal sampling progress bar
      {
        float       progress = 0.0;
        std::string buf      = "1/1";
        if(prmRtx.temporalSampling)
        {
          progress = (float)prmFrame.frameSampleId / (std::max(1, prmFrame.frameSampleMax));
          buf      = fmt::format("{}/{}", prmFrame.frameSampleId, prmFrame.frameSampleMax);
        }
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 255);
        ImGui::Text("%s", "SPP");
        nvgui::tooltip("Samples Per Pixel");
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.4f, 0.7f, 0.0f, 1.0f));  // Green of course :-)
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
}

///////////////////////////////////
// Loading and Saving Propjects

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

// some macros to fetch jsoin values only if exist and affect to val

#define LOAD1(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = (item)[name]

#define LOAD2(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1]}

#define LOAD3(val, item, name)                                                                                         \
  if((item).contains(name))                                                                                            \
  (val) = {(item)[name][0], (item)[name][1], (item)[name][2]}

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
    if(!m_loadedSceneFilename.empty())
      ImGui::OpenPopup("Load .vkg project file ?");

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    bool doReset = true;

    if(ImGui::BeginPopupModal("Load .vkg project file ?", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      doReset = false;

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
        prmScene.sceneToLoadFilename   = "";
        prmScene.projectToLoadFilename = "";
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    if(doReset)
    {
      std::cout << "Opening project file " << path << std::endl;

      std::ifstream i(path);
      if(!i.is_open())
      {
        std::cout << "Error : unable to open project file " << path << std::endl;
        prmScene.projectToLoadFilename = "";
        return false;
      }

      try
      {
        i >> data;
      }
      catch(...)
      {
        std::cout << "Error : invalid project file " << path << std::endl;
        prmScene.projectToLoadFilename = "";
        return false;
      }
      i.close();

      loadingProject = true;

      // Initiate SplatSet loading
      if(!data["splats"].empty())
      {
        const auto& item             = data["splats"][0];
        prmScene.sceneToLoadFilename = makeAbsolutePath(std::filesystem::path(path).parent_path(), item["path"]);
      }
    }

    // Will do the rest of the work on next call when splatset is loaded
    return true;
  }

  // we skip until the splat set is being loaded
  if(m_plyLoader.getStatus() != PlyLoaderAsync::State::E_READY)
    return true;

  // we finalize
  guiAddToRecentProjects(prmScene.projectToLoadFilename);
  loadingProject                 = false;
  prmScene.projectToLoadFilename = "";

  try
  {
    // Renderer
    if(data.contains("renderer"))
    {
      const auto& item = data["renderer"];

      if(item.contains("vsync"))
        m_app->setVsync(item["vsync"]);

      LOAD1(prmSelectedPipeline, item, "pipeline");

      LOAD1(prmRender.maxShDegree, item, "maxShDegree");
      LOAD1(prmRender.opacityGaussianDisabled, item, "opacityGaussianDisabled");
      LOAD1(prmRender.showShOnly, item, "showShOnly");
      LOAD1(prmRender.visualize, item, "visualize");
      LOAD1(prmRender.wireframe, item, "wireframe");

      LOAD1(prmRaster.cpuLazySort, item, "cpuLazySort");
      LOAD1(prmRaster.distShaderWorkgroupSize, item, "distShaderWorkgroupSize");
      LOAD1(prmRaster.fragmentBarycentric, item, "fragmentBarycentric");
      LOAD1(prmRaster.frustumCulling, item, "frustumCulling");
      LOAD1(prmRaster.meshShaderWorkgroupSize, item, "meshShaderWorkgroupSize");
      LOAD1(prmRaster.pointCloudModeEnabled, item, "pointCloudModeEnabled");
      LOAD1(prmRaster.sortingMethod, item, "sortingMethod");

      LOAD1(prmRtx.temporalSampling, item, "temporalSampling");
      LOAD1(prmFrame.frameSampleMax, item, "temporalSamplesCount");
      LOAD1(prmRtx.kernelAdaptiveClamping, item, "kernelAdaptiveClamping");
      LOAD1(prmRtx.kernelDegree, item, "kernelDegree");
      LOAD1(prmRtx.kernelMinResponse, item, "kernelMinResponse");
      LOAD1(prmRtx.payloadArraySize, item, "payloadArraySize");
    }
    // Splat global options
    if(data.contains("splatsGlobals"))
    {
      const auto& item = data["splatsGlobals"];

      LOAD1(prmData.dataStorage, item, "dataStorage");
      LOAD1(prmData.shFormat, item, "shFormat");

      LOAD1(prmRtxData.compressBlas, item, "compressBlas");
      LOAD1(prmRtxData.useAABBs, item, "useAABBs");
      LOAD1(prmRtxData.useTlasInstances, item, "useTlasInstances");

      m_requestUpdateSplatData = true;
      m_requestUpdateSplatAs   = true;
    }
    // Parse splat settings
    if(data.contains("splats"))
    {
      if(!data["splats"].empty())
      {
        const auto& item = data["splats"][0];
        LOAD3(m_splatSetVk.translation, item, "position");
        LOAD3(m_splatSetVk.rotation, item, "rotation");
        LOAD3(m_splatSetVk.scale, item, "scale");

        computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation, m_splatSetVk.transform,
                         m_splatSetVk.transformInverse);

        // delay update of Acceleration Structures if not using ray tracing
        m_requestDelayedUpdateSplatAs = true;
      }
    }

    // Load all the meshes
    if(data.contains("meshes"))
    {
      auto meshId = 0;
      for(const auto& item : data["meshes"])
      {
        std::string relPath;
        LOAD1(relPath, item, "path");
        if(relPath.empty())
          continue;

        auto meshPath = makeAbsolutePath(std::filesystem::path(path).parent_path(), relPath);
        if(!m_meshSetVk.loadModel(meshPath.string()))
        {
          meshId++;
          continue;
        }
        // Access to newly created mesh/instance
        auto& instance = m_meshSetVk.instances.back();
        auto& mesh     = m_meshSetVk.meshes[instance.objIndex];

        // Transform
        LOAD3(instance.translation, item, "position");
        LOAD3(instance.rotation, item, "rotation");
        LOAD3(instance.scale, item, "scale");
        computeTransform(instance.scale, instance.rotation, instance.translation, instance.transform, instance.transformInverse);

        // Materials
        if(item.contains("materials"))
        {
          auto matId = 0;
          for(const auto& matItem : item["materials"])
          {
            auto& mat = mesh.materials[matId];
            LOAD3(mat.ambient, matItem, "ambient");
            LOAD3(mat.diffuse, matItem, "diffuse");
            LOAD1(mat.illum, matItem, "illum");
            LOAD1(mat.ior, matItem, "ior");
            LOAD1(mat.shininess, matItem, "shininess");
            LOAD3(mat.specular, matItem, "specular");
            LOAD3(mat.transmittance, matItem, "transmittance");

            matId++;
          }
          m_meshSetVk.updateObjMaterialsBuffer(meshId);
        }

        meshId++;
      }
      m_requestUpdateMeshData = true;
      m_requestUpdateShaders  = true;
    }

    // Parse camera
    if(data.contains("camera"))
    {
      auto&  item = data["camera"];
      Camera cam;
      LOAD1(cam.model, item, "model");
      LOAD3(cam.ctr, item, "ctr");
      LOAD3(cam.eye, item, "eye");
      LOAD3(cam.up, item, "up");
      LOAD1(cam.fov, item, "fov");
      LOAD1(cam.dofEnabled, item, "dofEnabled");
      LOAD1(cam.focusDist, item, "focusDist");
      LOAD1(cam.aperture, item, "aperture");
      m_cameraSet.setCamera(cam);
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
        LOAD1(cam.dofEnabled, item, "dofEnabled");
        LOAD1(cam.focusDist, item, "focusDist");
        LOAD1(cam.aperture, item, "aperture");
        m_cameraSet.createPreset(cam);
      }
    }
    // Parse lights
    if(data.contains("lights"))
    {
      bool defaultLight = true;
      for(const auto& item : data["lights"])
      {
        // A default light already exists, we only modify it
        uint64_t id = 0;
        if(!defaultLight)
        {
          id = m_lightSet.createLight();
        }
        auto& light = m_lightSet.getLight(id);
        LOAD1(light.type, item, "type");
        LOAD3(light.position, item, "position");
        LOAD1(light.intensity, item, "intensity");
        defaultLight = false;
      }
      m_requestUpdateLightsBuffer = true;
    }

    return true;
  }
  catch(...)
  {
    return false;
  }
}

bool GaussianSplattingUI::saveProject(std::string path)
{
  std::ofstream o(path);
  if(!o.is_open())
    return false;

  try
  {
    json data;

    // Renderer
    {
      json item;

      item["vsync"] = m_app->isVsync();

      item["pipeline"] = prmSelectedPipeline;

      item["maxShDegree"]             = prmRender.maxShDegree;
      item["opacityGaussianDisabled"] = prmRender.opacityGaussianDisabled;
      item["showShOnly"]              = prmRender.showShOnly;
      item["visualize"]               = prmRender.visualize;
      item["wireframe"]               = prmRender.wireframe;

      item["cpuLazySort"]             = prmRaster.cpuLazySort;
      item["distShaderWorkgroupSize"] = prmRaster.distShaderWorkgroupSize;
      item["fragmentBarycentric"]     = prmRaster.fragmentBarycentric;
      item["frustumCulling"]          = prmRaster.frustumCulling;
      item["meshShaderWorkgroupSize"] = prmRaster.meshShaderWorkgroupSize;
      item["pointCloudModeEnabled"]   = prmRaster.pointCloudModeEnabled;
      item["sortingMethod"]           = prmRaster.sortingMethod;

      item["temporalSampling"]       = prmRtx.temporalSampling;
      item["temporalSamplesCount"]   = prmFrame.frameSampleMax;
      item["kernelAdaptiveClamping"] = prmRtx.kernelAdaptiveClamping;
      item["kernelDegree"]           = prmRtx.kernelDegree;
      item["kernelMinResponse"]      = prmRtx.kernelMinResponse;
      item["payloadArraySize"]       = prmRtx.payloadArraySize;

      data["renderer"] = item;
    }

    // Active Camera
    {
      const auto& cam = m_cameraSet.getCamera();
      json        item;
      item["model"]      = cam.model;
      item["ctr"]        = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
      item["eye"]        = {cam.eye.x, cam.eye.y, cam.eye.z};
      item["up"]         = {cam.up.x, cam.up.y, cam.up.z};
      item["fov"]        = cam.fov;
      item["dofEnabled"] = cam.dofEnabled;
      item["focusDist"]  = cam.focusDist;
      item["aperture"]   = cam.aperture;

      data["camera"] = item;
    }

    // Camera presets
    data["cameras"] = json::array();
    for(auto camId = 0; camId < m_cameraSet.size(); ++camId)
    {
      auto cam = m_cameraSet.getPreset(camId);

      json item;
      item["model"]      = cam.model;
      item["ctr"]        = {cam.ctr.x, cam.ctr.y, cam.ctr.z};
      item["eye"]        = {cam.eye.x, cam.eye.y, cam.eye.z};
      item["up"]         = {cam.up.x, cam.up.y, cam.up.z};
      item["fov"]        = cam.fov;
      item["dofEnabled"] = cam.dofEnabled;
      item["focusDist"]  = cam.focusDist;
      item["aperture"]   = cam.aperture;

      data["cameras"].push_back(item);
    }

    // Lights
    data["lights"] = json::array();
    for(auto lightId = 0; lightId < m_lightSet.numLights; ++lightId)
    {
      const auto& light = m_lightSet.getLight(lightId);

      json item;
      item["type"]      = light.type;
      item["position"]  = {light.position.x, light.position.y, light.position.z};
      item["intensity"] = light.intensity;

      data["lights"].push_back(item);
    }

    // Splat global options
    {
      json item;
      item["dataStorage"] = prmData.dataStorage;
      item["shFormat"]    = prmData.shFormat;

      item["compressBlas"]     = prmRtxData.compressBlas;
      item["useAABBs"]         = prmRtxData.useAABBs;
      item["useTlasInstances"] = prmRtxData.useTlasInstances;

      data["splatsGlobals"] = item;
    }

    // Splat sets
    data["splats"] = json::array();
    {
      json item;
      item["path"]     = getRelativePath(std::filesystem::path(path).parent_path(), m_loadedSceneFilename);
      item["position"] = {m_splatSetVk.translation.x, m_splatSetVk.translation.y, m_splatSetVk.translation.z};
      item["rotation"] = {m_splatSetVk.rotation.x, m_splatSetVk.rotation.y, m_splatSetVk.rotation.z};
      item["scale"]    = {m_splatSetVk.scale.x, m_splatSetVk.scale.y, m_splatSetVk.scale.z};

      data["splats"].push_back(item);
    }

    // Meshes
    data["meshes"] = json::array();
    for(auto instId = 0; instId < m_meshSetVk.instances.size(); ++instId)
    {
      const auto& instance = m_meshSetVk.instances[instId];
      const auto& mesh     = m_meshSetVk.meshes[instance.objIndex];

      json item;
      item["path"] = getRelativePath(std::filesystem::path(path).parent_path(), mesh.path);
      item["name"] = mesh.name;

      // Transform
      item["position"] = {instance.translation.x, instance.translation.y, instance.translation.z};
      item["rotation"] = {instance.rotation.x, instance.rotation.y, instance.rotation.z};
      item["scale"]    = {instance.scale.x, instance.scale.y, instance.scale.z};

      // Material override
      item["materials"] = json::array();

      for(auto matId = 0; matId < mesh.matNames.size(); ++matId)
      {
        json matItem;

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

      data["meshes"].push_back(item);
    }

    o << std::setw(4) << data << std::endl;
    o.close();
    return true;
  }
  catch(...)
  {
    return false;
  }
}

void GaussianSplattingUI::dumpSplat(uint32_t splatIdx)
{
  if(!(splatIdx >= 0 && splatIdx < m_splatSet.size()))
  {
    std::cout << "Error: no splat to dump" << std::endl;
    return;
  }

  std::ofstream out("c:\\Temp\\debug_splat.ply");
  if(!out)
  {
    std::cout << "Error: coud not open file c:\\Temp\\debug_splat.ply" << std::endl;
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
    out << m_splatSet.positions[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 3; ++i)
    out << "0 ";  // no normals
  for(auto i = 0; i < 3; ++i)
    out << m_splatSet.f_dc[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 45; ++i)
    out << m_splatSet.f_rest[splatIdx * 45 + i] << " ";
  out << m_splatSet.opacity[splatIdx] << " ";
  for(auto i = 0; i < 3; ++i)
    out << m_splatSet.scale[splatIdx * 3 + i] << " ";
  for(auto i = 0; i < 4; ++i)
    out << m_splatSet.rotation[splatIdx * 4 + i] << " ";

  //
  out.close();

  //
  std::cout << "Splat " << splatIdx << " was dumped to c:\\Temp\\debug_splat.ply" << std::endl;
}

}  // namespace vk_gaussian_splatting
