/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

// ImGUI ImVec maths
#define IMGUI_DEFINE_MATH_OPERATORS
// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <gaussian_splatting.h>

std::string formatMemorySize(size_t sizeInBytes)
{
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

void GaussianSplatting::initGui()
{
  // Pipeline selector
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_VERT, "Vertex shader");
  m_ui.enumAdd(GUI_PIPELINE, PIPELINE_MESH, "Mesh shader");
  // m_ui.enumAdd(GUI_PIPELINE, PIPELINE_RTX,  "Ray tracing", true);  // disabled for the time being, not implemented
  // Sorting method selector
  m_ui.enumAdd(GUI_SORTING, SORTING_GPU_SYNC_RADIX, "GPU radix sort");
  //m_ui.enumAdd(GUI_SORTING, SORTING_CPU_ASYNC_MONO, "CPU async std mono");
  m_ui.enumAdd(GUI_SORTING, SORTING_CPU_ASYNC_MULTI, "CPU async std multi");
}

void GaussianSplatting::onUIRender()
{

  if(!m_gBuffers)
    return;

  {  // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Display the G-Buffer image
    ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

    {
      float  size        = 25.F;
      ImVec2 window_pos  = ImGui::GetWindowPos();
      ImVec2 window_size = ImGui::GetWindowSize();
      ImVec2 offset      = ImVec2(size * 1.1F, -size * 1.1F) * ImGui::GetWindowDpiScale();
      ImVec2 pos         = ImVec2(window_pos.x, window_pos.y + window_size.y) + offset;
      ImGuiH::Axis(pos, CameraManip.getMatrix(), size);
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }

  // do we need to load a new scenes ?
  if(!m_sceneToLoadFilename.empty() && m_plyLoader.getStatus() == PlyAsyncLoader::Status::READY)
  {
    // reset if a scene already exists
    const auto splatCount = m_splatSet.positions.size() / 3;
    if(splatCount)
    {
      deinitAll();
    }

    m_loadedSceneFilename = m_sceneToLoadFilename;
    //
    vkDeviceWaitIdle(m_device);

    std::cout << "Start loading file " << m_sceneToLoadFilename << std::endl;
    if(!m_plyLoader.loadScene(m_sceneToLoadFilename, m_splatSet))
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
    m_sceneToLoadFilename.clear();
  }

  // display loading jauge modal window
  // Always center this window when appearing
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if(ImGui::BeginPopupModal("Loading", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    // managment of async load
    switch(m_plyLoader.getStatus())
    {
      case PlyAsyncLoader::Status::LOADING: {
        ImGui::Text(m_plyLoader.getFilename().c_str());
        ImGui::ProgressBar(m_plyLoader.getProgress(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f));
        /*
        if(ImGui::Button("Cancel", ImVec2(120, 0)))
        {
          // send cancelation order to loader
          // should then disable the button, until cancel occurs or finished
          m_plyLoader.cancel();
        }
        */
      }
      break;
      case PlyAsyncLoader::Status::FAILURE: {
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
      case PlyAsyncLoader::Status::LOADED: {
        initAll();
        // set ready for next load
        m_plyLoader.reset();
        ImGui::CloseCurrentPopup();
        addToRecentFiles(m_loadedSceneFilename);
      }
      break;
      default: {
        // nothing to do for READY or SHUTDOWN
      }
    }
    ImGui::EndPopup();
  }

  // will rebuild data set according
  // to parameter change
  if(m_updateData && m_splatSet.size())
  {
    reinitDataStorage();
    m_updateData = false;
  }

  // will rebuild shaders according
  // to parameter change
  if(m_updateShaders && m_splatSet.size())
  {
    reinitShaders();
    m_updateShaders = false;
  }

  //
  namespace PE = ImGuiH::PropertyEditor;
  
  if(ImGui::Begin("Settings"))
  {
    //
    if(ImGui::CollapsingHeader("Data format and storage", ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::Button("Reset"))
        m_useDataTextures = false;

      PE::begin("##3DGS format");

      if(PE::Checkbox("Use data textures", &m_useDataTextures))    
        m_updateData = true;

      PE::end();
    }
    //
    if(ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::Button("Reset"))
      {
        resetRenderSettings();
      }

      PE::begin("##3DGS rendering");

      bool vsync = m_app->isVsync();
      if(PE::Checkbox("V-Sync", &vsync))
        m_app->setVsync(vsync);

      if(PE::entry("Sorting method", [&]() { return m_ui.enumCombobox(GUI_SORTING, "##ID", &m_frameInfo.sortingMethod); }))
      {
        if(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX && m_defines.frustumCulling == FRUSTUM_CULLING_DIST)
        {
          if(m_selectedPipeline == PIPELINE_MESH)
          {
            m_defines.frustumCulling = FRUSTUM_CULLING_MESH;
            m_updateShaders          = true;
          }
          else if(m_selectedPipeline == PIPELINE_VERT)
          {
            m_defines.frustumCulling = FRUSTUM_CULLING_VERT;
            m_updateShaders          = true;
          }
        }
      }

      ImGui::BeginDisabled(m_frameInfo.sortingMethod == SORTING_GPU_SYNC_RADIX);
      PE::Text("CPU sorting ", m_cpuSorter.getStatus() == SplatSorterAsync::SORTING ? "Sorting" : "Idled");
      ImGui::EndDisabled();

      if(PE::entry("Rasterization", [&]() { return m_ui.enumCombobox(GUI_PIPELINE, "##ID", &m_selectedPipeline); }))
      {
        if(m_selectedPipeline == PIPELINE_MESH && m_defines.frustumCulling == FRUSTUM_CULLING_VERT)
        {
          m_defines.frustumCulling = FRUSTUM_CULLING_MESH;
          m_updateShaders          = true;
        }
        else if(m_selectedPipeline == PIPELINE_VERT && m_defines.frustumCulling == FRUSTUM_CULLING_MESH)
        {
          m_defines.frustumCulling = FRUSTUM_CULLING_VERT;
          m_updateShaders          = true;
        }
      }

      // Radio buttons for exclusive selection
      PE::entry("Frustum culling", [&]() {
        if(ImGui::RadioButton("Disabled", m_defines.frustumCulling == FRUSTUM_CULLING_NONE))
        {
          m_defines.frustumCulling = FRUSTUM_CULLING_NONE;
          m_updateShaders          = true;
        }

        ImGui::BeginDisabled(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX);
        if(ImGui::RadioButton("In distance shader", m_defines.frustumCulling == FRUSTUM_CULLING_DIST))
        {
          m_defines.frustumCulling = FRUSTUM_CULLING_DIST;
          m_updateShaders          = true;
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(m_selectedPipeline != PIPELINE_VERT);
        if(ImGui::RadioButton("In vertex shader", m_defines.frustumCulling == FRUSTUM_CULLING_VERT))
        {
          m_defines.frustumCulling = FRUSTUM_CULLING_VERT;
          m_updateShaders          = true;
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(m_selectedPipeline != PIPELINE_MESH);
        if(ImGui::RadioButton("In mesh shader", m_defines.frustumCulling == FRUSTUM_CULLING_MESH))
        {
          m_defines.frustumCulling = FRUSTUM_CULLING_MESH;
          m_updateShaders          = true;
        }
        ImGui::EndDisabled();
        return true;
      });

      PE::SliderFloat("Frustum dilation", &m_frameInfo.frustumDilation, 0.0f, 1.0f, "%.1f");

      int alphaThres = 255 * m_frameInfo.alphaCullThreshold;
      if(PE::SliderInt("Alpha culling threshold", &alphaThres, 0, 255))
      {
        m_frameInfo.alphaCullThreshold = (float)alphaThres / 255.0f;
      }

      // we set a different size range for point and splat rendering
      PE::SliderFloat("Splat scale", (float*)&m_frameInfo.splatScale, 0.1f, m_frameInfo.pointCloudModeEnabled != 0 ? 10.0f : 2.0f);

      PE::SliderInt("Maximum SH degree", (int*)&m_frameInfo.sphericalHarmonicsDegree, 0, 2);

      if (PE::Checkbox("Show SH deg > 0 only", &m_defines.showShOnly))
        m_updateShaders = true;

      bool disableSplatting = m_frameInfo.pointCloudModeEnabled != 0;
      if(PE::Checkbox("Disable splatting", &disableSplatting))
        m_frameInfo.pointCloudModeEnabled = disableSplatting ? 1 : 0;

      if(PE::Checkbox("Disable opacity gaussian ", &m_defines.opacityGaussianDisabled))
        m_updateShaders = true;

      PE::end();
    }
    //
    if(ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen))
    {
      // TODO: do not use disabled input object to display statistics
      PE::begin("##3DGS statistics");
      ImGui::BeginDisabled();
      uint32_t totalSplatCount = (uint32_t)m_splatIndices.size();
      PE::entry(
          "Total splats", [&]() { return ImGui::InputInt("##HiddenID", (int*)&totalSplatCount, 0, 100000); }, "TODOC");
      uint32_t renderedSplatCount =
          (m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX) ? totalSplatCount : m_indirectReadback.instanceCount;
      PE::entry(
          "Rendered splats", [&]() { return ImGui::InputInt("##HiddenID", (int*)&renderedSplatCount, 0, 100000); }, "TODOC");
      uint32_t wg = (m_selectedPipeline == PIPELINE_MESH) ?
                        ((m_frameInfo.sortingMethod == SORTING_GPU_SYNC_RADIX) ? m_indirectReadback.groupCountX :
                                                                                 (m_frameInfo.splatCount + 31) / 32) :
                        0;
      PE::entry(
          "Mesh shader work groups", [&]() { return ImGui::InputInt("##HiddenID", (int*)&wg, 0, 100000); }, "TODOC");
      PE::entry(
          "CPU Distances  (ms)", [&]() { return ImGui::InputInt("##HiddenID", (int*)&(m_distTime), 0, 100000); }, "TODOC");
      PE::entry(
          "CPU Sorting  (ms)", [&]() { return ImGui::InputInt("##HiddenID", (int*)&(m_sortTime), 0, 100000); }, "TODOC");
      ImGui::EndDisabled();

      PE::end();
    }
    ImGui::End();
  }
  /*
  if(ImGui::Begin("Misc"))
  {
    if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGuiH::CameraWidget();
    }
    ImGui::End();
  }
  */
  if (ImGui::Begin("Memory Statistics"))
  {
    if(ImGui::BeginTable("Scene stats", 4, ImGuiTableFlags_None))
    {
      ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Host used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device allocated", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Centers");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.srcCenters).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.odevCenters).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.devCenters).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Covariances");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.srcCov).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.odevCov).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.devCov).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("SH degree 0");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.srcSh0).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.odevSh0).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.devSh0).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("SH degree 1,2,3");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.srcShOther).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.odevShOther).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.devShOther).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("SH Total");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.srcShAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.odevShAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.devShAll).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Sub-total");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.srcAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.odevAll).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.devAll).c_str());
      ImGui::EndTable();
    }
    ImGui::Separator();
    if(ImGui::BeginTable("Scene stats", 4, ImGuiTableFlags_None))
    {
      ImGui::TableSetupColumn("Rendering", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Host used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device allocated", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("UBO frame info");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.usedUboFrameInfo).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Indirect params");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(0).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.usedIndirect).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.usedIndirect).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Distances");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.hostAllocDistances).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.usedDistances).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.allocDistances).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Indices");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.hostAllocIndices).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.usedIndices).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.allocIndices).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("GPU sort");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(0).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal)
                      .c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal)
                      .c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Sub-total");
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.hostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_renderMemoryStats.deviceAllocTotal).c_str());
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
      ImGui::Text(formatMemorySize(m_modelMemoryStats.srcAll + m_renderMemoryStats.hostTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.odevAll + m_renderMemoryStats.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(formatMemorySize(m_modelMemoryStats.devAll + m_renderMemoryStats.deviceAllocTotal).c_str());
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

void GaussianSplatting::onUIMenu()
{
  static bool close_app{false};
  bool        v_sync = m_app->isVsync();
#ifndef NDEBUG
  static bool s_showDemo{false};
  static bool s_showDemoPlot{false};
#endif
  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem("Open file", ""))
    {
      m_sceneToLoadFilename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load ply file", "PLY(.ply)");
    }
    if(ImGui::MenuItem("Re Open", "", false, m_loadedSceneFilename != ""))
    {
      m_sceneToLoadFilename = m_loadedSceneFilename;
    }
    if(ImGui::BeginMenu("Recent Files"))
    {
      for(const auto& file : m_recentFiles)
      {
        if(ImGui::MenuItem(file.c_str()))
        {
          m_sceneToLoadFilename = file;
        }
      }
      ImGui::EndMenu();
    }
    ImGui::Separator();
    if(ImGui::MenuItem("Close", ""))
    {
      deinitAll();
    }
    ImGui::Separator();
    if(ImGui::MenuItem("Exit", "Ctrl+Q"))
    {
      close_app = true;
    }
    ImGui::EndMenu();
  }
  if(ImGui::BeginMenu("View"))
  {
    ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &v_sync);
    ImGui::EndMenu();
  }
#ifndef NDEBUG
  if(ImGui::BeginMenu("Debug"))
  {
    ImGui::MenuItem("Show ImGui Demo", nullptr, &s_showDemo);
    ImGui::MenuItem("Show ImPlot Demo", nullptr, &s_showDemoPlot);
    ImGui::EndMenu();
  }
#endif  // !NDEBUG

  // Shortcuts
  if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
  {
    close_app = true;
  }

  if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
  {
    v_sync = !v_sync;
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
    ImPlot::ShowDemoWindow(&s_showDemoPlot);
  }
#endif  // !NDEBUG

  if(m_app->isVsync() != v_sync)
  {
    m_app->setVsync(v_sync);
  }
}


void GaussianSplatting::addToRecentFiles(const std::string& filePath, int historySize)
{
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

// Register handler
void GaussianSplatting::registerRecentFilesHandler()
{
  // mandatory to work, see ImGui::DockContextInitialize as an example
  auto readOpen = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* {
    if(strcmp(name, "Data") != 0)
      return NULL;
    return (void*)1;
  };

  // Save settings handler, not using capture so can be used as a function pointer
  auto saveRecentFilesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
    auto* self = static_cast<GaussianSplatting*>(handler->UserData);
    buf->appendf("[%s][Data]\n", handler->TypeName);
    for(const auto& file : self->m_recentFiles)
    {
      buf->appendf("File=%s\n", file.c_str());
    }
    buf->append("\n");
  };

  // Load settings handler, not using capture so can be used as a function pointer
  auto loadRecentFilesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
    auto* self = static_cast<GaussianSplatting*>(handler->UserData);
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