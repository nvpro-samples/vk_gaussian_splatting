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

void GaussianSplatting::onUIRender()
{
  if(!m_gBuffers)
    return;

  {  // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Deal with mouse interaction only if the window has focus
    //if(ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow) && ImGui::IsMouseDoubleClicked(0))
    //{
    //  rasterPicking();
    //}

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
      vkDeviceWaitIdle(m_device);
      destroyScene();
      destroy3dgsTextures();
      destroyVkBuffers();
      destroyPipeline();
    }
    //
    vkDeviceWaitIdle(m_device);

    std::cout << "Start loading file " << m_sceneToLoadFilename.string() << std::endl;
    if(!m_plyLoader.loadScene(m_sceneToLoadFilename.string(), m_splatSet))
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
          destroyScene();
          // TODO: use BBox of point cloud to set far plane
          CameraManip.setClipPlanes({0.1F, 2000.0F});
          // we know that most INRIA models are upside down so we set the up vector to 0,-1,0
          CameraManip.setLookat({0.0F, 0.0F, -2.0F}, {0.F, 0.F, 0.F}, {0.0F, -1.0F, 0.0F});
          // reset general parameters
          resetFrameInfo();
          //
          m_plyLoader.reset();
          //
          ImGui::CloseCurrentPopup();
        }
      }
      break;
      case PlyAsyncLoader::Status::LOADED: {
        // TODO: use BBox of point cloud to set far plane
        CameraManip.setClipPlanes({0.1F, 2000.0F});
        // we know that most INRIA models are upside down so we set the up vector to 0,-1,0
        CameraManip.setLookat({0.0F, 0.0F, -2.0F}, {0.F, 0.F, 0.F}, {0.0F, -1.0F, 0.0F});
        // reset general parameters
        resetFrameInfo();
        //
        createVkBuffers();
        createPipeline();
        create3dgsTextures();
        m_plyLoader.reset();
        //
        ImGui::CloseCurrentPopup();
      }
      break;
      default: {
        // nothing to do for READY or SHUTDOWN
      }
    }
    ImGui::EndPopup();
  }

  //
  namespace PE = ImGuiH::PropertyEditor;
  {  // Setting menu
    ImGui::Begin("Settings");
    if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGuiH::CameraWidget();
    }
    if(ImGui::CollapsingHeader("3DGS parameters", ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::Button("Reset"))
      {
        resetFrameInfo();
      }
      
      PE::begin("##3DGS sorting");

      bool disableGpuSorting = !m_gpuSortingEnabled;
      PE::entry(
          "Disable GPU sorting ", [&]() { return ImGui::Checkbox("##DisableGPUSorting", &disableGpuSorting); }, "TODOC");
      m_gpuSortingEnabled = !disableGpuSorting;
      
      ImGui::BeginDisabled(disableGpuSorting);
      bool culling = m_gpuSortingEnabled && frameInfo.culling != 0;
      PE::entry(
          "Frustum culling", [&]() { return ImGui::Checkbox("##hidenID", &culling); }, "TODOC");
      frameInfo.culling = culling ? 1 : 0;
      ImGui::EndDisabled();
      PE::end();

      PE::begin("##3DGS splatting");

      PE::entry(
          "Mesh shaders", [&]() { return ImGui::Checkbox("##HiddenID", &m_useMeshShaders); },
          "Switch betten use of vertex shader pipeline and mesh shader pipeline");

      PE::SliderFloat(
          "Splat scale",
          (float*)&frameInfo.splatScale, 0.1f,
          frameInfo.pointCloudModeEnabled != 0 ? 10.0f : 2.0f  // we set a different size range for point and splat rendering
          );

      PE::entry(
          "Spherical Harmonic degree",
          [&]() { return ImGui::SliderInt("##ShDegree", (int*)&frameInfo.sphericalHarmonicsDegree, 0, 2); }, "TODOC");

      bool showShOnly = frameInfo.showShOnly != 0;
      PE::entry(
          "Show SH only", [&]() { return ImGui::Checkbox("##ShowSHOnly", &showShOnly); }, "TODOC");
      frameInfo.showShOnly = showShOnly ? 1 : 0;

      bool disableSplatting = frameInfo.pointCloudModeEnabled != 0;
      PE::entry(
          "Disable splatting", [&]() { return ImGui::Checkbox("##DisableSplatting", &disableSplatting); }, "TODOC");
      frameInfo.pointCloudModeEnabled = disableSplatting ? 1 : 0;

      bool opacityGaussianDisabled = frameInfo.opacityGaussianDisabled != 0;
      PE::entry(
          "Disable opacity gaussian",
          [&]() { return ImGui::Checkbox("##opacityGaussianDisabled", &opacityGaussianDisabled); }, "TODOC");
      frameInfo.opacityGaussianDisabled = opacityGaussianDisabled ? 1 : 0;

      PE::end();
    }
    if(ImGui::CollapsingHeader("3DGS statistics", ImGuiTreeNodeFlags_DefaultOpen))
    {
      // TODO: do not use disabled input object to display statistics
      PE::begin("##3DGS statistics");
      ImGui::BeginDisabled();

      PE::entry(
          "Distances  (ms)", [&]() { return ImGui::InputFloat("##HiddenID", (float*)&m_distTime, 0, 100000); }, "TODOC");
      PE::entry(
          "Sorting  (ms)", [&]() { return ImGui::InputFloat("##HiddenID", (float*)&m_sortTime, 0, 100000); }, "TODOC");
      const auto totalSplatCount = gsIndex.size();
      PE::entry(
          "Total splats", [&]() { return ImGui::InputInt("##HiddenID", (int*)&totalSplatCount, 0, 100000); }, "TODOC");
      uint32_t renderedSplatCount = !m_gpuSortingEnabled ? totalSplatCount : m_indirectReadback.instanceCount;
      PE::entry(
          "Rendered splats", [&]() { return ImGui::InputInt("##HiddenID", (int*)&renderedSplatCount, 0, 100000); }, "TODOC");
      uint32_t wg =
          m_useMeshShaders ? (m_gpuSortingEnabled ? m_indirectReadback.groupCountX : (frameInfo.splatCount + 31) / 32) : 0;
      PE::entry(
          "Mesh shader work groups", [&]() { return ImGui::InputInt("##HiddenID", (int*)&wg, 0, 100000); }, "TODOC");
      ImGui::EndDisabled();

      PE::end();
    }
    ImGui::End();
  }
}
