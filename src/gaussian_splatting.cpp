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

// TODOC
#define VMA_IMPLEMENTATION
// ImGUI ImVec maths
#define IMGUI_DEFINE_MATH_OPERATORS

#include <gaussian_splatting.h>

// create, setup and run an nvvkhl::Application
int main(int argc, char** argv)
{
  // Vulkan creation context information (see nvvk::Context)
  nvvk::ContextCreateInfo vkSetup;
  vkSetup.setVersion(1, 3);
  vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

  // Create Vulkan context
  nvvk::Context vkContext;
  vkContext.init(vkSetup);

  // Application setup
  nvvkhl::ApplicationCreateInfo appSetup;
  appSetup.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appSetup.vSync          = true;
  appSetup.instance       = vkContext.m_instance;
  appSetup.device         = vkContext.m_device;
  appSetup.physicalDevice = vkContext.m_physicalDevice;
  appSetup.queues.push_back({vkContext.m_queueGCT.familyIndex, vkContext.m_queueGCT.queueIndex, vkContext.m_queueGCT.queue});

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appSetup);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  auto elementProfiler = std::make_shared<nvvkhl::ElementProfiler>(true);
  app->addElement(elementProfiler);
  app->addElement(std::make_shared<GaussianSplatting>(elementProfiler));

  app->run();
  app.reset();

  return test->errorCode();
}

void GaussianSplatting::onAttach(nvvkhl::Application* app)
{
  m_app    = app;
  m_device = m_app->getDevice();

  m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
  m_alloc       = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
            .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = app->getPhysicalDevice(),
            .device         = app->getDevice(),
            .instance       = app->getInstance(),
  });  // Allocator
  m_dset        = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
  m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
  //
  m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

  //
  createScene(std::string("C:\\Users\\jmarvie\\Datasets\\bicycle\\bicycle\\point_cloud\\iteration_7000\\point_cloud.ply"));
  createVkBuffers();
  createPipeline();
  create3dgsTextures();
  
};

void GaussianSplatting::onDetach()
{
  // stop the sorting thread
  std::unique_lock<std::mutex> lock(mutex);
  sortExit = true;
  cond_var.notify_one();
  lock.unlock();
  sortingThread.join();
  // release resources
  vkDeviceWaitIdle(m_device);
  destroyScene();
  destroy3dgsTextures();
  destroyVkBuffers();
  destroyPipeline();
  destroyGbuffers();
  //
  m_dset->deinit();
}

void GaussianSplatting::onResize(uint32_t width, uint32_t height)
{
  createGbuffers({width, height});
}

void GaussianSplatting::onRender(VkCommandBuffer cmd)
{
  if(!m_gBuffers)
    return;

  // do we need to load a ne scenes ?
  if(!m_sceneToLoadFilename.empty())
  {
    //
    vkDeviceWaitIdle(m_device);
    destroyScene();
    destroy3dgsTextures();
    destroyVkBuffers();
    destroyPipeline();

    //
    createScene(m_sceneToLoadFilename.string());
    createVkBuffers();
    createPipeline();
    create3dgsTextures();

    // reset request
    m_sceneToLoadFilename.clear();
  }

  //
  if(m_splatSet.positions.empty())
    return;

  //
  const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

  const float aspect_ratio = m_viewSize.x / m_viewSize.y;
  glm::vec3   eye;
  glm::vec3   center;
  glm::vec3   up;
  CameraManip.getLookat(eye, center, up);

  // update sorted indices if needed
  bool       newIndexAvailable = false;
  const auto splatCount        = m_splatSet.positions.size() / 3;

  if(!gpuSortingEnabled)
  {
    if(!frameInfo.opacityGaussianDisabled)
    {
      // Splatting/blending is on, we check for a newly sorted index table
      std::unique_lock<std::mutex> lock(mutex);
      bool                         sortDoneCopy = sortDone;
      if(sortDone)
        sortDone = false;
      lock.unlock();
      if(sortDoneCopy || !sortStart)
      {
        // sorter is sleeping, we can work on shared data
        // we take into account the result of the sort
        if(sortDoneCopy)
        {
          gsIndex.swap(sortGsIndex);
          //
          newIndexAvailable = true;
        }
        // then if view point has changed we restart a sort
        const auto nrmDir = glm::normalize(center - eye);
        if(sortDir != nrmDir || sortCop != eye)
        {
          // now we wake the sorting thread with new
          // camera information
          sortDir   = nrmDir;
          sortCop   = eye;
          sortStart = true;
          // let's wakeup teh sorting thread
          lock.lock();
          cond_var.notify_one();
          lock.unlock();
        }
      }
    }
    else
    {
      // splatting off, we disable the sorting
      // indices would not be needed for non splatted points
      // however, uisng the same mechanism allows to use exactly the same shader
      // so if splatting/blending is off we provide an ordered table of indices 0->splatcount
      bool refill = (gsIndex.size() != splatCount);
      if(refill)
      {
        gsIndex.resize(splatCount);
        for(int i = 0; i < splatCount; ++i)
        {
          gsIndex[i] = i;
        }
        newIndexAvailable = true;
      }
      m_sortTime = 0;
    }
  }

  // Update frame parameters uniform buffer
  // some attributes of frameInfo were set by the user interface
  const glm::vec2& clip      = CameraManip.getClipPlanes();
  frameInfo.splatCount       = splatCount;
  frameInfo.viewMatrix       = CameraManip.getMatrix();
  frameInfo.projectionMatrix = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, clip.x, clip.y);
  // OpenGL (0,0) is bottom left, Vulkan (0,0) is top left, and glm::perspectiveRH_ZO is for OpenGL so we mirror on y
  frameInfo.projectionMatrix[1][1] *= -1;
  frameInfo.cameraPosition         = eye;
  float       devicePixelRatio     = 1.0;
  const float focalLengthX         = frameInfo.projectionMatrix[0][0] * 0.5f * devicePixelRatio * m_viewSize.x;
  const float focalLengthY         = frameInfo.projectionMatrix[1][1] * 0.5f * devicePixelRatio * m_viewSize.y;
  const bool  isOrthographicCamera = false;
  const float focalMultiplier      = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
  const float focalAdjustment      = focalMultiplier;  //  this.focalAdjustment* focalMultiplier;
  frameInfo.orthoZoom              = 1.0f;
  frameInfo.orthographicMode       = 0;  // disabled (uses perspective) TODO: activate support for orthographic
  frameInfo.viewport               = glm::vec2(m_viewSize.x * devicePixelRatio, m_viewSize.x * devicePixelRatio);
  frameInfo.basisViewport          = glm::vec2(1.0f / m_viewSize.x, 1.0f / m_viewSize.y);
  frameInfo.focal                  = glm::vec2(focalLengthX, focalLengthY);
  frameInfo.inverseFocalAdjustment = 1.0f / focalAdjustment;

  vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(DH::FrameInfo), &frameInfo);

  { //
    std::vector<uint32_t> drawIndexedIndirectParams{6, 0, 0, 0, 0};
    vkCmdUpdateBuffer(cmd, m_indirect.buffer, 0, drawIndexedIndirectParams.size() * sizeof(uint32_t),
                      drawIndexedIndirectParams.data());

    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, NULL, 0, NULL);
  }

  // if CPU sorting we test for asyncronous sorting result to push to GPU
  if(!gpuSortingEnabled)
  {
    auto timerSection = m_profiler->timeRecurring("Copy indices to GPU", cmd);

    if(newIndexAvailable)
    {

      // CPU sort
      // Prepare buffer on host using sorted indices
      uint32_t* hostBuffer = static_cast<uint32_t*>(m_alloc->map(m_splatIndicesHost));
      for(int i = 0; i < splatCount; ++i)
      {
        hostBuffer[i] = gsIndex[i];
      }
      m_alloc->unmap(m_splatIndicesHost);
      // copy buffer to device
      VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = splatCount * sizeof(uint32_t)};
      vkCmdCopyBuffer(cmd, m_splatIndicesHost.buffer, m_splatIndicesDevice.buffer, 1, &bc);
      // sync with end of copy to device
      VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bmb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      bmb.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
      bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.buffer              = m_splatIndicesDevice.buffer;
      bmb.size                = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                           VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bmb, 0, nullptr);
    }
  }

  // when GPU sorting, we sort at each frame, all buffer in device memory, no copy
  if(gpuSortingEnabled)
  {
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

    // invoke the distance compute shader
    {
      auto timerSection = m_profiler->timeRecurring("GPU Dist", cmd);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);

      constexpr int local_size = 256;
      vkCmdDispatch(cmd, (splatCount + local_size - 1) / local_size, 1, 1);

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &barrier, 0, NULL, 0, NULL);
    }

    // sort
    {
      auto timerSection = m_profiler->timeRecurring("GPU Sort", cmd);

      vrdxCmdSortKeyValueIndirect(cmd, m_sorter, splatCount, 
        m_indirect.buffer, sizeof(uint32_t),
        m_keysDevice.buffer, 0, 
        m_keysDevice.buffer, splatCount * sizeof(uint32_t),
        m_storageDevice.buffer, 0, 0, 0);

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1,
                           &barrier, 0, NULL, 0, NULL);

    }

    // read back m_keysDevice for debug
    if(false)
    {
      // reset staging for debug
      uint32_t* stagingBuffer = static_cast<uint32_t*>(m_alloc->map(m_stagingHost));
      //std::memset(stagingBuffer, 0, (2 * splatCount + 1) * sizeof(uint32_t));


      // copy from device to host
      VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = (2 * splatCount + 1) * sizeof(uint32_t)};
      vkCmdCopyBuffer(cmd, m_keysDevice.buffer, m_stagingHost.buffer, 1, &bc);
      // sync with end of copy to device
      VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bmb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      bmb.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
      bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.buffer              = m_keysDevice.buffer;
      bmb.size                = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                           VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bmb, 0, nullptr);

      int i = 0;
      m_alloc->unmap(m_stagingHost);
    }
  }

  // Drawing the primitives in the G-Buffer
  {
    auto timerSection = m_profiler->timeRecurring("Splatting", cmd);

    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // let's throw some pixels !!
    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);
    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthTestEnable(cmd, (VkBool32)frameInfo.opacityGaussianDisabled);
    
    // display the quad as many times as we have visible splats
    {
      /* we do not use push_constant, everything passes though teh frameInfor unifrom buffer
      // transfo/color unused for the time beeing, could transform the whole 3DGS model
      // if used, could also be placed in the FrameOnfo or all the frameInfo placed in push_constant
      m_pushConst.transfo = glm::mat4(1.0);                 // identity
      m_pushConst.color   = glm::vec4(0.5, 0.5, 0.5, 1.0);  //

      vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(DH::PushConstant), &m_pushConst);
      */
      
      const VkDeviceSize offsets{0};
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      if(!gpuSortingEnabled)
      {
        vkCmdBindVertexBuffers(cmd, 1, 1, &m_splatIndicesDevice.buffer, &offsets);
        vkCmdDrawIndexed(cmd, 6, (uint32_t)splatCount, 0, 0, 0);
      }
      else
      {
        const VkDeviceSize valueOffsets{splatCount * sizeof(uint32_t)};
        vkCmdBindVertexBuffers(cmd, 1, 1, &m_keysDevice.buffer, &valueOffsets);
        vkCmdDrawIndexedIndirect(cmd, m_indirect.buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
      }
    }
    // TODOC
    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }
}

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
  namespace PE = ImGuiH::PropertyEditor;
  {  // Setting menu
    ImGui::Begin("Settings");
    if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGuiH::CameraWidget();
    }
    if(ImGui::CollapsingHeader("3DGS parameters", ImGuiTreeNodeFlags_DefaultOpen))
    {
      PE::begin();
      if(PE::entry("Default parameters", [&]() { return ImGui::Button("Reset"); }))
      {
        resetFrameInfo();
      }
      PE::entry(
          "Splat scale",
          [&]() {
            return ImGui::SliderFloat("##SPlatScale", (float*)&frameInfo.splatScale, 0.1f,
                                      frameInfo.pointCloudModeEnabled != 0 ? 10.0f : 2.0f);  // we set a different size range for point and splat rendering
          },
          "TODOC");

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

      bool disableGpuSorting = !gpuSortingEnabled;
      PE::entry(
          "Disable GPU sorting ", [&]() { return ImGui::Checkbox("##DisableGPUSorting", &disableGpuSorting); }, "TODOC");
      gpuSortingEnabled = !disableGpuSorting;

      PE::end();
    }
    if(ImGui::CollapsingHeader("3DGS statistics", ImGuiTreeNodeFlags_DefaultOpen))
    {
      // TODO: do not use disabled input object to display statistics
      PE::begin();
      ImGui::BeginDisabled();
      PE::entry(
          "Distances  (ms)", [&]() { return ImGui::InputFloat("##DistDuration", (float*)&m_distTime, 0, 100000); }, "TODOC");
      PE::entry(
          "Sorting  (ms)", [&]() { return ImGui::InputFloat("##SortDuration", (float*)&m_sortTime, 0, 100000); }, "TODOC");
      const auto splatCount = gsIndex.size();
      PE::entry(
          "Number of splats", [&]() { return ImGui::InputInt("##NbSplats", (int*)&splatCount, 0, 100000); }, "TODOC");
      ImGui::EndDisabled();

      PE::end();
    }
    ImGui::End();
  }
}

void GaussianSplatting::sortingThreadFunc(void)
{
  while(true)
  {
    // wait until a sort or an exit is requested
    std::unique_lock<std::mutex> lock(mutex);
    cond_var.wait(lock, [this] { return sortStart || sortExit; });
    const bool exitSortCopy = sortExit;
    lock.unlock();
    if(exitSortCopy)
      return;

    // There is no reason this arrives but we test just in case
    assert(!gpuSortingEnabled);

    // since it was not an exit it is a request for a sort
    // we suppose model does not change
    // however during the process dir and cop can change so we use the copy
    auto startTime = std::chrono::high_resolution_clock::now();
    // we do the sorting if needed
    // find plane passing through COP and with normal dir.
    // we use distance to plane instead of distance to COP as an approximation.
    // https://mathinsight.org/distance_point_plane
    const glm::vec4 plane(sortDir[0], sortDir[1], sortDir[2],
                          -sortDir[0] * sortCop[0] - sortDir[1] * sortCop[1] - sortDir[2] * sortCop[2]);
    const float     divider = 1.0f / sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);

    const auto splatCount = m_splatSet.positions.size() / 3;

    // prepare an array of pair <distance, original index>
    distArray.resize(splatCount);

    // Sequential version of compute distances
#if defined(SEQUENTIAL) || !defined(_WIN32)
    for(int i = 0; i < splatCount; ++i)
    {
      const auto pos = &(m_splatSet.positions[i * 3]);
      // distance to plane
      const float dist    = std::abs(plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3]) * divider;
      distArray[i].first  = dist;
      distArray[i].second = i;
    }
  }
#else
    // parallel for, compute distances
    auto& tmpArray = distArray;
    auto& splatSet = m_splatSet;
    std::for_each(std::execution::par_unseq, tmpArray.begin(), tmpArray.end(),
                  //concurrency::parallel_for_each(distArray.begin(), distArray.end(),
                  [&tmpArray, &splatSet, &plane, &divider](std::pair<float, int> const& val) {
                    size_t     i   = &val - &tmpArray[0];
                    const auto pos = &(splatSet.positions[i * 3]);
                    // distance to plane
                    const float dist = std::abs(plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3]) * divider;
                    tmpArray[i].first  = dist;
                    tmpArray[i].second = i;
                  });
#endif

  auto time1 = std::chrono::high_resolution_clock::now();
  m_distTime = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - startTime).count();

  // Sorting the array with respect to distance keys
#if defined(SEQUENTIAL) || !defined(_WIN32)
  std::sort(distArray.begin(), distArray.end(), compare);
#else
    std::sort(std::execution::par_unseq, distArray.begin(), distArray.end(), compare);
#endif
  // create the sorted index array
  sortGsIndex.resize(splatCount);
  for(int i = 0; i < splatCount; ++i)
  {
    sortGsIndex[i] = distArray[i].second;
  }

  auto time2 = std::chrono::high_resolution_clock::now();
  m_sortTime = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();

  lock.lock();
  sortDone  = true;
  sortStart = false;
  lock.unlock();
}
}

void GaussianSplatting::createScene(const std::string& path)
{

  loadPly(path, m_splatSet);

  // TODO: use BBox of point cloud to set far plane
  CameraManip.setClipPlanes({0.1F, 2000.0F});  
  // we know that INRIA models are upside down so we set the up vector to 0,-1,0
  CameraManip.setLookat({0.0F, 0.0F, -2.0F}, {0.F, 0.F, 0.F}, {0.0F, -1.0F, 0.0F});

  //
  resetFrameInfo();
}

void GaussianSplatting::destroyScene()
{
  m_splatSet = {};
}

void GaussianSplatting::createPipeline()
{
  /*
  m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  */
  m_dset->initLayout();
  m_dset->initPool(1);

  const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                    sizeof(DH::PushConstant)};
  m_dset->initPipeLayout(1, &push_constant_ranges);

  // Writing to descriptors
  const VkDescriptorBufferInfo      dbi_unif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_dset->makeWrite(0, 0, &dbi_unif));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  {  //  create the rendering pipeline

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Creating the Pipeline
    nvvk::GraphicsPipelineState pstate;
    pstate.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // activates blending and set blend func
    pstate.setBlendAttachmentCount(1);  // 1 color attachment
    {
      VkPipelineColorBlendAttachmentState blend_state{};
      blend_state.blendEnable = VK_TRUE;
      blend_state.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      blend_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blend_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      pstate.setBlendAttachmentState(0, blend_state);
    }

    pstate.addBindingDescriptions({{0, sizeof(Vertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, pos))},  // Position
        //{1, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, uv))},     // UVCoord
    });

    pstate.addBindingDescriptions({{1, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_INSTANCE}});
    pstate.addAttributeDescriptions({
        {1, 1, VK_FORMAT_R32_UINT, 0},  //
    });

    // By default disable depth test for the pipeline
    pstate.depthStencilState.depthTestEnable = VK_FALSE;
    // The dynamic state is used to change the depth test state dynamically
    pstate.addDynamicStateEnable(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

    // create the pipeline
    nvvk::GraphicsPipelineGenerator pgen(m_device, m_dset->getPipeLayout(), prend_info, pstate);

#if USE_SLANG
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
    pgen.addShader(shaderModule, VK_SHADER_STAGE_VERTEX_BIT, "vertexMain");
    pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain");
#else
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, USE_GLSL ? "main" : "vertexMain");
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_GLSL ? "main" : "fragmentMain");
#endif

    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();
#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
  }
  {  // create the compute pipeline

    /*-- Creating the pipeline to run the compute shader -*/
    const VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                              .codeSize = sizeof(rank_comp_glsl),
                                              .pCode    = &rank_comp_glsl[0]};
    VkShaderModule                 compute{};
    vkCreateShaderModule(m_device, &createInfo, nullptr, &compute);

    auto pipelineLayout = m_dset->getPipeLayout();

    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = compute,
                .pName  = "main",
            },
        .layout = pipelineLayout,
    };
    vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_computePipeline);

    /*-- Clean up the shader module -*/
    vkDestroyShaderModule(m_device, compute, nullptr);
  }
}

void GaussianSplatting::destroyPipeline() {

  m_dset->deinitPool();
  m_dset->deinitLayout();
  
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  
}

void GaussianSplatting::createGbuffers(const glm::vec2& size)
{
  m_viewSize = size;
  m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                 VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                 m_colorFormat, m_depthFormat);
}

void GaussianSplatting::destroyGbuffers() {
  
  m_gBuffers.reset();

}

// to be cleaned
constexpr uint32_t MAX_ELEMENT_COUNT = 1 << 25;

void GaussianSplatting::createVkBuffers()
{
  // TODO: free all other buffers so createVKBuffers can be used on a scene reset
  nvvkhl::Application::submitResourceFree([vertices = m_vertices, indices = m_indices, alloc = m_alloc]() {
    alloc->destroy(const_cast<nvvk::Buffer&>(vertices));
    alloc->destroy(const_cast<nvvk::Buffer&>(indices));
  });

  const auto splatCount = m_splatSet.positions.size() / 3;

  // this has nothing to do here
  distArray.resize(splatCount);
  distArray2.resize(splatCount);
  gsIndex.resize(splatCount);
  sortGsIndex.resize(splatCount);

  // All this block for the GPU sorter
  {

    // fence
    //VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    //vkCreateFence(m_app->getDevice(), &fence_info, NULL, &m_fence);

    // timestamp query pool
    constexpr int         timestamp_count = 15;
    VkQueryPoolCreateInfo query_pool_info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_pool_info.queryType             = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_info.queryCount            = timestamp_count;
    vkCreateQueryPool(m_app->getDevice(), &query_pool_info, NULL, &m_queryPool);

    // sorter
    m_sorter                    = VK_NULL_HANDLE;
    m_sorterInfo                = {};
    m_sorterInfo.physicalDevice = m_app->getPhysicalDevice();
    m_sorterInfo.device         = m_app->getDevice();
    vrdxCreateSorter(&m_sorterInfo, &m_sorter);

    {  // Create some buffer for GPU sorting

      m_stagingHost = m_alloc->createBuffer((splatCount * 2 + 1) * sizeof(uint32_t),
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      m_keysDevice = m_alloc->createBuffer((splatCount * 2 + 1) * sizeof(uint32_t),
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      VrdxSorterStorageRequirements requirements;
      vrdxGetSorterKeyValueStorageRequirements(m_sorter, MAX_ELEMENT_COUNT, &requirements);

      m_storageDevice = m_alloc->createBuffer(requirements.size, requirements.usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
  }

  {  // create the splat index buffer used for CPU sorting

    m_splatIndicesHost = m_alloc->createBuffer(splatCount * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    m_splatIndicesDevice = m_alloc->createBuffer(splatCount * sizeof(uint32_t),
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  // Quad with UV coordinates
  const std::vector<uint16_t> indices = {0, 2, 1, 2, 0, 3};
  /*
	std::vector<float> vertices = {
		-1.0, -1.0, 0.0,
			-1.0, 1.0, 0.0,
			1.0, 1.0, 0.0,
			1.0, -1.0, 0.0
	};*/
  std::vector<Vertex> vertices(4);
  vertices[0] = {{-1.0F, -1.0F, 0.0F}};  //{0.0F, 0.0F} };
  vertices[1] = {{1.0F, -1.0F, 0.0F}};   //{1.0F, 0.0F} };
  vertices[2] = {{1.0F, 1.0F, 0.0F}};    //{1.0F, 1.0F} };
  vertices[3] = {{-1.0F, 1.0F, 0.0F}};   //{0.0F, 1.0F} };

  // parameters for DrawIndexIndirect
  // uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance
  // vkCmdDrawIndexed(cmd, 6, (uint32_t)splatCount, 0, 0, 0);
  const std::vector<uint32_t> indirect = {6, 0, 0, 0, 0};  // the second value will be set to visible_splat_count after culling

  //
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // create the buffer for indirect
  m_indirect = m_alloc->createBuffer(cmd, indirect, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
  m_dutil->DBG_NAME(m_indirect.buffer);

  // create the quad buffers
  m_vertices = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  m_indices  = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  m_dutil->DBG_NAME(m_vertices.buffer);
  m_dutil->DBG_NAME(m_indices.buffer);

  //
  m_app->submitAndWaitTempCmdBuffer(cmd);

  //
  m_frameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dutil->DBG_NAME(m_frameInfo.buffer);

  m_pixelBuffer = m_alloc->createBuffer(sizeof(float) * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void GaussianSplatting::destroyVkBuffers()
{

  m_alloc->destroy(m_indirect);
  m_alloc->destroy(m_vertices);
  m_alloc->destroy(m_indices);
  m_vertices = {};  // <- is this needed ?
  m_indices  = {};  // <- is this needed ?
  m_alloc->destroy(m_splatIndicesDevice);
  m_alloc->destroy(m_splatIndicesHost);

  m_alloc->destroy(m_stagingHost);
  m_alloc->destroy(m_keysDevice);
  m_alloc->destroy(m_storageDevice);

  m_alloc->destroy(m_frameInfo);
  m_alloc->destroy(m_pixelBuffer);

}

glm::ivec2 GaussianSplatting::computeDataTextureSize(int elementsPerTexel, int elementsPerSplat, int maxSplatCount)
{
  glm::ivec2 texSize(4096, 1024);
  while(texSize.x * texSize.y * elementsPerTexel < maxSplatCount * elementsPerSplat)
    texSize.y *= 2;
  return texSize;
};

void GaussianSplatting::create3dgsTextures(void)
{

  const int splatCount = m_splatSet.positions.size() / 3;

  // create a texture sampler using nearest filtering mode.
  VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler_info.magFilter  = VK_FILTER_NEAREST;
  sampler_info.minFilter  = VK_FILTER_NEAREST;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  // centers - TODO: pack as covariances not to waste alpha chanel - but compare performance (1 lookup vs 2 lookups due to packing)
  centersMapSize = computeDataTextureSize(3, 3, splatCount);
  std::vector<float> centers(splatCount * 4);  // init with positions
  for(int i = 0; i < splatCount; ++i)
  {
    for(int cmp = 0; cmp < 3; ++cmp)
    {
      centers[i * 4 + cmp] = m_splatSet.positions[i * 3 + cmp];
    }
  }

  centers.resize(centersMapSize.x * centersMapSize.y * 4);  // adds the padding

  // place the result in the dedicated texture map 
  m_centersMap = std::make_shared<SampleTexture>(m_app->getDevice(), m_app->getQueue(0).familyIndex, m_alloc.get());
  m_centersMap->create(centersMapSize.x, centersMapSize.y, centers.size() * sizeof(float), (void*)centers.data(),
                       VK_FORMAT_R32G32B32A32_SFLOAT);
  assert(m_centersMap->isValid());
  m_centersMap->setSampler(m_alloc->acquireSampler(sampler_info));  // sampler will be released by texture

  // SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at eatch frame
  colorsMapSize = computeDataTextureSize(4, 4, splatCount);
  std::vector<uint8_t> colors(colorsMapSize.x * colorsMapSize.y * 4);  // includes some padding
  for(auto splatIdx = 0; splatIdx < splatCount; ++splatIdx)
  {
    const auto  stride3 = splatIdx * 3;
    const auto  stride4 = splatIdx * 4;
    const float SH_C0   = 0.28209479177387814;
    colors[stride4 + 0] = glm::clamp(std::floor((0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 0]) * 255), 0.0f, 255.0f);
    colors[stride4 + 1] = glm::clamp(std::floor((0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 1]) * 255), 0.0f, 255.0f);
    colors[stride4 + 2] = glm::clamp(std::floor((0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 2]) * 255), 0.0f, 255.0f);
    colors[stride4 + 3] = glm::clamp(std::floor((1.0f / (1.0f + std::exp(-m_splatSet.opacity[splatIdx]))) * 255), 0.0f, 255.0f);
  }
  // place the result in the dedicated texture map 
  m_colorsMap = std::make_shared<SampleTexture>(m_app->getDevice(), m_app->getQueue(0).familyIndex, m_alloc.get());
  m_colorsMap->create(colorsMapSize.x, colorsMapSize.y, colors.size(), (void*)colors.data(), VK_FORMAT_R8G8B8A8_UNORM);
  assert(m_colorsMap->isValid());
  m_colorsMap->setSampler(m_alloc->acquireSampler(sampler_info));

  // covariances
  covariancesMapSize = computeDataTextureSize(4, 6, splatCount);
  std::vector<float> covariances(covariancesMapSize.x * covariancesMapSize.y * 4, 0.0f);
  glm::vec3          scale;
  glm::quat          rotation;
  for(auto splatIdx = 0; splatIdx < splatCount; ++splatIdx)
  {
    const auto stride3 = splatIdx * 3;
    const auto stride4 = splatIdx * 4;
    const auto stride6 = splatIdx * 6;
    scale.x            = std::exp(m_splatSet.scale[stride3 + 0]);
    scale.y            = std::exp(m_splatSet.scale[stride3 + 1]);
    scale.z            = std::exp(m_splatSet.scale[stride3 + 2]);

    rotation.x = m_splatSet.rotation[stride4 + 1];
    rotation.y = m_splatSet.rotation[stride4 + 2];
    rotation.z = m_splatSet.rotation[stride4 + 3];
    rotation.w = m_splatSet.rotation[stride4 + 0];
    rotation   = glm::normalize(rotation);

    // computes the covariance
    const glm::mat3 scaleMatrix           = glm::mat3(glm::scale(scale));
    const glm::mat3 rotationMatrix        = glm::mat3_cast(rotation);  // where rotation is a quaternion
    const glm::mat3 covarianceMatrix      = rotationMatrix * scaleMatrix;
    glm::mat3       transformedCovariance = covarianceMatrix * glm::transpose(covarianceMatrix);

    covariances[stride6 + 0] = glm::value_ptr(transformedCovariance)[0];  
    covariances[stride6 + 1] = glm::value_ptr(transformedCovariance)[3];  
    covariances[stride6 + 2] = glm::value_ptr(transformedCovariance)[6];  

    covariances[stride6 + 3] = glm::value_ptr(transformedCovariance)[4];  
    covariances[stride6 + 4] = glm::value_ptr(transformedCovariance)[7];  
    covariances[stride6 + 5] = glm::value_ptr(transformedCovariance)[8];  
  }

  // place the result in the dedicated texture map 
  m_covariancesMap = std::make_shared<SampleTexture>(m_app->getDevice(), m_app->getQueue(0).familyIndex, m_alloc.get());
  m_covariancesMap->create(covariancesMapSize.x, covariancesMapSize.y, covariances.size() * sizeof(float),
                           (void*)covariances.data(), VK_FORMAT_R32G32B32A32_SFLOAT);
  assert(m_covariancesMap->isValid());
  m_covariancesMap->setSampler(m_alloc->acquireSampler(sampler_info));

  // Prepare the spherical harmonics
  const int sphericalHarmonicsElementsPerTexel       = 4;
  const int totalSphericalHarmonicsComponentCount    = m_splatSet.f_rest.size() / splatCount;
  const int sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
  // find the maximum SH degree stored in the file
  int sphericalHarmonicsDegree = 0;
  if(sphericalHarmonicsCoefficientsPerChannel >= 3)
    sphericalHarmonicsDegree = 1;
  if(sphericalHarmonicsCoefficientsPerChannel >= 8)
    sphericalHarmonicsDegree = 2;

  // add some padding at each splat if needed for easy texture lookups
  const int sphericalHarmonicsComponentCount = (sphericalHarmonicsDegree == 1) ? 9 : ((sphericalHarmonicsDegree == 2) ? 24 : 0);
  int paddedSphericalHarmonicsComponentCount = sphericalHarmonicsComponentCount;
  if(paddedSphericalHarmonicsComponentCount % 2 != 0)
    paddedSphericalHarmonicsComponentCount++;

  //
  sphericalHarmonicsMapSize =
      computeDataTextureSize(sphericalHarmonicsElementsPerTexel, paddedSphericalHarmonicsComponentCount, splatCount);

  std::vector<float> paddedSHArray(sphericalHarmonicsMapSize.x * sphericalHarmonicsMapSize.y * sphericalHarmonicsElementsPerTexel, 0.0f);

  for(auto splatIdx = 0; splatIdx < splatCount; ++splatIdx)
  {
    const auto srcBase   = totalSphericalHarmonicsComponentCount * splatIdx;
    const auto destBase  = paddedSphericalHarmonicsComponentCount * splatIdx;
    int        dstOffset = 0;
    // degree 1, three coefs per component
    for(auto i = 0; i < 3; i++)
    {
      for(auto rgb = 0; rgb < 3; rgb++)
      {
        const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + i);
        const auto dstIndex = destBase + dstOffset++;  // inc after add

        paddedSHArray[dstIndex] = m_splatSet.f_rest[srcIndex];
      }
    }

    // degree 2, five coefs per component
    for(auto i = 0; i < 5; i++)
    {
      for(auto rgb = 0; rgb < 3; rgb++)
      {
        const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
        const auto dstIndex = destBase + dstOffset++;  // inc after add

        paddedSHArray[dstIndex] = m_splatSet.f_rest[srcIndex];
      }
    }
    /*
    // degree 3 TODO
    for(auto i = 0; i < 7; i++)
    {
      for(auto rgb = 0; rgb < 3; rgb++)
      {
        const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 8 + i);
        const auto dstIndex = destBase + 24 + (i * 3 + rgb);
        
        paddedSHArray[dstIndex] = m_splatSet.f_rest[srcIndex];
      }
    }
    */
  }

  // place the result in the dedicated texture map 
  m_sphericalHarmonicsMap = std::make_shared<SampleTexture>(m_app->getDevice(), m_app->getQueue(0).familyIndex, m_alloc.get());
  m_sphericalHarmonicsMap->create(sphericalHarmonicsMapSize.x, sphericalHarmonicsMapSize.y, paddedSHArray.size() * sizeof(float),
                                  (void*)paddedSHArray.data(), VK_FORMAT_R32G32B32A32_SFLOAT);
  assert(m_sphericalHarmonicsMap->isValid());
  m_sphericalHarmonicsMap->setSampler(m_alloc->acquireSampler(sampler_info));

  // updateTexture
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_dset->makeWrite(0, 1, &m_centersMap->descriptor()));
  writes.emplace_back(m_dset->makeWrite(0, 2, &m_colorsMap->descriptor()));
  writes.emplace_back(m_dset->makeWrite(0, 3, &m_covariancesMap->descriptor()));
  writes.emplace_back(m_dset->makeWrite(0, 4, &m_sphericalHarmonicsMap->descriptor()));
  const VkDescriptorBufferInfo keys_desc{m_keysDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, 5, &keys_desc));
  const VkDescriptorBufferInfo indirect_desc{m_indirect.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, 6, &indirect_desc));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void GaussianSplatting::destroy3dgsTextures()
{
  // destructors will invoke destroy on next frame
  m_centersMap.reset();
  m_colorsMap.reset();
  m_covariancesMap.reset();
  m_sphericalHarmonicsMap.reset();
}

bool GaussianSplatting::loadPly(std::string filename, SplatSet& output)
{
  std::unique_ptr<std::istream> file_stream;
  file_stream.reset(new std::ifstream(filename.c_str(), std::ios::binary));
  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  std::shared_ptr<tinyply::PlyData> _vertices, _normals, _colors, _colorsRGBA, _texcoords, _faces, _tristrip;

  // The header information can be used to programmatically extract properties on elements
  // known to exist in the header prior to reading the data. For brevity of this sample, properties
  // like vertex position are hard-coded:
  try
  {
    _vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  }
  catch(const std::exception& e)
  {
    std::cerr << "Error: missing vertex positions. " << e.what() << std::endl;
  }
  try
  {
    _normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _colorsRGBA = file.request_properties_from_element("vertex", {"red", "green", "blue", "alpha"});
  }
  catch(const std::exception)
  {
  }

  if(!_colorsRGBA)
  {
    try
    {
      _colorsRGBA = file.request_properties_from_element("vertex", {"r", "g", "b", "a"});
    }
    catch(const std::exception)
    {
    }
  }
  try
  {
    _colors = file.request_properties_from_element("vertex", {"red", "green", "blue"});
  }
  catch(const std::exception)
  {
  }
  if(!_colors)
  {
    try
    {
      _colors = file.request_properties_from_element("vertex", {"r", "g", "b"});
    }
    catch(const std::exception)
    {
    }
  }
  try
  {
    _texcoords = file.request_properties_from_element("vertex", {"u", "v"});
  }
  catch(const std::exception)
  {
  }

  // 3DGS specifics
  std::shared_ptr<tinyply::PlyData> _f_dc, _f_rest, _opacity, _scale, _rotation;
  try
  {
    _f_dc = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _f_rest = file.request_properties_from_element(
        "vertex",
        {"f_rest_0",  "f_rest_1",  "f_rest_2",  "f_rest_3",  "f_rest_4",  "f_rest_5",  "f_rest_6",  "f_rest_7",
         "f_rest_8",  "f_rest_9",  "f_rest_10", "f_rest_11", "f_rest_12", "f_rest_13", "f_rest_14", "f_rest_15",
         "f_rest_16", "f_rest_17", "f_rest_18", "f_rest_19", "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23",
         "f_rest_24", "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29", "f_rest_30", "f_rest_31",
         "f_rest_32", "f_rest_33", "f_rest_34", "f_rest_35", "f_rest_36", "f_rest_37", "f_rest_38", "f_rest_39",
         "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _opacity = file.request_properties_from_element("vertex", {"opacity"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _scale = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _rotation = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
  }
  catch(const std::exception)
  {
  }

  //
  file.read(*file_stream);

  // now feed the data to the frame structure
  if(_vertices)
  {
    const size_t numVerticesBytes = _vertices->buffer.size_bytes();
    output.positions.resize(_vertices->count * 3);
    std::memcpy(output.positions.data(), _vertices->buffer.get(), numVerticesBytes);
  }
  if(_normals)
  {
    const size_t numNormalsBytes = _normals->buffer.size_bytes();
    output.normals.resize(_normals->count * 3);
    std::memcpy(output.normals.data(), _normals->buffer.get(), numNormalsBytes);
  }

  // 3DGS per vertex infos

  if(_f_dc && _f_rest && _opacity && _scale && _rotation)
  {
    const size_t numFDcBytes      = _f_dc->buffer.size_bytes();
    const size_t numFRestBytes    = _f_rest->buffer.size_bytes();
    const size_t numOpacityBytes  = _opacity->buffer.size_bytes();
    const size_t numScaleBytes    = _scale->buffer.size_bytes();
    const size_t numRotationBytes = _rotation->buffer.size_bytes();
    output.f_dc.resize(_f_dc->count * 3);
    output.f_rest.resize(_f_rest->count * 45);
    output.opacity.resize(_opacity->count);
    output.scale.resize(_scale->count * 3);
    output.rotation.resize(_rotation->count * 4);
    std::memcpy(output.f_dc.data(), _f_dc->buffer.get(), numFDcBytes);
    std::memcpy(output.f_rest.data(), _f_rest->buffer.get(), numFRestBytes);
    std::memcpy(output.opacity.data(), _opacity->buffer.get(), numOpacityBytes);
    std::memcpy(output.scale.data(), _scale->buffer.get(), numScaleBytes);
    std::memcpy(output.rotation.data(), _rotation->buffer.get(), numRotationBytes);
  }

  return true;
}