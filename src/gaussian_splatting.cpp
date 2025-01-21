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

// Vulkan Memory Allocator 
#define VMA_IMPLEMENTATION
// ImGUI ImVec maths
#define IMGUI_DEFINE_MATH_OPERATORS

#include <gaussian_splatting.h>

// shaders code

#if USE_SLANG
#include "_autogen/raster_slang.h"
#else
#include "_autogen/rank.comp.glsl.h"
#include "_autogen/raster.frag.glsl.h"
#include "_autogen/raster.vert.glsl.h"
#include "_autogen/raster.mesh.glsl.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(rank_comp_glsl), std::end(rank_comp_glsl)};
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert_glsl), std::end(raster_vert_glsl)};
const auto& mesh_shd = std::vector<uint32_t>{std::begin(raster_mesh_glsl), std::end(raster_mesh_glsl)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag_glsl), std::end(raster_frag_glsl)};
#endif

//
void GaussianSplatting::onAttach(nvvkhl::Application* app)
{
  initGui();

  // starts the loader
  m_plyLoader.initialize();
  
  //
  m_app    = app;
  m_device = m_app->getDevice();

  // Debug utility
  m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);
  // Allocator
  m_alloc = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
      .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = app->getPhysicalDevice(),
      .device         = app->getDevice(),
      .instance       = app->getInstance(),
  });
  //
  m_dset        = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
  m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
  // create descriptor bindings
  m_dset->addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_CENTERS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_COLORS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_COVARIANCES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_SH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_DISTANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_INDICES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_INDIRECT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

  m_dset->addBinding(BINDING_CENTERS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_COLORS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_COVARIANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_SH_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
};

void GaussianSplatting::onDetach()
{
  // stops the loader
  m_plyLoader.shutdown();
  // stop the sorting thread
  std::unique_lock<std::mutex> lock(mutex);
  sortExit = true;
  cond_var.notify_one();
  lock.unlock();
  sortingThread.join();
  // release resources
  reset();
  destroyGbuffers();
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

  //
  const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

  // for 0 if not ready so the rendering does not 
  // touch the splat set while loading
  size_t splatCount = 0;
  if(m_plyLoader.getStatus() == PlyAsyncLoader::Status::READY)
  {
    splatCount = m_splatSet.size();
  }
  
  const float aspect_ratio = m_viewSize.x / m_viewSize.y;
  glm::vec3   eye;
  glm::vec3   center;
  glm::vec3   up;
  CameraManip.getLookat(eye, center, up);

  if(splatCount)
  {
    {
      auto timerSection = m_profiler->timeRecurring("UBO update", cmd);

      // Update frame parameters uniform buffer
      // some attributes of frameInfo were set by the user interface
      const glm::vec2& clip = CameraManip.getClipPlanes();
      m_frameInfo.splatCount  = splatCount;
      m_frameInfo.viewMatrix = CameraManip.getMatrix();
      m_frameInfo.projectionMatrix = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, clip.x, clip.y);
      // OpenGL (0,0) is bottom left, Vulkan (0,0) is top left, and glm::perspectiveRH_ZO is for OpenGL so we mirror on y
      m_frameInfo.projectionMatrix[1][1] *= -1;
      m_frameInfo.cameraPosition       = eye;
      float       devicePixelRatio     = 1.0;
      const float focalLengthX         = m_frameInfo.projectionMatrix[0][0] * 0.5f * devicePixelRatio * m_viewSize.x;
      const float focalLengthY         = m_frameInfo.projectionMatrix[1][1] * 0.5f * devicePixelRatio * m_viewSize.y;
      const bool  isOrthographicCamera = false;
      const float focalMultiplier      = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
      const float focalAdjustment      = focalMultiplier;  //  this.focalAdjustment* focalMultiplier;
      m_frameInfo.orthoZoom              = 1.0f;
      m_frameInfo.orthographicMode     = 0;  // disabled (uses perspective) TODO: activate support for orthographic
      m_frameInfo.viewport               = glm::vec2(m_viewSize.x * devicePixelRatio, m_viewSize.x * devicePixelRatio);
      m_frameInfo.basisViewport          = glm::vec2(1.0f / m_viewSize.x, 1.0f / m_viewSize.y);
      m_frameInfo.focal                        = glm::vec2(focalLengthX, focalLengthY);
      m_frameInfo.inverseFocalAdjustment = 1.0f / focalAdjustment;

      vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(DH::FrameInfo), &m_frameInfo);

      // sync with end of copy to device
      VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bmb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      bmb.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
      bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.buffer              = m_frameInfoBuffer.buffer;
      bmb.size                = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
                           | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
                           VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bmb, 0, nullptr);
    }
    if(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX)
    {
      // upload CPU sorted indices to the GPU if needed
      bool newIndexAvailable = false;

      if(!m_frameInfo.opacityGaussianDisabled)
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
        // however, using the same mechanism allows to use exactly the same shader
        // so if splatting/blending is off we provide an ordered table of indices
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

      {  // upload to GPU is needed
        auto timerSection = m_profiler->timeRecurring("Copy indices to GPU", cmd);

        if(newIndexAvailable)
        { 
          // Prepare buffer on host using sorted indices
          uint32_t* hostBuffer = static_cast<uint32_t*>(m_alloc->map(m_splatIndicesHost));
          memcpy(hostBuffer, gsIndex.data(), gsIndex.size() * sizeof(uint32_t));
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
          vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT 
                                | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
                               VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bmb, 0, nullptr);
        }
      }
    }

    // when GPU sorting, we sort at each frame, all buffer in device memory, no copy
    if(m_frameInfo.sortingMethod == SORTING_GPU_SYNC_RADIX)
    {
      { // reset the draw indirect parameters and counters, will be updated by compute shader
        const IndirectParams drawIndexedIndirectParams;
        vkCmdUpdateBuffer(cmd, m_indirect.buffer, 0, sizeof(IndirectParams), (void*)&drawIndexedIndirectParams);

        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT ;
        barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0, 1,
                             &barrier,
                             0, NULL, 0, NULL);
      }

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

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT 
                             ,0, 1, &barrier, 0, NULL, 0, NULL);
      }

      // invoke the radix sort from vrdx lib
      {
        auto timerSection = m_profiler->timeRecurring("GPU Sort", cmd);

        vrdxCmdSortKeyValueIndirect(cmd, m_sorter, splatCount, 
          m_indirect.buffer, sizeof(uint32_t), 
          m_splatDistancesDevice.buffer, 0,
          m_splatIndicesDevice.buffer, 0, m_vrdxStorageDevice.buffer, 0, 0, 0);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, 0, 1,
                             &barrier, 0, NULL, 0, NULL);
      }
    }
  }
  // Drawing the primitives in the G-Buffer
  {
    auto timerSection = m_profiler->timeRecurring("Rendering", cmd);

    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // let's throw some pixels !!
    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);
    if(splatCount)
    {
      if(m_selectedPipeline == PIPELINE_VERT)
      { // Pipeline using vertex shader

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);
        // overrides the pipeline setup for depth test/write
        vkCmdSetDepthTestEnable(cmd, (VkBool32)m_frameInfo.opacityGaussianDisabled);

        /* we do not use push_constant, everything passes though the frameInfo unifrom buffer
          // transfo/color unused for the time beeing, could transform the whole 3DGS model
          // if used, could also be placed in the FrameInfo or all the frameInfo placed in push_constant
          m_pushConst.transfo = glm::mat4(1.0);                 // identity
          m_pushConst.color   = glm::vec4(0.5, 0.5, 0.5, 1.0);  //
          vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                             sizeof(DH::PushConstant), &m_pushConst);
          */

        // display the quad as many times as we have visible splats

        const VkDeviceSize offsets{0};
        vkCmdBindIndexBuffer(cmd, m_quadIndices.buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindVertexBuffers(cmd, 0, 1, &m_quadVertices.buffer, &offsets);
        if(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX)
        {
          vkCmdBindVertexBuffers(cmd, 1, 1, &m_splatIndicesDevice.buffer, &offsets);
          vkCmdDrawIndexed(cmd, 6, (uint32_t)splatCount, 0, 0, 0);
        }
        else
        {
          vkCmdBindVertexBuffers(cmd, 1, 1, &m_splatIndicesDevice.buffer, &offsets);
          vkCmdDrawIndexedIndirect(cmd, m_indirect.buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
        }
       
      }
      else
      { // Pipeline using mesh shader

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineMesh);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);
        // overrides the pipeline setup for depth test/write
        vkCmdSetDepthTestEnable(cmd, (VkBool32)m_frameInfo.opacityGaussianDisabled);
        if(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX)
        {
          // run the workgroups
          vkCmdDrawMeshTasksEXT(cmd, (m_frameInfo.splatCount + 31) / 32, 1, 1);
        }
        else
        {
          // run the workgroups
          vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer, 5 * sizeof(uint32_t), 1, sizeof(VkDrawIndexedIndirectCommand));
        }
        
      }
    }
    
    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  // read back m_indirect for statistics display in the UI
  if(true && (m_indirectHost.buffer != VK_NULL_HANDLE))
  {
    auto timerSection = m_profiler->timeRecurring("Indirect readback", cmd);

    // copy from device to host buffer
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = sizeof(IndirectParams)};
    vkCmdCopyBuffer(cmd, m_indirect.buffer, m_indirectHost.buffer, 1, &bc);

    // sync with end of copy to host, GPU timeline, 
    // value will be available between now and next frame
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer              = m_indirectHost.buffer;
    bmb.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bmb, 0, nullptr);

    // copy to main memory (this copy the value from last frame, CPU timeline)
    uint32_t* hostBuffer = static_cast<uint32_t*>(m_alloc->map(m_indirectHost));
    std::memcpy((void*)&m_indirectReadback, (void*)hostBuffer, sizeof(IndirectParams));
    m_alloc->unmap(m_indirectHost);
  }
  
  // update rendering memory statistics
  if(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX)
  {
    m_renderMemoryStats.hostAllocIndices = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.hostAllocDistances = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.allocIndices   = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedIndices    = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.allocDistances = 0;
    m_renderMemoryStats.usedDistances  = 0;
    m_renderMemoryStats.usedIndirect   = 0;
  }
  else
  {
    m_renderMemoryStats.hostAllocDistances = 0;
    m_renderMemoryStats.hostAllocIndices   = 0;
    m_renderMemoryStats.allocDistances = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedDistances  = m_indirectReadback.instanceCount * sizeof(uint32_t);
    m_renderMemoryStats.allocIndices   = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedIndices    = m_indirectReadback.instanceCount * sizeof(uint32_t);
    if(m_selectedPipeline == PIPELINE_VERT)
    {
      m_renderMemoryStats.usedIndirect = 5 * sizeof(uint32_t);
    }
    else
    {
      m_renderMemoryStats.usedIndirect = sizeof(IndirectParams);
    }
  }
  m_renderMemoryStats.usedUboFrameInfo = sizeof(DH::FrameInfo);

  m_renderMemoryStats.hostTotal = m_renderMemoryStats.hostAllocIndices + m_renderMemoryStats.hostAllocDistances
       + m_renderMemoryStats.usedUboFrameInfo;

  uint32_t vrdxSize = m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal;

  m_renderMemoryStats.deviceUsedTotal = m_renderMemoryStats.usedIndices + m_renderMemoryStats.usedDistances + vrdxSize
                                        + m_renderMemoryStats.usedIndirect + m_renderMemoryStats.usedUboFrameInfo;

  m_renderMemoryStats.deviceAllocTotal = m_renderMemoryStats.allocIndices + m_renderMemoryStats.allocDistances + vrdxSize
                                         + m_renderMemoryStats.usedIndirect
                                        + m_renderMemoryStats.usedUboFrameInfo;
}

void GaussianSplatting::sortingThreadFunc(void)
{
  while(true)
  {
    // wait until a sort or an exit is requested
    std::unique_lock<std::mutex> lock(mutex);
    m_cpuSortStatusUi = "Idled";
    cond_var.wait(lock, [this] { return sortStart || sortExit; });
    const bool exitSortCopy = sortExit;
    lock.unlock();
    if(exitSortCopy)
      return;
    m_cpuSortStatusUi = "Sorting";

    // There is no reason this arrives but we test just in case
    assert(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX);

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
                    size_t i = &val - &tmpArray[0];
                    const auto pos = &(splatSet.positions[i * 3]);
                    // distance to plane
                    const float dist = std::abs(plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3]) * divider;
                    tmpArray[i].first = dist;
                    tmpArray[i].second = i;
                  });
#endif

  auto time1 = std::chrono::high_resolution_clock::now();
  m_distTime = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - startTime).count();

  // comparison function working on the data <dist,idex>
  auto compare = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };

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

void GaussianSplatting::reset()
{
  vkDeviceWaitIdle(m_device);
  destroyScene();
  destroyDataTextures();
  destroyDataBuffers();
  destroyVkBuffers();
  destroyPipeline();
}

void GaussianSplatting::destroyScene()
{
  m_splatSet = {};
  m_loadedSceneFilename = "";
}

void GaussianSplatting::createPipeline()
{

  m_dset->initLayout();
  m_dset->initPool(1);
  const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_MESH_BIT_EXT,
                                                    0,
                                                    sizeof(DH::PushConstant)};
  m_dset->initPipeLayout(1, &push_constant_ranges);

  // Write descriptors for the buffers
  std::vector<VkWriteDescriptorSet> writes;
  // add texture maps
  writes.emplace_back(m_dset->makeWrite(0, BINDING_CENTERS_TEXTURE, &m_centersMap->descriptor()));
  writes.emplace_back(m_dset->makeWrite(0, BINDING_COLORS_TEXTURE, &m_colorsMap->descriptor()));
  writes.emplace_back(m_dset->makeWrite(0, BINDING_COVARIANCES_TEXTURE, &m_covariancesMap->descriptor()));
  writes.emplace_back(m_dset->makeWrite(0, BINDING_SH_TEXTURE, &m_sphericalHarmonicsMap->descriptor()));
  // add buffers
  const VkDescriptorBufferInfo      dbi_frameInfo{m_frameInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_FRAME_INFO_UBO, &dbi_frameInfo));
  const VkDescriptorBufferInfo keys_desc{m_splatDistancesDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_DISTANCES_BUFFER, &keys_desc));
  const VkDescriptorBufferInfo cpuKeys_desc{m_splatIndicesDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_INDICES_BUFFER, &cpuKeys_desc));
  const VkDescriptorBufferInfo indirect_desc{m_indirect.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_INDIRECT_BUFFER, &indirect_desc));

  const VkDescriptorBufferInfo centers_desc{m_centersDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_CENTERS_BUFFER, &centers_desc));
  const VkDescriptorBufferInfo colors_desc{m_colorsDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_COLORS_BUFFER, &colors_desc));
  const VkDescriptorBufferInfo covariances_desc{m_covariancesDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_COVARIANCES_BUFFER, &covariances_desc));
  const VkDescriptorBufferInfo sh_desc{m_sphericalHarmonicsDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_SH_BUFFER, &sh_desc));
  // write
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Create the pipeline to run the compute shader for distance & culling
  {
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

    // Shader module is not needed anymore
    vkDestroyShaderModule(m_device, compute, nullptr);
  }
  // Create the two rasterization pipelines
  {  

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
    
    // By default disable depth test for the pipeline
    pstate.depthStencilState.depthTestEnable = VK_FALSE;
    // The dynamic state is used to change the depth test state dynamically
    pstate.addDynamicStateEnable(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

    // create the pipeline that uses mesh shaders
    {
      nvvk::GraphicsPipelineGenerator pgen(m_device, m_dset->getPipeLayout(), prend_info, pstate);

#if USE_SLANG
      VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
      // TODO: what is the name foe mesh shader main if not GLSL ?
      pgen.addShader(shaderModule, VK_SHADER_STAGE_MESH_BIT_EXT, "meshMain");
      pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain");
#else
      // TODO: what is the name foe mesh shader main if not GLSL ?
      pgen.addShader(mesh_shd, VK_SHADER_STAGE_MESH_BIT_EXT, USE_GLSL ? "main" : "meshMain");
      pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_GLSL ? "main" : "fragmentMain");
#endif
      m_graphicsPipelineMesh = pgen.createPipeline();
      m_dutil->setObjectName(m_graphicsPipelineMesh, "PipelineMeshShader");
      pgen.clearShaders();
#if USE_SLANG
      vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
    }

    // create the pipeline that uses vertex shaders
    {
      // add vertex attributes descriptions (only in vertex shader mode)
      const auto POS_BINDING = 0;
      const auto IDX_BINDING = 1;
      
      pstate.addBindingDescriptions({{POS_BINDING, 3 * sizeof(float)}});  // 3 component per vertex position
      pstate.addAttributeDescriptions({{0, POS_BINDING, VK_FORMAT_R32G32B32_SFLOAT, 0}});

      pstate.addBindingDescriptions({{IDX_BINDING, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_INSTANCE}});
      pstate.addAttributeDescriptions({{1, IDX_BINDING, VK_FORMAT_R32_UINT, 0}});

      //
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
      m_dutil->setObjectName(m_graphicsPipeline, "PipelineVertexShader");
      pgen.clearShaders();
#if USE_SLANG
      vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
    }

  }
}

void GaussianSplatting::destroyPipeline()
{

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

void GaussianSplatting::destroyGbuffers()
{

  m_gBuffers.reset();
}

void GaussianSplatting::createVkBuffers()
{
  /* TODO JEM why putting that in a submitResourceFree
  // TODO: free all other buffers so createVKBuffers can be used on a scene reset
  nvvkhl::Application::submitResourceFree([vertices = m_quadVertices, indices = m_quadIndices, alloc = m_alloc]() {
    alloc->destroy(const_cast<nvvk::Buffer&>(vertices));
    alloc->destroy(const_cast<nvvk::Buffer&>(indices));
  });
  */

  const auto splatCount = m_splatSet.size();

  // TODO: this has nothing to do here, check where to put this
  distArray.resize(splatCount);
  gsIndex.resize(splatCount);
  sortGsIndex.resize(splatCount);

  // All this block for the sorting
  {
    // Vrdx sorter
    m_sorter                    = VK_NULL_HANDLE;
    m_sorterInfo                = {};
    m_sorterInfo.physicalDevice = m_app->getPhysicalDevice();
    m_sorterInfo.device         = m_app->getDevice();
    vrdxCreateSorter(&m_sorterInfo, &m_sorter);

    {  // Create some buffer for GPU and/or CPU sorting

      const VkDeviceSize bufferSize = splatCount * sizeof(uint32_t);

      m_splatIndicesHost = m_alloc->createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

      m_splatIndicesDevice =
          m_alloc->createBuffer(bufferSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

      m_splatDistancesDevice =
          m_alloc->createBuffer(bufferSize,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                  
      VrdxSorterStorageRequirements requirements;
      // vrdxGetSorterKeyValueStorageRequirements(m_sorter, MAX_ELEMENT_COUNT, &requirements);
      vrdxGetSorterKeyValueStorageRequirements(m_sorter, splatCount, &requirements);
      m_vrdxStorageDevice = m_alloc->createBuffer(requirements.size, requirements.usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      m_renderMemoryStats.allocVdrxInternal = requirements.size; // for stats reporting only

      // generate debug information for buffers
      m_dutil->DBG_NAME(m_splatIndicesHost.buffer);
      m_dutil->DBG_NAME(m_splatIndicesDevice.buffer);
      m_dutil->DBG_NAME(m_splatDistancesDevice.buffer);
      m_dutil->DBG_NAME(m_vrdxStorageDevice.buffer);
    }
  }

  // create the buffer for indirect parameters
  m_indirect = m_alloc->createBuffer(sizeof(IndirectParams),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                         | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

  // for statistics readback
  m_indirectHost = m_alloc->createBuffer(sizeof(IndirectParams),
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  m_dutil->DBG_NAME(m_indirect.buffer);
  m_dutil->DBG_NAME(m_indirectHost.buffer);

  //
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // The Quad 
  const std::vector<uint16_t> indices  = {0, 2, 1, 2, 0, 3};
  const std::vector<float>    vertices = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};

  // create the quad buffers
  m_quadVertices = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  m_quadIndices  = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  m_dutil->DBG_NAME(m_quadVertices.buffer);
  m_dutil->DBG_NAME(m_quadIndices.buffer);

  //
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Uniform buffer
  m_frameInfoBuffer = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dutil->DBG_NAME(m_frameInfoBuffer.buffer);

  // Frame buffer
  m_pixelBuffer = m_alloc->createBuffer(sizeof(float) * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

}

void GaussianSplatting::destroyVkBuffers()
{
  if(m_sorter != VK_NULL_HANDLE ) 
    vrdxDestroySorter(m_sorter);

  m_alloc->destroy(m_quadVertices);
  m_alloc->destroy(m_quadIndices);

  m_alloc->destroy(m_indirect);
  m_alloc->destroy(m_indirectHost);
    
  m_alloc->destroy(m_splatDistancesDevice);
  m_alloc->destroy(m_splatIndicesDevice);
  m_alloc->destroy(m_splatIndicesHost);
  m_alloc->destroy(m_vrdxStorageDevice);

  m_alloc->destroy(m_frameInfoBuffer);
  m_alloc->destroy(m_pixelBuffer);
}

void GaussianSplatting::createDataTextures(void)
{
  const int splatCount = m_splatSet.positions.size() / 3;

  // create a texture sampler using nearest filtering mode.
  VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler_info.magFilter  = VK_FILTER_NEAREST;
  sampler_info.minFilter  = VK_FILTER_NEAREST;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  // centers (3 components but texture map is only allowed with 4 components)
  // TODO: May pack as done for covariances not to waste alpha chanel ? but must 
  // compare performance (1 lookup vs 2 lookups due to packing)
  glm::vec2          centersMapSize = computeDataTextureSize(3, 3, splatCount);
  std::vector<float> centers(centersMapSize.x * centersMapSize.y * 4);  // includes some padding and unused w channel
  for(int i = 0; i < splatCount; ++i)
  {
    // we skip the alpha channel that is left undefined and not used in the shader
    for(int cmp = 0; cmp < 3; ++cmp)
    {
      centers[i * 4 + cmp] = m_splatSet.positions[i * 3 + cmp];
    }
  }
  // place the result in the dedicated texture map
  m_centersMap = std::make_shared<SampleTexture>(m_app->getDevice(), m_app->getQueue(0).familyIndex, m_alloc.get());
  m_centersMap->create(centersMapSize.x, centersMapSize.y, centers.size() * sizeof(float), (void*)centers.data(),
                       VK_FORMAT_R32G32B32A32_SFLOAT);
  assert(m_centersMap->isValid());
  m_centersMap->setSampler(m_alloc->acquireSampler(sampler_info));  // sampler will be released by texture
  // memory statistics
  m_modelMemoryStats.srcCenters  = splatCount * 3 * sizeof(float);
  m_modelMemoryStats.odevCenters = splatCount * 3 * sizeof(float); // no compression or quantization yet
  m_modelMemoryStats.devCenters  = centersMapSize.x * centersMapSize.y * 4 * sizeof(float);

  // covariances
  glm::vec2          covariancesMapSize = computeDataTextureSize(4, 6, splatCount);
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
  // memory statistics
  m_modelMemoryStats.srcCov  = ( splatCount * ( 4 + 3 ) ) * sizeof(float);
  m_modelMemoryStats.odevCov = splatCount * 6 * sizeof(float);  // covariance takes less space than rotation + scale
  m_modelMemoryStats.devCov  = covariancesMapSize.x * covariancesMapSize.y * 4 * sizeof(float);

  // SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  glm::vec2            colorsMapSize = computeDataTextureSize(4, 4, splatCount);
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
  // memory statistics
  m_modelMemoryStats.srcSh0  = splatCount * 4 * sizeof(float);
  m_modelMemoryStats.odevSh0 = splatCount * 4 * sizeof(float);  // no compression or quantization yet
  m_modelMemoryStats.devSh0  = colorsMapSize.x * colorsMapSize.y * 4 * sizeof(float);

  // Prepare the spherical harmonics of degree 1 to 3
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
  glm::vec2 sphericalHarmonicsMapSize =
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
  // memory statistics
  m_modelMemoryStats.srcShOther  = m_splatSet.f_rest.size() * sizeof(float);
  m_modelMemoryStats.odevShOther = splatCount * 8 * 3 * sizeof(float);  // we only use Sh1 and SH2 for now
  m_modelMemoryStats.devShOther =  
      sphericalHarmonicsMapSize.x * sphericalHarmonicsMapSize.y * sphericalHarmonicsElementsPerTexel * sizeof(float);

  // update statistics totals
  m_modelMemoryStats.srcShAll = m_modelMemoryStats.srcSh0 + m_modelMemoryStats.srcShOther;
  m_modelMemoryStats.odevShAll = m_modelMemoryStats.odevSh0 + m_modelMemoryStats.odevShOther;
  m_modelMemoryStats.devShAll  = m_modelMemoryStats.devSh0 + m_modelMemoryStats.devShOther;

  m_modelMemoryStats.srcAll = m_modelMemoryStats.srcCenters + m_modelMemoryStats.srcCov + m_modelMemoryStats.srcSh0 + m_modelMemoryStats.srcShOther;
  m_modelMemoryStats.odevAll = m_modelMemoryStats.odevCenters + m_modelMemoryStats.odevCov + m_modelMemoryStats.odevSh0 + m_modelMemoryStats.odevShOther;
  m_modelMemoryStats.devAll = m_modelMemoryStats.devCenters + m_modelMemoryStats.devCov + m_modelMemoryStats.devSh0 + m_modelMemoryStats.devShOther;

}

void GaussianSplatting::destroyDataTextures()
{
  // destructors will invoke destroy on next frame
  m_centersMap.reset();
  m_colorsMap.reset();
  m_covariancesMap.reset();
  m_sphericalHarmonicsMap.reset();
}

void GaussianSplatting::createDataBuffers(void)
{
  const int splatCount = m_splatSet.positions.size() / 3;
  
  //
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  VkBufferUsageFlags hostBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkMemoryPropertyFlags hostMemoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkBufferUsageFlags deviceBufferUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                              | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                              | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VkMemoryPropertyFlags deviceMemoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  // Centers
  {
    const VkDeviceSize bufferSize = splatCount * 3 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer = m_alloc->createBuffer(bufferSize, hostBufferUsageFlags, hostMemoryPropertyFlags);

    m_centersDevice = m_alloc->createBuffer(bufferSize, deviceBufferUsageFlags, deviceMemoryPropertyFlags);
    m_dutil->DBG_NAME(m_centersDevice.buffer);

    // map and fill host buffer
    float* hostBufferMapped = static_cast<float*>(m_alloc->map(hostBuffer));
    memcpy(hostBufferMapped, m_splatSet.positions.data(), bufferSize);
    m_alloc->unmap(hostBuffer);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, m_centersDevice.buffer, 1, &bc);

    // free host buffer (at next frame)
    nvvkhl::Application::submitResourceFree([buffer = hostBuffer, alloc = m_alloc]() {
      alloc->destroy(const_cast<nvvk::Buffer&>(buffer));
    });

    // memory statistics
    m_modelMemoryStats.srcCenters  = bufferSize;
    m_modelMemoryStats.odevCenters = bufferSize; // no compression or quantization
    m_modelMemoryStats.devCenters  = bufferSize; // same size as source
  }

  // covariances
  {
    const VkDeviceSize bufferSize = splatCount * 2 * 3 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer = m_alloc->createBuffer(bufferSize, hostBufferUsageFlags, hostMemoryPropertyFlags);

    m_covariancesDevice = m_alloc->createBuffer(bufferSize, deviceBufferUsageFlags, deviceMemoryPropertyFlags);
    m_dutil->DBG_NAME(m_covariancesDevice.buffer);

    // map and fill host buffer
    float* hostBufferMapped = static_cast<float*>(m_alloc->map(hostBuffer));

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

      hostBufferMapped[stride6 + 0] = glm::value_ptr(transformedCovariance)[0];
      hostBufferMapped[stride6 + 1] = glm::value_ptr(transformedCovariance)[3];
      hostBufferMapped[stride6 + 2] = glm::value_ptr(transformedCovariance)[6];

      hostBufferMapped[stride6 + 3] = glm::value_ptr(transformedCovariance)[4];
      hostBufferMapped[stride6 + 4] = glm::value_ptr(transformedCovariance)[7];
      hostBufferMapped[stride6 + 5] = glm::value_ptr(transformedCovariance)[8];
    }

    m_alloc->unmap(hostBuffer);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, m_covariancesDevice.buffer, 1, &bc);

   // free host buffer (at next frame)
    nvvkhl::Application::submitResourceFree(
        [buffer = hostBuffer, alloc = m_alloc]() { alloc->destroy(const_cast<nvvk::Buffer&>(buffer)); });

    // memory statistics
    m_modelMemoryStats.srcCov  = (splatCount * (4 + 3)) * sizeof(float);
    m_modelMemoryStats.odevCov = bufferSize; // no compression
    m_modelMemoryStats.devCov  = bufferSize; // covariance takes less space than rotation + scale
  }

  // Colors. SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  {
    const VkDeviceSize bufferSize = splatCount * 4 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer = m_alloc->createBuffer(bufferSize, hostBufferUsageFlags, hostMemoryPropertyFlags);

    m_colorsDevice = m_alloc->createBuffer(bufferSize, deviceBufferUsageFlags, deviceMemoryPropertyFlags);
    m_dutil->DBG_NAME(m_colorsDevice.buffer);

    // fill host buffer
    float* hostBufferMapped = static_cast<float*>(m_alloc->map(hostBuffer));
   
    for(auto splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    {
      const auto  stride3 = splatIdx * 3;
      const auto  stride4 = splatIdx * 4;
      const float SH_C0   = 0.28209479177387814;
      hostBufferMapped[stride4 + 0] = glm::clamp(0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 0], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 1] = glm::clamp(0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 1], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 2] = glm::clamp(0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 2], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 3] = glm::clamp(1.0f / (1.0f + std::exp(-m_splatSet.opacity[splatIdx])), 0.0f, 1.0f);
    }

    m_alloc->unmap(hostBuffer);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, m_colorsDevice.buffer, 1, &bc);

    // free host buffer (at next frame)
    nvvkhl::Application::submitResourceFree(
        [buffer = hostBuffer, alloc = m_alloc]() { alloc->destroy(const_cast<nvvk::Buffer&>(buffer)); });

    // memory statistics
    m_modelMemoryStats.srcSh0  = bufferSize;
    m_modelMemoryStats.odevSh0 = bufferSize;  // no compression or quantization 
    m_modelMemoryStats.devSh0  = bufferSize;
  }

  // Spherical harmonics of degree 1 to 3
  {
    const int totalSphericalHarmonicsComponentCount    = m_splatSet.f_rest.size() / splatCount;
    const int sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    int sphericalHarmonicsDegree = 0;
    int splatStride              = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
    {
      sphericalHarmonicsDegree = 1;
      splatStride += 3 * 3;
    }
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
    {
      sphericalHarmonicsDegree = 2;
      splatStride += 5 * 3;
    }
    if(sphericalHarmonicsCoefficientsPerChannel == 15)
    {
      sphericalHarmonicsDegree = 2;
      splatStride += 7 * 3;
    }
    
    int targetSplatStride = splatStride; // same for the time beeing, would be less if we do not upload all src degrees

    // allocate host and device buffers
    const VkDeviceSize bufferSize = splatCount * splatStride * sizeof(float);

    nvvk::Buffer hostBuffer = m_alloc->createBuffer(bufferSize, hostBufferUsageFlags, hostMemoryPropertyFlags);

    m_sphericalHarmonicsDevice = m_alloc->createBuffer(bufferSize, deviceBufferUsageFlags, deviceMemoryPropertyFlags);
    m_dutil->DBG_NAME(m_sphericalHarmonicsDevice.buffer);

    // fill host buffer
    float* hostBufferMapped = static_cast<float*>(m_alloc->map(hostBuffer));
       
    for(auto splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    {
      const auto srcBase   = splatStride * splatIdx;
      const auto destBase  = targetSplatStride * splatIdx;
      int        dstOffset = 0;
      // degree 1, three coefs per component
      for(auto i = 0; i < 3; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          hostBufferMapped[dstIndex] = m_splatSet.f_rest[srcIndex];
        }
      }

      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          hostBufferMapped[dstIndex] = m_splatSet.f_rest[srcIndex];
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add
        
          hostBufferMapped[dstIndex] = m_splatSet.f_rest[srcIndex];
        }
      }
    }

    m_alloc->unmap(hostBuffer);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, m_sphericalHarmonicsDevice.buffer, 1, &bc);

    // free host buffer (at next frame)
    nvvkhl::Application::submitResourceFree(
        [buffer = hostBuffer, alloc = m_alloc]() { alloc->destroy(const_cast<nvvk::Buffer&>(buffer)); });

    // memory statistics
    m_modelMemoryStats.srcShOther  = m_splatSet.f_rest.size() * sizeof(float);
    m_modelMemoryStats.odevShOther = bufferSize;  // no compression or quantization
    m_modelMemoryStats.devShOther  = bufferSize;
  }

  //
  // sync with end of copy to device
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);

  //
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // update statistics totals
  m_modelMemoryStats.srcShAll  = m_modelMemoryStats.srcSh0 + m_modelMemoryStats.srcShOther;
  m_modelMemoryStats.odevShAll = m_modelMemoryStats.odevSh0 + m_modelMemoryStats.odevShOther;
  m_modelMemoryStats.devShAll  = m_modelMemoryStats.devSh0 + m_modelMemoryStats.devShOther;

  m_modelMemoryStats.srcAll =
      m_modelMemoryStats.srcCenters + m_modelMemoryStats.srcCov + m_modelMemoryStats.srcSh0 + m_modelMemoryStats.srcShOther;
  m_modelMemoryStats.odevAll = m_modelMemoryStats.odevCenters + m_modelMemoryStats.odevCov + m_modelMemoryStats.odevSh0
                               + m_modelMemoryStats.odevShOther;
  m_modelMemoryStats.devAll =
      m_modelMemoryStats.devCenters + m_modelMemoryStats.devCov + m_modelMemoryStats.devSh0 + m_modelMemoryStats.devShOther;
}

void GaussianSplatting::destroyDataBuffers()
{
  m_alloc->destroy(m_centersDevice);
  m_alloc->destroy(m_colorsDevice);
  m_alloc->destroy(m_covariancesDevice);
  m_alloc->destroy(m_sphericalHarmonicsDevice);
}