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

// Vulkan Memory Allocator
#define VMA_IMPLEMENTATION

#include "gaussian_splatting.h"
#include "utilities.h"

#include <nvh/misc.hpp>
#include <glm/gtc/packing.hpp>  // Required for half-float operations

void GaussianSplatting::onAttach(nvvkhl::Application* app)
{
  initGui();

  // starts the asynchronous services
  m_plyLoader.initialize();
  m_cpuSorter.initialize();

  // shortcuts
  m_app    = app;
  m_device = m_app->getDevice();

  m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
  // Debug utility
  m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);
  //
  m_dset = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
  // Memory allocator
  m_alloc = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
      .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = app->getPhysicalDevice(),
      .device         = app->getDevice(),
      .instance       = app->getInstance(),
  });

  // Where to find shader' source code
  std::vector<std::string> shaderSearchPaths;
  std::string              path = NVPSystem::exePath();
  shaderSearchPaths.push_back(NVPSystem::exePath());
  shaderSearchPaths.push_back(std::string("./GLSL_" PROJECT_NAME));
  shaderSearchPaths.push_back(NVPSystem::exePath() + std::string("GLSL_" PROJECT_NAME));
  shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + "Shaders");
  shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + "nvpro_core");

  std::cout << "Add search path " << NVPSystem::exePath() + std::string("GLSL_" PROJECT_NAME) << std::endl;

  m_shaderManager.init(m_device, 1, 2);
  m_shaderManager.m_filetype        = nvh::ShaderFileManager::FILETYPE_GLSL;
  m_shaderManager.m_keepModuleSPIRV = true;
  for(std::string& path : shaderSearchPaths)
  {
    m_shaderManager.addDirectory(path);
  }
};

void GaussianSplatting::onDetach()
{
  // stops the threads
  m_plyLoader.shutdown();
  m_cpuSorter.shutdown();
  // release resources
  deinitAll();
  m_dset->deinit();
  deinitGbuffers();
}

void GaussianSplatting::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  initGbuffers({size.width, size.height});
}

void GaussianSplatting::initGbuffers(const glm::vec2& size)
{
  m_viewSize = size;
  m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                 VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                 m_colorFormat, m_depthFormat);
}

void GaussianSplatting::deinitGbuffers()
{
  m_gBuffers.reset();
}

void GaussianSplatting::onRender(VkCommandBuffer cmd)
{

  if(!m_gBuffers)
    return;

  const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

  // collect readback results from previous frame if any
  collectReadBackValuesIfNeeded();

  // 0 if not ready so the rendering does not
  // touch the splat set while loading
  uint32_t splatCount = 0;
  if(m_plyLoader.getStatus() == PlyAsyncLoader::State::E_READY)
  {
    splatCount = (uint32_t)m_splatSet.size();
  }

  // Handle device-host data update and sorting if a scene exist
  if(splatCount)
  {
    updateAndUploadFrameInfoUBO(cmd, splatCount);

    if(m_frameInfo.sortingMethod == SORTING_GPU_SYNC_RADIX)
    {
      // resets CPU sorting time info
      m_distTime = m_sortTime = 0.0;

      processSortingOnGPU(cmd, splatCount);
    }
    else
    {
      tryConsumeAndUploadCpuSortingResult(cmd, splatCount);
    }
  }
  // Drawing the primitives in the G-Buffer if any
  {
    auto timerSection = m_profiler->timeRecurring("Rendering", cmd);

    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);
    if(splatCount)
    {
      // let's throw some pixels !!
      drawSplatPrimitives(cmd, splatCount);
    }

    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  readBackIndirectParametersIfNeeded(cmd);

  updateRenderingMemoryStatistics(cmd, splatCount);
}

void GaussianSplatting::updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount)
{
  auto timerSection = m_profiler->timeRecurring("UBO update", cmd);

  CameraManip.getLookat(m_eye, m_center, m_up);

  // Update frame parameters uniform buffer
  // some attributes of frameInfo were set by the user interface
  const float      aspectRatio = m_viewSize.x / m_viewSize.y;
  const glm::vec2& clip        = CameraManip.getClipPlanes();
  m_frameInfo.splatCount       = splatCount;
  m_frameInfo.viewMatrix       = CameraManip.getMatrix();
  m_frameInfo.projectionMatrix = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspectRatio, clip.x, clip.y);
  // OpenGL (0,0) is bottom left, Vulkan (0,0) is top left, and glm::perspectiveRH_ZO is for OpenGL so we mirror on y
  m_frameInfo.projectionMatrix[1][1] *= -1;
  m_frameInfo.cameraPosition         = m_eye;
  float       devicePixelRatio       = 1.0;
  const float focalLengthX           = m_frameInfo.projectionMatrix[0][0] * 0.5f * devicePixelRatio * m_viewSize.x;
  const float focalLengthY           = m_frameInfo.projectionMatrix[1][1] * 0.5f * devicePixelRatio * m_viewSize.y;
  const bool  isOrthographicCamera   = false;
  const float focalMultiplier        = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
  const float focalAdjustment        = focalMultiplier;  //  this.focalAdjustment* focalMultiplier;
  m_frameInfo.orthoZoom              = 1.0f;
  m_frameInfo.orthographicMode       = 0;  // disabled (uses perspective) TODO: activate support for orthographic
  m_frameInfo.viewport               = glm::vec2(m_viewSize.x * devicePixelRatio, m_viewSize.x * devicePixelRatio);
  m_frameInfo.basisViewport          = glm::vec2(1.0f / m_viewSize.x, 1.0f / m_viewSize.y);
  m_frameInfo.focal                  = glm::vec2(focalLengthX, focalLengthY);
  m_frameInfo.inverseFocalAdjustment = 1.0f / focalAdjustment;

  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &m_frameInfo);

  // sync with end of copy to device
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
                           | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);
}

void GaussianSplatting::tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount)
{
  // upload CPU sorted indices to the GPU if needed
  bool newIndexAvailable = false;

  if(!m_defines.opacityGaussianDisabled)
  {
    // 1. Splatting/blending is on, we check for a newly sorted index table
    auto status = m_cpuSorter.getStatus();
    if(status != SplatSorterAsync::E_SORTING)
    {
      // sorter is sleeping, we can work on shared data
      // we take into account the result of the sort
      if(status == SplatSorterAsync::E_SORTED)
      {
        m_cpuSorter.consume(m_splatIndices, m_distTime, m_sortTime);
        newIndexAvailable = true;
      }

      // let's wakeup the sorting thread to run a new sort if needed
      // will start work only if camera direction or position has changed
      m_cpuSorter.sortAsync(glm::normalize(m_center - m_eye), m_eye, m_splatSet.positions, m_cpuLazySort);
    }
  }
  else
  {
    // splatting off, we disable the sorting
    // indices would not be needed for non splatted points
    // however, using the same mechanism allows to use exactly the same shader
    // so if splatting/blending is off we provide an ordered table of indices
    // if not already filled by any other previous frames (sorted or not)
    bool refill = (m_splatIndices.size() != splatCount);
    if(refill)
    {
      m_splatIndices.resize(splatCount);
      for(uint32_t i = 0; i < splatCount; ++i)
      {
        m_splatIndices[i] = i;
      }
      newIndexAvailable = true;
    }
  }

  // 2. upload to GPU is needed
  {
    auto timerSection = m_profiler->timeRecurring("Copy indices to GPU", cmd);

    if(newIndexAvailable)
    {
      // Prepare buffer on host using sorted indices
      uint32_t* hostBuffer = static_cast<uint32_t*>(m_alloc->map(m_splatIndicesHost));
      memcpy(hostBuffer, m_splatIndices.data(), m_splatIndices.size() * sizeof(uint32_t));
      m_alloc->unmap(m_splatIndicesHost);
      // copy buffer to device
      VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = splatCount * sizeof(uint32_t)};
      vkCmdCopyBuffer(cmd, m_splatIndicesHost.buffer, m_splatIndicesDevice.buffer, 1, &bc);
      // sync with end of copy to device
      VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
                           0, 1, &barrier, 0, NULL, 0, NULL);
    }
  }
}

void GaussianSplatting::processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount)
{
  // when GPU sorting, we sort at each frame, all buffer in device memory, no copy from RAM

  // 1. reset the draw indirect parameters and counters, will be updated by compute shader
  {
    const shaderio::IndirectParams drawIndexedIndirectParams;
    vkCmdUpdateBuffer(cmd, m_indirect.buffer, 0, sizeof(shaderio::IndirectParams), (void*)&drawIndexedIndirectParams);

    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }

  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

  // 2. invoke the distance compute shader
  {
    auto timerSection = m_profiler->timeRecurring("GPU Dist", cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);

    vkCmdDispatch(cmd, (splatCount + DISTANCE_COMPUTE_WORKGROUP_SIZE - 1) / DISTANCE_COMPUTE_WORKGROUP_SIZE, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }

  // 3. invoke the radix sort from vrdx lib
  {
    auto timerSection = m_profiler->timeRecurring("GPU Sort", cmd);

    vrdxCmdSortKeyValueIndirect(cmd, m_gpuSorter, splatCount, m_indirect.buffer,
                                offsetof(shaderio::IndirectParams, instanceCount), m_splatDistancesDevice.buffer, 0,
                                m_splatIndicesDevice.buffer, 0, m_vrdxStorageDevice.buffer, 0, 0, 0);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }
}

void GaussianSplatting::drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount)
{
  if(m_selectedPipeline == PIPELINE_VERT)
  {  // Pipeline using vertex shader

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);
    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthTestEnable(cmd, (VkBool32)m_defines.opacityGaussianDisabled);

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
  {  // Pipeline using mesh shader

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineMesh);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);
    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthTestEnable(cmd, (VkBool32)m_defines.opacityGaussianDisabled);
    if(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX)
    {
      // run the workgroups
      vkCmdDrawMeshTasksEXT(cmd, (m_frameInfo.splatCount + RASTER_MESH_WORKGROUP_SIZE - 1) / RASTER_MESH_WORKGROUP_SIZE, 1, 1);
    }
    else
    {
      // run the workgroups
      vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer, offsetof(shaderio::IndirectParams, groupCountX), 1,
                                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
    }
  }
}

void GaussianSplatting::collectReadBackValuesIfNeeded(void)
{
  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && m_frameInfo.sortingMethod == SORTING_GPU_SYNC_RADIX && m_canCollectReadback)
  {
    uint32_t* hostBuffer = static_cast<uint32_t*>(m_alloc->map(m_indirectReadbackHost));
    std::memcpy((void*)&m_indirectReadback, (void*)hostBuffer, sizeof(shaderio::IndirectParams));
    m_alloc->unmap(m_indirectReadbackHost);
  }
}

void GaussianSplatting::readBackIndirectParametersIfNeeded(VkCommandBuffer cmd)
{
  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && m_frameInfo.sortingMethod == SORTING_GPU_SYNC_RADIX)
  {
    auto timerSection = m_profiler->timeRecurring("Indirect readback", cmd);

    // ensures m_indirect buffer modified by GPU sort is available for transfer
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                         NULL, 0, NULL);

    // copy from device to host buffer
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = sizeof(shaderio::IndirectParams)};
    vkCmdCopyBuffer(cmd, m_indirect.buffer, m_indirectReadbackHost.buffer, 1, &bc);

    m_canCollectReadback = true;
  }
}

void GaussianSplatting::updateRenderingMemoryStatistics(VkCommandBuffer cmd, const uint32_t splatCount)
{
  // update rendering memory statistics
  if(m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX)
  {
    m_renderMemoryStats.hostAllocIndices   = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.hostAllocDistances = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.allocIndices       = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedIndices        = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.allocDistances     = 0;
    m_renderMemoryStats.usedDistances      = 0;
    m_renderMemoryStats.usedIndirect       = 0;
  }
  else
  {
    m_renderMemoryStats.hostAllocDistances = 0;
    m_renderMemoryStats.hostAllocIndices   = 0;
    m_renderMemoryStats.allocDistances     = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedDistances      = m_indirectReadback.instanceCount * sizeof(uint32_t);
    m_renderMemoryStats.allocIndices       = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedIndices        = m_indirectReadback.instanceCount * sizeof(uint32_t);
    if(m_selectedPipeline == PIPELINE_VERT)
    {
      m_renderMemoryStats.usedIndirect = 5 * sizeof(uint32_t);
    }
    else
    {
      m_renderMemoryStats.usedIndirect = sizeof(shaderio::IndirectParams);
    }
  }
  m_renderMemoryStats.usedUboFrameInfo = sizeof(shaderio::FrameInfo);

  m_renderMemoryStats.hostTotal =
      m_renderMemoryStats.hostAllocIndices + m_renderMemoryStats.hostAllocDistances + m_renderMemoryStats.usedUboFrameInfo;

  uint32_t vrdxSize = m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal;

  m_renderMemoryStats.deviceUsedTotal = m_renderMemoryStats.usedIndices + m_renderMemoryStats.usedDistances + vrdxSize
                                        + m_renderMemoryStats.usedIndirect + m_renderMemoryStats.usedUboFrameInfo;

  m_renderMemoryStats.deviceAllocTotal = m_renderMemoryStats.allocIndices + m_renderMemoryStats.allocDistances + vrdxSize
                                         + m_renderMemoryStats.usedIndirect + m_renderMemoryStats.usedUboFrameInfo;
}

void GaussianSplatting::deinitAll()
{
  m_canCollectReadback = false;
  vkDeviceWaitIdle(m_device);
  deinitScene();
  deinitDataTextures();
  deinitDataBuffers();
  deinitRendererBuffers();
  deinitShaders();
  deinitPipelines();
  resetRenderSettings();
  // reset camera to default
  CameraManip.setClipPlanes({0.1F, 2000.0F});
  const glm::vec3 eye(0.0F, 0.0F, -2.0F);
  const glm::vec3 center(0.F, 0.F, 0.F);
  const glm::vec3 up(0.F, 1.F, 0.F);
  CameraManip.setLookat(eye, center, up);
  // record default cam for reset in UI
  ImGuiH::SetHomeCamera({eye, center, up, CameraManip.getFov()});
}

void GaussianSplatting::initAll()
{
  // resize the CPU sorter indices buffer
  m_splatIndices.resize(m_splatIndices.size());
  // TODO: use BBox of point cloud to set far plane, eye and center
  CameraManip.setClipPlanes({0.1F, 2000.0F});
  // we know that most INRIA models are upside down so we set the up vector to 0,-1,0
  const glm::vec3 eye(0.0F, 0.0F, -2.0F);
  const glm::vec3 center(0.F, 0.F, 0.F);
  const glm::vec3 up(0.F, -1.F, 0.F);
  CameraManip.setLookat(eye, center, up);
  // record default cam for reset in UI
  ImGuiH::SetHomeCamera({eye, center, up, CameraManip.getFov()});
  // reset general parameters
  resetRenderSettings();
  // init a new setup
  initShaders();
  initRendererBuffers();
  if(m_defines.dataStorage == STORAGE_TEXTURES)
    initDataTextures();
  else
    initDataBuffers();
  initPipelines();
}

void GaussianSplatting::reinitDataStorage()
{
  vkDeviceWaitIdle(m_device);

  if(m_centersMap.image != VK_NULL_HANDLE)
  {
    deinitDataTextures();
  }
  else
  {
    deinitDataBuffers();
  }
  deinitPipelines();
  deinitShaders();

  if(m_defines.dataStorage == STORAGE_TEXTURES)
  {
    initDataTextures();
  }
  else
  {
    initDataBuffers();
  }
  initShaders();
  initPipelines();
}

void GaussianSplatting::reinitShaders()
{
  vkDeviceWaitIdle(m_device);

  deinitPipelines();
  deinitShaders();

  initShaders();
  initPipelines();
}

void GaussianSplatting::deinitScene()
{
  m_splatSet            = {};
  m_loadedSceneFilename = "";
}

bool GaussianSplatting::initShaders(void)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  // blank page
  deinitShaders();

  // prepare definitions prepend
  std::string prepends;
  if(m_defines.opacityGaussianDisabled)
    prepends += "#define DISABLE_OPACITY_GAUSSIAN\n";
  prepends += nvh::stringFormat("#define FRUSTUM_CULLING_MODE %d\n", m_defines.frustumCulling);
  prepends += "#define ORTHOGRAPHIC_MODE 0\n";  // Disabled, TODO do we enable ortho cam in the UI/camera controller
  prepends += nvh::stringFormat("#define SHOW_SH_ONLY %d\n", m_defines.showShOnly);
  prepends += nvh::stringFormat("#define MAX_SH_DEGREE %d\n", m_defines.maxShDegree);
  prepends += nvh::stringFormat("#define DATA_STORAGE %d\n", m_defines.dataStorage);
  prepends += nvh::stringFormat("#define SH_FORMAT %d\n", m_defines.shFormat);
  prepends += nvh::stringFormat("#define POINT_CLOUD_MODE %d\n", m_defines.pointCloudModeEnabled);

  // generate the shader modules
  m_shaders.distShader   = m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "dist.comp.glsl", prepends);
  m_shaders.vertexShader = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "raster.vert.glsl", prepends);
  m_shaders.meshShader = m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_EXT, "raster.mesh.glsl", prepends);
  m_shaders.fragmentShader = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "raster.frag.glsl", prepends);

  if(!m_shaderManager.areShaderModulesValid())
  {
    m_shaderManager.deleteShaderModules();
    return false;
  }
  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Shaders updated in " << buildTime << "ms" << std::endl;

  return true;
}

void GaussianSplatting::deinitShaders(void)
{
  if(m_shaderManager.areShaderModulesValid())
  {
    m_shaderManager.deleteShaderModules();
  }
}

void GaussianSplatting::initPipelines()
{
  // reset descriptor bindings
  std::vector<VkDescriptorSetLayoutBinding> empty;
  m_dset->setBindings(empty);

  m_dset->addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_DISTANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_INDICES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_dset->addBinding(BINDING_INDIRECT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  if(m_defines.dataStorage == STORAGE_TEXTURES)
  {
    m_dset->addBinding(BINDING_SH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(BINDING_CENTERS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(BINDING_COLORS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(BINDING_COVARIANCES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  }
  else
  {
    m_dset->addBinding(BINDING_CENTERS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(BINDING_COLORS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(BINDING_COVARIANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(BINDING_SH_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  }

  m_dset->initLayout();
  m_dset->initPool(1);

  const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_MESH_BIT_EXT,
                                                    0, sizeof(shaderio::PushConstant)};
  m_dset->initPipeLayout(1, &push_constant_ranges);

  // Write descriptors for the buffers and textures
  std::vector<VkWriteDescriptorSet> writes;

  // add common buffers
  const VkDescriptorBufferInfo dbi_frameInfo{m_frameInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_FRAME_INFO_UBO, &dbi_frameInfo));
  const VkDescriptorBufferInfo keys_desc{m_splatDistancesDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_DISTANCES_BUFFER, &keys_desc));
  const VkDescriptorBufferInfo cpuKeys_desc{m_splatIndicesDevice.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_INDICES_BUFFER, &cpuKeys_desc));
  const VkDescriptorBufferInfo indirect_desc{m_indirect.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_dset->makeWrite(0, BINDING_INDIRECT_BUFFER, &indirect_desc));

  if(m_defines.dataStorage == STORAGE_TEXTURES)
  {
    // add data texture maps
    writes.emplace_back(m_dset->makeWrite(0, BINDING_CENTERS_TEXTURE, &m_centersMap.descriptor));
    writes.emplace_back(m_dset->makeWrite(0, BINDING_COLORS_TEXTURE, &m_colorsMap.descriptor));
    writes.emplace_back(m_dset->makeWrite(0, BINDING_COVARIANCES_TEXTURE, &m_covariancesMap.descriptor));
    writes.emplace_back(m_dset->makeWrite(0, BINDING_SH_TEXTURE, &m_sphericalHarmonicsMap.descriptor));
  }
  else
  {
    // add data buffers
    const VkDescriptorBufferInfo centers_desc{m_centersDevice.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_dset->makeWrite(0, BINDING_CENTERS_BUFFER, &centers_desc));
    const VkDescriptorBufferInfo colors_desc{m_colorsDevice.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_dset->makeWrite(0, BINDING_COLORS_BUFFER, &colors_desc));
    const VkDescriptorBufferInfo covariances_desc{m_covariancesDevice.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_dset->makeWrite(0, BINDING_COVARIANCES_BUFFER, &covariances_desc));
    const VkDescriptorBufferInfo sh_desc{m_sphericalHarmonicsDevice.buffer, 0, VK_WHOLE_SIZE};
    writes.emplace_back(m_dset->makeWrite(0, BINDING_SH_BUFFER, &sh_desc));
  }

  // write
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Create the pipeline to run the compute shader for distance & culling
  {
    auto pipelineLayout = m_dset->getPipeLayout();

    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_shaderManager.get(m_shaders.distShader),
                .pName  = "main",
            },
        .layout = pipelineLayout,
    };
    vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_computePipeline);
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
      pgen.addShader(m_shaderManager.get(m_shaders.meshShader), VK_SHADER_STAGE_MESH_BIT_EXT);
      pgen.addShader(m_shaderManager.get(m_shaders.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
      m_graphicsPipelineMesh = pgen.createPipeline();
      m_dutil->setObjectName(m_graphicsPipelineMesh, "PipelineMeshShader");
    }

    // create the pipeline that uses vertex shaders
    {
      const auto BINDING_ATTR_POSITION    = 0;
      const auto BINDING_ATTR_SPLAT_INDEX = 1;

      pstate.addBindingDescriptions({{BINDING_ATTR_POSITION, 3 * sizeof(float)}});  // 3 component per vertex position
      pstate.addAttributeDescriptions({{ATTRIBUTE_LOC_POSITION, BINDING_ATTR_POSITION, VK_FORMAT_R32G32B32_SFLOAT, 0}});

      pstate.addBindingDescriptions({{BINDING_ATTR_SPLAT_INDEX, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_INSTANCE}});
      pstate.addAttributeDescriptions({{ATTRIBUTE_LOC_SPLAT_INDEX, BINDING_ATTR_SPLAT_INDEX, VK_FORMAT_R32_UINT, 0}});

      nvvk::GraphicsPipelineGenerator pgen(m_device, m_dset->getPipeLayout(), prend_info, pstate);
      pgen.addShader(m_shaderManager.get(m_shaders.vertexShader), VK_SHADER_STAGE_VERTEX_BIT);
      pgen.addShader(m_shaderManager.get(m_shaders.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
      m_graphicsPipeline = pgen.createPipeline();
      m_dutil->setObjectName(m_graphicsPipeline, "PipelineVertexShader");
    }
  }
}

void GaussianSplatting::deinitPipelines()
{

  m_dset->deinitPool();
  m_dset->deinitLayout();

  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipeline(m_device, m_graphicsPipelineMesh, nullptr);
  vkDestroyPipeline(m_device, m_computePipeline, nullptr);
}

void GaussianSplatting::initRendererBuffers()
{
  const auto splatCount = (uint32_t)m_splatSet.size();

  // All this block for the sorting
  {
    // Vrdx sorter
    VrdxSorterCreateInfo gpuSorterInfo{.physicalDevice = m_app->getPhysicalDevice(), .device = m_app->getDevice()};
    vrdxCreateSorter(&gpuSorterInfo, &m_gpuSorter);

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
      vrdxGetSorterKeyValueStorageRequirements(m_gpuSorter, splatCount, &requirements);
      m_vrdxStorageDevice = m_alloc->createBuffer(requirements.size, requirements.usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      m_renderMemoryStats.allocVdrxInternal = (uint32_t)requirements.size;  // for stats reporting only

      // generate debug information for buffers
      m_dutil->DBG_NAME(m_splatIndicesHost.buffer);
      m_dutil->DBG_NAME(m_splatIndicesDevice.buffer);
      m_dutil->DBG_NAME(m_splatDistancesDevice.buffer);
      m_dutil->DBG_NAME(m_vrdxStorageDevice.buffer);
    }
  }

  // create the buffer for indirect parameters
  m_indirect = m_alloc->createBuffer(sizeof(shaderio::IndirectParams),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

  // for statistics readback
  m_indirectReadbackHost = m_alloc->createBuffer(sizeof(shaderio::IndirectParams),
                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  m_dutil->DBG_NAME(m_indirect.buffer);
  m_dutil->DBG_NAME(m_indirectReadbackHost.buffer);

  // We create a command buffer in order to perform the copy to VRAM
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // The Quad
  const std::vector<uint16_t> indices  = {0, 2, 1, 2, 0, 3};
  const std::vector<float>    vertices = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};

  // create the quad buffers
  m_quadVertices = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  m_quadIndices  = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  m_dutil->DBG_NAME(m_quadVertices.buffer);
  m_dutil->DBG_NAME(m_quadIndices.buffer);

  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Uniform buffer
  m_frameInfoBuffer = m_alloc->createBuffer(sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dutil->DBG_NAME(m_frameInfoBuffer.buffer);
}

void GaussianSplatting::deinitRendererBuffers()
{
  // TODO can we rather move this to pipelines creation/deletion ?
  if(m_gpuSorter != VK_NULL_HANDLE)
  {
    vrdxDestroySorter(m_gpuSorter);
    m_gpuSorter = VK_NULL_HANDLE;
  }

  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_splatDistancesDevice));
  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_splatIndicesDevice));
  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_splatIndicesHost));
  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_vrdxStorageDevice));

  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_indirect));
  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_indirectReadbackHost));

  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_quadVertices));
  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_quadIndices));

  m_alloc->destroy(const_cast<nvvk::Buffer&>(m_frameInfoBuffer));
}

inline uint8_t toUint8(float v, float rangeMin, float rangeMax)
{
  float normalized = (v - rangeMin) / (rangeMax - rangeMin);
  return static_cast<uint8_t>(std::clamp(std::round(normalized * 255.0f), 0.0f, 255.0f));
};

inline int formatSize(uint32_t format)
{
  if(format == FORMAT_FLOAT32)
    return 4;
  if(format == FORMAT_FLOAT16)
    return 2;
  if(format == FORMAT_UINT8)
    return 1;
  return 0;
}

inline void storeSh(int format, float* srcBuffer, uint32_t srcIndex, void* dstBuffer, uint32_t dstIndex)
{
  if(format == FORMAT_FLOAT32)
    static_cast<float*>(dstBuffer)[dstIndex] = srcBuffer[srcIndex];
  else if(format == FORMAT_FLOAT16)
    static_cast<uint16_t*>(dstBuffer)[dstIndex] = glm::packHalf1x16(srcBuffer[srcIndex]);
  else if(format == FORMAT_UINT8)
    static_cast<uint8_t*>(dstBuffer)[dstIndex] = toUint8(srcBuffer[srcIndex], -1., 1.);
}

///////////////////
// using data buffers to store splatset in VRAM

void GaussianSplatting::initDataBuffers(void)
{
  auto       startTime  = std::chrono::high_resolution_clock::now();
  const auto splatCount = (uint32_t)m_splatSet.positions.size() / 3;

  VkCommandBuffer    cmd                  = m_app->createTempCmdBuffer();
  VkBufferUsageFlags hostBufferUsageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkMemoryPropertyFlags hostMemoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VkBufferUsageFlags deviceBufferUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                              | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                              | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VkMemoryPropertyFlags deviceMemoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  // set of buffer to be freed after command execution
  std::vector<nvvk::Buffer> buffersToDestroy;

  // Centers
  {
    const uint32_t bufferSize = splatCount * 3 * sizeof(float);

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

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBuffer);

    // memory statistics
    m_modelMemoryStats.srcCenters  = bufferSize;
    m_modelMemoryStats.odevCenters = bufferSize;  // no compression or quantization
    m_modelMemoryStats.devCenters  = bufferSize;  // same size as source
  }

  // covariances
  {
    const uint32_t bufferSize = splatCount * 2 * 3 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer = m_alloc->createBuffer(bufferSize, hostBufferUsageFlags, hostMemoryPropertyFlags);

    m_covariancesDevice = m_alloc->createBuffer(bufferSize, deviceBufferUsageFlags, deviceMemoryPropertyFlags);
    m_dutil->DBG_NAME(m_covariancesDevice.buffer);

    // map and fill host buffer
    float* hostBufferMapped = static_cast<float*>(m_alloc->map(hostBuffer));

    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto stride3 = splatIdx * 3;
      const auto stride4 = splatIdx * 4;
      const auto stride6 = splatIdx * 6;
      glm::vec3  scale{std::exp(m_splatSet.scale[stride3 + 0]), std::exp(m_splatSet.scale[stride3 + 1]),
                      std::exp(m_splatSet.scale[stride3 + 2])};

      glm::quat rotation{m_splatSet.rotation[stride4 + 0], m_splatSet.rotation[stride4 + 1],
                         m_splatSet.rotation[stride4 + 2], m_splatSet.rotation[stride4 + 3]};
      rotation = glm::normalize(rotation);

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
    END_PAR_LOOP();

    m_alloc->unmap(hostBuffer);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, m_covariancesDevice.buffer, 1, &bc);

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBuffer);

    // memory statistics
    m_modelMemoryStats.srcCov  = (splatCount * (4 + 3)) * sizeof(float);
    m_modelMemoryStats.odevCov = bufferSize;  // no compression
    m_modelMemoryStats.devCov  = bufferSize;  // covariance takes less space than rotation + scale
  }

  // Colors. SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  {
    const uint32_t bufferSize = splatCount * 4 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer = m_alloc->createBuffer(bufferSize, hostBufferUsageFlags, hostMemoryPropertyFlags);

    m_colorsDevice = m_alloc->createBuffer(bufferSize, deviceBufferUsageFlags, deviceMemoryPropertyFlags);
    m_dutil->DBG_NAME(m_colorsDevice.buffer);

    // fill host buffer
    float* hostBufferMapped = static_cast<float*>(m_alloc->map(hostBuffer));

    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto  stride3           = splatIdx * 3;
      const auto  stride4           = splatIdx * 4;
      const float SH_C0             = 0.28209479177387814f;
      hostBufferMapped[stride4 + 0] = glm::clamp(0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 0], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 1] = glm::clamp(0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 1], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 2] = glm::clamp(0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 2], 0.0f, 1.0f);
      hostBufferMapped[stride4 + 3] = glm::clamp(1.0f / (1.0f + std::exp(-m_splatSet.opacity[splatIdx])), 0.0f, 1.0f);
    }
    END_PAR_LOOP()

    m_alloc->unmap(hostBuffer);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, m_colorsDevice.buffer, 1, &bc);

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBuffer);

    // memory statistics
    m_modelMemoryStats.srcSh0  = bufferSize;
    m_modelMemoryStats.odevSh0 = bufferSize;
    m_modelMemoryStats.devSh0  = bufferSize;
  }

  // Spherical harmonics of degree 1 to 3
  {
    const uint32_t totalSphericalHarmonicsComponentCount    = (uint32_t)m_splatSet.f_rest.size() / splatCount;
    const uint32_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
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
      sphericalHarmonicsDegree = 3;
      splatStride += 7 * 3;
    }

    int targetSplatStride = splatStride;  // same for the time beeing, would be less if we do not upload all src degrees

    // allocate host and device buffers
    const uint32_t bufferSize = splatCount * splatStride * formatSize(m_defines.shFormat);

    nvvk::Buffer hostBuffer = m_alloc->createBuffer(bufferSize, hostBufferUsageFlags, hostMemoryPropertyFlags);

    m_sphericalHarmonicsDevice = m_alloc->createBuffer(bufferSize, deviceBufferUsageFlags, deviceMemoryPropertyFlags);
    m_dutil->DBG_NAME(m_sphericalHarmonicsDevice.buffer);

    // fill host buffer
    void* hostBufferMapped = m_alloc->map(hostBuffer);

    auto startShTime = std::chrono::high_resolution_clock::now();

    // for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
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

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
    }
    END_PAR_LOOP()

    auto      endShTime   = std::chrono::high_resolution_clock::now();
    long long buildShTime = std::chrono::duration_cast<std::chrono::milliseconds>(endShTime - startShTime).count();
    std::cout << "Sh data updated in " << buildShTime << "ms" << std::endl;

    m_alloc->unmap(hostBuffer);

    // copy from host buffer to device buffer
    // barrier at the end of this method.
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = bufferSize};
    vkCmdCopyBuffer(cmd, hostBuffer.buffer, m_sphericalHarmonicsDevice.buffer, 1, &bc);

    // free host buffer after command execution
    buffersToDestroy.push_back(hostBuffer);

    // memory statistics
    m_modelMemoryStats.srcShOther  = (uint32_t)m_splatSet.f_rest.size() * sizeof(float);
    m_modelMemoryStats.odevShOther = bufferSize;  // no compression or quantization
    m_modelMemoryStats.devShOther  = bufferSize;
  }

  // sync with end of copy to device
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);

  m_app->submitAndWaitTempCmdBuffer(cmd);

  // free temp buffers
  for(auto& buffer : buffersToDestroy)
  {
    m_alloc->destroy(buffer);
  }

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

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Data buffers updated in " << buildTime << "ms" << std::endl;
}

void GaussianSplatting::deinitDataBuffers()
{
  m_alloc->destroy(m_centersDevice);
  m_alloc->destroy(m_colorsDevice);
  m_alloc->destroy(m_covariancesDevice);
  m_alloc->destroy(m_sphericalHarmonicsDevice);
}

///////////////////
// using texture maps to store splatset in VRAM

void GaussianSplatting::initTexture(uint32_t         width,
                                    uint32_t         height,
                                    uint32_t         bufsize,
                                    void*            data,
                                    VkFormat         format,
                                    const VkSampler& sampler,
                                    nvvk::Texture&   texture)
{
  const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  const VkExtent2D          size        = {width, height};
  const VkImageCreateInfo   create_info = nvvk::makeImage2DCreateInfo(size, format, VK_IMAGE_USAGE_SAMPLED_BIT, false);

  nvvk::CommandPool cpool(m_device, m_app->getQueue(0).familyIndex);
  VkCommandBuffer   cmd = cpool.createCommandBuffer();

  texture = m_alloc->createTexture(cmd, bufsize, data, create_info, sampler_info);
  cpool.submitAndWait(cmd);

  texture.descriptor.sampler = sampler;
}

void GaussianSplatting::deinitTexture(nvvk::Texture& texture)
{
  m_alloc->destroy(texture);
}

void GaussianSplatting::initDataTextures(void)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  const auto splatCount = (uint32_t)m_splatSet.positions.size() / 3;

  // will create a texture sampler using nearest filtering mode foe each texture map
  // samplers will be released by texture destruction.
  VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler_info.magFilter  = VK_FILTER_NEAREST;
  sampler_info.minFilter  = VK_FILTER_NEAREST;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  // centers (3 components but texture map is only allowed with 4 components)
  // TODO: May pack as done for covariances not to waste alpha chanel ? but must
  // compare performance (1 lookup vs 2 lookups due to packing)
  {
    glm::ivec2         mapSize = computeDataTextureSize(3, 3, splatCount);
    std::vector<float> centers(mapSize.x * mapSize.y * 4);  // includes some padding and unused w channel
    //for(uint32_t i = 0; i < splatCount; ++i)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      // we skip the alpha channel that is left undefined and not used in the shader
      for(uint32_t cmp = 0; cmp < 3; ++cmp)
      {
        centers[splatIdx * 4 + cmp] = m_splatSet.positions[splatIdx * 3 + cmp];
      }
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    initTexture(mapSize.x, mapSize.y, (uint32_t)centers.size() * sizeof(float), (void*)centers.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, m_alloc->acquireSampler(sampler_info), m_centersMap);
    // memory statistics
    m_modelMemoryStats.srcCenters  = splatCount * 3 * sizeof(float);
    m_modelMemoryStats.odevCenters = splatCount * 3 * sizeof(float);  // no compression or quantization yet
    m_modelMemoryStats.devCenters  = mapSize.x * mapSize.y * 4 * sizeof(float);
  }
  // covariances
  {
    glm::ivec2         mapSize = computeDataTextureSize(4, 6, splatCount);
    std::vector<float> covariances(mapSize.x * mapSize.y * 4, 0.0f);
    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto stride3 = splatIdx * 3;
      const auto stride4 = splatIdx * 4;
      const auto stride6 = splatIdx * 6;
      glm::vec3  scale{std::exp(m_splatSet.scale[stride3 + 0]), std::exp(m_splatSet.scale[stride3 + 1]),
                      std::exp(m_splatSet.scale[stride3 + 2])};

      glm::quat rotation{m_splatSet.rotation[stride4 + 0], m_splatSet.rotation[stride4 + 1],
                         m_splatSet.rotation[stride4 + 2], m_splatSet.rotation[stride4 + 3]};
      rotation = glm::normalize(rotation);

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
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    initTexture(mapSize.x, mapSize.y, (uint32_t)covariances.size() * sizeof(float), (void*)covariances.data(),
                VK_FORMAT_R32G32B32A32_SFLOAT, m_alloc->acquireSampler(sampler_info), m_covariancesMap);
    // memory statistics
    m_modelMemoryStats.srcCov  = (splatCount * (4 + 3)) * sizeof(float);
    m_modelMemoryStats.odevCov = splatCount * 6 * sizeof(float);  // covariance takes less space than rotation + scale
    m_modelMemoryStats.devCov  = mapSize.x * mapSize.y * 4 * sizeof(float);
  }
  // SH degree 0 is not view dependent, so we directly transform to base color
  // this will make some economy of processing in the shader at each frame
  {
    glm::ivec2           mapSize = computeDataTextureSize(4, 4, splatCount);
    std::vector<uint8_t> colors(mapSize.x * mapSize.y * 4);  // includes some padding
    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const auto  stride3 = splatIdx * 3;
      const auto  stride4 = splatIdx * 4;
      const float SH_C0   = 0.28209479177387814f;
      colors[stride4 + 0] = (uint8_t)glm::clamp(std::floor((0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 0]) * 255), 0.0f, 255.0f);
      colors[stride4 + 1] = (uint8_t)glm::clamp(std::floor((0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 1]) * 255), 0.0f, 255.0f);
      colors[stride4 + 2] = (uint8_t)glm::clamp(std::floor((0.5f + SH_C0 * m_splatSet.f_dc[stride3 + 2]) * 255), 0.0f, 255.0f);
      colors[stride4 + 3] =
          (uint8_t)glm::clamp(std::floor((1.0f / (1.0f + std::exp(-m_splatSet.opacity[splatIdx]))) * 255), 0.0f, 255.0f);
    }
    END_PAR_LOOP()
    // place the result in the dedicated texture map
    initTexture(mapSize.x, mapSize.y, (uint32_t)colors.size(), (void*)colors.data(), VK_FORMAT_R8G8B8A8_UNORM,
                m_alloc->acquireSampler(sampler_info), m_colorsMap);
    // memory statistics
    m_modelMemoryStats.srcSh0  = splatCount * 4 * sizeof(float);  // original sh0 and opacity are floats
    m_modelMemoryStats.odevSh0 = splatCount * 4 * sizeof(uint8_t);
    m_modelMemoryStats.devSh0  = mapSize.x * mapSize.y * 4 * sizeof(uint8_t);
  }
  // Prepare the spherical harmonics of degree 1 to 3
  {
    const uint32_t sphericalHarmonicsElementsPerTexel       = 4;
    const uint32_t totalSphericalHarmonicsComponentCount    = (uint32_t)m_splatSet.f_rest.size() / splatCount;
    const uint32_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    int sphericalHarmonicsDegree = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
      sphericalHarmonicsDegree = 1;
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
      sphericalHarmonicsDegree = 2;
    if(sphericalHarmonicsCoefficientsPerChannel >= 15)
      sphericalHarmonicsDegree = 3;

    // add some padding at each splat if needed for easy texture lookups
    int sphericalHarmonicsComponentCount = 0;
    if(sphericalHarmonicsDegree == 1)
      sphericalHarmonicsComponentCount = 9;
    if(sphericalHarmonicsDegree == 2)
      sphericalHarmonicsComponentCount = 24;
    if(sphericalHarmonicsDegree == 3)
      sphericalHarmonicsComponentCount = 45;

    int paddedSphericalHarmonicsComponentCount = sphericalHarmonicsComponentCount;
    while(paddedSphericalHarmonicsComponentCount % 4 != 0)
      paddedSphericalHarmonicsComponentCount++;

    glm::ivec2 mapSize =
        computeDataTextureSize(sphericalHarmonicsElementsPerTexel, paddedSphericalHarmonicsComponentCount, splatCount);

    const uint32_t bufferSize = mapSize.x * mapSize.y * sphericalHarmonicsElementsPerTexel * formatSize(m_defines.shFormat);

    std::vector<uint8_t> paddedSHArray(bufferSize, 0);

    void* data = (void*)paddedSHArray.data();

    //for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
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

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, data, dstIndex);
        }
      }

      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, data, dstIndex);
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const auto srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const auto dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, data, dstIndex);
        }
      }
    }
    END_PAR_LOOP()

    // place the result in the dedicated texture map
    if(m_defines.shFormat == FORMAT_FLOAT32)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R32G32B32A32_SFLOAT,
                  m_alloc->acquireSampler(sampler_info), m_sphericalHarmonicsMap);
    }
    else if(m_defines.shFormat == FORMAT_FLOAT16)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R16G16B16A16_SFLOAT,
                  m_alloc->acquireSampler(sampler_info), m_sphericalHarmonicsMap);
    }
    else if(m_defines.shFormat == FORMAT_UINT8)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R8G8B8A8_UNORM,
                  m_alloc->acquireSampler(sampler_info), m_sphericalHarmonicsMap);
    }

    // memory statistics
    m_modelMemoryStats.srcShOther  = (uint32_t)m_splatSet.f_rest.size() * sizeof(float);
    m_modelMemoryStats.odevShOther = (uint32_t)m_splatSet.f_rest.size() * formatSize(m_defines.shFormat);
    m_modelMemoryStats.devShOther  = bufferSize;
  }

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

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Data textures updated in " << buildTime << "ms" << std::endl;
}

void GaussianSplatting::deinitDataTextures()
{
  deinitTexture(m_centersMap);
  deinitTexture(m_colorsMap);
  deinitTexture(m_covariancesMap);
  deinitTexture(m_sphericalHarmonicsMap);
}

void GaussianSplatting::benchmarkAdvance()
{
  m_benchmarkId++;

  std::cout << "BENCHMARK_ADV " << m_benchmarkId << " {" << std::endl;
  std::cout << " Memory Scene; Host used \t" << m_modelMemoryStats.srcAll << "; Device Used \t" << m_modelMemoryStats.odevAll
            << "; Device Allocated \t" << m_modelMemoryStats.devAll << "; (bytes)" << std::endl;
  std::cout << " Memory Rendering; Host used \t" << m_renderMemoryStats.hostTotal << "; Device Used \t"
            << m_renderMemoryStats.deviceUsedTotal << "; Device Allocated \t" << m_renderMemoryStats.deviceAllocTotal
            << "; (bytes)" << std::endl;
  std::cout << "}" << std::endl;
}