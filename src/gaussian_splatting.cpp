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
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include "gaussian_splatting.h"
#include "utilities.h"

#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/formats.hpp>

#include <nvutils/primitives.hpp>

// #include <nvh/misc.hpp>
#include <glm/gtc/packing.hpp>  // Required for half-float operations

using namespace vk_gaussian_splatting;

GaussianSplatting::GaussianSplatting(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry, bool* benchmarkEnabled)
    : m_profilerManager(profilerManager)
    , m_parameterRegistry(parameterRegistry)
    , m_pBenchmarkEnabled(benchmarkEnabled)
    , cameraManip(std::make_shared<nvutils::CameraManipulator>())
{
  // Register command line arguments
  // Done in this class instead of in main() so private members can be registered for direct modification
  parameterRegistry->add({"inputFile", "load a ply file"}, {".ply"}, &m_sceneToLoadFilename);
  parameterRegistry->add({"pipeline", "0=mesh 1=vert"}, &m_selectedPipeline);
  parameterRegistry->add({"shformat", "0=fp32 1=fp16 2=uint8"}, &m_defines.shFormat);
  parameterRegistry->add({"updateData", "1=triggers an update of data buffers or textures, used for benchmarking"}, &m_updateData);
  parameterRegistry->add({"maxShDegree", "max sh degree used for rendering in [0,1,2,3]"}, &m_defines.maxShDegree);
#ifdef WITH_DEFAULT_SCENE_FEATURE
  parameterRegistry->add({"loadDefaultScene", "0=disable the load of a default scene when no ply file is provided"},
                         &m_enableDefaultScene);
#endif

  parameterRegistry->add({.name = "screenshot",
                          .help = "takes a screenshot . Use only in benchmark script.",
                          .callbackSuccess =
                              [&](const nvutils::ParameterBase* const) {
                                if(m_app)
                                {
                                  m_app->screenShot(m_screenshotFilename);
                                }
                              }},
                         {".png"}, &m_screenshotFilename);
};

GaussianSplatting::~GaussianSplatting(){
    // all threads must be stopped,
    // work done in onDetach(),
    // could be done here, same result
};

void GaussianSplatting::onAttach(nvapp::Application* app)
{
  initGui();

  // we hide the UI dy default in benchmark mode
  m_showUI = !(*m_pBenchmarkEnabled);

  // shortcuts
  m_app    = app;
  m_device = m_app->getDevice();

  // profiling
  m_profilerTimeline = m_profilerManager->createTimeline({.name = "Primary Timeline"});
  m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(), m_app->getQueue(0).familyIndex, false);

  // starts the asynchronous services
  m_plyLoader.initialize();
  m_cpuSorter.initialize(m_profilerTimeline);

  // Memory allocator
  m_alloc.init(VmaAllocatorCreateInfo{
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = app->getPhysicalDevice(),
      .device           = app->getDevice(),
      .instance         = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });

  // uncomment and set id to find object leak
  // m_alloc.setLeakID(8);

  // TODO rename as buffer uploader ?
  // set up staging utility
  m_stagingUploader.init(&m_alloc, true);

  // Acquiring the sampler which will be used for displaying the GBuffer and accessing tetxures
  m_samplerPool.init(app->getDevice());
  NVVK_CHECK(m_samplerPool.acquireSampler(m_sampler));
  NVVK_DBG_NAME(m_sampler);

  // GBuffer
  m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
  m_gBuffers.init({
      .allocator      = &m_alloc,
      .colorFormats   = {m_colorFormat},  // Only one GBuffer color attachment
      .depthFormat    = m_depthFormat,
      .imageSampler   = m_sampler,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  });

  // Setting up the GLSL compiler
  // Where to find shader' source code
  m_glslCompiler.addSearchPaths(getShaderDirs());
  // SPIRV 1.6 and VULKAN 1.4
  m_glslCompiler.defaultTarget();
  m_glslCompiler.defaultOptions();
};

void GaussianSplatting::onDetach()
{
  // stops the threads
  m_plyLoader.shutdown();
  m_cpuSorter.shutdown();
  // release resources
  deinitAll();
  // m_dset->deinit();
  m_profilerGpuTimer.deinit();
  m_profilerManager->destroyTimeline(m_profilerTimeline);
  m_profilerTimeline = nullptr;
  m_gBuffers.deinit();
  m_samplerPool.releaseSampler(m_sampler);
  m_samplerPool.deinit();
  m_stagingUploader.deinit();
  m_alloc.deinit();
}

void GaussianSplatting::onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize)
{
  m_viewSize = {viewportSize.width, viewportSize.height};
  NVVK_CHECK(m_gBuffers.update(cmd, viewportSize));
}

void GaussianSplatting::onPreRender()
{
  m_profilerTimeline->frameAdvance();
}

void GaussianSplatting::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

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
      // remove eventual async CPU sorting timers
      // so that it will not appear since not sorting on CPU anymore
      m_profilerTimeline->asyncRemoveTimer("CPU Dist");
      m_profilerTimeline->asyncRemoveTimer("CPU Sort");
      // now work on GPU
      processSortingOnGPU(cmd, splatCount);
    }
    else
    {
      tryConsumeAndUploadCpuSortingResult(cmd, splatCount);
    }
  }
  // Drawing the primitives in the G-Buffer if any
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rendering");

    const VkExtent2D& viewportSize = m_app->getViewportSize();
    const VkViewport  viewport{0.0F, 0.0F, float(viewportSize.width), float(viewportSize.height), 0.0F, 1.0F};
    const VkRect2D    scissor{{0, 0}, viewportSize};

    // Drawing the primitives in a G-Buffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    colorAttachment.clearValue                = {m_clearColor};
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    vkCmdBeginRendering(cmd, &renderingInfo);

    vkCmdSetViewportWithCount(cmd, 1, &viewport);
    vkCmdSetScissorWithCount(cmd, 1, &scissor);

    if(splatCount)
    {
      // let's throw some pixels !!
      drawSplatPrimitives(cmd, splatCount);
    }

    vkCmdEndRendering(cmd);

    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  }

  readBackIndirectParametersIfNeeded(cmd);

  updateRenderingMemoryStatistics(cmd, splatCount);
}

void GaussianSplatting::updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UBO update");

  cameraManip->getLookat(m_eye, m_center, m_up);

  // Update frame parameters uniform buffer
  // some attributes of frameInfo were set by the user interface
  const glm::vec2& clip              = cameraManip->getClipPlanes();
  m_frameInfo.splatCount             = splatCount;
  m_frameInfo.viewMatrix             = cameraManip->getViewMatrix();
  m_frameInfo.projectionMatrix       = cameraManip->getPerspectiveMatrix();
  m_frameInfo.cameraPosition         = m_eye;
  float       devicePixelRatio       = 1.0;
  const float focalLengthX           = m_frameInfo.projectionMatrix[0][0] * 0.5f * devicePixelRatio * m_viewSize.x;
  const float focalLengthY           = m_frameInfo.projectionMatrix[1][1] * 0.5f * devicePixelRatio * m_viewSize.y;
  const bool  isOrthographicCamera   = false;
  const float focalMultiplier        = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
  const float focalAdjustment        = focalMultiplier;  //  this.focalAdjustment* focalMultiplier;
  m_frameInfo.orthoZoom              = 1.0f;
  m_frameInfo.orthographicMode       = 0;  // disabled (uses perspective) TODO: activate support for orthographic
  m_frameInfo.basisViewport          = glm::vec2(1.0f / m_viewSize.x, 1.0f / m_viewSize.y);
  m_frameInfo.focal                  = glm::vec2(focalLengthX, focalLengthY);
  m_frameInfo.inverseFocalAdjustment = 1.0f / focalAdjustment;

  // the buffer is small so we use vkCmdUpdateBuffer for the transfer
  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &m_frameInfo);

  // sync with end of copy to device
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                           | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);
}

void GaussianSplatting::tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

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
        m_cpuSorter.consume(m_splatIndices);
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
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Copy indices to GPU");

    if(newIndexAvailable)
    {
      // Prepare buffer on host using sorted indices
      memcpy(m_splatIndicesHost.mapping, m_splatIndices.data(), m_splatIndices.size() * sizeof(uint32_t));
      // copy buffer to device
      VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = splatCount * sizeof(uint32_t)};
      vkCmdCopyBuffer(cmd, m_splatIndicesHost.buffer, m_splatIndicesDevice.buffer, 1, &bc);
      // sync with end of copy to device
      VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                           0, 1, &barrier, 0, NULL, 0, NULL);
    }
  }
}

void GaussianSplatting::processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  // when GPU sorting, we sort at each frame, all buffer in device memory, no copy from RAM

  // 1. reset the draw indirect parameters and counters, will be updated by compute shader
  {
    const shaderio::IndirectParams drawIndexedIndirectParams;
    vkCmdUpdateBuffer(cmd, m_indirect.buffer, 0, sizeof(shaderio::IndirectParams), (void*)&drawIndexedIndirectParams);

    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }

  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

  // 2. invoke the distance compute shader
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "GPU Dist");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    vkCmdDispatch(cmd, (splatCount + DISTANCE_COMPUTE_WORKGROUP_SIZE - 1) / DISTANCE_COMPUTE_WORKGROUP_SIZE, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }

  // 3. invoke the radix sort from vrdx lib
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "GPU Sort");

    vrdxCmdSortKeyValueIndirect(cmd, m_gpuSorter, splatCount, m_indirect.buffer,
                                offsetof(shaderio::IndirectParams, instanceCount), m_splatDistancesDevice.buffer, 0,
                                m_splatIndicesDevice.buffer, 0, m_vrdxStorageDevice.buffer, 0, 0, 0);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }
}

void GaussianSplatting::drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  bool needDepth = (m_frameInfo.sortingMethod != SORTING_GPU_SYNC_RADIX) && m_defines.opacityGaussianDisabled;

  if(m_selectedPipeline == PIPELINE_VERT)
  {  // Pipeline using vertex shader

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

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
    std::memcpy((void*)&m_indirectReadback, (void*)m_indirectReadbackHost.mapping, sizeof(shaderio::IndirectParams));
  }
}

void GaussianSplatting::readBackIndirectParametersIfNeeded(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && m_frameInfo.sortingMethod == SORTING_GPU_SYNC_RADIX)
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Indirect readback");

    // ensures m_indirect buffer modified by GPU sort is available for transfer
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, 1, &barrier,
                         0, NULL, 0, NULL);

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
  cameraManip->setClipPlanes({0.1F, 2000.0F});
  const glm::vec3 eye(0.0F, 0.0F, -2.0F);
  const glm::vec3 center(0.F, 0.F, 0.F);
  const glm::vec3 up(0.F, 1.F, 0.F);
  cameraManip->setLookat(eye, center, up);
  // record default cam for reset in UI
  nvgui::SetHomeCamera({eye, center, up, cameraManip->getFov()});
}

void GaussianSplatting::initAll()
{
  // resize the CPU sorter indices buffer
  m_splatIndices.resize(m_splatIndices.size());
  // TODO: use BBox of point cloud to set far plane, eye and center
  cameraManip->setClipPlanes({0.1F, 2000.0F});
  // we know that most INRIA models are upside down so we set the up vector to 0,-1,0
  const glm::vec3 eye(0.0F, 0.0F, -2.0F);
  const glm::vec3 center(0.F, 0.F, 0.F);
  const glm::vec3 up(0.F, -1.F, 0.F);
  cameraManip->setLookat(eye, center, up);
  // record default cam for reset in UI
  nvgui::SetHomeCamera({eye, center, up, cameraManip->getFov()});
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

shaderc::SpvCompilationResult GaussianSplatting::compileGlslShader(const std::string& filename, shaderc_shader_kind shaderKind)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  m_glslCompiler.options().AddMacroDefinition("DISABLE_OPACITY_GAUSSIAN", std::to_string((int)m_defines.opacityGaussianDisabled));
  m_glslCompiler.options().AddMacroDefinition("FRUSTUM_CULLING_MODE", std::to_string((int)m_defines.frustumCulling));
  // Disabled, TODO do we enable ortho cam in the UI/camera controller
  m_glslCompiler.options().AddMacroDefinition("ORTHOGRAPHIC_MODE", "0");
  m_glslCompiler.options().AddMacroDefinition("SHOW_SH_ONLY", std::to_string((int)m_defines.showShOnly));
  m_glslCompiler.options().AddMacroDefinition("MAX_SH_DEGREE", std::to_string(m_defines.maxShDegree));
  m_glslCompiler.options().AddMacroDefinition("DATA_STORAGE", std::to_string(m_defines.dataStorage));
  m_glslCompiler.options().AddMacroDefinition("SH_FORMAT", std::to_string(m_defines.shFormat));
  m_glslCompiler.options().AddMacroDefinition("POINT_CLOUD_MODE", std::to_string((int)m_defines.pointCloudModeEnabled));
  m_glslCompiler.options().AddMacroDefinition("USE_BARYCENTRIC", std::to_string((int)m_defines.fragmentBarycentric));

  return m_glslCompiler.compileFile(filename, shaderKind);
}

bool GaussianSplatting::initShaders(void)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  std::string errorMsg;

  // generate the shader modules
  auto dist_comp   = compileGlslShader("dist.comp.glsl", shaderc_shader_kind::shaderc_compute_shader);
  auto raster_vert = compileGlslShader("raster.vert.glsl", shaderc_shader_kind::shaderc_vertex_shader);
  auto raster_mesh = compileGlslShader("raster.mesh.glsl", shaderc_shader_kind::shaderc_mesh_shader);
  auto raster_frag = compileGlslShader("raster.frag.glsl", shaderc_shader_kind::shaderc_fragment_shader);

  size_t errors = dist_comp.GetNumErrors() + raster_vert.GetNumErrors() + raster_mesh.GetNumErrors() + raster_frag.GetNumErrors();

  if(!errors)
  {
    VkShaderModuleCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};

    createInfo.codeSize = m_glslCompiler.getSpirvSize(dist_comp);
    createInfo.pCode    = m_glslCompiler.getSpirv(dist_comp);
    NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_shaders.distShader));
    NVVK_DBG_NAME(m_shaders.distShader);

    createInfo.codeSize = m_glslCompiler.getSpirvSize(raster_vert);
    createInfo.pCode    = m_glslCompiler.getSpirv(raster_vert);
    NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_shaders.vertexShader));
    NVVK_DBG_NAME(m_shaders.vertexShader);

    createInfo.codeSize = m_glslCompiler.getSpirvSize(raster_mesh);
    createInfo.pCode    = m_glslCompiler.getSpirv(raster_mesh);
    NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_shaders.meshShader));
    NVVK_DBG_NAME(m_shaders.meshShader);

    createInfo.codeSize = m_glslCompiler.getSpirvSize(raster_frag);
    createInfo.pCode    = m_glslCompiler.getSpirv(raster_frag);
    NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_shaders.fragmentShader));
    NVVK_DBG_NAME(m_shaders.fragmentShader);
  }
  else
  {
    errorMsg += dist_comp.GetErrorMessage();
    errorMsg += raster_vert.GetErrorMessage();
    errorMsg += raster_mesh.GetErrorMessage();
    errorMsg += raster_frag.GetErrorMessage();

    LOGE("%s", errorMsg.c_str());

    return false;
  }

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Shaders updated in " << buildTime << "ms" << std::endl;

  return true;
}

void GaussianSplatting::deinitShaders(void)
{
  vkDestroyShaderModule(m_device, m_shaders.distShader, nullptr);
  vkDestroyShaderModule(m_device, m_shaders.vertexShader, nullptr);
  vkDestroyShaderModule(m_device, m_shaders.meshShader, nullptr);
  vkDestroyShaderModule(m_device, m_shaders.fragmentShader, nullptr);

  m_shaders.distShader     = VK_NULL_HANDLE;
  m_shaders.vertexShader   = VK_NULL_HANDLE;
  m_shaders.meshShader     = VK_NULL_HANDLE;
  m_shaders.fragmentShader = VK_NULL_HANDLE;
}

void GaussianSplatting::initPipelines()
{
  nvvk::DescriptorBindings bindings;

  bindings.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_DISTANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_INDICES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_INDIRECT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  if(m_defines.dataStorage == STORAGE_TEXTURES)
  {
    bindings.addBinding(BINDING_SH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_CENTERS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COLORS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COVARIANCES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  }
  else
  {
    bindings.addBinding(BINDING_CENTERS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COLORS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COVARIANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SH_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  }

  const VkPushConstantRange pcRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_MESH_BIT_EXT,
                                        0, sizeof(shaderio::PushConstant)};

  NVVK_CHECK(bindings.createDescriptorSetLayout(m_device, 0, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  //
  std::vector<VkDescriptorPoolSize> poolSize;
  bindings.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));
  NVVK_DBG_NAME(m_descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_descriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet));
  NVVK_DBG_NAME(m_descriptorSet);

  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRanges,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  // Write descriptors for the buffers and textures
  nvvk::WriteSetContainer writeContainer;

  // add common buffers
  writeContainer.append(bindings.getWriteSet(BINDING_FRAME_INFO_UBO, m_descriptorSet), m_frameInfoBuffer);
  writeContainer.append(bindings.getWriteSet(BINDING_DISTANCES_BUFFER, m_descriptorSet), m_splatDistancesDevice);
  writeContainer.append(bindings.getWriteSet(BINDING_INDICES_BUFFER, m_descriptorSet), m_splatIndicesDevice);
  writeContainer.append(bindings.getWriteSet(BINDING_INDIRECT_BUFFER, m_descriptorSet), m_indirect);

  if(m_defines.dataStorage == STORAGE_TEXTURES)
  {
    // add data texture maps
    writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_TEXTURE, m_descriptorSet), m_centersMap);
    writeContainer.append(bindings.getWriteSet(BINDING_COLORS_TEXTURE, m_descriptorSet), m_colorsMap);
    writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_TEXTURE, m_descriptorSet), m_covariancesMap);
    writeContainer.append(bindings.getWriteSet(BINDING_SH_TEXTURE, m_descriptorSet), m_sphericalHarmonicsMap);
  }
  else
  {
    // add data buffers
    writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_BUFFER, m_descriptorSet), m_centersDevice);
    writeContainer.append(bindings.getWriteSet(BINDING_COLORS_BUFFER, m_descriptorSet), m_colorsDevice);
    writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_BUFFER, m_descriptorSet), m_covariancesDevice);
    writeContainer.append(bindings.getWriteSet(BINDING_SH_BUFFER, m_descriptorSet), m_sphericalHarmonicsDevice);
  }

  // write
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

  // Create the pipeline to run the compute shader for distance & culling
  {
    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_shaders.distShader,
                .pName  = "main",
            },
        .layout = m_pipelineLayout,
    };
    vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_computePipeline);
  }
  // Create the two rasterization pipelines
  {
    // Preparing the common states
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // activates blending and set blend func
    pipelineState.colorBlendEnables[0]                       = VK_TRUE;
    pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    // By default disable depth write and test for the pipeline
    // Since splats are sorted, screen aligned, and rendered back to front
    // we do not need depth test/write, which leads to faster rendering
    // however since CPU sorting mode is costly we disable it when not visualizing with alpha,
    // only in this case we will use depth test/write. this will be changed dynamically at rendering.
    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = VK_FALSE;
    pipelineState.depthStencilState.depthTestEnable  = VK_FALSE;

    // create the pipeline that uses mesh shaders
    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      // The dynamic state is used to change the depth test state dynamically
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.meshShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.fragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineMesh);
      NVVK_DBG_NAME(m_graphicsPipelineMesh);
    }

    // create the pipeline that uses vertex shaders
    {
      const auto BINDING_ATTR_POSITION    = 0;
      const auto BINDING_ATTR_SPLAT_INDEX = 1;

      pipelineState.vertexBindings   = {{// 3 component per vertex position
                                         .binding = BINDING_ATTR_POSITION,
                                         .stride  = 3 * sizeof(float),
                                       //.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                                         .divisor = 1},
                                        {// All the vertices of each splat instance will get the same index
                                         .binding   = BINDING_ATTR_SPLAT_INDEX,
                                         .stride    = sizeof(uint32_t),
                                         .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
                                         .divisor   = 1}};
      pipelineState.vertexAttributes = {
          {.location = ATTRIBUTE_LOC_POSITION, .binding = BINDING_ATTR_POSITION, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
          {.location = ATTRIBUTE_LOC_SPLAT_INDEX, .binding = BINDING_ATTR_SPLAT_INDEX, .format = VK_FORMAT_R32_UINT, .offset = 0}};

      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout                  = m_pipelineLayout;
      creator.colorFormats                         = {m_colorFormat};
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      // The dynamic state is used to change the depth test state dynamically
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.vertexShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.fragmentShader);

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipeline);
      NVVK_DBG_NAME(m_graphicsPipeline);
    }
  }
}

void GaussianSplatting::deinitPipelines()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipeline(m_device, m_graphicsPipelineMesh, nullptr);
  vkDestroyPipeline(m_device, m_computePipeline, nullptr);

  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
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

      m_alloc.createBuffer(m_splatIndicesHost, bufferSize, VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                           VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

      m_alloc.createBuffer(m_splatIndicesDevice, bufferSize,
                           VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                               | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

      m_alloc.createBuffer(m_splatDistancesDevice, bufferSize,
                           VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                               | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

      VrdxSorterStorageRequirements requirements;
      vrdxGetSorterKeyValueStorageRequirements(m_gpuSorter, splatCount, &requirements);
      m_alloc.createBuffer(m_vrdxStorageDevice, requirements.size, requirements.usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

      // for stats reporting only
      m_renderMemoryStats.allocVdrxInternal = (uint32_t)requirements.size;

      // generate debug information for buffers
      NVVK_DBG_NAME(m_splatIndicesHost.buffer);
      NVVK_DBG_NAME(m_splatIndicesDevice.buffer);
      NVVK_DBG_NAME(m_splatDistancesDevice.buffer);
      NVVK_DBG_NAME(m_vrdxStorageDevice.buffer);
    }
  }

  // create the device buffer for indirect parameters
  m_alloc.createBuffer(m_indirect, sizeof(shaderio::IndirectParams),
                       VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
                           | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

  // for statistics readback
  m_alloc.createBuffer(m_indirectReadbackHost, sizeof(shaderio::IndirectParams),
                       VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                       VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  NVVK_DBG_NAME(m_indirect.buffer);
  NVVK_DBG_NAME(m_indirectReadbackHost.buffer);

  // We create a command buffer in order to perform the copy to VRAM
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // The Quad
  const std::vector<uint16_t> indices  = {0, 2, 1, 2, 0, 3};
  const std::vector<float>    vertices = {-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0};

  // create the quad buffers
  m_alloc.createBuffer(m_quadVertices, vertices.size() * sizeof(float), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  m_alloc.createBuffer(m_quadIndices, indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

  NVVK_DBG_NAME(m_quadVertices.buffer);
  NVVK_DBG_NAME(m_quadIndices.buffer);

  //m_stagingUploader.appendBuffer(m_quadVertices, 0, std::span(vertices));
  //m_stagingUploader.appendBuffer(m_quadIndices, 0, std::span(indices));

  // buffers are small so we use vkCmdUpdateBuffer for the transfers
  vkCmdUpdateBuffer(cmd, m_quadVertices.buffer, 0, vertices.size() * sizeof(float), vertices.data());
  vkCmdUpdateBuffer(cmd, m_quadIndices.buffer, 0, indices.size() * sizeof(uint16_t), indices.data());

  //m_stagingUploader.cmdUploadAppended(cmd);
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Uniform buffer
  m_alloc.createBuffer(m_frameInfoBuffer, sizeof(shaderio::FrameInfo),
                       VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  NVVK_DBG_NAME(m_frameInfoBuffer.buffer);
}

void GaussianSplatting::deinitRendererBuffers()
{
  // TODO can we rather move this to pipelines creation/deletion ?
  if(m_gpuSorter != VK_NULL_HANDLE)
  {
    vrdxDestroySorter(m_gpuSorter);
    m_gpuSorter = VK_NULL_HANDLE;
  }

  m_alloc.destroyBuffer(m_splatDistancesDevice);
  m_alloc.destroyBuffer(m_splatIndicesDevice);
  m_alloc.destroyBuffer(m_splatIndicesHost);
  m_alloc.destroyBuffer(m_vrdxStorageDevice);

  m_alloc.destroyBuffer(m_indirect);
  m_alloc.destroyBuffer(m_indirectReadbackHost);

  m_alloc.destroyBuffer(m_quadVertices);
  m_alloc.destroyBuffer(m_quadIndices);

  m_alloc.destroyBuffer(m_frameInfoBuffer);
}

// utility functions for splat data processing

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

inline void storeSh(int format, float* srcBuffer, uint64_t srcIndex, void* dstBuffer, uint64_t dstIndex)
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

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // host buffers flags
  VkBufferUsageFlagBits2   hostBufferUsageFlags = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;
  VmaMemoryUsage           hostMemoryUsageFlags = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  VmaAllocationCreateFlags hostAllocCreateFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                                                  // for parallel access
                                                  | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  // device buffers flags
  VkBufferUsageFlagBits2 deviceBufferUsageFlags =
      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
      | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT;
  VmaMemoryUsage deviceMemoryUsageFlags = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // set of buffer to be freed after command execution
  std::vector<nvvk::Buffer> buffersToDestroy;

  // Centers
  {
    const uint32_t bufferSize = splatCount * 3 * sizeof(float);

    // allocate host and device buffers
    nvvk::Buffer hostBuffer;
    m_alloc.createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc.createBuffer(m_centersDevice, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(m_centersDevice.buffer);

    // map and fill host buffer
    memcpy(hostBuffer.mapping, m_splatSet.positions.data(), bufferSize);

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
    nvvk::Buffer hostBuffer;
    m_alloc.createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc.createBuffer(m_covariancesDevice, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(m_covariancesDevice.buffer);

    // map and fill host buffer
    float* hostBufferMapped = (float*)(hostBuffer.mapping);

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
    nvvk::Buffer hostBuffer;
    m_alloc.createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc.createBuffer(m_colorsDevice, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(m_colorsDevice.buffer);

    // fill host buffer
    float* hostBufferMapped = (float*)(hostBuffer.mapping);

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

    // same for the time beeing, would be less if we do not upload all src degrees
    int targetSplatStride = splatStride;

    // allocate host and device buffers
    const uint32_t bufferSize = splatCount * splatStride * formatSize(m_defines.shFormat);

    nvvk::Buffer hostBuffer;
    m_alloc.createBuffer(hostBuffer, bufferSize, hostBufferUsageFlags, hostMemoryUsageFlags, hostAllocCreateFlags);
    NVVK_DBG_NAME(hostBuffer.buffer);

    m_alloc.createBuffer(m_sphericalHarmonicsDevice, bufferSize, deviceBufferUsageFlags, deviceMemoryUsageFlags);
    NVVK_DBG_NAME(m_sphericalHarmonicsDevice.buffer);

    // fill host buffer
    float* hostBufferMapped = (float*)(hostBuffer.mapping);

    auto startShTime = std::chrono::high_resolution_clock::now();

    // for(uint32_t splatIdx = 0; splatIdx < splatCount; ++splatIdx)
    START_PAR_LOOP(splatCount, splatIdx)
    {
      const uint64_t srcBase   = splatStride * splatIdx;
      const uint64_t destBase  = targetSplatStride * splatIdx;
      uint64_t       dstOffset = 0;
      // degree 1, three coefs per component
      for(auto i = 0; i < 3; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const uint64_t srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + i);
          const uint64_t dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
      // degree 2, five coefs per component
      for(auto i = 0; i < 5; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const uint64_t srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + i);
          const uint64_t dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
      // degree 3, seven coefs per component
      for(auto i = 0; i < 7; i++)
      {
        for(auto rgb = 0; rgb < 3; rgb++)
        {
          const uint64_t srcIndex = srcBase + (sphericalHarmonicsCoefficientsPerChannel * rgb + 3 + 5 + i);
          const uint64_t dstIndex = destBase + dstOffset++;  // inc after add

          storeSh(m_defines.shFormat, m_splatSet.f_rest.data(), srcIndex, hostBufferMapped, dstIndex);
        }
      }
    }
    END_PAR_LOOP()

    auto      endShTime   = std::chrono::high_resolution_clock::now();
    long long buildShTime = std::chrono::duration_cast<std::chrono::milliseconds>(endShTime - startShTime).count();
    std::cout << "Sh data updated in " << buildShTime << "ms" << std::endl;

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
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);

  m_app->submitAndWaitTempCmdBuffer(cmd);

  // free temp buffers
  for(auto& buffer : buffersToDestroy)
  {
    m_alloc.destroyBuffer(buffer);
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
  m_alloc.destroyBuffer(m_centersDevice);
  m_alloc.destroyBuffer(m_colorsDevice);
  m_alloc.destroyBuffer(m_covariancesDevice);
  m_alloc.destroyBuffer(m_sphericalHarmonicsDevice);
}

///////////////////
// using texture maps to store splatset in VRAM

void GaussianSplatting::initTexture(uint32_t         width,
                                    uint32_t         height,
                                    uint32_t         bufsize,
                                    void*            data,
                                    VkFormat         format,
                                    const VkSampler& sampler,
                                    nvvk::Image&     texture)
{
  //const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  const VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkImageCreateInfo createInfo = DEFAULT_VkImageCreateInfo;
  createInfo.mipLevels         = 1;
  createInfo.extent            = {width, height, 1};
  createInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  createInfo.format            = format;

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_alloc.createImage(texture, createInfo, DEFAULT_VkImageViewCreateInfo));
  NVVK_DBG_NAME(texture.image);
  NVVK_DBG_NAME(texture.descriptor.imageView);

  NVVK_CHECK(m_stagingUploader.appendImage(texture, std::span<uint8_t>((uint8_t*)data, bufsize), imageLayout));
  m_stagingUploader.cmdUploadAppended(cmd);

  texture.descriptor.sampler = sampler;

  m_app->submitAndWaitTempCmdBuffer(cmd);
}

void GaussianSplatting::deinitTexture(nvvk::Image& texture)
{
  m_alloc.destroyImage(texture);
}

void GaussianSplatting::initDataTextures(void)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  const auto splatCount = (uint32_t)m_splatSet.positions.size() / 3;

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
                VK_FORMAT_R32G32B32A32_SFLOAT, m_sampler, m_centersMap);
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
                VK_FORMAT_R32G32B32A32_SFLOAT, m_sampler, m_covariancesMap);
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
    initTexture(mapSize.x, mapSize.y, (uint32_t)colors.size(), (void*)colors.data(), VK_FORMAT_R8G8B8A8_UNORM, m_sampler, m_colorsMap);
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
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R32G32B32A32_SFLOAT, m_sampler, m_sphericalHarmonicsMap);
    }
    else if(m_defines.shFormat == FORMAT_FLOAT16)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R16G16B16A16_SFLOAT, m_sampler, m_sphericalHarmonicsMap);
    }
    else if(m_defines.shFormat == FORMAT_UINT8)
    {
      initTexture(mapSize.x, mapSize.y, bufferSize, data, VK_FORMAT_R8G8B8A8_UNORM, m_sampler, m_sphericalHarmonicsMap);
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
  std::cout << "BENCHMARK_ADV " << m_benchmarkId << " {" << std::endl;
  std::cout << " Memory Scene; Host used \t" << m_modelMemoryStats.srcAll << "; Device Used \t" << m_modelMemoryStats.odevAll
            << "; Device Allocated \t" << m_modelMemoryStats.devAll << "; (bytes)" << std::endl;
  std::cout << " Memory Rendering; Host used \t" << m_renderMemoryStats.hostTotal << "; Device Used \t"
            << m_renderMemoryStats.deviceUsedTotal << "; Device Allocated \t" << m_renderMemoryStats.deviceAllocTotal
            << "; (bytes)" << std::endl;
  std::cout << "}" << std::endl;

  m_benchmarkId++;
}