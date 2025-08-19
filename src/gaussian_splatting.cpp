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

#define GLM_ENABLE_SWIZZLE
#include <glm/gtc/packing.hpp>  // Required for half-float operations

#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/formats.hpp>

namespace vk_gaussian_splatting {

GaussianSplatting::GaussianSplatting(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry)
    : m_profilerManager(profilerManager)
    , m_parameterRegistry(parameterRegistry)
    , cameraManip(std::make_shared<nvutils::CameraManipulator>()) {

    };

GaussianSplatting::~GaussianSplatting(){
    // all threads must be stopped,
    // work done in onDetach(),
    // could be done here, same result
};

void GaussianSplatting::onAttach(nvapp::Application* app)
{
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

  // DEBUG: uncomment and set id to find object leak
  // m_alloc.setLeakID(70);

  // set up buffer uploading utility
  m_uploader.init(&m_alloc, true);

  // Acquiring the sampler which will be used for displaying the GBuffer and accessing textures
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
  // Where to find shaders source code
  m_glslCompiler.addSearchPaths(getShaderDirs());
  // SPIRV 1.6 and VULKAN 1.4
  m_glslCompiler.defaultTarget();
  m_glslCompiler.defaultOptions();

  // Get ray tracing properties
  m_rtProperties.pNext = &m_accelStructProps;
  VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
  vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

  // init the Vulkan splatSet and the mesh set for mesh compositing
  m_splatSetVk.init(m_app, &m_alloc, &m_uploader, &m_sampler, &m_accelStructProps);
  m_meshSetVk.init(m_app, &m_alloc, &m_uploader, &m_accelStructProps);
  m_cameraSet.init(cameraManip.get());
};

void GaussianSplatting::onDetach()
{
  // stops the threads
  m_plyLoader.shutdown();
  m_cpuSorter.shutdown();
  // release scene and rendering related resources
  deinitAll();
  // release application wide related resources
  m_splatSetVk.deinit();
  m_meshSetVk.deinit();
  m_profilerGpuTimer.deinit();
  m_profilerManager->destroyTimeline(m_profilerTimeline);
  m_profilerTimeline = nullptr;
  m_gBuffers.deinit();
  m_samplerPool.releaseSampler(m_sampler);
  m_samplerPool.deinit();
  m_uploader.deinit();
  m_alloc.deinit();
}

void GaussianSplatting::onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize)
{
  m_viewSize = {viewportSize.width, viewportSize.height};
  NVVK_CHECK(m_gBuffers.update(cmd, viewportSize));
  updateRtDescriptorSet();
}

void GaussianSplatting::onPreRender()
{
  m_profilerTimeline->frameAdvance();
}

void GaussianSplatting::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  // update buffers, rebuild shaders and pipelines if needed
  processUpdateRequests();

  // 0 if not ready so the rendering does not
  // touch the splat set while loading
  // getStatus is thread safe.
  uint32_t splatCount = 0;
  if(m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
  {
    splatCount = (uint32_t)m_splatSet.size();
  }

  //////////////////
  // Full raytrace pipeline

  if(m_shaders.valid && splatCount && prmSelectedPipeline == PIPELINE_RTX)
  {
    collectReadBackValuesIfNeeded();

    updateAndUploadFrameInfoUBO(cmd, splatCount);

    raytrace(cmd, glm::vec4(1, 1, 1, 1));

    readBackIndirectParametersIfNeeded(cmd);

    updateRenderingMemoryStatistics(cmd, splatCount);

    // Attention: early return
    return;
  }

  ///////////////////
  // From this point we are using full raster or hybrid.

  // Handle device-host data update and splat sorting if a scene exist
  if(m_shaders.valid && splatCount)
  {
    // collect readback results from previous frame if any
    collectReadBackValuesIfNeeded();

    //
    updateAndUploadFrameInfoUBO(cmd, splatCount);

    if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
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
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization");

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

    // mesh first so that occluded splats fragments will be discarded by depth test
    if(m_shaders.valid && !m_meshSetVk.instances.empty())
    {
      drawMeshPrimitives(cmd);
    }

    // splat set
    if(m_shaders.valid && splatCount)
    {

      drawSplatPrimitives(cmd, splatCount);
    }

    vkCmdEndRendering(cmd);

    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  }

  // raytrace the secondary rays if needed
  if(m_shaders.valid && splatCount && !m_meshSetVk.instances.empty() && prmSelectedPipeline == PIPELINE_HYBRID)
  {
    raytrace(cmd, glm::vec4(1, 1, 1, 1));
  }

  readBackIndirectParametersIfNeeded(cmd);

  updateRenderingMemoryStatistics(cmd, splatCount);
}

void GaussianSplatting::processUpdateRequests(void)
{

  if((prmSelectedPipeline == PIPELINE_RTX || prmSelectedPipeline == PIPELINE_HYBRID) && m_requestDelayedUpdateSplatAs)
  {
    m_requestUpdateSplatAs        = true;
    m_requestDelayedUpdateSplatAs = false;
  }

  bool needUpdate = m_requestUpdateSplatData || m_requestUpdateSplatAs || m_requestUpdateMeshData
                    || m_requestUpdateShaders || m_requestUpdateLightsBuffer || m_requestDeleteSelectedMesh;

  if(!m_splatSet.size() || !needUpdate)
    return;

  vkDeviceWaitIdle(m_device);

  // updates that requires update of descriptor sets
  if(m_requestUpdateSplatData || m_requestUpdateSplatAs || m_requestUpdateMeshData || m_requestUpdateShaders || m_requestDeleteSelectedMesh)
  {

    deinitPipelines();
    deinitShaders();

    if(m_requestUpdateSplatData)
    {
      m_splatSetVk.deinitDataStorage();
      m_splatSetVk.initDataStorage(m_splatSet, prmData.dataStorage, prmData.shFormat);
    }
    if(m_requestUpdateSplatData || m_requestUpdateSplatAs)
    {
      // RTX specific
      m_splatSetVk.rtxDeinitAccelerationStructures();
      m_splatSetVk.rtxDeinitSplatModel();
      m_splatSetVk.rtxInitSplatModel(m_splatSet, prmRtxData.useTlasInstances, prmRtxData.useAABBs, prmRtxData.compressBlas,
                                     prmRtx.kernelDegree, prmRtx.kernelMinResponse, prmRtx.kernelAdaptiveClamping);
      m_splatSetVk.rtxInitAccelerationStructures(m_splatSet);
    }

    if(m_requestUpdateMeshData || m_requestDeleteSelectedMesh)
    {
      if(m_requestDeleteSelectedMesh)
        m_meshSetVk.deleteInstance(m_selectedMeshInstanceIndex);

      m_meshSetVk.rtxDeinitAccelerationStructures();
      m_meshSetVk.updateObjDescriptionBuffer();
      m_meshSetVk.rtxInitAccelerationStructures();
    }

    if(initShaders())
    {
      initPipelines();
      initRtDescriptorSet();
      initRtPipeline();
    }
  }

  // light buffer is never reallocated
  // updates does not require descripto set changes
  if(m_requestUpdateLightsBuffer)
  {
    m_lightSet.updateBuffer();
    m_requestUpdateLightsBuffer = false;
  }

  // reset request
  m_requestUpdateSplatData = m_requestUpdateSplatAs = m_requestUpdateMeshData = m_requestUpdateShaders =
      m_requestUpdateLightsBuffer = m_requestDeleteSelectedMesh = false;
}

void GaussianSplatting::updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UBO update");

  cameraManip->getLookat(m_eye, m_center, m_up);

  // Update frame parameters uniform buffer
  // some attributes of frameInfo were set by the user interface
  const glm::vec2& clip            = cameraManip->getClipPlanes();
  prmFrame.splatCount              = splatCount;
  prmFrame.viewMatrix              = cameraManip->getViewMatrix();
  prmFrame.projectionMatrix        = cameraManip->getPerspectiveMatrix();
  prmFrame.viewInverse             = glm::inverse(prmFrame.viewMatrix);
  prmFrame.projInverse             = glm::inverse(prmFrame.projectionMatrix);
  prmFrame.cameraPosition          = m_eye;
  float       devicePixelRatio     = 1.0;
  const float focalLengthX         = prmFrame.projectionMatrix[0][0] * 0.5f * devicePixelRatio * m_viewSize.x;
  const float focalLengthY         = prmFrame.projectionMatrix[1][1] * 0.5f * devicePixelRatio * m_viewSize.y;
  const bool  isOrthographicCamera = false;
  const float focalMultiplier      = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
  const float focalAdjustment      = focalMultiplier;  //  this.focalAdjustment* focalMultiplier;
  prmFrame.orthoZoom               = 1.0f;
  prmFrame.orthographicMode        = 0;  // disabled (uses perspective) TODO: activate support for orthographic
  prmFrame.viewport                = glm::vec2(m_viewSize.x * devicePixelRatio, m_viewSize.y * devicePixelRatio);
  prmFrame.basisViewport           = glm::vec2(1.0f / m_viewSize.x, 1.0f / m_viewSize.y);
  prmFrame.focal                   = glm::vec2(focalLengthX, focalLengthY);
  prmFrame.inverseFocalAdjustment  = 1.0f / focalAdjustment;
  prmFrame.lightCount              = m_lightSet.size();

  // the buffer is small so we use vkCmdUpdateBuffer for the transfer
  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);

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

  if(!prmRender.opacityGaussianDisabled)
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
      m_cpuSorter.sortAsync(glm::normalize(m_center - m_eye), m_eye, m_splatSet.positions, m_splatSetVk.transform,
                            prmRaster.cpuLazySort);
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

    // Model transform
    m_pcRaster.modelMatrix        = m_splatSetVk.transform;
    m_pcRaster.modelMatrixInverse = m_splatSetVk.transformInverse;

    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(shaderio::PushConstant), &m_pcRaster);

    vkCmdDispatch(cmd, (splatCount + prmRaster.distShaderWorkgroupSize - 1) / prmRaster.distShaderWorkgroupSize, 1, 1);

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

  // Do we need to activate depth test and Write ?
  bool needDepth = ((prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX) && prmRender.opacityGaussianDisabled)
                   || !m_meshSetVk.instances.empty();

  // Model transform
  m_pcRaster.modelMatrix        = m_splatSetVk.transform;
  m_pcRaster.modelMatrixInverse = m_splatSetVk.transformInverse;

  vkCmdPushConstants(cmd, m_pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(shaderio::PushConstant), &m_pcRaster);

  if(prmSelectedPipeline == PIPELINE_VERT)
  {  // Pipeline using vertex shader

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsVert);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

    // display the quad as many times as we have visible splats
    const VkDeviceSize offsets{0};
    vkCmdBindIndexBuffer(cmd, m_quadIndices.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_quadVertices.buffer, &offsets);
    if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
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
  {  // in mesh pipeline mode or in hybrid mode
    // Pipeline using mesh shader

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsMesh);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

    if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
    {
      // run the workgroups
      vkCmdDrawMeshTasksEXT(cmd, (prmFrame.splatCount + prmRaster.meshShaderWorkgroupSize - 1) / prmRaster.meshShaderWorkgroupSize,
                            1, 1);
    }
    else
    {
      // run the workgroups
      vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer, offsetof(shaderio::IndirectParams, groupCountX), 1,
                                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
    }
  }
}

void GaussianSplatting::drawMeshPrimitives(VkCommandBuffer cmd)
{

  NVVK_DBG_SCOPE(cmd);

  VkDeviceSize offset{0};

  // Drawing all triangles
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineMesh);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
  // overrides the pipeline setup for depth test/write
  vkCmdSetDepthWriteEnable(cmd, (VkBool32) true);
  vkCmdSetDepthTestEnable(cmd, (VkBool32) true);

  for(const Instance& inst : m_meshSetVk.instances)
  {
    auto& model                   = m_meshSetVk.meshes[inst.objIndex];
    m_pcRaster.objIndex           = inst.objIndex;  // Telling which object is drawn
    m_pcRaster.modelMatrix        = inst.transform;
    m_pcRaster.modelMatrixInverse = inst.transformInverse;

    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(shaderio::PushConstant), &m_pcRaster);
    vkCmdBindVertexBuffers(cmd, 0, 1, &model.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, model.nbIndices, 1, 0, 0, 0);
  }
}

void GaussianSplatting::collectReadBackValuesIfNeeded(void)
{
  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX && m_canCollectReadback)
  {
    std::memcpy((void*)&m_indirectReadback, (void*)m_indirectReadbackHost.mapping, sizeof(shaderio::IndirectParams));
  }
}

void GaussianSplatting::readBackIndirectParametersIfNeeded(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
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
  if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
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
    if(prmSelectedPipeline == PIPELINE_VERT)
    {
      m_renderMemoryStats.usedIndirect = 5 * sizeof(uint32_t);
    }
    else
    {
      m_renderMemoryStats.usedIndirect = sizeof(shaderio::IndirectParams);
    }
  }
  m_renderMemoryStats.usedUboFrameInfo = sizeof(shaderio::FrameInfo);
  //
  m_renderMemoryStats.rasterHostTotal =
      m_renderMemoryStats.hostAllocIndices + m_renderMemoryStats.hostAllocDistances + m_renderMemoryStats.usedUboFrameInfo;

  uint32_t vrdxSize = prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal;

  m_renderMemoryStats.rasterDeviceUsedTotal = m_renderMemoryStats.usedIndices + m_renderMemoryStats.usedDistances + vrdxSize
                                              + m_renderMemoryStats.usedIndirect + m_renderMemoryStats.usedUboFrameInfo;

  m_renderMemoryStats.rasterDeviceAllocTotal = m_renderMemoryStats.allocIndices + m_renderMemoryStats.allocDistances + vrdxSize
                                               + m_renderMemoryStats.usedIndirect + m_renderMemoryStats.usedUboFrameInfo;

  // RTX Acceleration Structures
  m_renderMemoryStats.rtxUsedTlas = m_splatSetVk.tlasSizeBytes;
  m_renderMemoryStats.rtxUsedBlas = m_splatSetVk.blasSizeBytes;

  m_renderMemoryStats.rtxHostTotal        = 0;
  m_renderMemoryStats.rtxDeviceUsedTotal  = m_renderMemoryStats.rtxUsedTlas + m_renderMemoryStats.rtxUsedBlas;
  m_renderMemoryStats.rtxDeviceAllocTotal = m_renderMemoryStats.rtxUsedTlas + m_renderMemoryStats.rtxUsedBlas;

  // Total
  m_renderMemoryStats.hostTotal = m_renderMemoryStats.rasterHostTotal + m_renderMemoryStats.rtxHostTotal;
  m_renderMemoryStats.deviceUsedTotal = m_renderMemoryStats.rasterDeviceUsedTotal + m_renderMemoryStats.rtxDeviceUsedTotal;
  m_renderMemoryStats.deviceAllocTotal = m_renderMemoryStats.rasterDeviceAllocTotal + m_renderMemoryStats.rtxDeviceAllocTotal;
}

void GaussianSplatting::deinitAll()
{
  vkDeviceWaitIdle(m_device);

  m_canCollectReadback = false;
  deinitScene();
  m_splatSetVk.resetTransform();
  m_splatSetVk.deinitDataStorage();
  m_splatSetVk.rtxDeinitSplatModel();
  m_splatSetVk.rtxDeinitAccelerationStructures();
  m_meshSetVk.deinitDataStorage();
  m_meshSetVk.rtxDeinitAccelerationStructures();
  m_lightSet.deinit();
  deinitShaders();
  deinitPipelines();
  deinitRendererBuffers();
  resetRenderSettings();
  // reset camera to default
  cameraManip->setClipPlanes({0.1F, 2000.0F});
  const glm::vec3 eye(0.0F, 0.0F, 2.0F);
  const glm::vec3 center(0.F, 0.F, 0.F);
  const glm::vec3 up(0.F, 1.F, 0.F);
  cameraManip->setLookat(eye, center, up);
  // record default cam for reset in UI
  m_cameraSet.setHomeCamera({eye, center, up, cameraManip->getFov()});
}

bool GaussianSplatting::initAll()
{
  vkDeviceWaitIdle(m_device);

  // resize the CPU sorter indices buffer
  m_splatIndices.resize(m_splatIndices.size());
  // TODO: use BBox of point cloud to set far plane, eye and center
  cameraManip->setClipPlanes({0.1F, 2000.0F});
  // we know that most INRIA models are upside down so we set the up vector to 0,-1,0
  const glm::vec3 eye(0.0F, 0.0F, 2.0F);
  const glm::vec3 center(0.F, 0.F, 0.F);
  const glm::vec3 up(0.F, 1.F, 0.F);
  cameraManip->setLookat(eye, center, up);
  // record default cam for reset in UI
  m_cameraSet.setHomeCamera({eye, center, up, cameraManip->getFov()});
  // reset general parameters
  resetRenderSettings();

  m_lightSet.init(m_app, &m_alloc, &m_uploader);
  // init a new setup
  if(!initShaders())
  {
    return false;
  }
  initRendererBuffers();
  m_splatSetVk.initDataStorage(m_splatSet, prmData.dataStorage, prmData.shFormat);
  initPipelines();

  // RTX specifics
  m_splatSetVk.rtxInitSplatModel(m_splatSet, prmRtxData.useTlasInstances, prmRtxData.useAABBs, prmRtxData.compressBlas,
                                 prmRtx.kernelDegree, prmRtx.kernelMinResponse, prmRtx.kernelAdaptiveClamping);

  m_splatSetVk.rtxInitAccelerationStructures(m_splatSet);

  initRtDescriptorSet();
  initRtPipeline();

  return true;
}

void GaussianSplatting::deinitScene()
{
  m_splatSet            = {};
  m_loadedSceneFilename = "";
}

shaderc::SpvCompilationResult GaussianSplatting::compileGlslShader(const std::string& filename, shaderc_shader_kind shaderKind)
{
  m_glslCompiler.options().AddMacroDefinition("VISUALIZE", std::to_string((int)prmRender.visualize));
  m_glslCompiler.options().AddMacroDefinition("DISABLE_OPACITY_GAUSSIAN", std::to_string((int)prmRender.opacityGaussianDisabled));
  m_glslCompiler.options().AddMacroDefinition("FRUSTUM_CULLING_MODE", std::to_string(prmRaster.frustumCulling));
  // Disabled, TODO do we enable ortho cam in the UI/camera controller
  m_glslCompiler.options().AddMacroDefinition("ORTHOGRAPHIC_MODE", "0");
  m_glslCompiler.options().AddMacroDefinition("SHOW_SH_ONLY", std::to_string((int)prmRender.showShOnly));
  m_glslCompiler.options().AddMacroDefinition("MAX_SH_DEGREE", std::to_string(prmRender.maxShDegree));
  m_glslCompiler.options().AddMacroDefinition("DATA_STORAGE", std::to_string(prmData.dataStorage));
  m_glslCompiler.options().AddMacroDefinition("SH_FORMAT", std::to_string(prmData.shFormat));
  m_glslCompiler.options().AddMacroDefinition("POINT_CLOUD_MODE", std::to_string((int)prmRaster.pointCloudModeEnabled));
  m_glslCompiler.options().AddMacroDefinition("USE_BARYCENTRIC", std::to_string((int)prmRaster.fragmentBarycentric));
  m_glslCompiler.options().AddMacroDefinition("WIREFRAME", std::to_string((int)prmRender.wireframe));
  m_glslCompiler.options().AddMacroDefinition("DISTANCE_COMPUTE_WORKGROUP_SIZE",
                                              std::to_string((int)prmRaster.distShaderWorkgroupSize));
  m_glslCompiler.options().AddMacroDefinition("RASTER_MESH_WORKGROUP_SIZE", std::to_string((int)prmRaster.meshShaderWorkgroupSize));
  m_glslCompiler.options().AddMacroDefinition("MS_ANTIALIASING", std::to_string((int)prmRaster.msAntialiasing));

  // RTX
  m_glslCompiler.options().AddMacroDefinition("KERNEL_DEGREE", std::to_string(prmRtx.kernelDegree));
  m_glslCompiler.options().AddMacroDefinition("KERNEL_MIN_RESPONSE", std::to_string(prmRtx.kernelMinResponse));
  m_glslCompiler.options().AddMacroDefinition("KERNEL_ADAPTIVE_CLAMPING", std::to_string((int)prmRtx.kernelAdaptiveClamping));
  m_glslCompiler.options().AddMacroDefinition("PAYLOAD_ARRAY_SIZE", std::to_string(prmRtx.payloadArraySize));
  m_glslCompiler.options().AddMacroDefinition("USE_RTX_PAYLOAD_BUFFER", std::to_string((int)prmRtx.usePayloadBuffer));
  m_glslCompiler.options().AddMacroDefinition("RTX_USE_INSTANCES", std::to_string((int)prmRtxData.useTlasInstances));
  m_glslCompiler.options().AddMacroDefinition("RTX_USE_AABBS", std::to_string((int)prmRtxData.useAABBs));
  m_glslCompiler.options().AddMacroDefinition("RTX_USE_MESHES", std::to_string((int)m_meshSetVk.instances.size()));
  // Hybrid
  m_glslCompiler.options().AddMacroDefinition("HYBRID_ENABLED", std::to_string((int)(prmSelectedPipeline == PIPELINE_HYBRID)));

  return m_glslCompiler.compileFile(filename, shaderKind);
}

void GaussianSplatting::createVkShaderModule(shaderc::SpvCompilationResult& spvShader, VkShaderModule& vkShaderModule)
{

  VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = m_glslCompiler.getSpirvSize(spvShader),
                                      .pCode    = m_glslCompiler.getSpirv(spvShader)};

  NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &vkShaderModule));
}

bool GaussianSplatting::initShaders(void)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  m_shaders.valid = false;

  // GS ratser
  m_allShaders.emplace_back(compileGlslShader("dist.comp.glsl", shaderc_shader_kind::shaderc_compute_shader), &m_shaders.distShader);
  m_allShaders.emplace_back(compileGlslShader("raster.vert.glsl", shaderc_shader_kind::shaderc_vertex_shader),
                            &m_shaders.vertexShader);
  m_allShaders.emplace_back(compileGlslShader("raster.mesh.glsl", shaderc_shader_kind::shaderc_mesh_shader), &m_shaders.meshShader);
  m_allShaders.emplace_back(compileGlslShader("raster.frag.glsl", shaderc_shader_kind::shaderc_fragment_shader),
                            &m_shaders.fragmentShader);
  // Mesh raster
  m_allShaders.emplace_back(compileGlslShader("mesh_raster.vert.glsl", shaderc_shader_kind::shaderc_vertex_shader),
                            &m_shaders.meshVertexShader);
  m_allShaders.emplace_back(compileGlslShader("mesh_raster.frag.glsl", shaderc_shader_kind::shaderc_fragment_shader),
                            &m_shaders.meshFragmentShader);
  // Ray trace
  m_allShaders.emplace_back(compileGlslShader("raytrace.rgen.glsl", shaderc_shader_kind::shaderc_raygen_shader),
                            &m_shaders.rtxRgenShader);
  m_allShaders.emplace_back(compileGlslShader("raytrace.rmiss.glsl", shaderc_shader_kind::shaderc_miss_shader),
                            &m_shaders.rtxRmissShader);
  m_allShaders.emplace_back(compileGlslShader("raytraceShadow.rmiss.glsl", shaderc_shader_kind::shaderc_miss_shader),
                            &m_shaders.rtxRmiss2Shader);
  m_allShaders.emplace_back(compileGlslShader("raytrace.rchit.glsl", shaderc_shader_kind::shaderc_closesthit_shader),
                            &m_shaders.rtxRchitShader);
  m_allShaders.emplace_back(compileGlslShader("raytrace.rahit.glsl", shaderc_shader_kind::shaderc_anyhit_shader),
                            &m_shaders.rtxRahitShader);
  m_allShaders.emplace_back(compileGlslShader("raytrace.rint.glsl", shaderc_shader_kind::shaderc_intersection_shader),
                            &m_shaders.rtxRintShader);

  int         errors = 0;
  std::string errorMsg;
  for(auto& shader : m_allShaders)
  {
    if(const auto numErrors = shader.spv.GetNumErrors())
    {
      errors += numErrors;
      errorMsg += shader.spv.GetErrorMessage();
    }
  }

  if(errors)
  {
    std::cerr << "\033[31m"
              << "Shader compilation failed:" << std::endl;
    std::cerr << errorMsg.c_str() << "\033[0m" << std::endl;
    return false;
  }

  for(auto& shader : m_allShaders)
  {
    createVkShaderModule(shader.spv, *shader.mod);
    NVVK_DBG_NAME(*shader.mod);
  }

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Shaders updated in " << buildTime << "ms" << std::endl;

  return (m_shaders.valid = true);
}

void GaussianSplatting::deinitShaders(void)
{
  for(auto& shader : m_allShaders)
  {
    vkDestroyShaderModule(m_device, *shader.mod, nullptr);
    *shader.mod = VK_NULL_HANDLE;
  }

  m_shaders.valid = false;
  m_allShaders.clear();
}

void GaussianSplatting::initPipelines()
{
  nvvk::DescriptorBindings bindings;

  bindings.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_DISTANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_INDICES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_INDIRECT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

  if(prmData.dataStorage == STORAGE_TEXTURES)
  {
    bindings.addBinding(BINDING_CENTERS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SCALES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_ROTATIONS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COVARIANCES_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);

    bindings.addBinding(BINDING_COLORS_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SH_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
  }
  else
  {
    bindings.addBinding(BINDING_CENTERS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SCALES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_ROTATIONS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_COVARIANCES_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

    bindings.addBinding(BINDING_COLORS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(BINDING_SH_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  }

  // Obj Mesh objectDescriptions
  bindings.addBinding(BINDING_MESH_DESCRIPTORS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bindings.addBinding(BINDING_LIGHT_SET, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

  //
  const VkPushConstantRange pcRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                            | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT,
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

  if(prmData.dataStorage == STORAGE_TEXTURES)
  {
    // add data texture maps
    writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_TEXTURE, m_descriptorSet), m_splatSetVk.centersMap);
    writeContainer.append(bindings.getWriteSet(BINDING_SCALES_TEXTURE, m_descriptorSet), m_splatSetVk.scalesMap);
    writeContainer.append(bindings.getWriteSet(BINDING_ROTATIONS_TEXTURE, m_descriptorSet), m_splatSetVk.rotationsMap);
    writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_TEXTURE, m_descriptorSet), m_splatSetVk.covariancesMap);

    writeContainer.append(bindings.getWriteSet(BINDING_COLORS_TEXTURE, m_descriptorSet), m_splatSetVk.colorsMap);
    writeContainer.append(bindings.getWriteSet(BINDING_SH_TEXTURE, m_descriptorSet), m_splatSetVk.sphericalHarmonicsMap);
  }
  else
  {
    // add data buffers
    writeContainer.append(bindings.getWriteSet(BINDING_CENTERS_BUFFER, m_descriptorSet), m_splatSetVk.centersBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_SCALES_BUFFER, m_descriptorSet), m_splatSetVk.scalesBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_ROTATIONS_BUFFER, m_descriptorSet), m_splatSetVk.rotationsBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_COVARIANCES_BUFFER, m_descriptorSet), m_splatSetVk.covariancesBuffer);

    writeContainer.append(bindings.getWriteSet(BINDING_COLORS_BUFFER, m_descriptorSet), m_splatSetVk.colorsBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_SH_BUFFER, m_descriptorSet), m_splatSetVk.sphericalHarmonicsBuffer);
  }

  if(m_meshSetVk.instances.size())
  {
    writeContainer.append(bindings.getWriteSet(BINDING_MESH_DESCRIPTORS, m_descriptorSet),
                          m_meshSetVk.objectDescriptionsBuffer.buffer);
  }

  if(m_lightSet.size())
  {
    writeContainer.append(bindings.getWriteSet(BINDING_LIGHT_SET, m_descriptorSet), m_lightSet.lightsBuffer);
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
    NVVK_DBG_NAME(m_computePipeline);
  }
  // Create the two GS rasterization pipelines
  {
    // Preparing the common states
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // activates blending and set blend func
    pipelineState.colorBlendEnables[0]                       = VK_TRUE;
    pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  //VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  //VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsMesh);
      NVVK_DBG_NAME(m_graphicsPipelineGsMesh);
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

      creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineGsVert);
      NVVK_DBG_NAME(m_graphicsPipelineGsVert);
    }
  }
  // Create the mesh rasterization pipeline
  {

    // Preparing the pipeline states
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // deactivates blending and set blend func
    pipelineState.colorBlendEnables[0]                       = VK_FALSE;
    pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    // TODOC
    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = VK_TRUE;
    pipelineState.depthStencilState.depthTestEnable  = VK_TRUE;

    // create the pipeline
    const auto BINDING_ATTR_VERTEX = 0;

    pipelineState.vertexBindings   = {{// 3 pos and 3 nrm per vertex
                                       .binding = BINDING_ATTR_VERTEX,
                                       .stride  = 6 * sizeof(float),
                                       .divisor = 1}};
    pipelineState.vertexAttributes = {{.location = ATTRIBUTE_LOC_MESH_POSITION,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(ObjVertex, pos))},
                                      {.location = ATTRIBUTE_LOC_MESH_NORMAL,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(ObjVertex, nrm))}};

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;
    // The dynamic state is used to change the depth test state dynamically
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.meshVertexShader);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.meshFragmentShader);

    creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_graphicsPipelineMesh);
    NVVK_DBG_NAME(m_graphicsPipelineMesh);
  }
}

// include RTX one
void GaussianSplatting::deinitPipelines()
{
  if(m_graphicsPipelineGsVert == VK_NULL_HANDLE)
    return;

  vkDestroyPipeline(m_device, m_graphicsPipelineGsVert, nullptr);
  m_graphicsPipelineGsVert = VK_NULL_HANDLE;
  vkDestroyPipeline(m_device, m_graphicsPipelineGsMesh, nullptr);
  m_graphicsPipelineGsMesh = VK_NULL_HANDLE;
  vkDestroyPipeline(m_device, m_graphicsPipelineMesh, nullptr);
  m_graphicsPipelineGsMesh = VK_NULL_HANDLE;
  vkDestroyPipeline(m_device, m_computePipeline, nullptr);
  m_computePipeline = VK_NULL_HANDLE;

  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);

  // RTX TODO move this in rtDeinitPipelin and invoke in proper location
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);

  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_rtDescriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescriptorSetLayout, nullptr);

  m_alloc.destroyBuffer(m_rtSBTBuffer);
  m_rtShaderGroups.clear();
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

  //m_uploader.appendBuffer(m_quadVertices, 0, std::span(vertices));
  //m_uploader.appendBuffer(m_quadIndices, 0, std::span(indices));

  // buffers are small so we use vkCmdUpdateBuffer for the transfers
  vkCmdUpdateBuffer(cmd, m_quadVertices.buffer, 0, vertices.size() * sizeof(float), vertices.data());
  vkCmdUpdateBuffer(cmd, m_quadIndices.buffer, 0, indices.size() * sizeof(uint16_t), indices.data());


  //m_uploader.cmdUploadAppended(cmd);
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

void GaussianSplatting::benchmarkAdvance()
{
  std::cout << "BENCHMARK_ADV " << m_benchmarkId << " {" << std::endl;
  std::cout << " Memory Scene; Host used \t" << m_splatSetVk.memoryStats.srcAll << "; Device Used \t"
            << m_splatSetVk.memoryStats.odevAll << "; Device Allocated \t" << m_splatSetVk.memoryStats.devAll
            << "; (bytes)" << std::endl;
  std::cout << " Memory Rasterization; Host used \t" << m_renderMemoryStats.rasterHostTotal << "; Device Used \t"
            << m_renderMemoryStats.rasterDeviceUsedTotal << "; Device Allocated \t"
            << m_renderMemoryStats.rasterDeviceAllocTotal << "; (bytes)" << std::endl;
  std::cout << " Memory Raytracing; Host used \t" << m_renderMemoryStats.rtxHostTotal << "; Device Used \t"
            << m_renderMemoryStats.rtxDeviceUsedTotal << "; Device Allocated \t"
            << m_renderMemoryStats.rtxDeviceAllocTotal << "; (bytes)" << std::endl;
  std::cout << "}" << std::endl;

  m_benchmarkId++;
}

/////////////////////////////////////////////
/// RTX

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void GaussianSplatting::initRtDescriptorSet()
{
  //SCOPED_TIMER(__FUNCTION__"\n");

  //////////////////////
  // Bindings

  m_rtDescriptorBindings.clear();

  m_rtDescriptorBindings.addBinding(RTX_BINDING_OUTIMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);

  m_rtDescriptorBindings.addBinding(RTX_BINDING_TLAS_SPLATS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                    VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_TLAS_MESH, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                    VK_SHADER_STAGE_RAYGEN_BIT_KHR);

  if(prmRtx.usePayloadBuffer)
  {
    m_rtDescriptorBindings.addBinding(RTX_BINDING_PAYLOAD_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  }

  NVVK_CHECK(m_rtDescriptorBindings.createDescriptorSetLayout(m_device, 0, &m_rtDescriptorSetLayout));
  NVVK_DBG_NAME(m_rtDescriptorSetLayout);

  //
  std::vector<VkDescriptorPoolSize> poolSize;
  m_rtDescriptorBindings.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_rtDescriptorPool));
  NVVK_DBG_NAME(m_rtDescriptorPool);

  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_rtDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_rtDescriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_rtDescriptorSet));
  NVVK_DBG_NAME(m_rtDescriptorSet);

  //////////////////////
  // Writes

  nvvk::WriteSetContainer writeContainer;

  // Output image buffer
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTIMAGE, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(), VK_IMAGE_LAYOUT_GENERAL);

  // splats TLAS
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_TLAS_SPLATS, m_rtDescriptorSet),
                        m_splatSetVk.rtAccelerationStructures.tlas);
  // mesh TLAS
  if(m_meshSetVk.instances.size())
  {
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_TLAS_MESH, m_rtDescriptorSet),
                          m_meshSetVk.rtAccelerationStructures.tlas);
  }
  // payload buffer if any
  if(prmRtx.usePayloadBuffer)
  {
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_PAYLOAD_BUFFER, m_rtDescriptorSet), m_payloadDevice);
  }

  // actually write
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void GaussianSplatting::updateRtDescriptorSet()
{
  //SCOPED_TIMER(__FUNCTION__"\n");

  // update only if the descriptor set is already initialized
  if(m_rtDescriptorSet != VK_NULL_HANDLE)
  {
    nvvk::WriteSetContainer writeContainer;

    // Output image buffer
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTIMAGE, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(), VK_IMAGE_LAYOUT_GENERAL);
    // let's update
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void GaussianSplatting::initRtPipeline()
{
  //SCOPED_TIMER(__FUNCTION__"\n");

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eAnyHit,
    eIntersection,
    eStageIndicesCount
  };

  // if not using AABBs we do not use the intersection shader (last stage listed)
  uint32_t stagesCount = prmRtxData.useAABBs ? eStageIndicesCount : eStageIndicesCount - 1;

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eStageIndicesCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module    = m_shaders.rtxRgenShader;
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  // Miss
  stage.module  = m_shaders.rtxRmissShader;
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module   = m_shaders.rtxRmiss2Shader;
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;
  // Hit Group - Closest Hit
  stage.module        = m_shaders.rtxRchitShader;
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;
  // Hit Group - Any Hit
  stage.module    = m_shaders.rtxRahitShader;
  stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  stages[eAnyHit] = stage;
  // Hit Group - Intersection (used only if useAABBs is true)
  stage.module          = m_shaders.rtxRintShader;
  stage.stage           = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
  stages[eIntersection] = stage;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  m_rtShaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  m_rtShaderGroups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  m_rtShaderGroups.push_back(group);

  if(prmRtxData.useAABBs)
  {
    // Hit 0 any hit shader with procedural intersections
    group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.anyHitShader       = eAnyHit;
    group.intersectionShader = eIntersection;
    m_rtShaderGroups.push_back(group);
  }
  else
  {
    // Hit 0 any hit shader with mesh ICOSA
    group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader       = eAnyHit;
    m_rtShaderGroups.push_back(group);
  }

  // Hit 1 Closest-hit only (for eMeshTlas)
  group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = eClosestHit;
  m_rtShaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                                       | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                                   0, sizeof(shaderio::PushConstantRay)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_descriptorSetLayout, m_rtDescriptorSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);


  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount = stagesCount;  // Stages are shaders
  rayPipelineInfo.pStages    = stages.data();

  // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
  // two miss shader groups, and one hit group.
  rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
  rayPipelineInfo.pGroups    = m_rtShaderGroups.data();

  // The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
  // hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
  // as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
  // in the ray generation to avoid deep recursion.
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;

  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);


  // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
  if(m_rtProperties.maxRayRecursionDepth <= 1)
  {
    throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 1)");
  }

  // Creating the SBT
  {
    // Shader Binding Table (SBT) setup
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(m_app->getDevice(), m_rtProperties);

    // Prepare SBT data from ray pipeline
    size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_rtPipeline, rayPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(m_alloc.createBuffer(m_rtSBTBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_rtSBTBuffer.buffer);

    // Pass the manual mapped pointer to fill the sbt data
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_rtSBTBuffer.address, bufferSize, m_rtSBTBuffer.mapping));

    // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
    m_sbtRegions = sbtGenerator.getSBTRegions();

    sbtGenerator.deinit();
  }
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void GaussianSplatting::raytrace(const VkCommandBuffer& cmdBuf, const glm::vec4& clearColor)
{
  NVVK_DBG_SCOPE(cmdBuf);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmdBuf, "Raytracing");

  // Initializing push constant values
  // m_pcRay.clearColor           = clearColor;
  m_pcRay.modelMatrix          = m_splatSetVk.transform;
  m_pcRay.modelMatrixInverse   = m_splatSetVk.transformInverse;
  m_pcRay.modelMatrixTranspose = transpose(m_splatSetVk.transform);

  std::vector<VkDescriptorSet> descSets{m_descriptorSet, m_rtDescriptorSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);

  m_pcRay.vertexAddress = m_splatSetVk.m_splatModel.vertexBuffer.address;
  m_pcRay.indexAddress  = m_splatSetVk.m_splatModel.indexBuffer.address;

  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                         | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                     0, sizeof(shaderio::PushConstantRay), &m_pcRay);


  vkCmdTraceRaysKHR(cmdBuf, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                    m_viewSize[0], m_viewSize[1], 1);
}

}  // namespace vk_gaussian_splatting