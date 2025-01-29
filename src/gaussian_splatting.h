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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _GAUSSIAN_SPLATTING_H_
#define _GAUSSIAN_SPLATTING_H_

#include <iostream>
#include <string>
#include <array>
#include <chrono>
#include <filesystem>
//
#include <vulkan/vulkan_core.h>
// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
// for parallel processing
#ifdef _WIN32
#include <ppl.h>
#include <execution>
#include <algorithm>
#endif
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
// GPU radix sort
#include <vk_radix_sort.h>
//
#include <imgui/imgui_camera_widget.h>
#include <imgui/imgui_helper.h>
#include <imgui/imgui_axis.hpp>
//
#include <nvh/primitives.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/renderpasses_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/dynamicrendering_vk.hpp>
#include <nvvk/extensions_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvkhl/alloc_vma.hpp>
#include <nvvkhl/application.hpp>
#include <nvvkhl/element_benchmark_parameters.hpp>
#include <nvvkhl/element_camera.hpp>
#include <nvvkhl/element_gui.hpp>
#include <nvvkhl/element_profiler.hpp>
#include <nvvkhl/element_nvml.hpp>
#include <nvvkhl/gbuffer.hpp>
#include <nvvkhl/pipeline_container.hpp>

namespace DH {
using namespace glm;
// Shared between host and device
#include "shaders/device_host.h" 
}

#include "splat_set.h"
#include "ply_async_loader.h"
#include "splat_sorter_async.h"
#include "sampler_texture.h"

//
class GaussianSplatting : public nvvkhl::IAppElement
{
public:  // Methods specializing IAppElement
  GaussianSplatting(std::shared_ptr<nvvkhl::ElementProfiler> profiler)
      // starts the splat sorting thread
      : m_profiler(profiler){};

  ~GaussianSplatting() override{
      // all threads must be stoped,
      // work done in onDetach(),
      // could be done here, same result
  };

  void onAttach(nvvkhl::Application* app) override;

  void onDetach() override;

  void onResize(uint32_t width, uint32_t height) override;

  void onRender(VkCommandBuffer cmd) override;

  void onUIRender() override;

  void onUIMenu() override;

  void onFileDrop(const char* filename) override { m_sceneToLoadFilename = filename; }

  // handle recent files save/load at imgui level
  void registerRecentFilesHandler();

private:  // Methods
  // main loop of the sorting thread for gaussians
  // the thread is started by the class constructor
  // then wait for triggers
  void sortingThreadFunc(void);

  void reset();

  void destroyScene();

  void createPipeline();

  void destroyPipeline();

  void createGbuffers(const glm::vec2& size);

  void destroyGbuffers();

  void createVkBuffers();

  void destroyVkBuffers();

  // create the texture maps on the device and upload the splat set data from host to device
  void createDataTextures(void);

  void destroyDataTextures(void);

  // create the buffers on the device and upload the splat set data from host to device
  void createDataBuffers(void);
  
  void destroyDataBuffers(void);

  bool initShaders(void);
  void deinitShaders(void);

  // Utility function to compute the texture size according to the size of the data to be stored
  // By default use map of 4K Width and 1K heightn then adjust the height according to the data size
  inline glm::ivec2 computeDataTextureSize(int elementsPerTexel, int elementsPerSplat, int maxSplatCount, glm::ivec2 texSize = {4096, 1024} )
  {
    while(texSize.x * texSize.y * elementsPerTexel < maxSplatCount * elementsPerSplat)
      texSize.y *= 2;
    return texSize;
  };

  // reset the attributes of the frameInformation that can
  // be modified by the user interface
  inline void resetFrameInfo()
  {
    m_frameInfo.splatScale                 = 1.0f;  // in [0.1,2.0]
    m_frameInfo.orthoZoom                  = 1.0f;  // in ?
    m_frameInfo.orthographicMode           = 0;     // disabled, in {0,1}
    m_frameInfo.pointCloudModeEnabled      = 0;     // disabled, in {0,1}
    m_frameInfo.sphericalHarmonicsDegree   = 2;     // in {0,1,2}
    m_frameInfo.sphericalHarmonics8BitMode = 0;     // disabled, in {0,1}
    m_frameInfo.showShOnly                 = 0;     // disabled, in {0,1}
    m_frameInfo.opacityGaussianDisabled    = 0;     // disabled, in {0,1}
    m_frameInfo.sortingMethod              = SORTING_GPU_SYNC_RADIX;
    m_frameInfo.frustumCulling             = FRUSTUM_CULLING_DIST;
  }

  // reset the memory usage stats
  inline void resetModelMemoryStats() { memset((void*)&m_modelMemoryStats, 0, sizeof(ModelMemoryStats)); }

  // for multiple choice selectors
  enum GuiEnums
  {
    GUI_SORTING,         // the sorting method to use
    GUI_PIPELINE,        // the rendering pipeline to use
    GUI_FRUSTUM_CULLING  // where to perform frustum culling (or disabled)
  };

  void initGui(void);

  // methods to handle recent files in file menu
  void addToRecentFiles(const std::string& filePath, int historySize = 20);

public:  // Attributes
  
  // triggers a scene load when set to non empty string
  std::string m_sceneToLoadFilename;

private:  // Attributes

  // name of the loaded scene if successfull
  std::string m_loadedSceneFilename;
  // Recent files list
  std::vector<std::string> m_recentFiles;
  // scene loader
  PlyAsyncLoader m_plyLoader;
  // loaded model
  SplatSet m_splatSet;

  // UI utility for choice menus
  ImGuiH::Registry m_ui;

  //
  nvvkhl::Application*                     m_app{nullptr};
  std::shared_ptr<nvvkhl::ElementProfiler> m_profiler;
  std::unique_ptr<nvvk::DebugUtil>         m_dutil;
  std::shared_ptr<nvvkhl::AllocVma>        m_alloc;

  glm::vec2                        m_viewSize    = {0, 0};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.0F, 0.0F, 0.0F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient sortcut to device
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                            // Descriptor set

  //
  nvvk::Buffer m_frameInfoBuffer;
  nvvk::Buffer m_pixelBuffer;

  // indirect parameters for
  // - vkCmdDrawIndexedIndirect (first 6 attr)
  // - vkCmdDrawMeshTasksIndirectEXT (last 3 attr)
  struct IndirectParams
  {
    // for vkCmdDrawIndexedIndirect
    uint32_t indexCount    = 6; // 6 indices for the quad (2 triangles)
    uint32_t instanceCount = 0; // will be incremented by the distance compute shader
    uint32_t firstIndex    = 0; 
    uint32_t vertexOffset  = 0;
    uint32_t firstInstance = 0;
    // for vkCmdDrawMeshTasksIndirectEXT
    uint32_t groupCountX = 0; // Will be incremented by distance the compute shader
    uint32_t groupCountY = 1; // Allways one workgroup on Y
    uint32_t groupCountZ = 1; // Allways one workgroup on Z
  };

  nvvk::Buffer   m_indirect;          // indirect parameter buffer
  nvvk::Buffer   m_indirectHost;      // buffer for readback
  IndirectParams m_indirectReadback;  // readback values

  //
  nvvk::Buffer m_quadVertices;  // Buffer of vertices for the splat quad
  nvvk::Buffer m_quadIndices;   // Buffer of indices for the splat quad
  
  // trigger a rebuild of the shaders and pipelines at next frame
  bool m_updateShaders = false;

  // trigger a rebuild of the data in vram (textures or buffers) at next frame
  // also triggers shaders and pipeline rebuild
  bool m_updateData = false;

  // switch between data textures and data buffers
  bool m_useDataTextures = false;

  // Data textures
  VkSampler                      m_sampler;  // texture sampler
  std::shared_ptr<SampleTexture> m_centersMap;
  std::shared_ptr<SampleTexture> m_colorsMap;
  std::shared_ptr<SampleTexture> m_covariancesMap;
  std::shared_ptr<SampleTexture> m_sphericalHarmonicsMap;

  // Data buffers
  nvvk::Buffer m_centersDevice;    
  nvvk::Buffer m_colorsDevice;  
  nvvk::Buffer m_covariancesDevice;
  nvvk::Buffer m_sphericalHarmonicsDevice;   

public:
  // rendering pipeline selector
  uint32_t m_selectedPipeline = PIPELINE_MESH;

private:
  // CPU async sorting
  SplatSorterAsync m_cpuSplatSorter;
  std::vector<uint32_t> gsIndex;      // the array of cpu sorted indices to use for rendering

  // GPU radix sort
  VrdxSorter           m_sorter = VK_NULL_HANDLE;
  VrdxSorterCreateInfo m_sorterInfo;

  // buffers used by GPU and/or CPU sort
  nvvk::Buffer m_splatIndicesHost;      // Buffer of splat indices on host for transfers (used by CPU sort)
  nvvk::Buffer m_splatIndicesDevice;    // Buffer of splat indices on device (used by CPU and GPU sort)
  nvvk::Buffer m_splatDistancesDevice;  // Buffer of splat indices on device (used by CPU and GPU sort)
  nvvk::Buffer m_vrdxStorageDevice;     // Used internally by VrdxSorter, GPU sort

  //
  nvvk::ShaderModuleManager m_shaderManager;
  
  //
  struct Shaders
  {
    nvvk::ShaderModuleID distShader;
    nvvk::ShaderModuleID meshShader;
    nvvk::ShaderModuleID vertexShader;
    nvvk::ShaderModuleID fragmentShader;
  } m_shaders;

  // Pipeline
  VkPipeline       m_graphicsPipeline     = VK_NULL_HANDLE;  // The graphic pipeline to render using vertex shaders
  VkPipeline       m_graphicsPipelineMesh = VK_NULL_HANDLE;  // The graphic pipeline to render using mesh shaders
  DH::PushConstant m_pushConst{};                            // Information sent to the shader using constant
  DH::FrameInfo    m_frameInfo{};                            // frame parameters, sent to device using a uniform buffer
  VkPipeline       m_computePipeline{};                      // The compute pipeline

  // Model related memory usage statistics
  struct ModelMemoryStats
  {
    // Memory footprint on host memory

    uint32_t srcAll     = 0;  // RAM bytes used for all the data of source model
    uint32_t srcCenters = 0;  // RAM bytes used for splat centers of source model
    // covariance
    uint32_t srcCov = 0;
    // spherical harmonics coeficients
    uint32_t srcShAll   = 0;  // RAM bytes used for all the SH coefs of source model
    uint32_t srcSh0     = 0;  // RAM bytes used for SH degree 0 of source model
    uint32_t srcShOther = 0;  // RAM bytes used for SH degree 1 of source model

    // Memory footprint on device memory (allocated)

    uint32_t devAll     = 0;  // GRAM bytes used for all the data of source model
    uint32_t devCenters = 0;  // GRAM bytes used for splat centers of source model
    // covariance
    uint32_t devCov = 0;
    // spherical harmonics coeficients
    uint32_t devShAll   = 0;  // GRAM bytes used for all the SH coefs of source model
    uint32_t devSh0     = 0;  // GRAM bytes used for SH degree 0 of source model
    uint32_t devShOther = 0;  // GRAM bytes used for SH degree 1 of source model


    // Actual data size within textures (a.k.a. mem footprint minus padding and
    // eventual unused components)

    uint32_t odevAll     = 0;  // GRAM bytes used for all the data of source model
    uint32_t odevCenters = 0;  // GRAM bytes used for splat centers of source model
    // covariance
    uint32_t odevCov = 0;
    // spherical harmonics coeficients
    uint32_t odevShAll   = 0;  // GRAM bytes used for all the SH coefs of source model
    uint32_t odevSh0     = 0;  // GRAM bytes used for SH degree 0 of source model
    uint32_t odevShOther = 0;  // GRAM bytes used for SH degree 1 of source model
  };

  // Model related memory usage statistics
  ModelMemoryStats m_modelMemoryStats;

  // Rendering (sorting and splatting) related memory usage statistics
  struct RenderMemoryStats
  {
    uint32_t usedUboFrameInfo = 0;  // used = alloc all the time
    uint32_t usedIndirect     = 0;  // used = alloc all the time, for the active pipeline

    uint32_t hostAllocDistances = 0;  // used = alloc
    uint32_t hostAllocIndices   = 0;  // used = alloc

    uint32_t allocIndices      = 0;
    uint32_t usedIndices       = 0;
    uint32_t allocDistances    = 0;
    uint32_t usedDistances     = 0;
    uint32_t allocVdrxInternal = 0;  // used is unknown

    uint32_t hostTotal        = 0;
    uint32_t deviceUsedTotal  = 0;
    uint32_t deviceAllocTotal = 0;

  };

  // Rendering (sorting and splatting) related memory usage statistics
  RenderMemoryStats m_renderMemoryStats;

  //
  float m_distTime = 0.0f;
  float m_sortTime = 0.0f;

};
#endif
