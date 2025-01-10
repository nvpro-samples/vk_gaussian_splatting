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
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "imgui/imgui_axis.hpp"
//
#include "nvh/primitives.hpp"
//
#include "nvvk/context_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/extensions_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"
//
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_profiler.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#include "splat_set.h"
#include "ply_async_loader.h"
#include "sampler_texture.h"

// TODO: for parallel sort, can be defined as a lambda in place
inline bool compare(const std::pair<float, int>& a, const std::pair<float, int>& b)
{
  return a.first > b.first;
}

// 
class GaussianSplatting : public nvvkhl::IAppElement
{
public:  // Methods specializing IAppElement
  GaussianSplatting(std::shared_ptr<nvvkhl::ElementProfiler> profiler)
      : sortingThread([this] { this->sortingThreadFunc(); })
      ,  // starts the splat sorting thread
      m_profiler(profiler){};

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

  void onUIMenu() override {}

  void onFileDrop(const char* filename) override { m_sceneToLoadFilename = filename; }

private:  // Methods
  // main loop of the sorting thread for gaussians
  // the thread is started by the class constructor
  // then wait for triggers
  void sortingThreadFunc(void);

  void destroyScene();

  void createPipeline();

  void destroyPipeline();

  void createGbuffers(const glm::vec2& size);

  void destroyGbuffers();

  void createVkBuffers();

  void destroyVkBuffers();

  // reset the attributes of the frameInformation that are
  // modified by the user interface
  inline void resetFrameInfo()
  {
    frameInfo.splatScale                 = 1.0f;  // in [0.1,2.0]
    frameInfo.orthoZoom                  = 1.0f;  // in ?
    frameInfo.orthographicMode           = 0;     // disabled, in {0,1}
    frameInfo.pointCloudModeEnabled      = 0;     // disabled, in {0,1}
    frameInfo.sphericalHarmonicsDegree   = 2;     // in {0,1,2}
    frameInfo.sphericalHarmonics8BitMode = 0;     // disabled, in {0,1}
    frameInfo.showShOnly                 = 0;     // disabled, in {0,1}
    frameInfo.opacityGaussianDisabled    = 0;     // disabled, in {0,1}
    frameInfo.gpuSorting                 = 1;     // enabled, in {0,1}
    frameInfo.culling                    = 1;     // enabled, in {0,1}
  }

  // Utility function to compute the texture size according to the size of the data to be stored
  // TODO: doc of parameters
  glm::ivec2 computeDataTextureSize(int elementsPerTexel, int elementsPerSplat, int maxSplatCount);

  // create the texture maps on the device and upload the splat set data from host to device
  void create3dgsTextures(void);

  void destroy3dgsTextures(void);

private:  // Attributes
  // indirect parameters
  struct IndirectParams
  {
    // for vkCmdDrawIndexedIndirect
    uint32_t indexCount    = 0;
    uint32_t instanceCount = 0;
    uint32_t firstIndex    = 0;
    int32_t  vertexOffset  = 0;
    uint32_t firstInstance = 0;
    // for vkCmdDrawMeshTasksIndirectEXT
    uint32_t groupCountX = 0;
    uint32_t groupCountY = 0;
    uint32_t groupCountZ = 0;
  };

  std::filesystem::path m_sceneToLoadFilename;
  PlyAsyncLoader        m_plyLoader;

  nvvkhl::Application*                     m_app{nullptr};
  std::shared_ptr<nvvkhl::ElementProfiler> m_profiler;
  std::unique_ptr<nvvk::DebugUtil>         m_dutil;
  std::shared_ptr<nvvkhl::AllocVma>        m_alloc;

  glm::vec2                        m_viewSize    = {0, 0};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.0F, 0.0F, 0.0F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                            // Descriptor set

  // Resources
  nvvk::Buffer m_frameInfo;
  nvvk::Buffer m_pixelBuffer;

  nvvk::Buffer m_splatIndicesHost;    // Buffer of splat indices on host for transfers
  nvvk::Buffer m_splatIndicesDevice;  // Buffer of splat indices on device

  nvvk::Buffer   m_indirect;          // indirect parametter buffer
  nvvk::Buffer   m_indirectHost;      // buffer for readback
  IndirectParams m_indirectReadback;  // readback values

  //
  nvvk::Buffer m_vertices;  // Buffer of vertices for the splat quad
  nvvk::Buffer m_indices;   // Buffer of indices for the splat quad

  // Data and setting
  SplatSet m_splatSet;

  // Data textures
  VkSampler                      m_sampler;  // texture sampler
  std::shared_ptr<SampleTexture> m_centersMap;
  std::shared_ptr<SampleTexture> m_colorsMap;
  std::shared_ptr<SampleTexture> m_covariancesMap;
  std::shared_ptr<SampleTexture> m_sphericalHarmonicsMap;

  // mesh shaders
  bool m_useMeshShaders = true;  // switch between vertex and mesh shaders

  bool m_supportsEXT = false;
  bool m_disableEXT  = false;
  VkPhysicalDeviceMeshShaderPropertiesEXT m_meshPropertiesEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  bool m_supportsSubgroupControl = false;
  VkPhysicalDeviceSubgroupSizeControlProperties m_subgroupSizeProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};

  // gpu/cpu sort switch
  bool m_gpuSortingEnabled = true;

  // CPU async sorting
  std::vector<std::pair<float, int>> distArray;  // dist, id

  std::thread             sortingThread;
  std::mutex              mutex;
  std::condition_variable cond_var;
  bool                    sortStart = false;
  bool                    sortDone  = false;
  bool                    sortExit  = false;
  glm::vec3               sortDir   = {1.0f, 0.0f, 0.0f};
  glm::vec3               sortCop   = {0.0f, 0.0f, 0.0f};
  std::vector<uint32_t>   gsIndex;
  std::vector<uint32_t>   sortGsIndex;
  float                   m_distTime = 0.0f;  // distance update timer
  float                   m_sortTime = 0.0f;  // distance sorting timer

  // GPU radix sort
  VrdxSorter           m_sorter = VK_NULL_HANDLE;
  VrdxSorterCreateInfo m_sorterInfo;
  std::vector<float>   m_dist;

  nvvk::Buffer m_keysDevice;    // will contain keys (distances), values (splat indices) and VkDrawIndexedIndirectCommand at the end
  nvvk::Buffer m_stagingHost;   // will contain values and splat count
  nvvk::Buffer m_storageDevice; // used internally by VrdxSorter (never read or write from to/from host)

  // Pipeline
  VkPipeline       m_graphicsPipeline     = VK_NULL_HANDLE;  // The graphic pipeline to render using vertex shaders
  VkPipeline       m_graphicsPipelineMesh = VK_NULL_HANDLE;  // The graphic pipeline to render using mesh shaders
  DH::PushConstant m_pushConst{};                            // Information sent to the shader using constant
  DH::FrameInfo    frameInfo{};                              // frame parameters, sent to device using a uniform buffer
  VkPipeline       m_computePipeline{};                      // The compute pipeline
};

#endif
