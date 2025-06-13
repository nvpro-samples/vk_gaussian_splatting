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
#include <span>
//
#include <vulkan/vulkan_core.h>
// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
// GPU radix sort
#include <vk_radix_sort.h>
//
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/pipeline.hpp>

#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>

#include <nvvk/helpers.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/default_structs.hpp>

#include <nvvkglsl/glsl.hpp>

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_default_menu.hpp>
//
#include <nvgui/camera.hpp>
#include <nvgui/axis.hpp>
#include <nvgui/enum_registry.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/file_dialog.hpp>
//
#include <nvgpu_monitor/elem_gpu_monitor.hpp>

// Shared between host and device
#include "shaderio.h"

#include "utilities.h"
#include "splat_set.h"
#include "ply_async_loader.h"
#include "splat_sorter_async.h"

namespace vk_gaussian_splatting {

class GaussianSplatting : public nvapp::IAppElement
{
public:  // Methods specializing IAppElement
  GaussianSplatting(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry, bool* benchmarkMode);

  ~GaussianSplatting() override;

  void onAttach(nvapp::Application* app) override;

  void onDetach() override;

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;

  void onPreRender() override;

  void onRender(VkCommandBuffer cmd) override;

  void onUIRender() override;

  void onUIMenu() override;

  void onFileDrop(const std::filesystem::path& filename) override { m_sceneToLoadFilename = filename; }

  // handle recent files save/load at imgui level
  void registerRecentFilesHandler();

  // Benchmarking, print extended info
  void benchmarkAdvance();

private:  // Methods
  // Initializes all that is related to the scene based
  // on current parameters. VRAM Data, shaders, pipelines.
  // Invoked on scene load success.
  void initAll();

  // Denitializes all that is related to the scene.
  // VRAM Data, shaders, pipelines.
  // Invoked on scene close or on exit.
  void deinitAll();

  // reinitializes the data related to the scene (the
  // splat set) in VRAM following a change of parameters
  // in the UI, to use data buffers or data textures.
  // this requires to regenerate shaders and pipelines.
  void reinitDataStorage();

  // reinitializes the shaders following a change of parameters
  // in the UI, this also requires to regenerate the pipelines.
  void reinitShaders();

  // free scene (splat set) from RAM
  void deinitScene();

  // create the buffers on the device and upload
  // the splat set data from host to device
  void initDataBuffers(void);

  // release buffers at next frame
  void deinitDataBuffers(void);

  // create the texture maps on the device and upload
  // the splat set data from host to device
  void initDataTextures(void);

  // release textures at next frame
  void deinitDataTextures(void);

  void initPipelines();

  void deinitPipelines();

  void initRendererBuffers();

  void deinitRendererBuffers();

  shaderc::SpvCompilationResult compileGlslShader(const std::string& filename, shaderc_shader_kind shaderKind);

  bool initShaders(void);

  void deinitShaders(void);

  // Create texture, upload data and assign sampler
  // sampler will be released by deinitTexture
  void initTexture(uint32_t width, uint32_t height, uint32_t bufsize, void* data, VkFormat format, const VkSampler& sampler, nvvk::Image& texture);

  // Destroy texture at once, texture must not be in use
  void deinitTexture(nvvk::Image& texture);

  // Utility function to compute the texture size according to the size of the data to be stored
  // By default use map of 4K Width and 1K height then adjust the height according to the data size
  inline glm::ivec2 computeDataTextureSize(int elementsPerTexel, int elementsPerSplat, int maxSplatCount, glm::ivec2 texSize = {4096, 1024})
  {
    while(texSize.x * texSize.y * elementsPerTexel < maxSplatCount * elementsPerSplat)
      texSize.y *= 2;
    return texSize;
  };

  // reset the redering settings that can
  // be modified by the user interface
  inline void resetRenderSettings()
  {
    m_frameInfo   = {};
    m_defines     = {};
    m_cpuLazySort = true;
  }

  // reset the memory usage stats
  inline void resetModelMemoryStats() { memset((void*)&m_modelMemoryStats, 0, sizeof(ModelMemoryStats)); }

  /////////////
  // Rendering submethods

  // Updates frame information uniform buffer and frame camera info
  void updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount);

  void tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount);

  void processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount);

  void drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount);

  // for statistics display in the UI
  // copy form m_indirectReadbackHost updated at previous frame to m_indirectReadback
  void collectReadBackValuesIfNeeded(void);
  // for statistics display in the UI
  // read back updated indirect parameters from m_indirect into m_indirectReadbackHost
  void readBackIndirectParametersIfNeeded(VkCommandBuffer cmd);

  void updateRenderingMemoryStatistics(VkCommandBuffer cmd, const uint32_t splatCount);

  ////////
  // UI

  // for multiple choice selectors in the UI
  enum GuiEnums
  {
    GUI_STORAGE,          // model storage in VRAM (in texture or buffer)
    GUI_SORTING,          // the sorting method to use
    GUI_PIPELINE,         // the rendering pipeline to use
    GUI_FRUSTUM_CULLING,  // where to perform frustum culling (or disabled)
    GUI_SH_FORMAT         // data format for storage of SH in VRAM
  };

  // initialize UI specifics
  void initGui(void);

  // methods to handle recent files in file menu
  void addToRecentFiles(const std::filesystem::path& filePath, int historySize = 20);

public:
  // public so that it can be accessed by main
  // Camera manipulatorr
  std::shared_ptr<nvutils::CameraManipulator> cameraManip{};

private:  // Attributes
  // benchmark mode (enabled by command line), loadings will be synchronous and vsync off
  bool* m_pBenchmarkEnabled = {};
  // screenshot file name (used by benchmark)
  std::filesystem::path m_screenshotFilename;

  // triggers a scene load at next frame when set to non empty string
  std::filesystem::path m_sceneToLoadFilename;
  // name of the loaded scene if successfull
  std::filesystem::path m_loadedSceneFilename;
  // do we load a default scene at startup if none is provided through CLI
  bool m_enableDefaultScene = true;
  // Recent files list
  std::vector<std::filesystem::path> m_recentFiles;
  // scene loader
  PlyAsyncLoader m_plyLoader;
  // loaded model
  SplatSet m_splatSet;

  // counting benchmark steps
  int m_benchmarkId = 0;

  // hide/show ui elements
  bool m_showUI = true;
  // UI utility for choice menus
  nvgui::EnumRegistry m_ui;

  //
  nvapp::Application*         m_app{nullptr};
  nvutils::ProfilerManager*   m_profilerManager;
  nvutils::ParameterRegistry* m_parameterRegistry;
  nvvk::StagingUploader       m_stagingUploader{};  // utility to upload buffers to device
  nvvk::SamplerPool           m_samplerPool{};      // The sampler pool, used to create texture samplers
  nvvk::ResourceAllocator     m_alloc;

  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;

  glm::vec2         m_viewSize    = {0, 0};
  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;    // Color format of the image
  VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;         // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {{0.0F, 0.0F, 0.0F, 1.0F}};  // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;              // Convenient sortcut to device
  nvvk::GBuffer     m_gBuffers;                                  // G-Buffers: color + depth
  //std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                            // Descriptor set

  // camera info for current frame, updated by onRender
  glm::vec3 m_eye{};
  glm::vec3 m_center{};
  glm::vec3 m_up{};

  // IndirectParams structure defined in device_host.h
  nvvk::Buffer             m_indirect;              // indirect parameter buffer
  nvvk::Buffer             m_indirectReadbackHost;  // buffer for readback
  shaderio::IndirectParams m_indirectReadback;      // readback values
  bool m_canCollectReadback = false;  // tells wether readback will be available in Host buffer at next frame

  //
  nvvk::Buffer m_quadVertices;  // Buffer of vertices for the splat quad
  nvvk::Buffer m_quadIndices;   // Buffer of indices for the splat quad

  // trigger a rebuild of the shaders and pipelines at next frame
  bool m_updateShaders = false;

  // trigger a rebuild of the data in VRAM (textures or buffers) at next frame
  // also triggers shaders and pipeline rebuild
  bool m_updateData = false;

  // Data textures
  VkSampler   m_sampler;  // texture sampler (nearest)
  nvvk::Image m_centersMap;
  nvvk::Image m_colorsMap;
  nvvk::Image m_covariancesMap;
  nvvk::Image m_sphericalHarmonicsMap;

  // Data buffers
  nvvk::Buffer m_centersDevice;
  nvvk::Buffer m_colorsDevice;
  nvvk::Buffer m_covariancesDevice;
  nvvk::Buffer m_sphericalHarmonicsDevice;

  // rasterization pipeline selector
  uint32_t m_selectedPipeline = PIPELINE_MESH;

  // CPU async sorting
  SplatSorterAsync      m_cpuSorter;
  bool                  m_cpuLazySort = true;  // if true, sorting starts only if viewpoint changed
  std::vector<uint32_t> m_splatIndices;        // the array of cpu sorted indices to use for rendering
  // GPU radix sort
  VrdxSorter m_gpuSorter = VK_NULL_HANDLE;

  // buffers used by GPU and/or CPU sort
  nvvk::Buffer m_splatIndicesHost;      // Buffer of splat indices on host for transfers (used by CPU sort)
  nvvk::Buffer m_splatIndicesDevice;    // Buffer of splat indices on device (used by CPU and GPU sort)
  nvvk::Buffer m_splatDistancesDevice;  // Buffer of splat indices on device (used by CPU and GPU sort)
  nvvk::Buffer m_vrdxStorageDevice;     // Used internally by VrdxSorter, GPU sort

  // used to load and compile shaders
  nvvkglsl::GlslCompiler m_glslCompiler{};

  // The different shaders that are used in the pipelines
  struct Shaders
  {
    VkShaderModule distShader{VK_NULL_HANDLE};
    VkShaderModule meshShader{VK_NULL_HANDLE};
    VkShaderModule vertexShader{VK_NULL_HANDLE};
    VkShaderModule fragmentShader{VK_NULL_HANDLE};
  } m_shaders;

  // This fields will be transformed to compilation definitions
  // and prepend to the shader code by initShaders
  struct ShaderDefines
  {
    int  frustumCulling          = FRUSTUM_CULLING_AT_DIST;
    bool opacityGaussianDisabled = false;
    bool showShOnly              = false;
    int  maxShDegree             = 3;  // in [0,3]
    bool pointCloudModeEnabled   = false;
    int  shFormat                = FORMAT_FLOAT32;
    int  dataStorage             = STORAGE_BUFFERS;
    bool fragmentBarycentric     = true;
  } m_defines;

  // Pipelines
  VkPipeline       m_graphicsPipeline     = VK_NULL_HANDLE;  // The graphic pipeline to render using vertex shaders
  VkPipeline       m_graphicsPipelineMesh = VK_NULL_HANDLE;  // The graphic pipeline to render using mesh shaders
  VkPipeline       m_computePipeline{};                      // The compute pipeline to compute distances and cull
  VkPipelineLayout m_pipelineLayout{};                       // Raster Pipelines layout

  VkDescriptorSetLayout m_descriptorSetLayout{};  // Descriptor set layout
  VkDescriptorSet       m_descriptorSet{};        // Raster Descriptor set
  VkDescriptorPool      m_descriptorPool{};       // Raster Descriptor pool

  shaderio::FrameInfo m_frameInfo{};      // Frame parameters, sent to device using a uniform buffer
  nvvk::Buffer        m_frameInfoBuffer;  // uniform buffer to store frame info

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
  } m_modelMemoryStats;

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

  } m_renderMemoryStats;
};

}  // namespace vk_gaussian_splatting

#endif
