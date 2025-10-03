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
// Important: include Igmlui before Vulkan
// Or undef "Status" before including imgui
#include <imgui/imgui.h>
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
#include <nvutils/logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/alignment.hpp>

#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/pipeline.hpp>
#include <nvvk/physical_device.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/sbt_generator.hpp>

#include <nvvkglsl/glsl.hpp>

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_default_menu.hpp>
//
#include <nvgui/axis.hpp>
#include <nvgui/enum_registry.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/file_dialog.hpp>
//
#include <nvgpu_monitor/elem_gpu_monitor.hpp>

// Shared between host and device
#include "shaderio.h"

#include "parameters.h"
#include "utilities.h"
#include "splat_set.h"
#include "splat_set_vk.h"
#include "ply_loader_async.h"
#include "splat_sorter_async.h"
#include "mesh_set_vk.h"
#include "light_set_vk.h"
#include "camera_set.h"

namespace vk_gaussian_splatting {

class GaussianSplatting
{
public:
  // Benchmarking, print extended info
  // invoked by parameter sequencer
  void benchmarkAdvance();

public:
  // Camera manipulator
  // public so that it can be accessed by main
  std::shared_ptr<nvutils::CameraManipulator> cameraManip{};

protected:
  GaussianSplatting(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry);

  ~GaussianSplatting();

  void onAttach(nvapp::Application* app);

  void onDetach();

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size);

  void onPreRender();

  // reset frame counter for temporal accumulated multi-sampling
  // will cause a restart of the frame construction
  inline void resetFrameCounter() { prmFrame.frameSampleId = -1; }

  void onRender(VkCommandBuffer cmd);

  // reset the rendering settings that can
  // be modified by the user interface
  inline void resetRenderSettings()
  {
    resetFrameParameters();
    resetRenderParameters();
    resetRasterParameters();
    resetRtxParameters();
  }

  // Initializes all that is related to the scene based
  // on current parameters. VRAM Data, shaders, pipelines.
  // Invoked on scene load success.
  bool initAll();

  // Denitializes all that is related to the scene.
  // VRAM Data, shaders, pipelines.
  // Invoked on scene close or on exit.
  void deinitAll();

  // free scene (splat set) from RAM
  void deinitScene();

private:
  // init the raster pipelines
  void initPipelines();

  // deinit raster and rtx pipelines TODO move rtx in separate method
  void deinitPipelines();

  void initRendererBuffers();

  void deinitRendererBuffers();

  shaderc::SpvCompilationResult compileGlslShader(const std::string& filename, shaderc_shader_kind shaderKind);
  void createVkShaderModule(shaderc::SpvCompilationResult& spvShader, VkShaderModule& vkShaderModule);

  bool initShaders(void);

  void deinitShaders(void);

  /////////////
  // Rendering submethods

  // process eventual update requests comming from UI or benchmark
  // that requires to be performed before a new rendering after a DeviceWaitIdle
  void processUpdateRequests(void);

  // Updates frame information uniform buffer and frame camera info
  void updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount);

  void tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount);

  void processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount);

  void drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount);

  void drawMeshPrimitives(VkCommandBuffer cmd);

  // for statistics display in the UI
  // copy form m_indirectReadbackHost updated at previous frame to m_indirectReadback
  void collectReadBackValuesIfNeeded(void);
  // for statistics display in the UI
  // read back updated indirect parameters from m_indirect into m_indirectReadbackHost
  void readBackIndirectParametersIfNeeded(VkCommandBuffer cmd);

  void updateRenderingMemoryStatistics(VkCommandBuffer cmd, const uint32_t splatCount);

  //////////////
  // RTX specific

  // updates the frame counter and returns true if a new raytracing pass is needed
  bool updateFrameCounter();

  void initRtDescriptorSet();
  void updateRtDescriptorSet();
  void initRtPipeline();
  void raytrace(const VkCommandBuffer& cmdBuf, bool meshDepthOnly = false);

  //////////////
  // Post processing

  void initDescriptorSetPostProcessing();
  void updateDescriptorSetPostProcessing();
  void initPipelinePostProcessing();
  void postProcess(VkCommandBuffer cmd);

protected:
  // name of the loaded scene if load is successfull
  std::filesystem::path m_loadedSceneFilename;

  // scene loader
  PlyLoaderAsync m_plyLoader;
  // 3DGS/3DGRT model in RAM
  SplatSet m_splatSet = {};
  // 3DGS/3DGRT model in VRAM
  SplatSetVk m_splatSetVk = {};
  // Set of meshes in VRAM
  MeshSetVk m_meshSetVk = {};
  // Set of lights in RAM and VRAM
  LightSetVk m_lightSet = {};
  // Set of cameras in RAM
  CameraSet m_cameraSet = {};

  // Index of the item selected in a root node of scene graph or -1 if none
  int64_t m_selectedItemIndex = -1;
  // Index of the last camera loaded
  uint64_t m_lastLoadedCamera = 0;

  // Push constant for rasterizer
  shaderio::PushConstant m_pcRaster{};

  // counting benchmark steps
  int m_benchmarkId = 0;

  // trigger a rebuild of the data in VRAM (textures or buffers) at next frame
  // also triggers shaders and pipeline rebuild
  bool m_requestUpdateSplatData = false;
  // trigger a rebuild of the splat set RTX Acceleration Structure at next frame
  bool m_requestUpdateSplatAs = false;
  // request delayed update of Acceleration Structures if not using ray tracing
  bool m_requestDelayedUpdateSplatAs = false;
  // trigger a rebuild of the shaders and pipelines at next frame
  bool m_requestUpdateShaders = false;
  // trigger the reinit of mesh acceleration structures at next frame
  bool m_requestUpdateMeshData = false;
  // trigger the update of light buffer at next frame
  bool m_requestUpdateLightsBuffer = false;
  // trigger the deletion of the selected mesh object
  bool m_requestDeleteSelectedMesh = false;

  nvapp::Application*         m_app{nullptr};
  nvutils::ProfilerManager*   m_profilerManager;
  nvutils::ParameterRegistry* m_parameterRegistry;
  nvvk::StagingUploader       m_uploader{};     // utility to upload buffers to device
  nvvk::SamplerPool           m_samplerPool{};  // The sampler pool, used to create texture samplers
  VkSampler                   m_sampler{};      // texture sampler (nearest)
  nvvk::ResourceAllocator     m_alloc;
  nvvk::PhysicalDeviceInfo    m_physicalDeviceInfo;

  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;

  glm::vec2         m_viewSize    = {0, 0};
  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;    // Color format of the image
  VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;         // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {{0.0F, 0.0F, 0.0F, 0.0F}};  // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;              // Convenient sortcut to device

  // Convenient enum to dereference color buffers in GBuffers
  enum
  {
    COLOR_MAIN = 0,
    COLOR_AUX1 = 1,
  };
  // G-Buffers: 2 color buffers + 1 depth buffer
  nvvk::GBuffer m_gBuffers;

  // camera info for current frame, updated by onRender
  glm::vec3 m_eye{};
  glm::vec3 m_center{};
  glm::vec3 m_up{};

  // IndirectParams structure defined in shaderio.h
  nvvk::Buffer             m_indirect;              // indirect parameter buffer
  nvvk::Buffer             m_indirectReadbackHost;  // buffer for readback
  shaderio::IndirectParams m_indirectReadback;      // readback values
  bool m_canCollectReadback = false;  // tells wether readback will be available in Host buffer at next frame

  // TODO maybe move that in SplatSetVK next to icosa, and the associated init/deinit
  nvvk::Buffer m_quadVertices;  // Buffer of vertices for the splat quad
  nvvk::Buffer m_quadIndices;   // Buffer of indices for the splat quad


  SplatSorterAsync      m_cpuSorter;                   // CPU async sorting
  std::vector<uint32_t> m_splatIndices;                // the array of cpu sorted indices to use for rendering
  VrdxSorter            m_gpuSorter = VK_NULL_HANDLE;  // GPU radix sort

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
    // 3D Gaussians Raster
    VkShaderModule distShader{};
    VkShaderModule meshShader{};
    VkShaderModule vertexShader{};
    VkShaderModule fragmentShader{};
    VkShaderModule threedgutMeshShader{};
    VkShaderModule threedgutFragmentShader{};
    // 3D Meshes raster
    VkShaderModule meshVertexShader{};
    VkShaderModule meshFragmentShader{};
    // for RTX
    VkShaderModule rtxRgenShader{};    // The ray generator
    VkShaderModule rtxRmissShader{};   // The miss shader
    VkShaderModule rtxRmiss2Shader{};  // For shadows (no support yet)
    VkShaderModule rtxRchitShader{};   // Closest Hit
    VkShaderModule rtxRahitShader{};   // Any Hit
    VkShaderModule rtxRintShader{};    // Interrsection
    // Post processings
    VkShaderModule postComputeShader{};
    // true if all the shaders are succesfully build
    bool valid = false;
  } m_shaders;

  // Utility storage to process m_shaders in loop
  struct ShaderEntry
  {
    shaderc::SpvCompilationResult spv;
    VkShaderModule*               mod;
  };
  std::vector<ShaderEntry> m_allShaders{};

  // 3D Gaussians Pipelines
  VkPipeline m_computePipelineGsDistCull{};  // The compute pipeline to compute gaussian splats distances to eye and cull
  VkPipeline m_graphicsPipelineGsVert = VK_NULL_HANDLE;  // The graphic pipeline to rasterize gaussian splats using vertex shaders
  VkPipeline m_graphicsPipelineGsMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize gaussian splats using mesh shaders
  VkPipeline m_graphicsPipeline3dgutMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize 3DGUT splats using mesh shaders
  // 3D Meshes Pipelines
  VkPipeline m_graphicsPipelineMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize meshes

  // Common to 3D meshes and 3D Gaussians pipeline
  VkPipelineLayout      m_pipelineLayout{};       // Raster Pipelines layout
  VkDescriptorSetLayout m_descriptorSetLayout{};  // Descriptor set layout
  VkDescriptorSet       m_descriptorSet{};        // Raster Descriptor set
  VkDescriptorPool      m_descriptorPool{};       // Raster Descriptor pool

  nvvk::Buffer m_frameInfoBuffer;  // uniform buffer to store frame parameters defined by global variable prmFrame

  // Rendering (sorting and splatting) related memory usage statistics
  struct RenderMemoryStats
  {
    // Rasterization

    uint64_t usedUboFrameInfo = 0;  // used = alloc all the time
    uint64_t usedIndirect     = 0;  // used = alloc all the time, for the active pipeline

    uint64_t hostAllocDistances = 0;  // used = alloc
    uint64_t hostAllocIndices   = 0;  // used = alloc

    uint64_t allocIndices      = 0;
    uint64_t usedIndices       = 0;
    uint64_t allocDistances    = 0;
    uint64_t usedDistances     = 0;
    uint64_t allocVdrxInternal = 0;  // used is unknown

    uint64_t rasterHostTotal        = 0;
    uint64_t rasterDeviceUsedTotal  = 0;
    uint64_t rasterDeviceAllocTotal = 0;

    // RTX
    uint64_t rtxUsedTlas = 0;
    uint64_t rtxUsedBlas = 0;

    uint64_t rtxHostTotal        = 0;
    uint64_t rtxDeviceUsedTotal  = 0;
    uint64_t rtxDeviceAllocTotal = 0;

    // Totals
    uint64_t hostTotal        = 0;
    uint64_t deviceUsedTotal  = 0;
    uint64_t deviceAllocTotal = 0;

  } m_renderMemoryStats;

  /////////////////////////
  // RTX specific

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelStructProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline m_rtPipeline;  // The RTX pipeline to ray trace gaussian splats and meshes

  nvvk::DescriptorBindings m_rtDescriptorBindings{};
  VkDescriptorSetLayout    m_rtDescriptorSetLayout{};
  VkDescriptorSet          m_rtDescriptorSet{};
  VkDescriptorPool         m_rtDescriptorPool{};

  nvvk::Buffer m_payloadDevice;

  nvvk::Buffer m_rtSBTBuffer;  // common to GS and Mesh
  // The 4 SBT regions (raygen, miss, chit, call in this order)
  nvvk::SBTGenerator::Regions m_sbtRegions{};  // common to GS and Mesh

  shaderio::PushConstantRay m_pcRay{};  // Push constant for ray tracer

  ///////////////////////////////
  // Post processing

  VkPipeline       m_computePipelinePostProcess{};
  VkPipelineLayout m_pipelineLayoutPostProcess{};

  nvvk::DescriptorBindings m_descriptorBindingsPostProcess{};
  VkDescriptorSetLayout    m_descriptorSetLayoutPostProcess{};
  VkDescriptorSet          m_descriptorSetPostProcess{};
  VkDescriptorPool         m_descriptorPoolPostProcess{};
};

}  // namespace vk_gaussian_splatting

#endif
