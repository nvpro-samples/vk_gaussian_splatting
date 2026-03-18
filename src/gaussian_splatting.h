/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
#include <nvslang/slang.hpp>

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
#include "memory_statistics.h"
#include "utilities.h"
#include "splat_set.h"
#include "splat_set_vk.h"
#include "asset_manager_vk.h"
#include "ply_loader_async.h"
#include "splat_sorter_async.h"
#include "visual_helpers_vk.h"  // 3D gizmo and grid visualization

// #DLSS
#if defined(USE_DLSS)
#include "dlss_denoiser.hpp"
#endif

#include "image_compare.h"

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

  // Check if current pipeline uses ray tracing (RTX, Hybrid, or Hybrid 3DGUT)
  inline bool isRtxPipelineActive() const
  {
    return (prmSelectedPipeline == PIPELINE_RTX || prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
  }

  // Check if DLSS is supported in current pipeline (RTX and hybrid pipelines)
  inline bool isDlssSupportedPipeline() const
  {
    return (prmSelectedPipeline == PIPELINE_RTX || prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
  }

  // Check if auto-focus is supported in current pipeline
  // Requires ray tracing for distance feedback (pure 3DGRT or hybrid 3DGUT+3DGRT)
  inline bool supportsAutoFocus() const
  {
    return (prmSelectedPipeline == PIPELINE_RTX || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
  }

  // Check if surface info (depth, normal, splat ID) is needed by any feature
  // Must match the NEED_SURFACE_INFO shader macro computation in updateSlangMacros()
  inline bool needSurfaceInfo()
  {
    bool need = (prmRender.lightingMode != LightingMode::eLightingDisabled) || (m_assets.cameras.getCamera().dofMode != DOF_DISABLED);
#if defined(USE_DLSS)
    need = need || m_dlss.isEnabled();
#endif
    const int v = prmRender.visualize;
    need        = need || (v >= VISUALIZE_DEPTH && v <= VISUALIZE_DEPTH_FOR_DLSS)
           || (v >= VISUALIZE_NORMAL && v <= VISUALIZE_NORMAL_FOR_DLSS) || (v == VISUALIZE_SPLAT_ID);
    return need;
  }

  // Reset all scene and rendering related resources (for scene/project reset)
  // NOTE: vkDeviceWaitIdle shall be invoked before calling this method
  virtual void reset();

  // free scene (splat set) from RAM
  void deinitScene();

private:
  // init the raster pipelines
  void initPipelines();

  // Update BINDING_SPLAT_TEXTURES in the descriptor set after textures are (re)created.
  // Called after processVramUpdates() to ensure descriptor set matches current GPU textures.
  void updateSplatTextureDescriptors();

  // deinit raster and rtx pipelines TODO move rtx in separate method
  void deinitPipelines();

  void initRendererBuffers();

  void deinitRendererBuffers();

  void initHelperPass();

  void deinitHelperPass();

  void updateSlangMacros(void);

  bool compileSlangShader(const std::string& filename, VkShaderModule& module);

  bool initShaders(void);

  void deinitShaders(void);

  /////////////
  // Rendering submethods

  void renderPureRaytracingPipeline(VkCommandBuffer cmd, uint32_t splatCount, bool temporalConverged);

  void renderHybridPipeline(VkCommandBuffer cmd, uint32_t splatCount, bool temporalConverged);

  void renderVisualHelpers(VkCommandBuffer cmd);

  // Output image getters (returns helper buffer if rendered, else COLOR_MAIN)
  VkImage         getOutputColorImage() const;
  VkImageView     getOutputColorImageView() const;
  VkDescriptorSet getOutputDescriptorSet() const;

  // process eventual update requests comming from UI or benchmark
  // that requires to be performed before a new rendering after a DeviceWaitIdle
  // forceAll: if true, process all requests including RTX even if not in RTX pipeline (for cleanup on reset/exit)
  void processUpdateRequests(bool forceAll = false);


  // Process GBuffer reinitialization requests (format changes)
  // Called from onPreRender before the frame command buffer is recorded
  void processGBufferUpdateRequests();

  // Updates frame information uniform buffer and frame camera info
  void updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount);

  void processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount);

  void drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount);

  void drawMeshPrimitives(VkCommandBuffer cmd, bool ftbColorPass = false);

  // Helper to check if mesh rendering pipeline should be active (meshes OR lights exist)
  inline bool shouldUseMeshPipeline() const { return !m_assets.meshes.instances.empty(); }

  // for statistics display in the UI
  // copy form m_indirectReadbackHost updated at previous frame to m_indirectReadback
  void collectReadBackValuesIfNeeded(void);
  // for statistics display in the UI
  // read back updated indirect parameters from m_indirect into m_indirectReadbackHost
  void readBackIndirectParametersIfNeeded(VkCommandBuffer cmd);

  void updateRenderingMemoryStatistics(const uint32_t splatCount);

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

  // Current load request being processed (for queue-based loading)
  SceneLoadRequest m_currentLoadRequest;

  // scene loader
  PlyLoaderAsync m_plyLoader;

  // Centralized asset management
  AssetManagerVk m_assets = {};

  // Index of the camera preset selected (-1 = active camera, >= 0 = preset index)
  int64_t m_selectedCameraPresetIndex = -1;
  // Track selected splat set instance (for UI selection)
  std::shared_ptr<SplatSetInstanceVk> m_selectedSplatInstance = nullptr;
  // Selected mesh instance (nullptr if none selected)
  std::shared_ptr<MeshInstanceVk> m_selectedMeshInstance = nullptr;
  // Selected light instance (nullptr if none selected)
  std::shared_ptr<LightSourceInstanceVk> m_selectedLightInstance = nullptr;
  // Index of the last camera loaded
  uint64_t m_lastLoadedCamera = 0;

  // Push constant for rasterizer
  shaderio::PushConstant m_pcRaster{};

  // counting benchmark steps
  int m_benchmarkId = 0;

  // Trigger a rebuild of the shaders and pipelines at next frame
  bool m_requestUpdateShaders = false;
  // Defer shader rebuild until camera animation completes
  bool m_requestUpdateShadersAfterCameraAnim = false;
  // Pipeline-specific optimization: defer RTX AS rebuild when in raster mode
  // When transforms are modified in raster mode, we don't rebuild RTX structures immediately.
  // This flag tracks that RTX needs rebuilding, and will trigger rebuild when switching to RTX pipeline.
  // This avoids expensive RTX structure updates when not actively using ray tracing.
  bool m_deferredRtxRebuildPending = false;
  // trigger update of assets buffer (bindless SceneAssets structure)
  // Set when underlying buffers change (mesh descriptors, splat descriptors, lights, etc.)
  bool m_requestUpdateAssetsBuffer = false;
  // GBuffer reinitialization (for color format changes)
  bool m_requestGBufferReinit     = false;  // Set by UI, deferred to avoid ImGui stale descriptor
  bool m_pendingGBufferReinitSeen = false;  // Two-pass deferral (onUIRender runs before onPreRender)

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

  glm::vec2 m_viewSize = {0, 0};
  VkFormat m_colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;  // Color format of the image (minimum 16-bit float for temporal accumulation precision)
  VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;       // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {0.0F, 0.0F, 0.0F, 0.0F};  // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;            // Convenient sortcut to device

  // Convenient enum to dereference color buffers in GBuffers
  enum
  {
    COLOR_MAIN              = 0,  // Main output / temporal accumulation buffer
    COLOR_AUX1              = 1,  // Temporal sampling intermediate buffer
    COLOR_COMPARISON_OUTPUT = 2,  // Comparison mode composite output
    COLOR_RASTER_NORMAL     = 3,  // Rasterization: integrated normals (RGB16F)
    COLOR_RASTER_DEPTH      = 4,  // Rasterization: picked depth (R) + transmittance (G) for FTB
    COLOR_RASTER_SPLATID    = 5,  // Rasterization: global splat ID (R32_UINT)
  };
  // G-Buffers: 9 color buffers + 1 depth buffer
  nvvk::GBuffer m_gBuffers;
  VkFormat      m_normalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;  // Normal buffer format (RGB16F + alpha for roughness)
  VkFormat m_rasterDepthFormat = VK_FORMAT_R32G32_SFLOAT;  // Raster depth buffer format (R=depth, G=transmittance for FTB)
  VkFormat m_splatIdFormat = VK_FORMAT_R32_UINT;           // Splat ID buffer format (single uint32)

  // camera info for current frame, updated by onRender
  glm::vec3 m_eye{};
  glm::vec3 m_center{};
  glm::vec3 m_up{};

  // IndirectParams structure defined in shaderio.h
  nvvk::Buffer             m_indirect;              // indirect parameter buffer
  nvvk::Buffer             m_indirectReadbackHost;  // buffer for readback
  shaderio::IndirectParams m_indirectReadback;      // readback values
  bool m_canCollectReadback = false;  // tells wether readback will be available in Host buffer at next frame

  nvvk::Buffer m_quadVertices;  // Buffer of vertices for the splat quad
  nvvk::Buffer m_quadIndices;   // Buffer of indices for the splat quad

  // macro definitions shared by all shaders
  std::vector<std::pair<std::string, std::string>> m_shaderMacros;
  // used to load and compile shaders
  nvslang::SlangCompiler m_slangCompiler{};

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
    VkShaderModule rtxRchitShader{};   // Closest Hit (for meshes)
    VkShaderModule rtxRahitShader{};   // Any Hit
    VkShaderModule rtxRintShader{};    // Interrsection
    // Post processings
    VkShaderModule postComputeShader{};
    // Deferred shading
    VkShaderModule deferredShadingShader{};
    // Particle AS build compute
    VkShaderModule particleAsBuildShader{};
    // Depth consolidation (FTB: write picked splat depth to hw depth buffer)
    VkShaderModule depthConsolidateVertShader{};
    VkShaderModule depthConsolidateFragShader{};
    // Utility storage to process shaders in loop
    std::vector<VkShaderModule*> modules{};
    // true if all the shaders are succesfully build
    bool valid = false;
  } m_shaders;

  // 3D Gaussians Pipelines
  VkPipeline m_computePipelineGsDistCull = VK_NULL_HANDLE;  // The compute pipeline to compute gaussian splats distances to eye and cull
  VkPipeline m_computePipelineParticleAs = VK_NULL_HANDLE;  // Compute pipeline to generate particle AS buffers
  VkPipeline m_graphicsPipelineGsVert = VK_NULL_HANDLE;  // The graphic pipeline to rasterize gaussian splats using vertex shaders
  VkPipeline m_graphicsPipelineGsMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize gaussian splats using mesh shaders
  VkPipeline m_graphicsPipeline3dgutMesh = VK_NULL_HANDLE;  // The graphic pipeline to rasterize 3DGUT splats using mesh shaders
  // 3D Meshes Pipelines
  VkPipeline m_graphicsPipelineMesh             = VK_NULL_HANDLE;  // The graphic pipeline to rasterize meshes
  VkPipeline m_graphicsPipelineMeshFtbColor     = VK_NULL_HANDLE;  // FTB: mesh color pass with additive blend
  VkPipeline m_graphicsPipelineDepthConsolidate = VK_NULL_HANDLE;  // FTB: consolidate picked depth to hw depth

  // Common to 3D meshes and 3D Gaussians pipeline
  VkPipelineLayout         m_pipelineLayout           = VK_NULL_HANDLE;  // Raster Pipelines layout
  VkPipelineLayout         m_particleAsPipelineLayout = VK_NULL_HANDLE;  // Particle AS compute pipeline layout
  nvvk::DescriptorBindings m_descriptorBindings       = {};              // Raster Descriptor bindings
  VkDescriptorSetLayout    m_descriptorSetLayout      = VK_NULL_HANDLE;  // Descriptor set layout
  VkDescriptorSet          m_descriptorSet            = VK_NULL_HANDLE;  // Raster Descriptor set
  VkDescriptorPool         m_descriptorPool           = VK_NULL_HANDLE;  // Raster Descriptor pool

  nvvk::Buffer m_frameInfoBuffer;  // uniform buffer to store frame parameters defined by global variable prmFrame

  /////////////////////////
  // RTX specific

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelStructProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout = VK_NULL_HANDLE;
  VkPipeline m_rtPipeline = VK_NULL_HANDLE;  // The RTX pipeline to ray trace gaussian splats and meshes

  nvvk::DescriptorBindings m_rtDescriptorBindings  = {};
  VkDescriptorSetLayout    m_rtDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet          m_rtDescriptorSet       = VK_NULL_HANDLE;
  VkDescriptorPool         m_rtDescriptorPool      = VK_NULL_HANDLE;

  nvvk::Buffer m_payloadDevice;

  nvvk::Buffer m_rtSBTBuffer;  // common to GS and Mesh
  // The 4 SBT regions (raygen, miss, chit, call in this order)
  nvvk::SBTGenerator::Regions m_sbtRegions{};  // common to GS and Mesh

  shaderio::PushConstantRay m_pcRay{};  // Push constant for ray tracer

  ///////////////////////////////
  // Deferred shading (compute shader for lighting in raster-only pipelines)
  VkPipeline m_computePipelineDeferredShading = VK_NULL_HANDLE;

  // Post processing

  VkPipeline       m_computePipelinePostProcess = VK_NULL_HANDLE;
  VkPipelineLayout m_pipelineLayoutPostProcess  = VK_NULL_HANDLE;

  nvvk::DescriptorBindings m_descriptorBindingsPostProcess{};
  VkDescriptorSetLayout    m_descriptorSetLayoutPostProcess = VK_NULL_HANDLE;
  VkDescriptorSet          m_descriptorSetPostProcess       = VK_NULL_HANDLE;
  VkDescriptorPool         m_descriptorPoolPostProcess      = VK_NULL_HANDLE;

  ///////////////////////////////
  // DLSS
#if defined(USE_DLSS)
  DlssDenoiser m_dlss;
  glm::mat4    m_prevMVP{1.0f};              // Previous frame's model-view-projection for motion vectors
  glm::vec2    m_currentJitter{0.0f, 0.0f};  // Current frame's jitter for DLSS
#endif

  ///////////////////////////////
  // Image Comparison
  ImageCompare m_imageCompare;
  bool         m_requestCaptureComparison = false;  // Request to capture reference at end of next frame

  // Returns the appropriate buffer based on comparison mode and visualization settings
  VkDescriptorSet getPresentationImageDescriptorSet(void);

  ///////////////////////////////
  // Visual Helpers (3D Gizmo and Grid)
  VisualHelpers m_helpers;
  bool          m_showLightProxies = true;  // Show/hide light proxy meshes

  ///////////////////////////////
  // Helper Methods for Image Comparison

  // Returns the source image info for the current visualization mode (used by ImageCompare)
  ImageCompare::ImageInfo getCurrentVisualizationImageInfo() const;

#if defined(USE_DLSS)
  // Helper to map visualization mode to DLSS buffer
  shaderio::DlssImages getDlssBufferForVisuMode(int visualizeMode) const;
#endif
};

}  // namespace vk_gaussian_splatting

#endif
