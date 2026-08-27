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

// Vulkan Memory Allocator
#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include "gaussian_splatting.h"
#include "utilities.h"
#include "memory_statistics.h"
#include "hardware_support.h"

#define GLM_ENABLE_SWIZZLE
#include <glm/gtc/packing.hpp>  // Required for half-float operations

#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/formats.hpp>

#include <vulkan/vk_enum_string_helper.h>

// #DLSS - Include shader utilities for host-side functions
#include "dlss_util.h"

namespace vk_gaussian_splatting {

GaussianSplatting::GaussianSplatting(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry)
    : m_profilerManager(profilerManager)
    , m_parameterRegistry(parameterRegistry)
    , cameraManip(std::make_shared<nvutils::CameraManipulator>())
{
  cameraManip->setAnimationDuration(1.0);

#if defined(USE_DLSS)
  // Register DLSS parameters
  m_dlss.registerParameters(parameterRegistry);
#endif
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

  // Pipeline cache (in-memory only, avoids expensive recompilation of unchanged shader variants)
  VkPipelineCacheCreateInfo cacheCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  NVVK_CHECK(vkCreatePipelineCache(m_device, &cacheCreateInfo, nullptr, &m_pipelineCache));

  // profiling
  m_profilerTimeline = m_profilerManager->createTimeline({.name = "Primary Timeline"});
  m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(), m_app->getQueue(0).familyIndex, false);

  // starts the asynchronous services
  m_plyLoader.initialize();

  // Memory allocator
  m_alloc.init(VmaAllocatorCreateInfo{
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = app->getPhysicalDevice(),
      .device           = app->getDevice(),
      .instance         = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });

  // DEBUG: uncomment and set id to find object leak
  // m_alloc.setLeakID(231);

  // set up buffer uploading utility
  m_uploader.init(&m_alloc, true);

  // Acquiring the sampler which will be used for displaying the GBuffer and accessing textures
  m_samplerPool.init(app->getDevice());
  NVVK_CHECK(m_samplerPool.acquireSampler(m_sampler));
  NVVK_DBG_NAME(m_sampler);

  // GBuffer
  m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

  // Seven GBuffer color attachments:
  // - MAIN: primary output / temporal accumulation buffer
  // - AUX1: temporal sampling intermediate buffer
  // - COMPARISON_OUTPUT: dedicated buffer for comparison mode composite output
  // - RASTER_NORMAL: rasterization integrated normals (for surface reconstruction)
  // - RASTER_DEPTH: rasterization picked depth (R) + transmittance (G) for FTB
  // - RASTER_SPLATID: rasterization global splat ID (for material lookup in hybrid mode)
  // - LDR: tone-mapped output (R8G8B8A8_UNORM)
  m_gBuffers.init({
      .allocator      = &m_alloc,
      .colorFormats   = {prmRender.colorFormat, prmRender.colorFormat, prmRender.colorFormat, m_normalFormat,
                         m_rasterDepthFormat, m_splatIdFormat, VK_FORMAT_R8G8B8A8_UNORM},
      .depthFormat    = m_depthFormat,
      .imageSampler   = m_sampler,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  });

  // Setting up the Slang compiler
  {
    // Where to find shaders source code
    m_slangCompiler.addSearchPaths(getShaderDirs());
    // SPIRV 1.6 and VULKAN 1.4
    m_slangCompiler.defaultTarget();
    m_slangCompiler.defaultOptions();
    // Declare the subgroup capabilities used by wave intrinsics (e.g. WavePrefixSum /
    // WaveGetLaneCount in the tonemapper auto-exposure pass) up front, so Slang does not
    // "implicitly upgrade" the spirv_1_6 profile and emit warning E41012 on those entry points.
    m_slangCompiler.addCapability("spvGroupNonUniformBallot");
    m_slangCompiler.addCapability("spvGroupNonUniformArithmetic");
    m_slangCompiler.addOption({slang::CompilerOptionName::MatrixLayoutRow, {slang::CompilerOptionValueKind::Int, 1}});
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation,
                               {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL}});
    m_slangCompiler.addOption({slang::CompilerOptionName::Optimization,
                               {slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_DEFAULT}});
  }

  // Get device information
  m_physicalDeviceInfo.init(m_app->getPhysicalDevice(), VK_API_VERSION_1_4);

  // Get ray tracing properties
  m_rtProperties.pNext = &m_accelStructProps;
  VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
  vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

  // Initialize centralized asset manager
  m_assets.init(m_app, &m_alloc, &m_uploader, cameraManip.get(), &m_sampler, &m_physicalDeviceInfo, &m_accelStructProps,
                m_profilerTimeline);

  // add camera to default position
  m_assets.cameras.setCamera(Camera());
  m_assets.cameras.setHomePreset(m_assets.cameras.getCamera());

#if defined(USE_DLSS)
  // Initialize DLSS
  DlssDenoiser::InitResources dlssRes{
      .instance       = app->getInstance(),
      .allocator      = &m_alloc,
      .samplerPool    = &m_samplerPool,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  };
  m_dlss.init(dlssRes);

  // Initialize NGX context
  m_dlss.ensureInitialized(dlssRes);

  // Bridge the NGX runtime probe result into the global capability registry
  // so the rest of the application can guard DLSS-only paths via
  // isSupported.DLSS instead of duplicating the probe.
  isSupported.setDlssRuntimeAvailable(m_dlss.isRuntimeSupported());

  // Note: DLSS G-buffers and output image will be set in onResize()
  // when the viewport G-buffers are created
#endif

  // Initialize ImageCompare
  ImageCompare::Resources imageCompareRes{
      .device        = m_device,
      .allocator     = &m_alloc,
      .sampler       = m_sampler,
      .profiler      = &m_profilerGpuTimer,
      .slangCompiler = &m_slangCompiler,
      .parameters    = &prmComparison,
  };
  m_imageCompare.init(imageCompareRes);

  // Initialize renderer buffers
  initRendererBuffers();

  // Initialize Visual Helpers (3D gizmo and grid)
  VisualHelpers::Resources helperRes{
      .app           = m_app,
      .alloc         = &m_alloc,
      .uploader      = &m_uploader,
      .device        = m_device,
      .sampler       = m_sampler,
      .slangCompiler = &m_slangCompiler,
      .colorFormat   = prmRender.colorFormat,
      .depthFormat   = m_depthFormat,
  };
  m_helpers.init(helperRes);

  // Initialize Sky, Sun & IBL
  m_sky.init({
      .device        = m_device,
      .alloc         = &m_alloc,
      .samplerPool   = &m_samplerPool,
      .uploader      = &m_uploader,
      .slangCompiler = &m_slangCompiler,
      .profiler      = &m_profilerGpuTimer,
  });

  // Dummy buffer for RTX_BINDING_ENV_ACCEL when no HDR is loaded
  {
    NVVK_CHECK(m_alloc.createBuffer(m_envAccelDummyBuffer, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    NVVK_DBG_NAME(m_envAccelDummyBuffer.buffer);
  }

  // Initialize Tonemapper
  initTonemapper();

  // Request initial shader compilation
  m_requestUpdateShaders = true;
};


void GaussianSplatting::onDetach()
{
  vkDeviceWaitIdle(m_device);

  // stops the threads
  m_plyLoader.shutdown();

  // Release scene and rendering related resources (marks assets for deletion)
  // This calls processUpdateRequests(true) internally to ensure complete cleanup
  reset();

  // Deinitialize centralized asset manager
  m_assets.deinit();

  // Deinitialize pipelines, shaders, and related resources
  // (includes m_rtSBTBuffer, descriptor pools/layouts, etc.)
  deinitPipelines();
  deinitShaders();

  // Deinitialize renderer buffers
  deinitRendererBuffers();

  m_profilerGpuTimer.deinit();
  m_profilerManager->destroyTimeline(m_profilerTimeline);
  m_profilerTimeline = nullptr;
  m_gBuffers.deinit();

  // #DLSS - Cleanup DLSS denoiser
#if defined(USE_DLSS)
  DlssDenoiser::InitResources dlssRes{
      .instance       = m_app->getInstance(),
      .allocator      = &m_alloc,
      .samplerPool    = &m_samplerPool,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  };
  m_dlss.deinit(dlssRes);
#endif

  // Cleanup ImageCompare
  m_imageCompare.deinit();

  // Cleanup Visual Helpers
  m_helpers.deinit();

  // Cleanup Tonemapper
  deinitTonemapper();

  // Cleanup Sky, Sun & IBL
  m_sky.deinit();
  if(m_envAccelDummyBuffer.buffer != VK_NULL_HANDLE)
    m_alloc.destroyBuffer(m_envAccelDummyBuffer);

  m_samplerPool.releaseSampler(m_sampler);
  m_samplerPool.deinit();
  m_uploader.deinit();
  m_alloc.deinit();

  if(m_pipelineCache != VK_NULL_HANDLE)
  {
    vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
    m_pipelineCache = VK_NULL_HANDLE;
  }
}


void GaussianSplatting::onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize)
{
  m_viewSize = {viewportSize.width, viewportSize.height};
  NVVK_CHECK(m_gBuffers.update(cmd, viewportSize));

  // Track main GBuffer memory (updated on resize)
  uint32_t colorFormatSize = getColorFormatBytesPerPixel(prmRender.colorFormat);
  uint32_t depthFormatSize = 4;  // D32_SFLOAT = 4 bytes per pixel
  vk_gaussian_splatting::memRender.gBuffersColor = uint64_t(viewportSize.width) * viewportSize.height * colorFormatSize * 3;  // 3 color attachments
  vk_gaussian_splatting::memRender.gBuffersDepth = uint64_t(viewportSize.width) * viewportSize.height * depthFormatSize;

  // Update Visual Helpers (manages helper GBuffer and descriptor set)
  m_helpers.onResize(cmd, viewportSize, m_gBuffers.getDepthImage(), m_gBuffers.getDepthImageView(), m_sampler);

#if defined(USE_DLSS)
  // Always update DLSS size (following vk_gltf_renderer pattern)
  // This ensures DLSS is ready regardless of enable state
  m_dlss.updateSize(cmd, viewportSize);
  m_dlss.setResources();
  m_dlss.setResource(DlssRayReconstruction::ResourceType::eColorOut, m_gBuffers.getColorImage(COLOR_MAIN),
                     m_gBuffers.getColorImageView(COLOR_MAIN), m_gBuffers.getColorFormat(COLOR_MAIN));
#endif

  updateRtDescriptorSet();
  updateDescriptorSetPostProcessing();
  resetFrameCounter();

  // Notify ImageCompare of resize (invalidates reference and auto-disables comparison if size changed)
  m_imageCompare.resize(viewportSize);
}


void GaussianSplatting::onPreRender()
{
  m_profilerTimeline->frameAdvance();

  // Reset helper rendering flag at start of frame
  m_helpers.resetFrameState();

  // Handle GBuffer format change if requested (before frame command buffer starts)
  // Deferred GBuffer reinit (two-pass to avoid ImGui stale descriptor issue)
  // onUIRender() runs before onPreRender(), so we wait one extra frame before destroying resources
  if(m_requestGBufferReinit)
  {
    if(m_pendingGBufferReinitSeen)
    {
      m_requestGBufferReinit     = true;
      m_pendingGBufferReinitSeen = false;
      processGBufferUpdateRequests();
      m_requestUpdateShaders = true;
    }
    else
    {
      m_pendingGBufferReinitSeen = true;
    }
  }

  // Coerce ray tracing primitive to a supported value.
  // A previous session, a loaded project file, or a command-line argument
  // can request the LSS sphere primitive on hardware that does not expose
  // VK_NV_ray_tracing_linear_swept_spheres. Detect that here so the
  // correction is applied no matter how prmRtxData.useSpheres got set, and
  // queue the BLAS / shader rebuild needed to materialize the new mode.
  if(prmRtxData.useSpheres && !isSupported.rayTracingLinearSweptSpheres)
  {
    LOGW("VK_NV_ray_tracing_linear_swept_spheres not available - falling back to Icosahedron particle format.\n");
    prmRtxData.useSpheres = false;
    prmRtxData.useAABBs   = false;
    m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
    m_requestUpdateShaders = true;
  }

  // Coerce the selected pipeline to a supported value when VK_EXT_mesh_shader
  // is missing. The default, project files and the pipeline hotkeys all may
  // request a mesh-shader-based pipeline; on hardware without mesh shaders the
  // matching SPIR-V modules and graphics pipelines are not created
  // (initShaders / initPipelines) and the draw paths would crash. Falling
  // back to PIPELINE_VERT keeps the closest raster-3DGS behaviour available.
  if(!isSupported.meshShader && isMeshShaderPipeline(prmSelectedPipeline))
  {
    LOGW("VK_EXT_mesh_shader not available - falling back to vertex raster pipeline (PIPELINE_VERT).\n");
    prmSelectedPipeline    = PIPELINE_VERT;
    m_requestUpdateShaders = true;
  }

  // Coerce the selected pipeline to a non-RTX one when ray tracing is not
  // available on this device. The composite raytracing flag covers
  // VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure +
  // VK_KHR_deferred_host_operations; without it the RT shaders are not
  // compiled, m_rtPipeline stays VK_NULL_HANDLE and processVramUpdates skips
  // BLAS/TLAS work, so any RTX/Hybrid selection from the default, project
  // file or hotkeys would render nothing. Fall back to the vertex raster
  // pipeline, which is always available on the Vulkan 1.4 baseline.
  if(!isSupported.raytracing && isRtxPipelineActive())
  {
    LOGW("Ray tracing extensions not available - falling back to a raster pipeline.\n");
    prmSelectedPipeline    = PIPELINE_VERT;
    m_requestUpdateShaders = true;
  }

  coerceRasterParamsForSortingMethod();

  // Coerce particle depth away from billboard when the current configuration
  // does not support it. Billboard depth requires either AABBs (custom
  // intersection) or stochastic any-hit trace strategy.
  if(prmRtx.particleDepth == PARTICLE_DEPTH_BILLBOARD && !prmRtxData.useAABBs && prmRtx.rtxTraceStrategy != RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT)
  {
    prmRtx.particleDepth = PARTICLE_DEPTH_ELLIPSOID;
    m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
    m_requestUpdateShaders = true;
  }

  // Disable the SV_Barycentrics raster optimization when the device does not
  // expose VK_KHR_fragment_shader_barycentric. The flag is wired into the
  // USE_BARYCENTRIC slang macro (updateSlangMacros), so toggling it off
  // requires a shader rebuild to drop the SPIR-V capability.
  if(prmRaster.fragmentBarycentric && !isSupported.fragmentShaderBarycentric)
  {
    LOGW("VK_KHR_fragment_shader_barycentric not available - disabling fragment shader barycentric.\n");
    prmRaster.fragmentBarycentric = false;
    m_requestUpdateShaders        = true;
  }

  // Automatic temporal sampling settings
  {
    const bool prevTemporalSampling = prmRtx.temporalSampling;

    // Automatic temporal accumulation:
    // enabled if depth of field, soft shadows (RTX only), RTX lighting (uses stochastic NEE),
    // environment lighting, non-full-pass tracing (RTX only), or stochastic raster (raster only)
    if(prmRtx.temporalSamplingMode == TEMPORAL_SAMPLING_AUTO
       && (m_assets.cameras.getCamera().dofMode != DOF_DISABLED
           || (prmRender.shadowsMode == ShadowsMode::eShadowsSoft && isRtxPipelineActive() && effectiveLightingMode())
           || (m_sky.isEnabled() && m_sky.mode() != shaderio::EnvironmentMode::eNone && effectiveLightingMode())
           || (isRtxPipelineActive() && effectiveLightingMode())
           || (isRtxPipelineActive() && prmRtx.rtxTraceStrategy != RTX_TRACE_STRATEGY_FULL_ANYHIT)
           || (isRasterPipelineActive() && prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT)))
    {
      prmRtx.temporalSampling = true;
    }
    else  // or if forced ENABLED
    {
      prmRtx.temporalSampling = (prmRtx.temporalSamplingMode == TEMPORAL_SAMPLING_ENABLED);
    }

    // Update metrics history size if temporal sampling state changed and comparison is active
    if(prevTemporalSampling != prmRtx.temporalSampling && prmComparison.enabled && m_imageCompare.hasValidCaptureImage())
    {
      int historySize = prmRtx.temporalSampling ? prmFrame.frameSampleMax : 25;
      m_imageCompare.setMetricsHistorySize(historySize);
    }
  }

  // Coerce DLSS state before rebuilding shaders so that the pipelines
  // created by processUpdateRequests() match the state used in onRender().
#if defined(USE_DLSS)
  if(m_dlss.isEnabled() && !isDlssSupportedPipeline())
  {
    m_dlss.setEnabled(false);
    m_requestUpdateShaders = true;
  }
#endif

  // Process shader rebuilds, buffer updates, RTX AS updates
  processUpdateRequests();

  // Fall back to raster if RTX is requested but acceleration structures are
  // not ready (e.g., first load, OOM).  Skip while async loading is in progress.
  // Must run AFTER processUpdateRequests() so the TLAS has a chance to be
  // built; otherwise the fallback fires every frame and RTX can never activate.
  {
    const uint32_t splatCount = m_assets.splatSets.getTotalGlobalSplatCount();
    bool isLoading = (m_plyLoader.getStatus() == PlyLoaderAsync::State::E_LOADING) || !prmScene.sceneLoadQueue.empty()
                     || !prmScene.projectToLoadFilename.empty();
    bool splatRtxValid = m_assets.splatSets.isRtxValid();
    bool meshRtxValid =
        !m_assets.meshes.instances.empty() && (m_assets.meshes.rtAccelerationStructures.tlas.accel != VK_NULL_HANDLE);
    bool rtxValid    = splatRtxValid || meshRtxValid;
    bool hasGeometry = (splatCount > 0) || !m_assets.meshes.instances.empty();
    if(!rtxValid && hasGeometry && isRtxPipelineActive() && !isLoading)
    {
      prmSelectedPipeline    = isSupported.meshShader ? PIPELINE_MESH : PIPELINE_VERT;
      m_requestUpdateShaders = true;
    }
  }
}

void GaussianSplatting::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);


  // Collect metrics result from previous frame if available
  m_imageCompare.collectMetricsResult();

#if defined(USE_DLSS)
  // Check if DLSS size mode has changed and update if needed
  if(m_dlss.isEnabled() && m_dlss.needsSizeUpdate())
  {
    // Wait for GPU to finish using the current descriptors before updating them
    vkDeviceWaitIdle(m_device);

    // Update DLSS rendering size based on new size mode
    m_dlss.updateSize(cmd, m_gBuffers.getSize());

    // Update DLSS resources after size change
    m_dlss.setResources();

    // Set the output image resource (remains the same)
    m_dlss.setResource(DlssRayReconstruction::ResourceType::eColorOut, m_gBuffers.getColorImage(COLOR_MAIN),
                       m_gBuffers.getColorImageView(COLOR_MAIN), prmRender.colorFormat);

    // Update ray tracing descriptors to bind new DLSS G-buffer sizes
    updateRtDescriptorSet();

    // Reset frame counter for temporal accumulation
    resetFrameCounter();
  }
#endif

  // 0 if not at least one valide splat set
  const uint32_t splatCount = m_assets.splatSets.getTotalGlobalSplatCount();

  // Update frame counter once per frame and check if temporal sampling has converged
  // IMPORTANT: Call updateFrameCounter() only once to avoid incrementing frameSampleId multiple times
  bool shouldContinueSampling = updateFrameCounter();
  bool temporalConverged      = prmRtx.temporalSampling && !shouldContinueSampling;
#if defined(USE_DLSS)
  temporalConverged = !m_dlss.isEnabled() && temporalConverged;
#endif

  // Bake sky equirect texture if needed (only when parameters change)
  if(m_sky.bake(cmd))
    resetFrameCounter();

  // Update descriptor set if the environment texture changed (bake, HDR load, mode switch)
  if(m_sky.imageViewChanged())
  {
    updateRtDescriptorSet();
    m_sky.clearImageViewChanged();
  }

  // Do the actual rendering
  // RTX pipeline works for any scene composition (particles, meshes, environment-only, or empty)
  if(m_shaders.valid && isRtxPipelineOnly())
  {
    renderPureRaytracingPipeline(cmd, splatCount, temporalConverged);
  }
  else
  {
    // also handles the fallback display when no valid scene or shader compilation errors
    renderHybridPipeline(cmd, splatCount, temporalConverged);
  }

  // Apply tone mapping (after DLSS/postProcess, before visual helpers)
  tonemap(cmd);

  // Render visual helpers (3D gizmo and grid)
  renderVisualHelpers(cmd);

  readBackIndirectParametersIfNeeded(cmd);

  updateRasterizationMemoryStatistics(splatCount);

  updateMemoryStatistics();

  // Capture comparison reference if requested
  if(m_requestCaptureComparison)
  {
    // Configure metrics history buffer size based on temporal sampling
    int historySize = prmRtx.temporalSampling ? prmFrame.frameSampleMax : 1000;
    m_imageCompare.setMetricsHistorySize(historySize);

    // capture
    m_imageCompare.capture(cmd, getCurrentVisualizationImageInfo());
    m_requestCaptureComparison = false;
  }

  // Perform comparison composite if comparison mode is active
  if(prmComparison.enabled && m_imageCompare.hasValidCaptureImage())
  {
    VkImageView outputView = m_gBuffers.getColorImageView(COLOR_COMPARISON_OUTPUT);
    // Skip metrics computation when temporal sampling has converged (just presenting same frame)
    m_imageCompare.render(cmd, getCurrentVisualizationImageInfo(), m_gBuffers.getSize(), outputView, temporalConverged);
  }
}


void GaussianSplatting::renderPureRaytracingPipeline(VkCommandBuffer cmd, uint32_t splatCount, bool temporalConverged)
{
  NVVK_DBG_SCOPE(cmd);

  // Skip rendering while GBuffer reinit is pending (deferred two-pass).
  if(m_requestGBufferReinit)
    return;

  // If converged, just return (no need to re-render the same converged frame)
  if(temporalConverged)
    return;
#if defined(USE_DLSS)
  if(m_dlss.isEnabled())
    prmFrame.frameSampleId++;
#endif
  collectReadBackValuesIfNeeded();

  updateAndUploadFrameInfoUBO(cmd, splatCount);

  raytrace(cmd);

  // Apply DLSS denoising for ray tracing (only for Final/Clay — debug visualizations bypass denoising)
#if defined(USE_DLSS)
  if(m_dlss.isEnabled() && (prmRender.visualize == VISUALIZE_FINAL || prmRender.visualize == VISUALIZE_CLAY))
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "DLSS");

    // Get camera matrices
    const glm::mat4& view = cameraManip->getViewMatrix();
    const glm::mat4& proj = cameraManip->getPerspectiveMatrix();

    // Compute MVP for current frame
    glm::mat4 currentMVP = proj * view;

    // Reset denoiser on first frame
    bool reset = (prmFrame.frameSampleId == 0);

    // Denoise - use the same jitter that was passed to the shader
    m_dlss.denoise(cmd, m_currentJitter, view, proj, reset);

    // Store current MVP for next frame
    m_prevMVP = currentMVP;

    // Memory barrier after DLSS
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  }
#endif
}


void GaussianSplatting::renderHybridPipeline(VkCommandBuffer cmd, uint32_t splatCount, bool temporalConverged)
{
  NVVK_DBG_SCOPE(cmd);

  // Skip rendering while GBuffer reinit is pending (deferred two-pass).
  // The pipeline color format may already reflect the new prmRender.colorFormat
  // while the GBuffer images still use the old format, causing a validation error.
  if(m_requestGBufferReinit)
    return;

  // If converged, just return (no need to re-render the same converged frame)
  if(temporalConverged)
    return;

#if defined(USE_DLSS)
  if(m_dlss.isEnabled())
    prmFrame.frameSampleId++;
#endif

  // FTB (front-to-back) mode: used when surface info is needed and NOT stochastic splat
  // Stochastic splat can generate surface data but uses back-to-front with different depth handling
  const bool useFTB = needSurfaceInfo() && (prmRaster.sortingMethod != SORTING_STOCHASTIC_SPLAT);

  // Always update frame info UBO
  if(m_shaders.valid)
  {
    // collect readback results from previous frame if any
    collectReadBackValuesIfNeeded();

    // Update UBO with camera matrices (needed for both splats AND meshes)
    updateAndUploadFrameInfoUBO(cmd, splatCount);

    // Handle splat sorting only if splats exist
    if(splatCount)
    {
      // Force stochastic mode when CPU sorting is selected but the host staging buffer
      // couldn't be allocated (buffer exceeds maxMemoryAllocationSize for very large scenes).
      // Stochastic avoids both the CPU host buffer and the VRDX >4GB descriptor limit.
      if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX && prmRaster.sortingMethod != SORTING_STOCHASTIC_SPLAT
         && !m_assets.splatSets.hasCpuSortingBuffer())
      {
        LOGW("CPU sorting host buffer unavailable — forcing stochastic splat mode.\n");
        prmRaster.sortingMethod = SORTING_STOCHASTIC_SPLAT;
        m_requestUpdateShaders  = true;
      }

      if(usesGpuDistSort())
      {
        m_profilerTimeline->asyncRemoveTimer("CPU Dist");
        m_profilerTimeline->asyncRemoveTimer("CPU Sort");
        processSortingOnGPU(cmd, splatCount);
      }
      else
      {
        m_assets.splatSets.tryConsumeAndUploadCpuSortingResult(cmd, splatCount,
                                                               glm::normalize(m_center - m_eye),  // viewDir
                                                               m_eye,                             // eyePos
                                                               prmRaster.cpuLazySort, prmRender.opacityGaussianDisabled,
                                                               m_selectedSplatInstance, useFTB);
      }
    }
  }

  // In which color buffer are we going to render ?
  // Shader reads from image (MAIN) when frameSampleId <= 0, else imageAux (AUX1)
  // So we must write to the same buffer the shader will read from
  uint32_t colorBufferId = COLOR_MAIN;
  if(prmFrame.frameSampleId > 0)  // DLSS or temporal sampling increments frameSampleId
    colorBufferId = COLOR_AUX1;

  // raytrace the mesh depth using primary rays if needed (if meshes OR lights exist)
  bool raytraceMeshDepth = m_shaders.valid && shouldUseMeshPipeline() && prmSelectedPipeline == PIPELINE_HYBRID_3DGUT;

  if(raytraceMeshDepth)
  {
    // Hybrid 3DGUT writes mesh depth via ray tracing into the G-buffer depth image (storage/general).
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
    raytrace(cmd, true);
  }

  // Drawing the primitives in the G-Buffer
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization");

    // Use DLSS render size when DLSS is enabled in hybrid mode
    VkExtent2D rasterSize = m_app->getViewportSize();
#if defined(USE_DLSS)
    if(m_dlss.isEnabled())
    {
      rasterSize = m_dlss.getRenderSize();
    }
#endif
    const VkViewport viewport{0.0F, 0.0F, float(rasterSize.width), float(rasterSize.height), 0.0F, 1.0F};
    const VkRect2D   scissor{{0, 0}, rasterSize};

    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView(colorBufferId);
    colorAttachment.clearValue                = {m_clearColor};
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    if(raytraceMeshDepth)
    {
      depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // <-- preserve existing depth
    }
    depthAttachment.imageView   = m_gBuffers.getDepthImageView();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    depthAttachment.clearValue  = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    // Setup color attachments array (main color + optional normal and depth buffers)
    std::vector<VkRenderingAttachmentInfo> colorAttachments = {colorAttachment};

    // Add normal, depth, and splat ID buffer attachments if generating surface
    // For FTB: depth buffer is used as storage image, not color attachment
    // For stochastic splat: depth buffer is color attachment with blending
    if(needSurfaceInfo())
    {
      VkRenderingAttachmentInfo normalAttachment = DEFAULT_VkRenderingAttachmentInfo;
      normalAttachment.imageView                 = m_gBuffers.getColorImageView(COLOR_RASTER_NORMAL);
      normalAttachment.clearValue                = {{0.0f, 0.0f, 0.0f, 0.0f}};  // Clear to zero
      colorAttachments.push_back(normalAttachment);

      if(!useFTB)
      {
        // BTF mode (stochastic splat): depth buffer is color attachment with blending
        VkRenderingAttachmentInfo rasterDepthAttachment = DEFAULT_VkRenderingAttachmentInfo;
        rasterDepthAttachment.imageView                 = m_gBuffers.getColorImageView(COLOR_RASTER_DEPTH);
        rasterDepthAttachment.clearValue                = {{0.0f, 0.0f, 0.0f, 0.0f}};  // Clear to zero
        colorAttachments.push_back(rasterDepthAttachment);
      }

      VkRenderingAttachmentInfo splatIdAttachment  = DEFAULT_VkRenderingAttachmentInfo;
      splatIdAttachment.imageView                  = m_gBuffers.getColorImageView(COLOR_RASTER_SPLATID);
      splatIdAttachment.clearValue.color.uint32[0] = 0xFFFFFFFF;  // Clear to invalid ID
      colorAttachments.push_back(splatIdAttachment);
    }

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
    renderingInfo.pColorAttachments    = colorAttachments.data();
    renderingInfo.pDepthAttachment     = &depthAttachment;


    // Batch all pre-render barriers into a single synchronization point
    {
      std::vector<VkImageMemoryBarrier2> barriers;
      barriers.reserve(5);

      barriers.push_back(nvvk::makeImageMemoryBarrier(
          {m_gBuffers.getColorImage(colorBufferId), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}));

      if(needSurfaceInfo())
      {
        barriers.push_back(nvvk::makeImageMemoryBarrier({m_gBuffers.getColorImage(COLOR_RASTER_NORMAL), VK_IMAGE_LAYOUT_GENERAL,
                                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}));
        if(!useFTB)
        {
          // BTF mode: depth buffer is color attachment
          barriers.push_back(nvvk::makeImageMemoryBarrier({m_gBuffers.getColorImage(COLOR_RASTER_DEPTH), VK_IMAGE_LAYOUT_GENERAL,
                                                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}));
        }
        // For FTB: depth buffer stays in GENERAL layout for storage image access
        barriers.push_back(nvvk::makeImageMemoryBarrier({m_gBuffers.getColorImage(COLOR_RASTER_SPLATID), VK_IMAGE_LAYOUT_GENERAL,
                                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}));
      }

      VkDependencyInfo depInfo{.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
                               .pImageMemoryBarriers    = barriers.data()};
      vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // For FTB: clear depth buffer via vkCmdClearColorImage (R=0 for depth, G=1 for transmittance)
    // The buffer stays in GENERAL layout for storage image access during rendering
    if(useFTB && needSurfaceInfo())
    {
      VkClearColorValue       clearValue = {{0.0f, 1.0f, 0.0f, 0.0f}};  // R=depth(0), G=transmittance(1)
      VkImageSubresourceRange range      = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(cmd, m_gBuffers.getColorImage(COLOR_RASTER_DEPTH), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);

      // Barrier to ensure clear is complete before splat shader reads/writes
      VkImageMemoryBarrier2 clearBarrier{
          .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
          .srcStageMask        = VK_PIPELINE_STAGE_2_CLEAR_BIT,
          .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
          .dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          .dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
          .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
          .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image               = m_gBuffers.getColorImage(COLOR_RASTER_DEPTH),
          .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      VkDependencyInfo clearDepInfo{
          .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .imageMemoryBarrierCount = 1,
          .pImageMemoryBarriers    = &clearBarrier,
      };
      vkCmdPipelineBarrier2(cmd, &clearDepInfo);
    }

    vkCmdBeginRendering(cmd, &renderingInfo);

    vkCmdSetViewportWithCount(cmd, 1, &viewport);
    vkCmdSetScissorWithCount(cmd, 1, &scissor);

    // Render order depends on blending mode:
    // - BTF (back-to-front): mesh first, then splats (standard "over" blending)
    // - FTB (front-to-back): three-pass approach for correct mesh compositing
    //   1. Mesh depth pre-pass: write depth, output black (for correct splat depth test)
    //   2. Splats FTB: depth test against mesh, accumulate color + transmittance
    //   3. Mesh color pass: read transmittance, add mesh_color * transmittance
    const bool hasMeshes = m_shaders.valid && !m_assets.meshes.instances.empty() && !raytraceMeshDepth;

    if(useFTB)
    {
      // FTB mode: three-pass rendering for correct mesh/splat compositing
      // Pass 1: Mesh depth pre-pass
      if(hasMeshes)
      {
        auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization meshes depth");
        drawMeshPrimitives(cmd, false);  // depth pre-pass
      }

      // Barrier: ensure mesh depth writes are visible to splat depth test
      if(hasMeshes)
      {
        VkMemoryBarrier2 depthBarrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        };
        VkDependencyInfo depInfo{
            .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags    = VK_DEPENDENCY_BY_REGION_BIT,
            .memoryBarrierCount = 1,
            .pMemoryBarriers    = &depthBarrier,
        };
        vkCmdPipelineBarrier2(cmd, &depInfo);
      }

      // Pass 2: Splats FTB with depth test
      if(m_shaders.valid && splatCount)
      {
        auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization splats");
        drawSplatPrimitives(cmd, splatCount);
      }

      // Barriers: ensure splat writes are visible to mesh color pass
      // 1. Image barrier for storage image (depth/transmittance buffer)
      // 2. Memory barrier for color attachment
      // VK_DEPENDENCY_BY_REGION_BIT is required for framebuffer-space stages within render pass
      if(hasMeshes && useFTB)
      {
        // Image barrier for the depth/transmittance storage image
        VkImageMemoryBarrier2 imgBarrier{
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_gBuffers.getColorImage(COLOR_RASTER_DEPTH),
            .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };

        // Memory barrier for color attachment
        VkMemoryBarrier2 memBarrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        };

        VkDependencyInfo depInfo{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags         = VK_DEPENDENCY_BY_REGION_BIT,
            .memoryBarrierCount      = 1,
            .pMemoryBarriers         = &memBarrier,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &imgBarrier,
        };
        vkCmdPipelineBarrier2(cmd, &depInfo);
      }
      else if(hasMeshes)
      {
        // No storage image, just color attachment barrier
        VkMemoryBarrier2 memBarrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        };
        VkDependencyInfo depInfo{
            .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags    = VK_DEPENDENCY_BY_REGION_BIT,
            .memoryBarrierCount = 1,
            .pMemoryBarriers    = &memBarrier,
        };
        vkCmdPipelineBarrier2(cmd, &depInfo);
      }

      // Pass 3: Mesh color pass (blend with transmittance)
      if(hasMeshes)
      {
        auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization meshes color");
        drawMeshPrimitives(cmd, true);  // FTB color pass with additive blend
      }

      // Pass 4: Depth consolidation (write picked splat depth to hw depth buffer)
      // This enables visual helpers and other post-effects to use complete scene depth
      if(useFTB && m_graphicsPipelineDepthConsolidate != VK_NULL_HANDLE)
      {
        auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Depth Consolidation");

        // Barrier: ensure splat storage image writes are visible to depth consolidation reads
        VkMemoryBarrier2 depthReadBarrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        };
        VkDependencyInfo depInfoRead{
            .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags    = VK_DEPENDENCY_BY_REGION_BIT,
            .memoryBarrierCount = 1,
            .pMemoryBarriers    = &depthReadBarrier,
        };
        vkCmdPipelineBarrier2(cmd, &depInfoRead);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineDepthConsolidate);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

        // Draw fullscreen triangle (3 vertices, vertex shader generates positions from vertex ID)
        vkCmdDraw(cmd, 3, 1, 0, 0);
      }
    }
    else
    {
      // BTF mode: mesh first, then splats (standard approach)
      if(hasMeshes)
      {
        auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization meshes");
        drawMeshPrimitives(cmd, false);
      }

      if(m_shaders.valid && splatCount)
      {
        auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization splats");
        drawSplatPrimitives(cmd, splatCount);
      }
    }

    vkCmdEndRendering(cmd);

    // Batch all post-render barriers into a single synchronization point
    std::vector<VkImageMemoryBarrier2> barriers;
    barriers.reserve(5);

    barriers.push_back(nvvk::makeImageMemoryBarrier(
        {m_gBuffers.getColorImage(colorBufferId), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL}));

    if(needSurfaceInfo())
    {
      barriers.push_back(nvvk::makeImageMemoryBarrier({m_gBuffers.getColorImage(COLOR_RASTER_NORMAL),
                                                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL}));
      if(!useFTB)
      {
        // BTF mode: depth buffer was color attachment
        barriers.push_back(nvvk::makeImageMemoryBarrier({m_gBuffers.getColorImage(COLOR_RASTER_DEPTH),
                                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL}));
      }
      // For FTB: depth buffer stays in GENERAL layout (storage image access)
      barriers.push_back(nvvk::makeImageMemoryBarrier({m_gBuffers.getColorImage(COLOR_RASTER_SPLATID),
                                                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL}));
    }

    if(!barriers.empty())
    {
      VkDependencyInfo depInfo{.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
                               .pImageMemoryBarriers    = barriers.data()};
      vkCmdPipelineBarrier2(cmd, &depInfo);
    }
  }

  // Compute shader workgroup size and dispatch dimensions for deferred shading
  uint32_t wgSize    = 16;
  uint32_t dispatchX = (static_cast<uint32_t>(m_viewSize.x) + wgSize - 1) / wgSize;
  uint32_t dispatchY = (static_cast<uint32_t>(m_viewSize.y) + wgSize - 1) / wgSize;

  // Deferred shading pass for raster-only pipelines (MESH, VERT, MESH_3DGUT)
  // Hybrid pipelines use raytracing for lighting instead
  // Skip if shading is disabled (splats already have their raw color in the color buffer)
  bool isRasterOnlyPipeline = isRasterPipelineOnly();
  if(prmRender.lightingEnabled && isRasterOnlyPipeline && m_computePipelineDeferredShading != VK_NULL_HANDLE)
  {
    auto timerSectionDeferred = m_profilerGpuTimer.cmdFrameSection(cmd, "Deferred shading splats");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineDeferredShading);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
    vkCmdDispatch(cmd, dispatchX, dispatchY, 1);

    // Barrier to ensure output image is written before presentation
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});
  }

  // raytrace the secondary rays if needed (if meshes OR lights exist for RTX)
  // Check RTX validity from manager (not individual splat set)
  bool rtxValid = m_assets.splatSets.isRtxValid();

  // Raytrace secondary rays if meshes exist OR if surface info is needed (for lighting, DLSS, or DOF)
  // (splat set is optional but must be valid if present)
  if(m_shaders.valid && (shouldUseMeshPipeline() || needSurfaceInfo())
     && (prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT)
     && (splatCount == 0 || rtxValid))  // Allow if no splats, or if splats exist they must be RTX-valid
  {
    raytrace(cmd);
  }

  // Apply DLSS denoising for hybrid pipelines (only for Final/Clay — debug visualizations bypass denoising)
#if defined(USE_DLSS)
  if(m_dlss.isEnabled() && (prmRender.visualize == VISUALIZE_FINAL || prmRender.visualize == VISUALIZE_CLAY))
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "DLSS");

    // Get camera matrices
    const glm::mat4& view = cameraManip->getViewMatrix();
    const glm::mat4& proj = cameraManip->getPerspectiveMatrix();

    // Reset denoiser on first frame
    bool reset = (prmFrame.frameSampleId == 0);

    // Denoise - use the same jitter that was passed to the shader
    m_dlss.denoise(cmd, m_currentJitter, view, proj, reset);

    // Memory barrier after DLSS
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  }
#endif

  // Perform post processings if needed (e.g. temporal accumulation)
  // Only accumulate for final image visualization; debug modes (depth, normals, clock, etc.)
  // display the current frame directly from AUX1 without accumulation.
  // Note: When DLSS is enabled, it handles temporal accumulation internally
  // Note: frameSampleId check moved inside postProcess() to keep profiler section count stable
  bool shouldAccumulate = (prmRender.visualize == VISUALIZE_FINAL || prmRender.visualize == VISUALIZE_CLAY);
#if defined(USE_DLSS)
  if(shouldAccumulate && !m_dlss.isEnabled() && prmRtx.temporalSampling)
#else
  if(shouldAccumulate && prmRtx.temporalSampling)
#endif
  {
    postProcess(cmd);
  }
}

//--------------------------------------------------------------------------------------------------
// Tonemapper: init, deinit, per-frame pass
//--------------------------------------------------------------------------------------------------
void GaussianSplatting::initTonemapper()
{
  if(!m_slangCompiler.compileFile("nvshaders/tonemapper.slang"))
  {
    LOGE("Tonemapper: failed to compile tonemapper.slang\n");
    m_tonemapperData.isActive = 0;
    return;
  }
  auto spirv = std::span<const uint32_t>(m_slangCompiler.getSpirv(), m_slangCompiler.getSpirvSize() / sizeof(uint32_t));
  if(m_tonemapper.init(&m_alloc, spirv) != VK_SUCCESS)
  {
    LOGE("Tonemapper: Vulkan init failed - disabling tone mapping.\n");
    m_tonemapperData.isActive = 0;
  }
}

void GaussianSplatting::deinitTonemapper()
{
  m_tonemapper.deinit();
}

void GaussianSplatting::tonemap(VkCommandBuffer cmd)
{
  if(!m_tonemapperData.isActive)
    return;

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Tonemap");

  VkExtent2D size = m_gBuffers.getSize();

  // Tonemap: read from COLOR_MAIN (HDR), write to COLOR_LDR (R8G8B8A8_UNORM)
  VkDescriptorImageInfo inImage  = m_gBuffers.getDescriptorImageInfo(COLOR_MAIN);
  VkDescriptorImageInfo outImage = m_gBuffers.getDescriptorImageInfo(COLOR_LDR);
  outImage.imageLayout           = VK_IMAGE_LAYOUT_GENERAL;

  m_tonemapper.runCompute(cmd, size, m_tonemapperData, inImage, outImage);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

//--------------------------------------------------------------------------------------------------
// Render visual helpers (3D gizmo and grid) if needed
//--------------------------------------------------------------------------------------------------
void GaussianSplatting::renderVisualHelpers(VkCommandBuffer cmd)
{
  // Check camera is not fisheye (helpers don't work with fisheye projection)
  Camera camera = m_assets.cameras.getCamera();
  if(camera.model != CAMERA_FISHEYE && m_helpers.shouldRender())
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Visual helpers");

    const VkExtent2D& viewportSize = m_app->getViewportSize();

    // Calculate depth buffer size (may differ from viewport for DLSS)
    glm::vec2 depthBufferSize = glm::vec2(viewportSize.width, viewportSize.height);
#if defined(USE_DLSS)
    if(m_dlss.isEnabled())
    {
      VkExtent2D dlssRenderSize = m_dlss.getRenderSize();
      depthBufferSize           = glm::vec2(dlssRenderSize.width, dlssRenderSize.height);
    }
#endif

    VkImage helperTarget = m_tonemapperData.isActive ? m_gBuffers.getColorImage(COLOR_LDR) : m_gBuffers.getColorImage(COLOR_MAIN);

#if defined(USE_DLSS)
    // When visualizing DLSS guide images, blit the guide buffer to helperTarget first
    // so visual helpers are composited on top of the DLSS visualization
    if(m_dlss.isEnabled() && prmRender.visualize >= VISUALIZE_DLSS_INPUT && prmRender.visualize <= VISUALIZE_DLSS_DEPTH)
    {
      const auto&          dlssGBuffers = m_dlss.getGBuffers();
      shaderio::DlssImages dlssBuffer   = getDlssBufferForVisuMode(prmRender.visualize);
      VkImage              dlssImage    = dlssGBuffers.getColorImage((uint32_t)dlssBuffer);
      VkExtent2D           dlssSize     = dlssGBuffers.getSize();

      nvvk::cmdImageMemoryBarrier(cmd, {dlssImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
      nvvk::cmdImageMemoryBarrier(cmd, {helperTarget, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

      VkImageBlit blitRegion{};
      blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      blitRegion.srcOffsets[1]  = {int32_t(dlssSize.width), int32_t(dlssSize.height), 1};
      blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      blitRegion.dstOffsets[1]  = {int32_t(viewportSize.width), int32_t(viewportSize.height), 1};

      vkCmdBlitImage(cmd, dlssImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, helperTarget,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);

      nvvk::cmdImageMemoryBarrier(cmd, {dlssImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
      nvvk::cmdImageMemoryBarrier(cmd, {helperTarget, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    }
#endif

    m_helpers.render(cmd, helperTarget, m_descriptorSet, cameraManip->getViewMatrix(), cameraManip->getPerspectiveMatrix(),
                     glm::vec2(viewportSize.width, viewportSize.height), depthBufferSize);
  }
}

void GaussianSplatting::processGBufferUpdateRequests()
{
  // Wait for all GPU work to complete before destroying/recreating resources
  vkDeviceWaitIdle(m_device);

  VkExtent2D currentSize = m_gBuffers.getSize();

  // Release comparison capture image if it exists (incompatible with new format)
  if(m_imageCompare.hasValidCaptureImage())
  {
    m_imageCompare.releaseCaptureImage();
    prmComparison.enabled = false;
  }

  // Full deinit/reinit of GBuffer with new format
  m_gBuffers.deinit();
  m_gBuffers.init({
      .allocator      = &m_alloc,
      .colorFormats   = {prmRender.colorFormat, prmRender.colorFormat, prmRender.colorFormat, m_normalFormat,
                         m_rasterDepthFormat, m_splatIdFormat, VK_FORMAT_R8G8B8A8_UNORM},
      .depthFormat    = m_depthFormat,
      .imageSampler   = m_sampler,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  });

  // Use temp command buffer for GBuffer update
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  NVVK_CHECK(m_gBuffers.update(cmd, currentSize));

  // Full deinit/reinit of visual helpers with new format
  m_helpers.deinit();
  m_helpers.init({
      .app           = m_app,
      .alloc         = &m_alloc,
      .uploader      = &m_uploader,
      .device        = m_device,
      .sampler       = m_sampler,
      .slangCompiler = &m_slangCompiler,
      .colorFormat   = prmRender.colorFormat,
      .depthFormat   = m_depthFormat,
  });
  m_helpers.onResize(cmd, currentSize, m_gBuffers.getDepthImage(), m_gBuffers.getDepthImageView(), m_sampler);

  // Submit and wait for temp command buffer
  m_app->submitAndWaitTempCmdBuffer(cmd);

#if defined(USE_DLSS)
  if(m_dlss.isEnabled())
  {
    m_dlss.setResource(DlssRayReconstruction::ResourceType::eColorOut, m_gBuffers.getColorImage(COLOR_MAIN),
                       m_gBuffers.getColorImageView(COLOR_MAIN), prmRender.colorFormat);
  }
#endif

  resetFrameCounter();
  m_requestGBufferReinit = false;
}

void GaussianSplatting::processUpdateRequests(bool forceAll)
{
  coerceRasterParamsForSortingMethod();

  // Check if camera animation has completed and we have a deferred shader rebuild
  if(m_requestUpdateShadersAfterCameraAnim && !cameraManip->isAnimated())
  {
    // Animation is done, trigger the actual shader rebuild
    m_requestUpdateShaders                = true;
    m_requestUpdateShadersAfterCameraAnim = false;
  }

  // Release GPU buffers not needed by the current pipeline to reclaim VRAM.
  // Must run before deferred RTX rebuild and needUpdate checks, because:
  //   - Freeing sorting buffers reclaims VRAM before RTX AS allocation
  //   - Freeing RTX AS may set m_deferredRtxRebuildPending (triggers rebuild when switching back)
  //   - Allocating sorting buffers may set m_requestUpdateAssetsBuffer (picked up by needUpdate)
  if(!forceAll)
    releaseUnusedBuffers();

  // Check if we have any requests that require buffer updates.
  // Determine if we're using RTX pipeline (used for deferred RTX optimization).
  // When forceAll is true (e.g., on reset/exit) we still want to process RTX
  // bookkeeping so any allocated AS gets torn down — but only if the device
  // supports raytracing; otherwise no AS was ever built so there is nothing
  // to do for Phase 3 of processVramUpdates.
  bool isRtxPipeline = (isRtxPipelineActive() || forceAll) && isSupported.raytracing;

  // Process deferred RTX rebuild request when switching to RTX pipeline
  // This handles transforms that were modified in raster mode but need RTX AS updates
  // NOTE: Must happen BEFORE computing splatSetRequestsToCheck so the snapshot is accurate
  if(isRtxPipelineActive() && m_deferredRtxRebuildPending)
  {
    // Only trigger rebuild if RTX structures don't exist yet
    // (if they exist, updateInstanceTransform already queued UpdateTransformsOnly)
    if(!m_assets.splatSets.isRtxValid())
    {
      m_assets.splatSets.pendingRequests |= SplatSetManagerVk::Request::eRebuildBLAS;
    }
    m_deferredRtxRebuildPending = false;
  }

  // When checking for buffer updates, exclude deferred RTX requests in raster mode
  // RTX requests (UpdateTransformsOnly, RebuildTLAS, RebuildBLAS) are deferred and don't need device wait
  // unless forceAll is true (cleanup on reset/exit)
  // NOTE: Computed AFTER deferred rebuild so the snapshot includes any newly-set flags
  uint32_t splatSetRequestsToCheck = static_cast<uint32_t>(m_assets.splatSets.pendingRequests);
  if(!isRtxPipeline)
  {
    // In raster mode, mask out RTX-related requests (they're deferred, don't need device wait)
    constexpr uint32_t RTX_REQUESTS_MASK = static_cast<uint32_t>(SplatSetManagerVk::Request::eUpdateTransformsOnly)
                                           | static_cast<uint32_t>(SplatSetManagerVk::Request::eRebuildTLAS)
                                           | static_cast<uint32_t>(SplatSetManagerVk::Request::eRebuildBLAS);
    splatSetRequestsToCheck &= ~RTX_REQUESTS_MASK;
  }

  bool hasBufferUpdateRequests = m_requestUpdateShaders || static_cast<uint32_t>(m_assets.lights.pendingRequests)
                                 || static_cast<uint32_t>(m_assets.meshes.pendingRequests) || splatSetRequestsToCheck;

  // CRITICAL: Wait for ALL previous GPU work to complete ONLY if we have buffer update requests
  // This ensures that any buffers we're about to destroy/recreate are no longer in use
  if(hasBufferUpdateRequests)
  {
    vkDeviceWaitIdle(m_device);
  }

  // Detect scene composition changes that affect RTX_HAS_MESHES / RTX_HAS_PARTICLES macros.
  // Use getEffectiveGlobalSplatCount(): m_totalGlobalSplatCount is still 0 on the load frame
  // until rebuildGlobalIndexTables runs inside processVramUpdates (after this check).
  bool hasMeshesNow    = !m_assets.meshes.instances.empty();
  bool hasParticlesNow = m_assets.splatSets.getEffectiveGlobalSplatCount() > 0;
  if(hasMeshesNow != m_lastHadMeshes || hasParticlesNow != m_lastHadParticles)
  {
    m_requestUpdateShaders = true;
    m_lastHadMeshes        = hasMeshesNow;
    m_lastHadParticles     = hasParticlesNow;
  }

  const bool createVrdxSorter = usesGpuDistSort() && (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX);
  m_assets.splatSets.setCreateVrdxSorter(createVrdxSorter);
  if(m_assets.splatSets.hasSortingBuffers())
  {
    bool sortingBackendChanged = false;
    if(!m_assets.splatSets.ensureSortingBackend(createVrdxSorter, &sortingBackendChanged))
    {
      LOGW("VRDX setup failed — falling back to stochastic splat mode.\n");
      prmRaster.sortingMethod = SORTING_STOCHASTIC_SPLAT;
      m_requestUpdateShaders  = true;
    }
    else if(sortingBackendChanged)
    {
      m_requestUpdateAssetsBuffer = true;
    }
  }

  // Lightweight SceneAssets UBO refresh (e.g. after VRDX alloc) — no full VRAM rebuild.
  if(m_requestUpdateAssetsBuffer && !m_requestUpdateShaders && !m_assets.hasPendingRequests(!isRtxPipeline))
  {
    m_assets.updateAssetsBuffer();
    m_requestUpdateAssetsBuffer = false;
  }

  // Check if any updates are needed
  // In raster mode, exclude RTX-only requests (they're deferred and don't need processing)
  bool needUpdate = m_requestUpdateShaders || m_assets.hasPendingRequests(!isRtxPipeline);

  if(!needUpdate)
    return;

  LOGD("processUpdateRequests: Buffer updates requested (MeshPending=%d, Shaders=%d, LightsPending=%d, SplatSetPending=%d, Assets=%d)\n",
       static_cast<uint32_t>(m_assets.meshes.pendingRequests), m_requestUpdateShaders,
       static_cast<uint32_t>(m_assets.lights.pendingRequests), splatSetRequestsToCheck, m_requestUpdateAssetsBuffer);

  resetFrameCounter();
  vkDeviceWaitIdle(m_device);

  if(m_requestUpdateShaders)
  {
    deinitPipelines();
    deinitShaders();

    if(initShaders())
    {
      // Upload splat data before creating graphics pipelines when RTX is off — the upload
      // path does not need pipeline objects and this avoids Intel driver stalls when large
      // VRAM allocations are immediately followed by pipeline creation + device idle.
      if(!isRtxPipeline)
      {
        m_assets.processVramUpdates(false);
      }

      initPipelines();
      initRtDescriptorSet();
      initRtPipeline();
      initDescriptorSetPostProcessing();
      initPipelinePostProcessing();
      m_imageCompare.rebuildPipelines();

      if(isRtxPipeline)
      {
        // Process all asset updates after pipelines are ready (GPU AS compute depends on this)
        m_assets.processVramUpdates(isRtxPipeline);
      }
      else
      {
        // Descriptors were rebuilt during the upload above; refresh the assets UBO pointers.
        m_assets.updateAssetsBuffer();
      }
      // Upload paths already synchronize via submitAndWaitTempCmdBuffer; a global device
      // idle here has hung some Intel drivers after large splat uploads.
      m_requestUpdateShaders = false;
    }
    else
    {
      LOGE("Shader rebuild failed (initShaders) — will retry next frame.\n");
    }
  }
  else
  {
    m_assets.processVramUpdates(isRtxPipeline);
  }

  // Graceful fallback: if RTX build just failed (e.g., OOM during TLAS allocation),
  // immediately switch to raster pipeline and ensure sorting buffers are available.
  // Without this, the pipeline switch would only happen in onRender() (next frame),
  // but sorting buffers may already have been freed — causing a crash on the first
  // raster frame.
  if(m_assets.splatSets.isRtxError() && isRtxPipelineActive())
  {
    LOGW("RTX build failed - falling back to raster pipeline.\n");
    prmSelectedPipeline    = isSupported.meshShader ? PIPELINE_MESH : PIPELINE_VERT;
    m_requestUpdateShaders = true;
  }

  // Ensure sorting buffers exist when the pipeline needs rasterization (raster or hybrid).
  // After project load, clearSceneGpuBuffers sets m_sortingBuffersReleased = true, and
  // releaseUnusedBuffers (which runs before processVramUpdates) can't allocate because
  // hasSplats is still 0 at that point. processVramUpdates then rebuilds global index
  // tables (setting the splat count) but uploadGlobalIndexTablesToGPU skips allocation
  // due to m_sortingBuffersReleased. Without this check, the first rendered frame would
  // attempt rasterization with no sorting buffers, causing VK_ERROR_DEVICE_LOST.
  if(prmSelectedPipeline != PIPELINE_RTX && !m_assets.splatSets.hasSortingBuffers()
     && m_assets.splatSets.getEffectiveGlobalSplatCount() > 0)
  {
    if(!m_assets.splatSets.updateSortingBuffers(createVrdxSorter))
    {
      // Sorting buffer allocation failed (OOM). Free RTX AS to reclaim VRAM, then retry.
      // Fall back to raster mesh with stochastic splat (no VRDX sort needed).
      if(m_assets.splatSets.isRtxValid())
      {
        LOGW("Sorting buffer allocation failed — freeing RTX AS and retrying.\n");
        m_assets.splatSets.rtxDeinitAccelerationStructures();
        m_assets.splatSets.setRtxState(RtxState::eRtxNone);
        m_deferredRtxRebuildPending = true;

        if(!m_assets.splatSets.updateSortingBuffers(createVrdxSorter))
        {
          LOGE("Sorting buffer allocation still failed after freeing RTX AS.\n");
        }
      }

      LOGW("Falling back to raster pipeline with stochastic splat.\n");
      prmSelectedPipeline     = isSupported.meshShader ? PIPELINE_MESH : PIPELINE_VERT;
      prmRaster.sortingMethod = SORTING_STOCHASTIC_SPLAT;
      m_requestUpdateShaders  = true;
    }
    m_assets.updateAssetsBuffer();
    resetFrameCounter();
  }

  // Update splat texture descriptors AFTER VRAM updates, only when textures were (re)created.
  // processVramUpdates() sets this flag when data storage is regenerated (e.g., switching
  // storage mode or changing RGBA/SH format). We must rebind BINDING_SPLAT_TEXTURES so the
  // descriptor set points to the current GPU texture image views.
  if(m_assets.splatSets.consumeTextureDescriptorsDirty())
  {
    updateSplatTextureDescriptors();
  }

  // Update mesh material texture descriptors when textures were loaded or freed
  if(m_assets.meshes.consumeMeshTexturesDirty())
  {
    updateMeshTextureDescriptors();
  }

  // Mesh raster can fail on some drivers even when VK_EXT_mesh_shader is reported; fall back to
  // vertex raster and schedule one more shader rebuild (PIPELINE macro changes).
  if((prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_MESH_3DGUT)
     && m_graphicsPipelineGsMesh == VK_NULL_HANDLE && m_graphicsPipeline3dgutMesh == VK_NULL_HANDLE
     && m_graphicsPipelineGsVert != VK_NULL_HANDLE)
  {
    LOGW("Mesh-shader splat pipeline unavailable - switching to vertex raster (PIPELINE_VERT).\n");
    prmSelectedPipeline    = PIPELINE_VERT;
    m_requestUpdateShaders = true;
  }
}

void GaussianSplatting::releaseUnusedBuffers()
{
  bool isPureRtx    = isRtxPipelineOnly();
  bool isPureRaster = !isRtxPipelineActive();
  // Hybrid modes (PIPELINE_HYBRID, PIPELINE_HYBRID_3DGUT) need both.

  // Pure RTX does not need the global index table (saves ~8 bytes per splat)
  // or covariance buffers (raster-only, saves 24 bytes per splat).
  // Hybrid and raster pipelines need both for sorting and raster-pass rendering.
  m_assets.splatSets.setNeedsGlobalIndexTable(!isPureRtx);
  m_assets.splatSets.setNeedsCovarianceData(!isPureRtx);

  bool hasSplats     = m_assets.splatSets.getEffectiveGlobalSplatCount() > 0;
  bool hasSorting    = m_assets.splatSets.hasSortingBuffers();
  bool hasRtx        = m_assets.splatSets.isRtxValid();
  bool hasGitBuffers = m_assets.splatSets.hasGlobalIndexTableBuffers();

  bool releaseSorting    = isPureRtx && hasSorting;
  bool releaseGitBuffers = isPureRtx && hasGitBuffers;
  bool releaseCov        = isPureRtx && !m_assets.splatSets.isCovarianceDataReleased() && hasSplats;
  bool allocateSorting   = !isPureRtx && !hasSorting && hasSplats;
  bool allocateCov       = !isPureRtx && m_assets.splatSets.isCovarianceDataReleased() && hasSplats;
  bool releaseRtx        = isPureRaster && hasRtx;

  if(!releaseSorting && !allocateSorting && !releaseRtx && !releaseGitBuffers && !releaseCov && !allocateCov)
    return;

  vkDeviceWaitIdle(m_device);

  // Free RTX AS first to reclaim VRAM before allocating sorting buffers.
  // Without this ordering, switching from pure RTX to raster on large scenes
  // would OOM because sorting buffers are allocated while AS still occupies VRAM.
  if(releaseRtx)
  {
    m_assets.splatSets.rtxDeinitAccelerationStructures();
    m_assets.splatSets.setRtxState(RtxState::eRtxNone);
    m_deferredRtxRebuildPending = true;
    m_requestUpdateAssetsBuffer = true;
  }

  if(releaseGitBuffers)
  {
    m_assets.splatSets.releaseGlobalIndexTableBuffers();
    m_requestUpdateAssetsBuffer = true;
  }

  if(releaseCov)
  {
    m_assets.splatSets.releaseCovarianceData();
    m_requestUpdateAssetsBuffer = true;
  }

  if(releaseSorting)
  {
    m_assets.splatSets.releaseSortingBuffers();
    m_requestUpdateAssetsBuffer = true;
  }
  else if(allocateSorting)
  {
    // Allocate covariance buffers before sorting, since descriptor rebuild
    // (triggered by covariance address changes) should complete first.
    if(allocateCov)
    {
      m_assets.splatSets.allocateCovarianceData();
      allocateCov = false;
    }

    const bool allocateVrdx = usesGpuDistSort() && (prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX);
    if(!m_assets.splatSets.updateSortingBuffers(allocateVrdx))
    {
      // Free RTX AS to reclaim VRAM and retry
      if(m_assets.splatSets.isRtxValid())
      {
        LOGW("Sorting buffer allocation failed — freeing RTX AS and retrying.\n");
        m_assets.splatSets.rtxDeinitAccelerationStructures();
        m_assets.splatSets.setRtxState(RtxState::eRtxNone);
        m_deferredRtxRebuildPending = true;

        if(!m_assets.splatSets.updateSortingBuffers(allocateVrdx))
        {
          LOGE("Sorting buffer allocation still failed after freeing RTX AS.\n");
        }
      }

      LOGW("Falling back to raster pipeline with stochastic splat.\n");
      prmSelectedPipeline     = isSupported.meshShader ? PIPELINE_MESH : PIPELINE_VERT;
      prmRaster.sortingMethod = SORTING_STOCHASTIC_SPLAT;
      m_requestUpdateShaders  = true;
    }
    m_requestUpdateAssetsBuffer = true;
  }

  // Allocate covariance buffers if not already done above (e.g. switching to hybrid
  // when sorting buffers already existed)
  if(allocateCov)
  {
    m_assets.splatSets.allocateCovarianceData();
    m_requestUpdateAssetsBuffer = true;
  }

  if(m_requestUpdateAssetsBuffer)
  {
    m_assets.updateAssetsBuffer();
    m_requestUpdateAssetsBuffer = false;
    resetFrameCounter();
  }
}

void GaussianSplatting::updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UBO update");

  Camera camera = m_assets.cameras.getCamera();

  // CameraManipulator is double precision; narrow to our single-precision members.
  glm::dvec3 eyeD, centerD, upD;
  cameraManip->getLookat(eyeD, centerD, upD);
  m_eye    = glm::vec3(eyeD);
  m_center = glm::vec3(centerD);
  m_up     = glm::vec3(upD);

  // Update frame parameters uniform buffer
  // some attributes of prmFrame were directly set by the user interface
  prmFrame.splatCount = splatCount;

  prmFrame.cameraPosition = m_eye;
  prmFrame.viewMatrix     = cameraManip->getViewMatrix();
  prmFrame.viewInverse    = glm::inverse(prmFrame.viewMatrix);

  prmFrame.fovRad  = cameraManip->getRadFov();
  prmFrame.nearFar = cameraManip->getClipPlanes();
  // Projection matrix only viable in pinhole mode,
  // but is used as a fallback for 3DGS when Fisheye is on
  // projectionMatrix stays unjittered (used by raygen, motion vectors, etc.)
  prmFrame.projectionMatrix = cameraManip->getPerspectiveMatrix();

#if defined(USE_DLSS)
  // Create jittered projection matrix for rasterization when DLSS is enabled
  // projectionMatrix remains unjittered for raygen and motion vector calculation
  // projectionMatrixJittered is used by raster shaders for DLSS sub-pixel sampling
  if(m_dlss.isEnabled() && isDlssSupportedPipeline())
  {
    m_currentJitter = shaderio::dlssJitter(prmFrame.frameSampleId);

    // Start with unjittered projection, then apply jitter
    prmFrame.projectionMatrixJittered = prmFrame.projectionMatrix;
    // Apply jitter in NDC space (range [-1,1])
    // Use DLSS render size since rasterization happens at render size
    VkExtent2D dlssRenderSize = m_dlss.getRenderSize();
    prmFrame.projectionMatrixJittered[2][0] += m_currentJitter.x * 2.0f / float(dlssRenderSize.width);
    prmFrame.projectionMatrixJittered[2][1] += m_currentJitter.y * 2.0f / float(dlssRenderSize.height);
  }
  else
  {
    // When DLSS is disabled, jittered matrix is same as unjittered
    prmFrame.projectionMatrixJittered = prmFrame.projectionMatrix;
  }
#else
  prmFrame.projectionMatrixJittered = prmFrame.projectionMatrix;
#endif

  prmFrame.projInverse = glm::inverse(prmFrame.projectionMatrix);

  float       devicePixelRatio     = 1.0;
  const bool  isOrthographicCamera = false;
  const float focalMultiplier      = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
  const float focalAdjustment      = focalMultiplier;
  prmFrame.orthoZoom               = 1.0f;
  prmFrame.orthographicMode        = 0;  // disabled (uses perspective) TODO: activate support for orthographic

  // Use DLSS render size for viewport when DLSS is enabled (for both raster and ray tracing)
  glm::vec2 renderSize = glm::vec2(m_viewSize.x, m_viewSize.y);
#if defined(USE_DLSS)
  if(m_dlss.isEnabled() && isDlssSupportedPipeline())
  {
    VkExtent2D dlssSize = m_dlss.getRenderSize();
    renderSize          = glm::vec2(dlssSize.width, dlssSize.height);
    // DLSS minimum radiance threshold
    prmFrame.dlssMinRadianceThreshold = prmRtx.dlssMinRadianceThreshold;
  }
#endif
  prmFrame.viewport               = renderSize * devicePixelRatio;
  prmFrame.basisViewport          = glm::vec2(1.0f / renderSize.x, 1.0f / renderSize.y);
  prmFrame.inverseFocalAdjustment = 1.0f / focalAdjustment;

  // Compute ray mask for mesh visibility (CPU-side computation)
  // 0xFF = show all meshes (including light proxies with mask 0x01)
  // 0xFE = hide light proxies (exclude mask 0x01)
  prmFrame.rayMask = m_showLightProxies ? 0xFF : 0xFE;

  // Particle shadow parameters
  prmFrame.particleShadowOffset                 = prmRtx.particleShadowOffset;
  prmFrame.particleShadowTransmittanceThreshold = prmRtx.particleShadowTransmittanceThreshold;
  prmFrame.particleShadowColorStrength          = prmRtx.particleShadowColorStrength;

  // Ambient occlusion for emissive splat sets
  prmFrame.particleEmissiveAoRadius   = prmRtx.particleEmissiveAoRadius;
  prmFrame.particleEmissiveAoStrength = prmRtx.particleEmissiveAoStrength;

  // GS shadow mask floor
  prmFrame.gsShadowMaskMin = prmRender.gsShadowMaskMin;

  // Depth iso threshold (transmittance threshold for depth picking)
  prmFrame.depthIsoThreshold    = prmRaster.depthIsoThreshold;
  prmFrame.depthIsoThresholdRTX = prmRtx.depthIsoThresholdRTX;

  // Thin particle threshold for normal computation
  prmFrame.thinParticleThreshold = prmRender.thinParticleThreshold;

  // Visualization mode parameters
  prmFrame.clayColor = {prmRender.clayColor.x, prmRender.clayColor.y, prmRender.clayColor.z};
  switch(prmRender.visualize)
  {
    case VISUALIZE_CLOCK:
      prmFrame.visuMinMax = prmRender.clockVisuMinMax;
      prmFrame.visuShift  = prmRender.clockVisuShift;
      break;
    case VISUALIZE_DEPTH:
    case VISUALIZE_DEPTH_INTEGRATED:
    case VISUALIZE_DEPTH_FOR_DLSS:
      prmFrame.visuMinMax = prmRender.depthVisuMinMax;
      prmFrame.visuShift  = prmRender.depthVisuShift;
      break;
    case VISUALIZE_RAYHITS:
      prmFrame.visuMinMax = prmRender.hitsVisuMinMax;
      prmFrame.visuShift  = prmRender.hitsVisuShift;
      break;
    default:
      break;
  }

  // Firefly clamp
  prmFrame.rtxFireflyClampThreshold = prmRtx.fireflyClampThreshold;

  // Environment (sky / IBL)
  prmFrame.envMode      = static_cast<int32_t>(m_sky.mode());
  prmFrame.envEnabled   = m_sky.isEnabled() ? 1 : 0;
  prmFrame.skyParams    = m_sky.skyParams();
  prmFrame.envRotation  = {m_sky.iblRotation().x, m_sky.iblRotation().y, m_sky.iblRotation().z};
  prmFrame.envIntensity = m_sky.iblIntensity();
  // samplePhysicalSky() internally uses Z-up; convert params so all sampled
  // directions are consistently Z-up, then the raygen shader swaps back to Y-up.
  prmFrame.skyParams.yIsUp = 0;
  std::swap(prmFrame.skyParams.sunDirection.y, prmFrame.skyParams.sunDirection.z);

  if(camera.model == CAMERA_FISHEYE && prmSelectedPipeline != PIPELINE_VERT && prmSelectedPipeline != PIPELINE_MESH
     && prmSelectedPipeline != PIPELINE_HYBRID)
  {
    // FISHEYE focal
    prmFrame.focal = glm::vec2(1.0, -1.0) * prmFrame.viewport / prmFrame.fovRad;
  }
  else
  {
    // PINHOLE focal - use renderSize for consistent projection calculations
    const float focalLengthX = prmFrame.projectionMatrix[0][0] * 0.5f * devicePixelRatio * renderSize.x;
    const float focalLengthY = prmFrame.projectionMatrix[1][1] * 0.5f * devicePixelRatio * renderSize.y;
    prmFrame.focal           = glm::vec2(focalLengthX, focalLengthY);
  }

  // Camera pose, used by unscented transform
  {
    prmFrame.viewTrans = prmFrame.viewMatrix[3];
    glm::quat viewQuat = glm::quat_cast(prmFrame.viewMatrix);
    // glm quaternion storage is scalar last, so we forward as is
    prmFrame.viewQuat = glm::vec4(viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w);
  }

  if(camera.dofMode == DOF_AUTO_FOCUS)
  {
    // Only update auto-focus distance when the feedback reports a valid hit
    const float readbackDist = m_indirectReadback.particleIntegratedDist;
    if(readbackDist > 0.0f && readbackDist < 1.0e+30f)
    {
      prmFrame.focusDist = readbackDist;
    }
  }
  else
  {
    prmFrame.focusDist = camera.focusDist;
  }
  prmFrame.aperture = camera.aperture;

  // #DLSS - Store previous view-projection matrix for motion vectors
  // On first frame or if m_prevMVP is identity, initialize with current
  static glm::mat4 prevViewProj    = glm::mat4(1.0f);
  glm::mat4        currentViewProj = prmFrame.projectionMatrix * prmFrame.viewMatrix;
  prmFrame.prevViewProjMatrix      = prevViewProj;
  prevViewProj                     = currentViewProj;  // Store for next frame

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


void GaussianSplatting::coerceRasterParamsForSortingMethod()
{
  // Distance compute (GPU radix / stochastic) supports frustum culling at the dist stage;
  // CPU sorting does not run that pass, so culling must happen in the raster shader instead.
  const bool usesDistShader = usesGpuDistSort();

  if(!usesDistShader && prmRaster.frustumCulling == FRUSTUM_CULLING_AT_DIST)
  {
    prmRaster.frustumCulling = FRUSTUM_CULLING_AT_RASTER;
    m_requestUpdateShaders   = true;
  }
  if(!usesDistShader && prmRaster.sizeCulling == SIZE_CULLING_ENABLED)
  {
    prmRaster.sizeCulling  = SIZE_CULLING_DISABLED;
    m_requestUpdateShaders = true;
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
    if(m_computePipelineGsDistCull == VK_NULL_HANDLE)
    {
      static bool loggedDistPipelineMissing = false;
      if(!loggedDistPipelineMissing)
      {
        LOGE("GPU distance/cull compute pipeline is not available on this device.\n");
        loggedDistPipelineMissing = true;
      }
      return;
    }

    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "GPU Dist");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineGsDistCull);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // Model transform for splats now comes from SplatSetDesc in descriptor, not push constants
    // Push constants are only used for meshes (see drawMeshPrimitives)

    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(shaderio::PushConstant), &m_pcRaster);

    vkCmdDispatch(cmd, (splatCount + prmRaster.distShaderWorkgroupSize - 1) / prmRaster.distShaderWorkgroupSize, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }

  // 3. invoke the radix sort from vrdx lib (skip if stochastic mode - no sorting needed)
  if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX && usesGpuDistSort())
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "GPU Sort");

    // Get sorting buffers from SplatSetManagerVk
    VrdxSorter  gpuSorter         = m_assets.splatSets.getSplatSortingVrdxSorter();
    const auto& distancesBuffer   = m_assets.splatSets.getSplatSortingDistancesDevice();
    const auto& indicesBuffer     = m_assets.splatSets.getSplatSortingIndicesDevice();
    const auto& vrdxStorageBuffer = m_assets.splatSets.getSplatSortingVrdxStorageBuffer();

    if(gpuSorter != VK_NULL_HANDLE)
    {
      vrdxCmdSortKeyValueIndirect(cmd, gpuSorter, splatCount, m_indirect.buffer,
                                  offsetof(shaderio::IndirectParams, instanceCount), distancesBuffer.buffer, 0,
                                  indicesBuffer.buffer, 0, vrdxStorageBuffer.buffer, 0, VK_NULL_HANDLE, 0);
    }
    else
    {
      static bool loggedMissingVrdx = false;
      if(!loggedMissingVrdx)
      {
        LOGE("GPU radix sort requested but VRDX sorter is not initialized.\n");
        loggedMissingVrdx = true;
      }
    }

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }
  // Note: For stochastic mode, the barrier after distance compute provides
  // sufficient synchronization since it includes DRAW_INDIRECT_BIT and MESH_SHADER_BIT.
}

void GaussianSplatting::drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  // Early exit if the pipeline for the active raster path is missing (async load or create failure).
  if(prmSelectedPipeline == PIPELINE_VERT)
  {
    if(m_graphicsPipelineGsVert == VK_NULL_HANDLE)
      return;
  }
  else if(prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_HYBRID
          || prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT)
  {
    if(m_graphicsPipelineGsMesh == VK_NULL_HANDLE && m_graphicsPipeline3dgutMesh == VK_NULL_HANDLE)
      return;
  }

  // Do we need to activate depth test and Write ?
  Camera     camera           = m_assets.cameras.getCamera();
  const bool helpersNeedDepth = (camera.model != CAMERA_FISHEYE) && m_helpers.shouldRender();
  const bool isHybridPipeline = (prmSelectedPipeline == PIPELINE_HYBRID) || (prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);

  // FTB mode: only when surface info is needed and NOT stochastic splat
  const bool useFTB = needSurfaceInfo() && (prmRaster.sortingMethod != SORTING_STOCHASTIC_SPLAT);

  // Determine if depth test/write would normally be needed
  const bool baseNeedDepth = ((prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX) && prmRender.opacityGaussianDisabled)
                             || shouldUseMeshPipeline() || helpersNeedDepth
                             || (prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT) || isHybridPipeline;

  // For front-to-back rendering:
  // - Depth TEST enabled when meshes exist (splats get occluded by mesh in hw depth buffer)
  // - Depth WRITE disabled (picked depth stored in side reconstruction buffer, not hw depth)
  // For back-to-front: depth test and write are coupled via needDepth
  const bool needDepthTest  = baseNeedDepth;             // Always test when needed (including FTB with meshes)
  const bool needDepthWrite = !useFTB && baseNeedDepth;  // FTB uses storage image for picked depth


  // Model transform for splats now comes from SplatSetDesc in descriptor, not push constants
  // Push constants are only used for meshes (see drawMeshPrimitives)

  vkCmdPushConstants(cmd, m_pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(shaderio::PushConstant), &m_pcRaster);

  // Depth handling:
  // - Depth TEST: enabled when meshes exist (splats behind mesh get discarded)
  // - Depth WRITE: BTF writes to hw depth buffer, FTB stores picked depth in side buffer

  if(prmSelectedPipeline == PIPELINE_VERT)
  {  // Pipeline using vertex shader
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsVert);
    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepthWrite);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepthTest);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    // display the quad as many times as we have visible splats
    const VkDeviceSize offsets{0};
    vkCmdBindIndexBuffer(cmd, m_quadIndices.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_quadVertices.buffer, &offsets);

    // Use sorting indices buffer from SplatSetManagerVk
    const auto& indicesBuffer = m_assets.splatSets.getSplatSortingIndicesDevice();

    const bool usesGpuDist = usesGpuDistSort();
    if(!usesGpuDist && !m_assets.splatSets.hasCpuSortIndicesOnGpu())
      return;

    vkCmdBindVertexBuffers(cmd, 1, 1, &indicesBuffer.buffer, &offsets);
    if(!usesGpuDist)
    {
      vkCmdDrawIndexed(cmd, 6, (uint32_t)splatCount, 0, 0, 0);
    }
    else
    {
      vkCmdDrawIndexedIndirect(cmd, m_indirect.buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
    }
  }
  else
  {  // in mesh pipeline mode or in hybrid mode
    // Pipeline using mesh shader

    if(prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_HYBRID)
    {
      if(m_graphicsPipelineGsMesh == VK_NULL_HANDLE)
        return;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsMesh);
    }
    if(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT)
    {
      if(m_graphicsPipeline3dgutMesh == VK_NULL_HANDLE)
        return;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline3dgutMesh);
    }

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepthWrite);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepthTest);

    const bool usesGpuDist = usesGpuDistSort();
    if(!usesGpuDist && !m_assets.splatSets.hasCpuSortIndicesOnGpu())
      return;

    if(!usesGpuDist)
    {
      // 2D mesh-task grid (width = device maxMeshWorkGroupCount[0]), linearized in-shader.
      // groupCountY == 1 for scenes within the per-dimension limit (i.e. like 1D).
      const glm::uvec2 grid = getMeshTaskDispatchGrid(splatCount);
      vkCmdDrawMeshTasksEXT(cmd, grid.x, grid.y, 1);
    }
    else
    {
      // run the workgroups
      vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer, offsetof(shaderio::IndirectParams, groupCountX), 1,
                                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
    }
  }
}

void GaussianSplatting::drawMeshPrimitives(VkCommandBuffer cmd, bool ftbColorPass)
{

  NVVK_DBG_SCOPE(cmd);

  // Early exit if pipeline not initialized (can happen during async loading)
  if(m_graphicsPipelineMesh == VK_NULL_HANDLE)
    return;

  // For FTB color pass, use the additive blend pipeline
  if(ftbColorPass && m_graphicsPipelineMeshFtbColor == VK_NULL_HANDLE)
    return;

  VkDeviceSize offset{0};

  // Drawing all triangles
  // FTB color pass uses additive blend pipeline; regular pass uses standard pipeline
  VkPipeline pipeline = ftbColorPass ? m_graphicsPipelineMeshFtbColor : m_graphicsPipelineMesh;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

  // Depth state:
  // - Depth pre-pass (BTF or FTB pass 1): depth test ON (LESS), depth write ON
  // - FTB color pass: depth test ON (LESS_OR_EQUAL to not self-occlude), depth write OFF
  vkCmdSetDepthTestEnable(cmd, (VkBool32) true);
  vkCmdSetDepthWriteEnable(cmd, (VkBool32)!ftbColorPass);
  vkCmdSetDepthCompareOp(cmd, ftbColorPass ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_LESS);

  // Set FTB color pass flag for shader
  m_pcRaster.ftbColorPass = ftbColorPass ? 1 : 0;

  uint32_t instanceIndex = 0;
  for(const auto& instance : m_assets.meshes.instances)
  {
    if(!instance || !instance->mesh)
      continue;  // Skip invalid instances

    // Skip hidden instances (but still increment instanceIndex to keep descriptor alignment)
    if(!instance->shouldRender())
    {
      instanceIndex++;
      continue;
    }

    // Skip light proxies if visibility is disabled (but still increment instanceIndex to keep descriptor alignment)
    if(instance->type == MeshType::eLightProxy && !m_showLightProxies)
    {
      instanceIndex++;
      continue;
    }

    const auto& mesh = instance->mesh;

    // Only objIndex is needed (transforms come from MeshDesc in bindless assets)
    m_pcRaster.objIndex = instanceIndex;  // Index into per-instance descriptor array

    instanceIndex++;

    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(shaderio::PushConstant), &m_pcRaster);
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, mesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, mesh->gpuIndexCount, 1, 0, 0, 0);
  }
}


void GaussianSplatting::collectReadBackValuesIfNeeded(void)
{
  if((m_indirectReadbackHost.buffer != VK_NULL_HANDLE) && m_canCollectReadback)
  {
    std::memcpy((void*)&m_indirectReadback, (void*)m_indirectReadbackHost.mapping, sizeof(shaderio::IndirectParams));
  }
}

void GaussianSplatting::readBackIndirectParametersIfNeeded(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE)
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Indirect readback");

    // ensures m_indirect buffer modified by any shader stage is available for transfer
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);

    // copy from device to host buffer
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = sizeof(shaderio::IndirectParams)};
    vkCmdCopyBuffer(cmd, m_indirect.buffer, m_indirectReadbackHost.buffer, 1, &bc);

    m_canCollectReadback = true;
  }
}

void GaussianSplatting::updateRasterizationMemoryStatistics(const uint32_t splatCount)
{
  // If sorting buffers don't exist (e.g. pure RTX mode), all rasterization
  // "used" stats must be zero — there's no actual GPU memory behind them.
  if(!m_assets.splatSets.hasSortingBuffers())
  {
    memRasterization.hostAllocDistances  = 0;
    memRasterization.hostAllocIndices    = 0;
    memRasterization.deviceUsedDistances = 0;
    memRasterization.DeviceUsedIndices   = 0;
    memRasterization.usedIndirect        = 0;
    return;
  }

  // Update sorting statistics in manager
  // GPU distance/culling is used for both GPU radix sort and stochastic splat modes
  const bool usesGpuDist    = usesGpuDistSort();
  uint32_t usedSplatCount = usesGpuDist ? m_indirectReadback.instanceCount : splatCount;

  if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
  {
    // GPU sorting: distances buffer used, no host buffers
    memRasterization.hostAllocDistances  = 0;
    memRasterization.hostAllocIndices    = 0;
    memRasterization.deviceUsedDistances = usedSplatCount * sizeof(uint32_t);
    memRasterization.DeviceUsedIndices   = usedSplatCount * sizeof(uint32_t);
  }
  else if(prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT)
  {
    // Stochastic splat: GPU distance/culling runs, but no sorting buffers used
    memRasterization.hostAllocDistances  = 0;
    memRasterization.hostAllocIndices    = 0;
    memRasterization.deviceUsedDistances = 0;  // distances computed but not stored for sorting
    memRasterization.DeviceUsedIndices   = 0;  // no sorted indices needed
  }
  else
  {
    // CPU sorting: host indices buffer used, no distances
    memRasterization.hostAllocIndices    = splatCount * sizeof(uint32_t);
    memRasterization.hostAllocDistances  = 0;
    memRasterization.DeviceUsedIndices   = usedSplatCount * sizeof(uint32_t);
    memRasterization.deviceUsedDistances = 0;
  }

  // Update indirect buffer used size (depends on pipeline mode)
  if(!usesGpuDist)
  {
    memRasterization.usedIndirect = 0;
  }
  else
  {
    if(prmSelectedPipeline == PIPELINE_VERT)
    {
      memRasterization.usedIndirect = 5 * sizeof(uint32_t);
    }
    else
    {
      memRasterization.usedIndirect = sizeof(shaderio::IndirectParams);
    }
  }
}

// Reset all scene and rendering related resources (for scene/project reset)
// NOTE: vkDeviceWaitIdle shall be invoked before calling this method
// NOTE: This is virtual - UI layer overrides to clear selections first
void GaussianSplatting::reset()
{
  vkDeviceWaitIdle(m_device);

  m_canCollectReadback = false;
  m_projectPath.clear();
  deinitScene();

  // Reset indirect readback structure (contains stale values from previous scene)
  m_indirectReadback = shaderio::IndirectParams();

  // Reset all asset managers (splat sets, lights, meshes, cameras)
  // This calls processVramUpdates which destroys/rebuilds descriptor buffers
  m_assets.reset();
  // Note: Selection clearing handled by UI layer override (resetSelection() called before this)

  // Process any pending VRAM updates to ensure complete cleanup
  // forceAll=true to process RTX requests even if not in RTX pipeline
  processUpdateRequests(true);

  // Reset render settings (camera, parameters, etc.)
  resetRenderSettings();

  // Reset tonemapper to defaults
  m_tonemapperData = shaderio::TonemapperData{.isActive = 0, .method = shaderio::ToneMapMethod::eClip};

  // Reset camera to default position
  m_assets.cameras.setCamera(Camera());
  m_assets.cameras.setHomePreset(m_assets.cameras.getCamera());

  // Request shader rebuild for next frame
  // Pipelines and shaders remain alive (application lifetime)
  // They will be rebound to the new empty scene state, then to new scene when loaded
  m_requestUpdateShaders = true;
}

void GaussianSplatting::deinitScene()
{
  m_currentLoadingSplatSetFilename = "";
}

int GaussianSplatting::getPayloadArraySize() const
{
  bool hasMeshes = !m_assets.meshes.instances.empty();
  int meshMin = hasMeshes ? (prmRtx.quantizeMeshPayload ? MESH_PAYLOAD_MIN_SIZE_QUANTIZED : MESH_PAYLOAD_MIN_SIZE_FULL) : 0;
  int particleSpp = prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT ? 1 : prmRtx.particleSamplesPerPass;
  return std::max({1, particleSpp, meshMin});
}

uint32_t GaussianSplatting::getMeshTaskWorkgroupCount(uint32_t splatCount) const
{
  const uint32_t wgSize = std::max(1u, uint32_t(prmRaster.meshShaderWorkgroupSize));
  return (splatCount + wgSize - 1) / wgSize;
}

glm::uvec2 GaussianSplatting::getMeshTaskDispatchGrid(uint32_t splatCount) const
{
  const uint32_t totalGroups = getMeshTaskWorkgroupCount(splatCount);
  const uint32_t gridWidth   = std::max(1u, isSupported.maxMeshWorkGroupCountX);
  const uint32_t groupCountX = std::min(totalGroups, gridWidth);
  const uint32_t groupCountY = (totalGroups + gridWidth - 1) / gridWidth;
  return {groupCountX, groupCountY};
}

bool GaussianSplatting::isMeshTaskDispatchOverflow(uint32_t splatCount) const
{
  const glm::uvec2 grid = getMeshTaskDispatchGrid(splatCount);
  // Per-dimension Y limit and the X*Y (*Z=1) total-count limit must both hold for the dispatch to
  // be valid. grid.x is capped to maxMeshWorkGroupCountX by construction, so X never overflows.
  if(grid.y > isSupported.maxMeshWorkGroupCountY)
    return true;
  if(uint64_t(grid.x) * uint64_t(grid.y) > uint64_t(isSupported.maxMeshWorkGroupTotalCount))
    return true;
  return false;
}

void GaussianSplatting::updateSlangMacros()
{
  m_shaderMacros =  // comment to force clang new line and better indent
      {
          {"PIPELINE", std::to_string(prmSelectedPipeline)},
          // GPU vendor (GPU_VENDOR_NVIDIA/AMD/INTEL/OTHER) for vendor-gated shader paths
          {"GPU_VENDOR", std::to_string(isSupported.gpuVendor)},
          // Whether the mesh shaders may emit per-primitive attributes (NVIDIA only, see Slang #7019)
          {"MESH_SHADER_PER_PRIMITIVE_ATTRIBS", std::to_string((int)isSupported.meshShaderPerPrimitiveAttribute)},
          {"HYBRID_ENABLED",
           std::to_string((int)(prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT))},
          {"CAMERA_TYPE", std::to_string(m_assets.cameras.getCamera().model)},
          {"VISUALIZE", std::to_string((int)prmRender.visualize)},
          {"DISABLE_OPACITY_GAUSSIAN", std::to_string((int)prmRender.opacityGaussianDisabled)},
          {"FRUSTUM_CULLING_MODE", std::to_string(prmRaster.frustumCulling)},
          {"SIZE_CULLING_MODE", std::to_string(prmRaster.sizeCulling)},
          // Disabled, TODO do we enable ortho cam in the UI/camera controller
          {"ORTHOGRAPHIC_MODE", "0"},
          {"SHOW_SH_ONLY", std::to_string((int)prmRender.showShOnly)},
          //{"MAX_SH_DEGREE", std::to_string(prmRender.maxShDegree)}, // now in prmFrame
          //{"DATA_STORAGE", std::to_string(prmData.dataStorage)},    // now in splat set description
          {"SH_FORMAT", std::to_string(prmData.shFormat)},
          {"RGBA_FORMAT", std::to_string(prmData.rgbaFormat)},
          {"POINT_CLOUD_MODE", std::to_string((int)prmRaster.pointCloudModeEnabled)},
          {"USE_BARYCENTRIC", std::to_string((int)prmRaster.fragmentBarycentric)},
          // Wireframe relies on per-fragment barycentric coordinates and on
          // double-precision helpers pulled in via nvshaders/wireframe.h.slang
          // (computeDerivativeHighp / aabbFaceDerivativeHighp). On devices that
          // expose neither VK_KHR_fragment_shader_barycentric nor
          // shaderFloat64 (e.g. Intel Arc), enabling WIREFRAME would force the
          // fragment shader to declare unsupported SPIR-V capabilities. Coerce
          // the macro to 0 in that case; the user's prmRender.wireframe value
          // is preserved so loading a project on capable hardware restores it.
          {"WIREFRAME", std::to_string((int)(prmRender.wireframe && isSupported.shaderFloat64 && isSupported.fragmentShaderBarycentric))},
          {"DISTANCE_COMPUTE_WORKGROUP_SIZE", std::to_string((int)prmRaster.distShaderWorkgroupSize)},
          {"RASTER_MESH_WORKGROUP_SIZE", std::to_string((int)prmRaster.meshShaderWorkgroupSize)},
          // Mesh-task 2D grid width = device maxMeshWorkGroupCount[0]; shaders linearize the grid id
          {"MESH_SHADER_MAX_GROUPS_X", std::to_string(isSupported.maxMeshWorkGroupCountX)},
          {"MS_ANTIALIASING", std::to_string((int)prmRaster.msAntialiasing)},
          {"COVARIANCE_DILATION", std::to_string(prmRaster.covarianceDilation)},
          {"EXTENT_METHOD", std::to_string((int)prmRaster.extentProjection)},
          {"STOCHASTIC_SPLAT", std::to_string((int)(prmRaster.sortingMethod == SORTING_STOCHASTIC_SPLAT))},
          {"USE_GPU_DIST_SORT", std::to_string((int)usesGpuDistSort())},
          {"QUANTIZE_NORMALS", std::to_string((int)prmRaster.quantizeNormals)},
          // Normal computation method
          {"NORMAL_METHOD", std::to_string((int)prmRender.normalMethod)},
          // Global lighting, shadows, and DOF modes
          {"LIGHTING_MODE", std::to_string(effectiveLightingMode())},
          {"SHADOWS_MODE", std::to_string((int)prmRender.shadowsMode)},
          // GS shadow mask: gs-shadow lights darken the splat emissive at the surfel.
          // Needs at least one occluder source compiled (meshes by default, particles opt-in).
          {"GS_SHADOW_MASK", std::to_string((int)(prmRender.gsShadowMask
                                                  && (!m_assets.meshes.instances.empty() || prmRender.gsShadowMaskFromParticles)))},
          {"GS_SHADOW_MASK_PARTICLES", std::to_string((int)(prmRender.gsShadowMask && prmRender.gsShadowMaskFromParticles))},
          {"DOF_MODE", std::to_string((int)(m_assets.cameras.getCamera().dofMode))},
          // Surface info needed for lighting, DLSS, or DOF
          {"NEED_SURFACE_INFO", std::to_string((int)needSurfaceInfo())},
          // FTB only when surface info is needed and NOT stochastic splat (stochastic uses BTF)
          {"FRONT_TO_BACK", std::to_string((int)(needSurfaceInfo() && prmRaster.sortingMethod != SORTING_STOCHASTIC_SPLAT))},
          // FTB synchronization mode for depth buffer storage image access
          {"FTB_SYNC_MODE", std::to_string(prmRaster.ftbSyncMode)},
          {"TEMPORAL_SAMPLING", std::to_string((int)prmRtx.temporalSampling)},
          {"KERNEL_DEGREE", std::to_string(prmRtx.kernelDegree)},
          {"KERNEL_MIN_RESPONSE", std::to_string(prmRtx.kernelMinResponse)},
          {"KERNEL_ADAPTIVE_CLAMPING", std::to_string((int)prmRtx.kernelAdaptiveClamping)},
          {"PARTICLES_SPP", std::to_string(prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT ? 1 : prmRtx.particleSamplesPerPass)},
          {"PAYLOAD_ARRAY_SIZE", std::to_string(getPayloadArraySize())},
          {"RTX_QUANTIZE_MESH_PAYLOAD", std::to_string((int)prmRtx.quantizeMeshPayload)},
          {"RTX_HAS_MESHES", std::to_string((int)(!m_assets.meshes.instances.empty()))},
          {"RTX_HAS_PARTICLES", std::to_string((int)(m_assets.splatSets.getEffectiveGlobalSplatCount() > 0))},
          {"RTX_TRACE_STRATEGY", std::to_string(prmRtx.rtxTraceStrategy)},
          {"TRACE_PROFILE", std::to_string((int)prmRtx.traceProfile)},
          {"RTX_USE_INSTANCES", std::to_string((int)prmRtxData.useTlasInstances)},
          {"RTX_USE_AABBS", std::to_string((int)prmRtxData.useAABBs)},
          {"RTX_USE_SPHERES", std::to_string((int)prmRtxData.useSpheres)},
          {"RTX_PARTICLE_DEPTH", std::to_string(prmRtx.particleDepth)},
          {"RTX_BILLBOARD_FRUSTUM_CULLING",
           std::to_string((int)(prmRtx.billboardFrustumCulling && prmRtx.particleDepth == PARTICLE_DEPTH_BILLBOARD))},
          {"RTX_SHORTEN_RAY", std::to_string((int)[&]() {
             const bool userControlled = (prmRtx.rtxTraceStrategy == RTX_TRACE_STRATEGY_STOCHASTIC_ANYHIT
                                          && prmRtx.particleDepth == PARTICLE_DEPTH_BILLBOARD);
             if(!userControlled)
               prmRtx.shortenRay = true;
             return prmRtx.shortenRay;
           }())},
          {"PARTICLE_AO_ENABLED", std::to_string((int)(prmRtx.particleEmissiveAoEnabled && !m_assets.meshes.instances.empty()
                                                       && effectiveLightingMode() == LIGHTING_ENABLED))},
#if defined(USE_DLSS)
          {"DLSS_ENABLED", std::to_string((int)m_dlss.isEnabled())},
#else
          {"DLSS_ENABLED", "0"},
#endif
      };

  // Print all macro values to console for debugging
  LOGI("=== Slang Shader Macros ===\n");
  for(const auto& macro : m_shaderMacros)
  {
    LOGI("  %s = %s\n", macro.first.c_str(), macro.second.c_str());
  }
  LOGI("===========================\n");

  m_slangCompiler.clearMacros();

  // then provide the char* strings to the compiler
  for(auto& macro : m_shaderMacros)
  {
    m_slangCompiler.addMacro({macro.first.c_str(), macro.second.c_str()});
  }
}

bool GaussianSplatting::compileSlangShader(const std::string& filename, VkShaderModule& module)
{

  if(!m_slangCompiler.compileFile(filename))
  {
    return false;
  }

  if(module != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_device, module, nullptr);

  // Create the VK module
  VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = m_slangCompiler.getSpirvSize(),
                                      .pCode    = m_slangCompiler.getSpirv()};

  if(m_slangCompiler.getSpirvSize() == 0)
  {
    LOGE("Missing entry point in shader %s\n", filename.c_str());
    return false;
  }
  NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &module));
  NVVK_DBG_NAME(module);

  m_shaders.modules.emplace_back(&module);

  return true;
}

bool GaussianSplatting::initShaders(void)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  bool success = true;

  updateSlangMacros();

  // Particles distance to viewpoint and frustum culling
  success &= compileSlangShader("dist.comp.slang", m_shaders.distShader);
  // 3DGS raster
  success &= compileSlangShader("threedgs_raster.vert.slang", m_shaders.vertexShader);
  success &= compileSlangShader("threedgs_raster.frag.slang", m_shaders.fragmentShader);
  // 3DGUT raster fragment stage (mesh stage compiled below when supported)
  success &= compileSlangShader("threedgut_raster.frag.slang", m_shaders.threedgutFragmentShader);
  // Mesh shader stages: skip compilation entirely when VK_EXT_mesh_shader is
  // not available so we never produce SPIR-V the device cannot consume.
  // Pipeline creation (initPipelines) and the matching draw paths are gated
  // by the same flag, so leaving the modules at VK_NULL_HANDLE is safe.
  if(isSupported.meshShader)
  {
    success &= compileSlangShader("threedgs_raster.mesh.slang", m_shaders.meshShader);
    success &= compileSlangShader("threedgut_raster.mesh.slang", m_shaders.threedgutMeshShader);
  }
  // Mesh raster
  success &= compileSlangShader("threedmesh_raster.vert.slang", m_shaders.meshVertexShader);
  success &= compileSlangShader("threedmesh_raster.frag.slang", m_shaders.meshFragmentShader);
  // Ray tracing stages and the particle AS build compute shader are only
  // compiled when KHR ray tracing pipeline / acceleration structure are
  // available. initRtPipeline / initRtDescriptorSet and the AS build pipeline
  // creation are gated on the same flag, and processVramUpdates skips Phase 3
  // when raytracing is unsupported, so leaving the SPIR-V modules at
  // VK_NULL_HANDLE is safe.
  if(isSupported.raytracing)
  {
    success &= compileSlangShader("threedgrt_raytrace.rgen.slang", m_shaders.rtxRgenShader);
    success &= compileSlangShader("threedgrt_raytrace.rmiss.slang", m_shaders.rtxRmissShader);
    success &= compileSlangShader("threedgrt_raytrace_shadow.rmiss.slang", m_shaders.rtxRmiss2Shader);
    success &= compileSlangShader("threedgrt_raytrace.rchit.slang", m_shaders.rtxRchitShader);
    success &= compileSlangShader("threedgrt_raytrace.rahit.slang", m_shaders.rtxRahitShader);
    success &= compileSlangShader("threedgrt_raytrace.rint.slang", m_shaders.rtxRintShader);
    if(prmRtxData.useAABBs && prmRtx.particleDepth == PARTICLE_DEPTH_BILLBOARD)
      success &= compileSlangShader("threedgrt_raytrace.rint_billboard.slang", m_shaders.rtxRintBillboardShader);
    success &= compileSlangShader("particle_as_build.comp.slang", m_shaders.particleAsBuildShader);
  }
  // Post processings
  success &= compileSlangShader("post.comp.slang", m_shaders.postComputeShader);
  // Deferred shading (for raster-only pipelines with surface reconstruction)
  if(prmRender.lightingEnabled)
  {
    success &= compileSlangShader("deferred_shading.comp.slang", m_shaders.deferredShadingShader);
  }

  // FTB depth consolidation shaders (write picked splat depth to hw depth buffer)
  // Only needed for FTB mode (surface info without stochastic splat)
  const bool useFTB = needSurfaceInfo() && (prmRaster.sortingMethod != SORTING_STOCHASTIC_SPLAT);
  if(useFTB)
  {
    success &= compileSlangShader("depth_consolidate.vert.slang", m_shaders.depthConsolidateVertShader);
    success &= compileSlangShader("depth_consolidate.frag.slang", m_shaders.depthConsolidateFragShader);
  }

  if(!success)
    return (m_shaders.valid = false);

  auto      endTime   = std::chrono::high_resolution_clock::now();
  long long buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  LOGI("Shaders updated in %lldms\n", buildTime);

  return (m_shaders.valid = true);
}

void GaussianSplatting::deinitShaders(void)
{
  for(auto& shader : m_shaders.modules)
  {
    vkDestroyShaderModule(m_device, *shader, nullptr);
    *shader = VK_NULL_HANDLE;
  }

  m_shaders.valid = false;
  m_shaders.modules.clear();
}

//--------------------------------------------------------------------------------------------------
// Utility: create a graphics pipeline, logging and nulling out the handle on failure.
//
bool GaussianSplatting::createGraphicsPipelineChecked(VkDevice                           device,
                                                      VkPipelineCache                    cache,
                                                      nvvk::GraphicsPipelineCreator&     creator,
                                                      const nvvk::GraphicsPipelineState& state,
                                                      VkPipeline*                        pipeline,
                                                      const char*                        debugName)
{
  const VkResult result = creator.createGraphicsPipeline(device, cache, state, pipeline);
  if(result != VK_SUCCESS)
  {
    LOGE("Failed to create graphics pipeline '%s': %s\n", debugName, string_VkResult(result));
    *pipeline = VK_NULL_HANDLE;
    return false;
  }
  NVVK_DBG_NAME(*pipeline);
  return true;
}

void GaussianSplatting::initPipelines()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  // Use member variable so we can update bindings on resize
  m_descriptorBindings.clear();

  m_descriptorBindings.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_descriptorBindings.addBinding(BINDING_INDIRECT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

  // Legacy splat bindings removed - all data accessed via BINDING_ASSETS now
  // BINDING_CENTERS_BUFFER, BINDING_COLORS_BUFFER, BINDING_SCALES_BUFFER, etc. REMOVED
  // BINDING_SPLAT_MATERIAL REMOVED (per-instance material in SplatSetDesc)

  // Bindless assets buffer (unified access to all scene assets, including meshes and lights)
  m_descriptorBindings.addBinding(BINDING_ASSETS, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);

  // Bindless texture array for STORAGE_TEXTURES mode
  // Always create this binding since storage mode can be switched at runtime
  // Multi-splat-set support: 6 textures per instance, max 1000 instances = 6000 textures
  uint32_t maxTextureDescriptors = 6000;  // Conservative max (1000 instances * 6 textures)
  m_descriptorBindings.addBinding(BINDING_SPLAT_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                  maxTextureDescriptors, VK_SHADER_STAGE_ALL);

  // Mesh material textures (base color, metallic-roughness, normal, emissive, occlusion).
  // PARTIALLY_BOUND: only populated slots are valid; shaders guard access with texture ID >= 0.
  uint32_t maxMeshTextureDescriptors = 512;
  m_descriptorBindings.addBinding(BINDING_MESH_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxMeshTextureDescriptors,
                                  VK_SHADER_STAGE_ALL, nullptr, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

  // Rasterization surface reconstruction buffers (integrated normal, depth, and splat ID)
  // Need FRAGMENT_BIT for raster output, RAYGEN_BIT_KHR for hybrid mode, COMPUTE_BIT for deferred shading
  m_descriptorBindings.addBinding(BINDING_RASTER_NORMAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                  VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindings.addBinding(BINDING_RASTER_DEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                  VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindings.addBinding(BINDING_RASTER_SPLATID, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                  VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindings.addBinding(BINDING_RASTER_COLOR, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindings.addBinding(BINDING_DEFERRED_OUTPUT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  // AUX buffers for temporal accumulation (deferred shading selects based on frameSampleId)
  m_descriptorBindings.addBinding(BINDING_RASTER_COLOR_AUX, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindings.addBinding(BINDING_DEFERRED_OUTPUT_AUX, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

  //
  const VkPushConstantRange pcRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                            | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT,
                                        0, sizeof(shaderio::PushConstant)};

  NVVK_CHECK(m_descriptorBindings.createDescriptorSetLayout(m_device, 0, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  //
  std::vector<VkDescriptorPoolSize> poolSize;
  m_descriptorBindings.appendPoolSizes(poolSize);
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

  // Particle AS compute pipeline layout
  // Uses the same descriptor set layout as the main pipeline so the AS build
  // compute shader can access splat textures (BINDING_SPLAT_TEXTURES) via the
  // storage accessors in threedgs_particle_storage.h.slang.
  // Only created when ray tracing is available: the AS build path is the sole
  // consumer of this layout, and SplatSetManagerVk::processVramUpdates skips
  // BLAS/TLAS builds (Phase 3) when raytracing is unsupported.
  if(isSupported.raytracing)
  {
    const VkPushConstantRange pcRange = {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(shaderio::ParticleAsBuildPushConstants),
    };
    VkPipelineLayoutCreateInfo pcLayoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcRange,
    };
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pcLayoutInfo, nullptr, &m_particleAsPipelineLayout));
    NVVK_DBG_NAME(m_particleAsPipelineLayout);
  }

  // Write descriptors for the buffers and textures
  nvvk::WriteSetContainer writeContainer;

  // add common buffers
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_FRAME_INFO_UBO, m_descriptorSet), m_frameInfoBuffer);
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_INDIRECT_BUFFER, m_descriptorSet), m_indirect);
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_ASSETS, m_descriptorSet), m_assets.assetsBuffer);

  // Rasterization surface reconstruction buffers (will be updated on resize)
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_NORMAL, m_descriptorSet),
                        m_gBuffers.getColorImageView(COLOR_RASTER_NORMAL), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_DEPTH, m_descriptorSet),
                        m_gBuffers.getColorImageView(COLOR_RASTER_DEPTH), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_SPLATID, m_descriptorSet),
                        m_gBuffers.getColorImageView(COLOR_RASTER_SPLATID), VK_IMAGE_LAYOUT_GENERAL);
  // Deferred shading: bind both MAIN and AUX buffers, shader selects based on frameSampleId
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_COLOR, m_descriptorSet),
                        m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_DEFERRED_OUTPUT, m_descriptorSet),
                        m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_COLOR_AUX, m_descriptorSet),
                        m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_DEFERRED_OUTPUT_AUX, m_descriptorSet),
                        m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);

  // write
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

  // Bind splat textures (STORAGE_TEXTURES mode) and mesh textures into the freshly
  // allocated descriptor set. These are written separately because they use variable-count
  // bindless arrays and can change independently of the other bindings.
  updateSplatTextureDescriptors();
  updateMeshTextureDescriptors();

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
    NVVK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_computePipelineGsDistCull));
    NVVK_DBG_NAME(m_computePipelineGsDistCull);
  }

  // Create the particle AS compute pipeline only when ray tracing is
  // supported. The AS build shader is exclusively used to populate BLAS/TLAS
  // inputs in SplatSetManagerVk::processVramUpdates Phase 3, which is skipped
  // when isSupported.raytracing is false (matching the gates in initShaders
  // and the layout creation above).
  if(isSupported.raytracing && m_shaders.particleAsBuildShader)
  {
    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_shaders.particleAsBuildShader,
                .pName  = "main",
            },
        .layout = m_particleAsPipelineLayout,
    };
    vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_computePipelineParticleAs);
    NVVK_DBG_NAME(m_computePipelineParticleAs);
  }

  // Create the deferred shading compute pipeline
  if(prmRender.lightingEnabled && m_shaders.deferredShadingShader)
  {
    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_shaders.deferredShadingShader,
                .pName  = "main",
            },
        .layout = m_pipelineLayout,
    };
    vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_computePipelineDeferredShading);
    NVVK_DBG_NAME(m_computePipelineDeferredShading);
  }

  // Provide particle AS compute pipeline state and descriptor set to splat manager
  // The descriptor set is needed so the AS build shader can access splat textures
  m_assets.splatSets.setParticleAsComputeState(m_computePipelineParticleAs, m_particleAsPipelineLayout, m_descriptorSet);

  // FTB mode: only when surface info is needed and NOT stochastic splat
  const bool useFTB = needSurfaceInfo() && (prmRaster.sortingMethod != SORTING_STOCHASTIC_SPLAT);

  // Create the GS rasterization pipelines
  {
    // Preparing the common states
    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // activates blending and set blend func
    // Back-to-front (default): standard "over" compositing
    //   Color: SrcColor * SrcAlpha + DstColor * (1 - SrcAlpha)
    //   Alpha: SrcAlpha * 1 + DstAlpha * 1 (additive accumulation)
    // Front-to-back: "under" compositing with premultiplied alpha
    //   Color: SrcPremultipliedColor * (1 - DstAlpha) + DstColor * 1
    //   Alpha: SrcAlpha * (1 - DstAlpha) + DstAlpha * 1
    pipelineState.colorBlendEnables[0]                = VK_TRUE;
    pipelineState.colorBlendEquations[0].alphaBlendOp = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].colorBlendOp = VK_BLEND_OP_ADD;

    if(useFTB)
    {
      // Front-to-back "under" operator with premultiplied alpha
      // Shader outputs: float4(color.rgb * opacity, opacity)
      pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
      pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
      pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    }
    else
    {
      // Back-to-front "over" operator (standard alpha blending)
      // Shader outputs: float4(color.rgb, opacity)
      pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    }

    // Add blend state for normal buffer (second color attachment) if generating surface
    if(needSurfaceInfo())
    {
      // Same blending for normal integration (pre-multiplied by opacity in fragment shader)
      pipelineState.colorBlendEnables.push_back(VK_TRUE);
      pipelineState.colorWriteMasks.push_back(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
      if(useFTB)
      {
        // Front-to-back "under" operator for normal integration
        pipelineState.colorBlendEquations.push_back({
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .colorBlendOp        = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .alphaBlendOp        = VK_BLEND_OP_ADD,
        });
      }
      else
      {
        // Back-to-front "over" operator for normal integration
        pipelineState.colorBlendEquations.push_back({
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,  // Already pre-multiplied by opacity
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp        = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,  // Additive for opacity accumulation
            .alphaBlendOp        = VK_BLEND_OP_ADD,
        });
      }

      // Add blend state for depth buffer (third color attachment) - RG32F (R=depth, G=transmittance)
      // For FTB: depth buffer is used as storage image only (manual blending), not as color attachment
      // For BTF: depth buffer is color attachment with hardware blending
      if(!useFTB)
      {
        pipelineState.colorBlendEnables.push_back(VK_TRUE);
        pipelineState.colorWriteMasks.push_back(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT);  // R for depth, G for transmittance
        // Back-to-front: standard depth blending (transmittance not used in BTF)
        pipelineState.colorBlendEquations.push_back({
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,  // Depth weighted by opacity from src alpha
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp        = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .alphaBlendOp        = VK_BLEND_OP_ADD,
        });
      }
      // Note: For FTB, depth buffer is not a color attachment - it's accessed via storage image

      // Add blend state for splat ID buffer (fourth color attachment) - NO blending, just write
      pipelineState.colorBlendEnables.push_back(VK_FALSE);                // No blending for integer ID
      pipelineState.colorWriteMasks.push_back(VK_COLOR_COMPONENT_R_BIT);  // Single component for uint32
      pipelineState.colorBlendEquations.push_back({});                    // Unused but required
    }

    // By default disable depth write and test for the pipeline
    // Since splats are sorted, screen aligned, and rendered back to front
    // we do not need depth test/write, which leads to faster rendering
    // however since CPU sorting mode is costly we disable it when not visualizing with alpha,
    // only in this case we will use depth test/write. this will be changed dynamically at rendering.
    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = VK_FALSE;
    pipelineState.depthStencilState.depthTestEnable  = VK_FALSE;

    // Mesh-shader graphics pipelines (3DGS and 3DGUT) are only created when
    // VK_EXT_mesh_shader is available; the matching draw paths and pipeline
    // selector are gated by isSupported.meshShader as well.
    if(isSupported.meshShader)
    {
      // create the pipeline that uses mesh shaders for 3DGS
      {
        nvvk::GraphicsPipelineCreator creator;
        creator.pipelineInfo.layout = m_pipelineLayout;
        creator.colorFormats        = {prmRender.colorFormat};
        // Add normal, depth, and splat ID buffer formats if generating surface
        // For FTB: depth buffer is storage image only, not color attachment
        if(needSurfaceInfo())
        {
          creator.colorFormats.push_back(m_normalFormat);
          if(!useFTB)
          {
            creator.colorFormats.push_back(m_rasterDepthFormat);
          }
          creator.colorFormats.push_back(m_splatIdFormat);
        }
        creator.renderingState.depthAttachmentFormat = m_depthFormat;
        creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
        creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

        creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.meshShader);
        creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main_mesh", m_shaders.fragmentShader);

        createGraphicsPipelineChecked(m_device, m_pipelineCache, creator, pipelineState, &m_graphicsPipelineGsMesh, "GsMesh");
      }

      // create the pipeline that uses mesh shaders for 3DGUT
      {
        nvvk::GraphicsPipelineCreator creator;
        creator.pipelineInfo.layout = m_pipelineLayout;
        creator.colorFormats        = {prmRender.colorFormat};
        // Add normal, depth, and splat ID buffer formats if generating surface
        // For FTB: depth buffer is storage image only, not color attachment
        if(needSurfaceInfo())
        {
          creator.colorFormats.push_back(m_normalFormat);
          if(!useFTB)
          {
            creator.colorFormats.push_back(m_rasterDepthFormat);
          }
          creator.colorFormats.push_back(m_splatIdFormat);
        }
        creator.renderingState.depthAttachmentFormat = m_depthFormat;
        creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
        creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

        creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", m_shaders.threedgutMeshShader);
        creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.threedgutFragmentShader);

        createGraphicsPipelineChecked(m_device, m_pipelineCache, creator, pipelineState, &m_graphicsPipeline3dgutMesh, "3dgutMesh");
      }
    }

    // create the pipeline that uses vertex shaders for 3DGS (instanced vertex divisors, core in Vulkan 1.4)
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
      creator.pipelineInfo.layout = m_pipelineLayout;
      creator.colorFormats        = {prmRender.colorFormat};
      // Add normal, depth, and splat ID buffer formats if generating surface
      // For FTB: depth buffer is storage image only, not color attachment
      if(needSurfaceInfo())
      {
        creator.colorFormats.push_back(m_normalFormat);
        if(!useFTB)
        {
          creator.colorFormats.push_back(m_rasterDepthFormat);
        }
        creator.colorFormats.push_back(m_splatIdFormat);
      }
      creator.renderingState.depthAttachmentFormat = m_depthFormat;
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);

      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.vertexShader);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.fragmentShader);

      createGraphicsPipelineChecked(m_device, m_pipelineCache, creator, pipelineState, &m_graphicsPipelineGsVert, "GsVert");
    }
  }
  // Create the 3D mesh rasterization pipeline
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

    // Add blend states for surface generation buffers (meshes don't write to them, but need matching attachment count)
    // For FTB: depth buffer is storage image only, not color attachment
    if(needSurfaceInfo())
    {
      // Normal buffer - no blending, no write (meshes don't contribute)
      pipelineState.colorBlendEnables.push_back(VK_FALSE);
      pipelineState.colorWriteMasks.push_back(0);  // Don't write anything
      pipelineState.colorBlendEquations.push_back({});

      // Depth buffer - no blending, no write (only in BTF mode - FTB uses storage image)
      if(!useFTB)
      {
        pipelineState.colorBlendEnables.push_back(VK_FALSE);
        pipelineState.colorWriteMasks.push_back(0);
        pipelineState.colorBlendEquations.push_back({});
      }

      // Splat ID buffer - no blending, no write
      pipelineState.colorBlendEnables.push_back(VK_FALSE);
      pipelineState.colorWriteMasks.push_back(0);
      pipelineState.colorBlendEquations.push_back({});
    }

    // TODOC
    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = VK_TRUE;
    pipelineState.depthStencilState.depthTestEnable  = VK_TRUE;

    // create the pipeline
    const auto BINDING_ATTR_VERTEX = 0;

    pipelineState.vertexBindings   = {{.binding = BINDING_ATTR_VERTEX, .stride = sizeof(Vertex), .divisor = 1}};
    pipelineState.vertexAttributes = {{.location = ATTRIBUTE_LOC_MESH_POSITION,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(Vertex, pos))},
                                      {.location = ATTRIBUTE_LOC_MESH_NORMAL,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(Vertex, nrm))},
                                      {.location = ATTRIBUTE_LOC_MESH_TEXCOORD,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(Vertex, texCoord))},
                                      {.location = ATTRIBUTE_LOC_MESH_TEXCOORD1,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(Vertex, texCoord1))},
                                      {.location = ATTRIBUTE_LOC_MESH_TANGENT,
                                       .binding  = BINDING_ATTR_VERTEX,
                                       .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
                                       .offset   = static_cast<uint32_t>(offsetof(Vertex, tangent))}};

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout = m_pipelineLayout;
    creator.colorFormats        = {prmRender.colorFormat};
    // Add surface generation buffer formats if enabled (meshes don't write to them but need matching count)
    // For FTB: depth buffer is storage image only, not color attachment
    if(needSurfaceInfo())
    {
      creator.colorFormats.push_back(m_normalFormat);
      if(!useFTB)
      {
        creator.colorFormats.push_back(m_rasterDepthFormat);
      }
      creator.colorFormats.push_back(m_splatIdFormat);
    }
    creator.renderingState.depthAttachmentFormat = m_depthFormat;
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);

    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.meshVertexShader);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.meshFragmentShader);

    createGraphicsPipelineChecked(m_device, m_pipelineCache, creator, pipelineState, &m_graphicsPipelineMesh, "SceneMesh");

    // Create FTB mesh color pass pipeline
    // Use dst alpha (accumulated splat opacity) as transmittance:
    // finalColor = meshColor * (1 - dstAlpha) + splatColor
    // This way transmittance = (1 - accumulated_opacity) is computed via blend hardware
    if(useFTB)
    {
      // Blend: src * (1 - dstAlpha) + dst * 1
      // meshColor weighted by transmittance, added to accumulated splat colors
      pipelineState.colorBlendEnables[0]                       = VK_TRUE;
      pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
      pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
      pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;  // Don't modify alpha
      pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;   // Keep accumulated alpha
      pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;

      nvvk::GraphicsPipelineCreator creatorFtb;
      creatorFtb.pipelineInfo.layout                  = m_pipelineLayout;
      creatorFtb.colorFormats                         = creator.colorFormats;  // Same formats
      creatorFtb.renderingState.depthAttachmentFormat = m_depthFormat;
      creatorFtb.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
      creatorFtb.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
      creatorFtb.dynamicStateValues.push_back(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);

      creatorFtb.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.meshVertexShader);
      creatorFtb.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.meshFragmentShader);

      createGraphicsPipelineChecked(m_device, m_pipelineCache, creatorFtb, pipelineState,
                                    &m_graphicsPipelineMeshFtbColor, "SceneMeshFtbColor");
    }
  }

  // FTB depth consolidation pipeline: write picked splat depth to hardware depth buffer
  // This runs after mesh color pass to merge splat depths with mesh depths for visual helpers
  if(useFTB && m_shaders.depthConsolidateVertShader && m_shaders.depthConsolidateFragShader)
  {
    nvvk::GraphicsPipelineState pipelineStateDepth;
    pipelineStateDepth.inputAssemblyState.topology        = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pipelineStateDepth.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineStateDepth.depthStencilState.depthTestEnable  = VK_TRUE;
    pipelineStateDepth.depthStencilState.depthWriteEnable = VK_TRUE;
    pipelineStateDepth.depthStencilState.depthCompareOp   = VK_COMPARE_OP_LESS;  // Write if closer than existing

    // Disable color writes for all attachments (depth-only pass)
    // Must match render pass color attachment count: main + normal + splatId = 3
    // All arrays must have matching sizes
    VkColorBlendEquationEXT noBlend{};  // Default values (no blending)
    pipelineStateDepth.colorBlendEnables   = {VK_FALSE, VK_FALSE, VK_FALSE};
    pipelineStateDepth.colorBlendEquations = {noBlend, noBlend, noBlend};
    pipelineStateDepth.colorWriteMasks     = {0, 0, 0};  // No color writes

    nvvk::GraphicsPipelineCreator creatorDepth;
    creatorDepth.pipelineInfo.layout = m_pipelineLayout;
    // Must match the render pass color attachments (FTB with generateSurface)
    creatorDepth.colorFormats                         = {prmRender.colorFormat, m_normalFormat, m_splatIdFormat};
    creatorDepth.renderingState.depthAttachmentFormat = m_depthFormat;

    creatorDepth.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_shaders.depthConsolidateVertShader);
    creatorDepth.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_shaders.depthConsolidateFragShader);

    createGraphicsPipelineChecked(m_device, m_pipelineCache, creatorDepth, pipelineStateDepth,
                                  &m_graphicsPipelineDepthConsolidate, "DepthConsolidate");
  }
}

void GaussianSplatting::updateSplatTextureDescriptors()
{
  m_assets.splatSets.updateBindingSplatTextures(m_descriptorSet);
}

void GaussianSplatting::updateMeshTextureDescriptors()
{
  if(m_descriptorSet == VK_NULL_HANDLE)
    return;

  const auto& textures = m_assets.meshes.getMeshTextures();
  if(textures.empty())
    return;

  std::vector<VkDescriptorImageInfo> imageInfos;
  imageInfos.reserve(textures.size());
  for(const auto& tex : textures)
  {
    imageInfos.push_back(tex->image.descriptor);
  }

  VkWriteDescriptorSet writeSet{};
  writeSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeSet.dstSet          = m_descriptorSet;
  writeSet.dstBinding      = BINDING_MESH_TEXTURES;
  writeSet.dstArrayElement = 0;
  writeSet.descriptorCount = static_cast<uint32_t>(imageInfos.size());
  writeSet.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writeSet.pImageInfo      = imageInfos.data();

  vkUpdateDescriptorSets(m_device, 1, &writeSet, 0, nullptr);
}

// include RTX one
void GaussianSplatting::deinitPipelines()
{
  m_assets.splatSets.setParticleAsComputeState(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);

  TEST_DESTROY_AND_RESET(m_graphicsPipelineGsVert, vkDestroyPipeline(m_device, m_graphicsPipelineGsVert, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineGsMesh, vkDestroyPipeline(m_device, m_graphicsPipelineGsMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipeline3dgutMesh, vkDestroyPipeline(m_device, m_graphicsPipeline3dgutMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineMesh, vkDestroyPipeline(m_device, m_graphicsPipelineMesh, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineMeshFtbColor, vkDestroyPipeline(m_device, m_graphicsPipelineMeshFtbColor, nullptr));
  TEST_DESTROY_AND_RESET(m_graphicsPipelineDepthConsolidate,
                         vkDestroyPipeline(m_device, m_graphicsPipelineDepthConsolidate, nullptr));
  TEST_DESTROY_AND_RESET(m_computePipelineGsDistCull, vkDestroyPipeline(m_device, m_computePipelineGsDistCull, nullptr));
  TEST_DESTROY_AND_RESET(m_computePipelineParticleAs, vkDestroyPipeline(m_device, m_computePipelineParticleAs, nullptr));
  TEST_DESTROY_AND_RESET(m_computePipelineDeferredShading, vkDestroyPipeline(m_device, m_computePipelineDeferredShading, nullptr));

  TEST_DESTROY_AND_RESET(m_pipelineLayout, vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr));
  TEST_DESTROY_AND_RESET(m_particleAsPipelineLayout, vkDestroyPipelineLayout(m_device, m_particleAsPipelineLayout, nullptr));
  TEST_DESTROY_AND_RESET(m_descriptorSetLayout, vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr));
  TEST_DESTROY_AND_RESET(m_descriptorPool, vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr));

  // Invalidate descriptor set handle (destroyed with pool)
  m_descriptorSet = VK_NULL_HANDLE;

  // RTX TODO move this in rtDeinitPipeline and invoke in proper location
  TEST_DESTROY_AND_RESET(m_rtPipeline, vkDestroyPipeline(m_device, m_rtPipeline, nullptr));

  TEST_DESTROY_AND_RESET(m_rtPipelineLayout, vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr));
  TEST_DESTROY_AND_RESET(m_rtDescriptorPool, vkDestroyDescriptorPool(m_device, m_rtDescriptorPool, nullptr));
  TEST_DESTROY_AND_RESET(m_rtDescriptorSetLayout, vkDestroyDescriptorSetLayout(m_device, m_rtDescriptorSetLayout, nullptr));

  // Invalidate RT descriptor set handle (destroyed with pool)
  m_rtDescriptorSet = VK_NULL_HANDLE;

  m_alloc.destroyBuffer(m_rtSBTBuffer);
  m_rtShaderGroups.clear();

  // Post process
  TEST_DESTROY_AND_RESET(m_computePipelinePostProcess, vkDestroyPipeline(m_device, m_computePipelinePostProcess, nullptr));

  TEST_DESTROY_AND_RESET(m_pipelineLayoutPostProcess, vkDestroyPipelineLayout(m_device, m_pipelineLayoutPostProcess, nullptr));
  TEST_DESTROY_AND_RESET(m_descriptorPoolPostProcess, vkDestroyDescriptorPool(m_device, m_descriptorPoolPostProcess, nullptr));
  TEST_DESTROY_AND_RESET(m_descriptorSetLayoutPostProcess,
                         vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayoutPostProcess, nullptr));

  m_assets.splatSets.setParticleAsComputeState(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);
}

//--------------------------------------------------------------------------------------------------
// Initialize renderer buffers (application lifetime - independent of splat count)
// Called once during onAttach(), destroyed in onDetach()
// These buffers do NOT depend on splat count:
//   - Indirect draw parameters
//   - Quad mesh (fixed 4 vertices, 6 indices)
//   - Frame info uniform buffer
//
void GaussianSplatting::initRendererBuffers()
{
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

  // buffers are small so we use vkCmdUpdateBuffer for the transfers
  vkCmdUpdateBuffer(cmd, m_quadVertices.buffer, 0, vertices.size() * sizeof(float), vertices.data());
  vkCmdUpdateBuffer(cmd, m_quadIndices.buffer, 0, indices.size() * sizeof(uint16_t), indices.data());
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Uniform buffer
  m_alloc.createBuffer(m_frameInfoBuffer, sizeof(shaderio::FrameInfo),
                       VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  NVVK_DBG_NAME(m_frameInfoBuffer.buffer);

  // Track memory for renderer buffers (application lifetime, fixed size)
  vk_gaussian_splatting::memRender.usedUboFrameInfo = sizeof(shaderio::FrameInfo);
  vk_gaussian_splatting::memRender.quadVertices     = vertices.size() * sizeof(float);
  vk_gaussian_splatting::memRender.quadIndices      = indices.size() * sizeof(uint16_t);
}

//--------------------------------------------------------------------------------------------------
// Destroy renderer buffers (application lifetime)
// Called once during onDetach()
//
void GaussianSplatting::deinitRendererBuffers()
{
  m_alloc.destroyBuffer(m_indirect);

  // Reset renderer buffer memory tracking
  vk_gaussian_splatting::memRender.usedUboFrameInfo = 0;
  vk_gaussian_splatting::memRender.quadVertices     = 0;
  vk_gaussian_splatting::memRender.quadIndices      = 0;
  m_alloc.destroyBuffer(m_indirectReadbackHost);

  m_alloc.destroyBuffer(m_quadVertices);
  m_alloc.destroyBuffer(m_quadIndices);

  m_alloc.destroyBuffer(m_frameInfoBuffer);
  // Bindless assets buffer now destroyed by AssetManagerVk::deinit()
}

//--------------------------------------------------------------------------------------------------
// Output image getters - return helper buffer if helpers were rendered,
// else COLOR_LDR when tonemapping is active, else COLOR_MAIN
//--------------------------------------------------------------------------------------------------
VkImage GaussianSplatting::getOutputColorImage() const
{
  VkImage helperImage = m_helpers.getOutputColorImage();
  if(helperImage)
    return helperImage;
  return m_tonemapperData.isActive ? m_gBuffers.getColorImage(COLOR_LDR) : m_gBuffers.getColorImage(COLOR_MAIN);
}

VkImageView GaussianSplatting::getOutputColorImageView() const
{
  VkImageView helperImageView = m_helpers.getOutputColorImageView();
  if(helperImageView)
    return helperImageView;
  return m_tonemapperData.isActive ? m_gBuffers.getColorImageView(COLOR_LDR) : m_gBuffers.getColorImageView(COLOR_MAIN);
}

VkDescriptorSet GaussianSplatting::getOutputDescriptorSet() const
{
  VkDescriptorSet helperDescSet = m_helpers.getOutputDescriptorSet();
  if(helperDescSet)
    return helperDescSet;
  return m_tonemapperData.isActive ? m_gBuffers.getDescriptorSet(COLOR_LDR) : m_gBuffers.getDescriptorSet(COLOR_MAIN);
}

void GaussianSplatting::benchmarkAdvance()
{
  std::cout << "BENCHMARK_ADV " << m_benchmarkId << " {" << std::endl;

  // Access global memory stats directly
  std::cout << " Memory Scene; Host used \t" << memModels.hostAll << "; Device Used \t" << memModels.deviceUsedAll
            << "; Device Allocated \t" << memModels.deviceAllocAll << "; (bytes)" << std::endl;

  std::cout << " Memory Rasterization; Host used \t" << memRasterization.hostTotal << "; Device Used \t"
            << memRasterization.deviceUsedTotal << "; Device Allocated \t" << memRasterization.deviceAllocTotal
            << "; (bytes)" << std::endl;
  std::cout << " Memory Raytracing; Host used \t" << memRaytracing.hostTotal << "; Device Used \t" << memRaytracing.deviceUsedTotal
            << "; Device Allocated \t" << memRaytracing.deviceAllocTotal << "; (bytes)" << std::endl;
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
  // No-op when the device does not expose the ray tracing extensions; the
  // matching descriptor set / pool / layout stay at VK_NULL_HANDLE and
  // deinitPipelines() handles them via TEST_DESTROY_AND_RESET.
  if(!isSupported.raytracing)
    return;

  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  //////////////////////
  // Bindings

  m_rtDescriptorBindings.clear();

  m_rtDescriptorBindings.addBinding(RTX_BINDING_OUTIMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_AUX1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  m_rtDescriptorBindings.addBinding(RTX_BINDING_OUTDEPTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);

  // TLAS bindings removed - now accessed via device addresses in SceneAssets (bindless)

  // #DLSS - Add binding for DLSS output images array
#if defined(USE_DLSS)
  m_rtDescriptorBindings.addBinding(RTX_BINDING_DLSS_OUT_IMAGES, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
#endif

  // Environment equirectangular texture (sky or HDR IBL)
  m_rtDescriptorBindings.addBinding(RTX_BINDING_SKY_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  // HDR importance sampling acceleration buffer
  m_rtDescriptorBindings.addBinding(RTX_BINDING_ENV_ACCEL, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);

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
                        m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_AUX1, m_rtDescriptorSet),
                        m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTDEPTH, m_rtDescriptorSet),
                        m_gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

  // TLAS - now accessed via device addresses in SceneAssets (bindless, no descriptor writes needed)

  // #DLSS - Bind DLSS output images (only if DLSS is enabled and initialized)
#if defined(USE_DLSS)
  if(m_dlss.isEnabled())
  {
    const auto& dlssGBuffers = m_dlss.getGBuffers();
    // Check if G-buffers are valid (size > 0 means they've been initialized)
    if(dlssGBuffers.getSize().width > 0)
    {
      std::vector<VkDescriptorImageInfo> dlssImageInfos;

      // Bind all DLSS G-buffer images (7 total)
      for(uint32_t i = 0; i < 7; ++i)
      {
        dlssImageInfos.push_back(dlssGBuffers.getDescriptorImageInfo(i));
      }

      writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_OUT_IMAGES, m_rtDescriptorSet),
                            dlssImageInfos.data());
    }
  }
#endif

  // Environment equirectangular texture (sky or HDR IBL)
  writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_SKY_TEXTURE, m_rtDescriptorSet),
                        m_sky.getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_sky.getSampler());

  // HDR importance sampling acceleration buffer (or dummy when no HDR loaded)
  {
    VkBuffer accelBuf = m_sky.hasEnvAccel() ? m_sky.getEnvAccelBuffer().buffer : m_envAccelDummyBuffer.buffer;
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_ENV_ACCEL, m_rtDescriptorSet), accelBuf);
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
                          m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_AUX1, m_rtDescriptorSet),
                          m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_OUTDEPTH, m_rtDescriptorSet),
                          m_gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

    // TLAS - now accessed via device addresses in SceneAssets (bindless, no updates needed)

    // #DLSS - Update DLSS output images (only if DLSS is enabled and initialized)
#if defined(USE_DLSS)
    if(m_dlss.isEnabled())
    {
      const auto& dlssGBuffers = m_dlss.getGBuffers();
      // Check if G-buffers are valid (size > 0 means they've been initialized)
      if(dlssGBuffers.getSize().width > 0)
      {
        std::vector<VkDescriptorImageInfo> dlssImageInfos;

        // Bind all DLSS G-buffer images (7 total)
        for(uint32_t i = 0; i < 7; ++i)
        {
          dlssImageInfos.push_back(dlssGBuffers.getDescriptorImageInfo(i));
        }

        writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_DLSS_OUT_IMAGES, m_rtDescriptorSet),
                              dlssImageInfos.data());
      }
    }
#endif

    // Environment equirectangular texture (sky or HDR IBL)
    writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_SKY_TEXTURE, m_rtDescriptorSet),
                          m_sky.getImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_sky.getSampler());

    // HDR importance sampling acceleration buffer (or dummy when no HDR loaded)
    {
      VkBuffer accelBuf = m_sky.hasEnvAccel() ? m_sky.getEnvAccelBuffer().buffer : m_envAccelDummyBuffer.buffer;
      writeContainer.append(m_rtDescriptorBindings.getWriteSet(RTX_BINDING_ENV_ACCEL, m_rtDescriptorSet), accelBuf);
    }

    // let's update
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }

  // Update rasterization surface reconstruction buffers (in main descriptor set)
  if(m_descriptorSet != VK_NULL_HANDLE)
  {
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_NORMAL, m_descriptorSet),
                          m_gBuffers.getColorImageView(COLOR_RASTER_NORMAL), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_DEPTH, m_descriptorSet),
                          m_gBuffers.getColorImageView(COLOR_RASTER_DEPTH), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_SPLATID, m_descriptorSet),
                          m_gBuffers.getColorImageView(COLOR_RASTER_SPLATID), VK_IMAGE_LAYOUT_GENERAL);
    // Deferred shading: bind both MAIN and AUX buffers, shader selects based on frameSampleId
    writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_COLOR, m_descriptorSet),
                          m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_DEFERRED_OUTPUT, m_descriptorSet),
                          m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_RASTER_COLOR_AUX, m_descriptorSet),
                          m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorBindings.getWriteSet(BINDING_DEFERRED_OUTPUT_AUX, m_descriptorSet),
                          m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void GaussianSplatting::initRtPipeline()
{
  // No-op when the device does not expose the ray tracing extensions; the
  // SPIR-V modules required below were not compiled (see initShaders) and
  // vkCreateRayTracingPipelinesKHR is not available.
  if(!isSupported.raytracing)
    return;

  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  const bool useBillboardRint = prmRtxData.useAABBs && (prmRtx.particleDepth == PARTICLE_DEPTH_BILLBOARD);

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eAnyHit,
    eIntersection,
    eIntersectionBillboard,
    eStageIndicesCount
  };

  // Determine how many stages are actually used
  uint32_t stagesCount = eStageIndicesCount;
  if(!prmRtxData.useAABBs)
    stagesCount -= 2;  // no intersection shaders at all
  else if(!useBillboardRint)
    stagesCount -= 1;  // only standard intersection shader

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
  // Hit Group - Closest Hit (for meshes)
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
  // Hit Group - Intersection Billboard (used only if useAABBs + billboard)
  if(useBillboardRint)
  {
    stage.module                   = m_shaders.rtxRintBillboardShader;
    stage.stage                    = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    stages[eIntersectionBillboard] = stage;
  }

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
    // Hit 0 (SBT offset 0): standard intersection — ellipsoid T
    group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.anyHitShader       = eAnyHit;
    group.intersectionShader = eIntersection;
    m_rtShaderGroups.push_back(group);

    // Hit 1 (SBT offset 1): billboard intersection — billboard T (only when billboard + AABB)
    if(useBillboardRint)
    {
      group.intersectionShader = eIntersectionBillboard;
      m_rtShaderGroups.push_back(group);
    }
  }
  else if(prmRtxData.useSpheres)
  {
    // Hit 0 any hit shader with hardware sphere intersection (no custom rint needed)
    group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.anyHitShader       = eAnyHit;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;
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

  // Mesh hit group: Closest-hit only (for eMeshTlas)
  // SBT offset is 1 normally, or 2 when billboard hit group is present
  group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = eClosestHit;
  m_rtShaderGroups.push_back(group);

  // Update mesh TLAS SBT record offset to account for the billboard hit group
  const uint32_t newMeshSbtOffset = useBillboardRint ? 2 : 1;
  if(newMeshSbtOffset != m_assets.meshes.meshSbtRecordOffset)
  {
    m_assets.meshes.meshSbtRecordOffset = newMeshSbtOffset;
    if(!m_assets.meshes.instances.empty())
      m_assets.meshes.pendingRequests |= MeshManagerVk::Request::eRebuildTLAS;
  }

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

  VkPipelineCreateFlags2CreateInfo flags2Info{VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO};
  if(prmRtxData.useSpheres)
  {
    flags2Info.flags      = VK_PIPELINE_CREATE_2_RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES_BIT_NV;
    rayPipelineInfo.pNext = &flags2Info;
  }

  {
    SCOPED_TIMER("vkCreateRayTracingPipelinesKHR \n");
    VkResult result = vkCreateRayTracingPipelinesKHR(m_device, {}, m_pipelineCache, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);
    if(result != VK_SUCCESS)
    {
      LOGE("vkCreateRayTracingPipelinesKHR failed with VkResult %d\n", result);
      m_rtPipeline = VK_NULL_HANDLE;
      return;
    }
  }


  // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
  if(m_rtProperties.maxRayRecursionDepth <= 1)
  {
    throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 1)");
  }

  // Creating the SBT
  {
    SCOPED_TIMER("Creating the SBT \n");

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
void GaussianSplatting::raytrace(const VkCommandBuffer& cmdBuf, bool meshDepthOnly)
{
  NVVK_DBG_SCOPE(cmdBuf);

  // Early exit if pipeline not initialized (can happen during async loading)
  if(m_rtPipeline == VK_NULL_HANDLE)
    return;

  const std::string name = meshDepthOnly ? "Raytracing prepass" : "Raytracing";

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmdBuf, name);

  // Transform matrices now come from SplatSetDesc in bindless assets
  // Only rendering flags remain in push constants
  m_pcRay.meshDepthOnly = meshDepthOnly;

  // DLSS jitter — shader applies it when compiled with DLSS_ENABLED
#if defined(USE_DLSS)
  if(m_dlss.isEnabled())
  {
    m_currentJitter = shaderio::dlssJitter(prmFrame.frameSampleId);
    m_pcRay.jitter  = m_currentJitter;
  }
  else
  {
    m_pcRay.jitter  = glm::vec2(0.0f);
    m_currentJitter = glm::vec2(0.0f);
  }
#else
  m_pcRay.jitter = glm::vec2(0.0f);
#endif

  std::vector<VkDescriptorSet> descSets{m_descriptorSet, m_rtDescriptorSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);

  // Vertex/index addresses now come from MeshDesc in bindless assets buffer
  // No longer needed in push constants

  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                         | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                     0, sizeof(shaderio::PushConstantRay), &m_pcRay);

  // #DLSS - Use DLSS rendering size when enabled (for both pure RTX and hybrid pipelines)
  VkExtent2D renderingSize = {uint32_t(m_viewSize[0]), uint32_t(m_viewSize[1])};
#if defined(USE_DLSS)
  if(m_dlss.isEnabled() && isDlssSupportedPipeline())
  {
    // Use DLSS render size for all DLSS-enabled pipelines
    // Rasterization also uses this size, so coordinates align
    renderingSize = m_dlss.getRenderSize();
  }
#endif

  vkCmdTraceRaysKHR(cmdBuf, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                    renderingSize.width, renderingSize.height, 1);
}


bool GaussianSplatting::updateFrameCounter()
{
  static float     ref_fov{0};
  static glm::mat4 ref_cam_matrix;

  const glm::mat4 m   = glm::mat4(cameraManip->getViewMatrix());
  const float     fov = float(cameraManip->getFov());

  if(ref_cam_matrix != m || ref_fov != fov)
  {
    resetFrameCounter();
    ref_cam_matrix = m;
    ref_fov        = fov;
  }

  // Reset temporal accumulation when cursor moves in auto-focus mode
  // (focus distance changes with cursor position)
  static glm::ivec2 ref_cursor{-1, -1};
  if(m_assets.cameras.getCamera().dofMode == DOF_AUTO_FOCUS && ref_cursor != prmFrame.cursor)
  {
    resetFrameCounter();
  }
  ref_cursor = prmFrame.cursor;

  // Only increment frame counter if temporal sampling or DLSS is active
  bool needsFrameCounting = prmRtx.temporalSampling;
#if defined(USE_DLSS)
  needsFrameCounting |= m_dlss.isEnabled();
#endif

  if(!needsFrameCounting)
  {
    // Keep frameSampleId at 0 when temporal features are not active
    prmFrame.frameSampleId = 0;
    return true;  // Always continue (no convergence check needed)
  }

  // DLSS always increments without convergence (real-time denoising)
  // Temporal sampling (with or without stochastic splat) converges at frameSampleMax
  bool bypassConvergence = false;
#if defined(USE_DLSS)
  bypassConvergence = m_dlss.isEnabled();
#endif

  // Check convergence: stop after frameSampleMax frames (0 to frameSampleMax-1)
  if(!bypassConvergence && prmFrame.frameSampleId >= prmFrame.frameSampleMax - 1)
  {
    return false;  // Converged
  }

  prmFrame.frameSampleId++;
  return true;  // Continue rendering
}

///////////////////////////////////
// Post processings

void GaussianSplatting::initDescriptorSetPostProcessing()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  // Descriptor Bindings
  m_descriptorBindingsPostProcess.clear();
  m_descriptorBindingsPostProcess.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsPostProcess.addBinding(POST_BINDING_MAIN_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_descriptorBindingsPostProcess.addBinding(POST_BINDING_AUX1_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_descriptorBindingsPostProcess.createDescriptorSetLayout(m_device, 0, &m_descriptorSetLayoutPostProcess));
  NVVK_DBG_NAME(m_descriptorSetLayoutPostProcess);

  // Descriptor Pool
  std::vector<VkDescriptorPoolSize> poolSize;
  m_descriptorBindingsPostProcess.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPoolPostProcess));
  NVVK_DBG_NAME(m_descriptorPoolPostProcess);

  // Descriptor Set
  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPoolPostProcess,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_descriptorSetLayoutPostProcess,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSetPostProcess));
  NVVK_DBG_NAME(m_descriptorSetPostProcess);

  // Pipelne layout
  const VkPushConstantRange pcRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                            | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT,
                                        0, sizeof(shaderio::PushConstant)};

  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayoutPostProcess,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRanges,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayoutPostProcess));
  NVVK_DBG_NAME(m_pipelineLayoutPostProcess);

  // Writes
  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(BINDING_FRAME_INFO_UBO, m_descriptorSetPostProcess),
                        m_frameInfoBuffer);
  writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_MAIN_IMAGE, m_descriptorSetPostProcess),
                        m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_AUX1_IMAGE, m_descriptorSetPostProcess),
                        m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
}

void GaussianSplatting::updateDescriptorSetPostProcessing()
{
  // update only if the descriptor set is already initialized
  if(m_descriptorSetPostProcess != VK_NULL_HANDLE)
  {
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_MAIN_IMAGE, m_descriptorSetPostProcess),
                          m_gBuffers.getColorImageView(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorBindingsPostProcess.getWriteSet(POST_BINDING_AUX1_IMAGE, m_descriptorSetPostProcess),
                          m_gBuffers.getColorImageView(COLOR_AUX1), VK_IMAGE_LAYOUT_GENERAL);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }
}

void GaussianSplatting::initPipelinePostProcessing()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = m_shaders.postComputeShader,
              .pName  = "main",
          },
      .layout = m_pipelineLayoutPostProcess,
  };
  vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_computePipelinePostProcess);
  NVVK_DBG_NAME(m_computePipelinePostProcess);
}

void GaussianSplatting::postProcess(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  if(m_computePipelinePostProcess == VK_NULL_HANDLE)
    return;

  // Always create the profiler section to keep section count stable across frames.
  // Without this, the section appears/disappears as frameSampleId oscillates between
  // 0 and 1 during camera drag (mouse events don't arrive every frame), which
  // continuously triggers the profiler's 8-frame reset delay.
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Post process");

  if(prmFrame.frameSampleId <= 0)
    return;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelinePostProcess);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutPostProcess, 0, 1,
                          &m_descriptorSetPostProcess, 0, nullptr);

  uint32_t wgSize = 32;

  vkCmdDispatch(cmd, (uint32_t(m_viewSize.x) + wgSize - 1) / wgSize, (uint32_t(m_viewSize.y) + wgSize - 1) / wgSize, 1);
}

//--------------------------------------------------------------------------------------------------
// Comparison Mode Implementation
//--------------------------------------------------------------------------------------------------

VkDescriptorSet GaussianSplatting::getPresentationImageDescriptorSet(void)
{
  // Default: use output buffer (helper color if helpers rendered, else COLOR_MAIN)
  VkDescriptorSet displayDescriptor = getOutputDescriptorSet();

  // For non-final visualization modes (depth, normals, clock, etc.) during temporal
  // accumulation in hybrid/raster pipelines, display the current frame (AUX1) instead
  // of the accumulated result (MAIN). Pure RTX writes all modes directly to MAIN (image),
  // but hybrid/raster pipelines use ping-pong: frame 0 → MAIN, frame 1+ → AUX1.
  if(prmRender.visualize != VISUALIZE_FINAL && prmRender.visualize != VISUALIZE_CLAY && prmRtx.temporalSampling
     && prmFrame.frameSampleId > 0 && prmSelectedPipeline != PIPELINE_RTX)
  {
    displayDescriptor = m_gBuffers.getDescriptorSet(COLOR_AUX1);
  }

  // When comparison mode is active, display the comparison output buffer
  if(prmComparison.enabled && m_imageCompare.hasValidCaptureImage())
  {
    displayDescriptor = m_gBuffers.getDescriptorSet(COLOR_COMPARISON_OUTPUT);
  }
#if defined(USE_DLSS)
  // Otherwise, check if a DLSS visualization mode is selected and DLSS is enabled
  else if(m_dlss.isEnabled())
  {
    shaderio::DlssImages dlssBuffer    = shaderio::DlssImages::eDlssInputImage;
    bool                 useDlssBuffer = false;

    switch(prmRender.visualize)
    {
      case VISUALIZE_DLSS_INPUT:
        dlssBuffer    = shaderio::DlssImages::eDlssInputImage;
        useDlssBuffer = true;
        break;
      case VISUALIZE_DLSS_ALBEDO:
        dlssBuffer    = shaderio::DlssImages::eDlssAlbedo;
        useDlssBuffer = true;
        break;
      case VISUALIZE_DLSS_SPECULAR:
        dlssBuffer    = shaderio::DlssImages::eDlssSpecAlbedo;
        useDlssBuffer = true;
        break;
      case VISUALIZE_DLSS_NORMAL:
        dlssBuffer    = shaderio::DlssImages::eDlssNormalRoughness;
        useDlssBuffer = true;
        break;
      case VISUALIZE_DLSS_MOTION:
        dlssBuffer    = shaderio::DlssImages::eDlssMotion;
        useDlssBuffer = true;
        break;
      case VISUALIZE_DLSS_DEPTH:
        dlssBuffer    = shaderio::DlssImages::eDlssDepth;
        useDlssBuffer = true;
        break;
      default:
        if(prmRender.visualize != VISUALIZE_FINAL && prmRender.visualize != VISUALIZE_CLAY)
        {
          dlssBuffer    = shaderio::DlssImages::eDlssInputImage;
          useDlssBuffer = true;
        }
        break;
    }

    if(useDlssBuffer)
    {
      // If helpers were rendered, they already composited on top of the DLSS viz
      if(m_helpers.getOutputColorImage())
      {
        displayDescriptor = m_helpers.getOutputDescriptorSet();
      }
      else
      {
        const auto& dlssGBuffers = m_dlss.getGBuffers();
        displayDescriptor        = dlssGBuffers.getDescriptorSet((uint32_t)dlssBuffer);
      }
    }
  }
#endif

  return displayDescriptor;
}

ImageCompare::ImageInfo GaussianSplatting::getCurrentVisualizationImageInfo() const
{
  ImageCompare::ImageInfo info;

#if defined(USE_DLSS)
  // DLSS visualization modes use separate DLSS buffers
  if(m_dlss.isEnabled() && prmRender.visualize >= VISUALIZE_DLSS_INPUT && prmRender.visualize <= VISUALIZE_DLSS_DEPTH)
  {
    const auto&          dlssGBuffers = m_dlss.getGBuffers();
    shaderio::DlssImages dlssBuffer   = getDlssBufferForVisuMode(prmRender.visualize);
    info.image                        = dlssGBuffers.getColorImage((uint32_t)dlssBuffer);
    info.format                       = dlssGBuffers.getColorFormat((uint32_t)dlssBuffer);
    info.size                         = dlssGBuffers.getSize();
    return info;
  }
#endif

  // All other visualization modes use output buffer (helper color if helpers rendered,
  // else COLOR_LDR when tonemapping is active, else COLOR_MAIN)
  info.image = getOutputColorImage();
  info.size  = m_gBuffers.getSize();

  // Format must match the actual image: helper buffer is always prmRender.colorFormat
  // (even when tonemapping blits LDR into it), COLOR_LDR is R8G8B8A8_UNORM, COLOR_MAIN is prmRender.colorFormat
  if(m_helpers.getOutputColorImage())
    info.format = prmRender.colorFormat;
  else
    info.format = m_tonemapperData.isActive ? VK_FORMAT_R8G8B8A8_UNORM : prmRender.colorFormat;

  return info;
}

std::vector<BufferDumpInfo> GaussianSplatting::getAllDumpableBuffers() const
{
  std::vector<BufferDumpInfo> buffers;
  const VkExtent2D            gBufSize = m_gBuffers.getSize();

  // Main color buffer (always)
  buffers.push_back({m_gBuffers.getColorImage(COLOR_MAIN), prmRender.colorFormat, gBufSize, "_main"});

  // Aux1 temporal intermediate buffer (always)
  buffers.push_back({m_gBuffers.getColorImage(COLOR_AUX1), prmRender.colorFormat, gBufSize, "_aux1"});

  // Comparison output (only when comparison mode is active)
  if(prmComparison.enabled && m_imageCompare.hasValidCaptureImage())
  {
    buffers.push_back({m_gBuffers.getColorImage(COLOR_COMPARISON_OUTPUT), prmRender.colorFormat, gBufSize, "_comparison"});
  }

  // Raster normals (always)
  buffers.push_back({m_gBuffers.getColorImage(COLOR_RASTER_NORMAL), m_normalFormat, gBufSize, "_normal"});

  // Raster depth (always)
  buffers.push_back({m_gBuffers.getColorImage(COLOR_RASTER_DEPTH), m_rasterDepthFormat, gBufSize, "_depth"});

  // LDR tone-mapped output (only when tonemapping is active)
  if(m_tonemapperData.isActive)
  {
    buffers.push_back({m_gBuffers.getColorImage(COLOR_LDR), VK_FORMAT_R8G8B8A8_UNORM, gBufSize, "_ldr"});
  }

#if defined(USE_DLSS)
  if(m_dlss.isEnabled())
  {
    const auto&      dlssGBuffers = m_dlss.getGBuffers();
    const VkExtent2D dlssSize     = dlssGBuffers.getSize();

    buffers.push_back({dlssGBuffers.getColorImage((uint32_t)shaderio::DlssImages::eDlssInputImage),
                       dlssGBuffers.getColorFormat((uint32_t)shaderio::DlssImages::eDlssInputImage), dlssSize, "_dlss_input"});
    buffers.push_back({dlssGBuffers.getColorImage((uint32_t)shaderio::DlssImages::eDlssAlbedo),
                       dlssGBuffers.getColorFormat((uint32_t)shaderio::DlssImages::eDlssAlbedo), dlssSize, "_dlss_albedo"});
    buffers.push_back({dlssGBuffers.getColorImage((uint32_t)shaderio::DlssImages::eDlssSpecAlbedo),
                       dlssGBuffers.getColorFormat((uint32_t)shaderio::DlssImages::eDlssSpecAlbedo), dlssSize, "_dlss_specular"});
    buffers.push_back({dlssGBuffers.getColorImage((uint32_t)shaderio::DlssImages::eDlssNormalRoughness),
                       dlssGBuffers.getColorFormat((uint32_t)shaderio::DlssImages::eDlssNormalRoughness), dlssSize, "_dlss_normal"});
    buffers.push_back({dlssGBuffers.getColorImage((uint32_t)shaderio::DlssImages::eDlssMotion),
                       dlssGBuffers.getColorFormat((uint32_t)shaderio::DlssImages::eDlssMotion), dlssSize, "_dlss_motion"});
    buffers.push_back({dlssGBuffers.getColorImage((uint32_t)shaderio::DlssImages::eDlssDepth),
                       dlssGBuffers.getColorFormat((uint32_t)shaderio::DlssImages::eDlssDepth), dlssSize, "_dlss_depth"});
  }
#endif

  // Helper composited output (only if helpers were rendered this frame)
  if(m_helpers.wasRenderedThisFrame())
  {
    buffers.push_back({m_helpers.getOutputColorImage(), prmRender.colorFormat, gBufSize, "_helpers"});
  }

  return buffers;
}

#if defined(USE_DLSS)
shaderio::DlssImages GaussianSplatting::getDlssBufferForVisuMode(int visualizeMode) const
{
  switch(visualizeMode)
  {
    case VISUALIZE_DLSS_INPUT:
      return shaderio::DlssImages::eDlssInputImage;
    case VISUALIZE_DLSS_ALBEDO:
      return shaderio::DlssImages::eDlssAlbedo;
    case VISUALIZE_DLSS_SPECULAR:
      return shaderio::DlssImages::eDlssSpecAlbedo;
    case VISUALIZE_DLSS_NORMAL:
      return shaderio::DlssImages::eDlssNormalRoughness;
    case VISUALIZE_DLSS_MOTION:
      return shaderio::DlssImages::eDlssMotion;
    case VISUALIZE_DLSS_DEPTH:
      return shaderio::DlssImages::eDlssDepth;
    default:
      return shaderio::DlssImages::eDlssInputImage;
  }
}
#endif

}  // namespace vk_gaussian_splatting
