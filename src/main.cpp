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

#include <gaussian_splatting_ui.h>

using namespace vk_gaussian_splatting;

// create, setup and run an nvapp::Application
// with a GaussianSplatting element.
int main(int argc, char** argv)
{
  nvutils::Logger::getInstance().breakOnError(false);
  //nvutils::Logger::getInstance().setLogLevel(nvutils::Logger::LogLevel::eDEBUG);

  nvutils::ProfilerManager              profilerManager;
  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              parameterParser(nvutils::getExecutablePath().stem().string(), {".txt"});
  nvutils::ParameterSequencer::InitInfo sequencerInfo{// sequencer always requires a parser and registry
                                                      .parameterParser   = &parameterParser,
                                                      .parameterRegistry = &parameterRegistry,
                                                      // sequencer uses the profiler for benchmarking
                                                      .profilerManager = &profilerManager};

  nvvk::Context                vkContext;  // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;    // Information to create the Vulkan context
  nvapp::Application           application;
  nvapp::ApplicationCreateInfo appInfo;  // Information to create the application
  bool                         benchmarkMode = false;

  /////////////////////////////////
  // Parse the command line to get the application creation information
  // those parameter will have no effect if changed via benchmark script
  // see GaussianSplatting constructor for other options
  parameterRegistry.addVector({"size", "Size of the window to be created"}, &appInfo.windowSize);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"verbose", "Verbose output of the Vulkan context"}, &vkSetup.verbose);
  parameterRegistry.add({"validation", "Enable validation layers"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"benchmark", "Enable benchmarking, prevents async loadings and turns off vsync"}, &benchmarkMode);
  parameterRegistry.add({"forcegpu", "Force the use of a specific GPU by probviding its ID"}, &vkSetup.forceGPU);

  registerCommandLineParameters(&parameterRegistry);

  /////////////////////////////////
  // Create elements of the application, including the core of the sample (gaussianSplatting)

  // The GaussianSplattingUI includes the core GaussianSplatting class by inheritance
  auto gaussianSplatting = std::make_shared<GaussianSplattingUI>(&profilerManager, &parameterRegistry, &benchmarkMode);

  // add a few more parameters to registry and parser to handle sequencer settings
  sequencerInfo.registerScriptParameters(parameterRegistry, parameterParser);

  // extends reporting output with memory consumption information
  sequencerInfo.postCallbacks.emplace_back(
      [&](const nvutils::ParameterSequencer::State& /* unused */) { gaussianSplatting->benchmarkAdvance(); });

  // After the creation of the elements we have more parameters in the registry than before (from gaussianSplatting).
  // Therefore add the entire registry to the commandline parser again, to add new ones.
  parameterParser.add(parameterRegistry);
  // commandline parsing
  parameterParser.parse(argc, argv);
  // backup the default applications parameters, including those modified by command line
  storeDefaultParameters();
  // set more verbose for benchmark usage later on
  parameterParser.setVerbose(true);

  // this element requires sequencerInfo that is potentially updated by parameterParser
  auto elemSequencer = std::make_shared<nvapp::ElementSequencer>(sequencerInfo);

  /////////////////////////////////
  // Vulkan creation context information
  vkSetup.enableAllFeatures = true;

  // - Instance extensions
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // - Device extensions
  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  static VkPhysicalDeviceMeshShaderFeaturesEXT meshFeaturesEXT = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
  };

  static VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
  };
  vkSetup.deviceExtensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);  // for vk_radix_sort (vrdx)
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshFeaturesEXT, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &fragFeaturesKHR, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeaturesKHR, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);  // for ImGui

  // Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature, true);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature, false);  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

  VkPhysicalDeviceShaderClockFeaturesKHR clockFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeatures);

  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures = {
      .sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
      .rayTracingInvocationReorder = VK_TRUE,
  };
  vkSetup.deviceExtensions.emplace_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &serFeatures, false);

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Setting up the validation layers
  nvvk::ValidationSettings vvlInfo{};
  // vvlInfo.validate_best_practices = true;
  vvlInfo.validate_core = false;
  //vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eSynchronization);
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();  // Adding the validation layer settings

  // Create Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  /////////////////////////////////
  // Application setup
  appInfo.name                  = TARGET_NAME;
  appInfo.instance              = vkContext.getInstance();
  appInfo.device                = vkContext.getDevice();
  appInfo.physicalDevice        = vkContext.getPhysicalDevice();
  appInfo.queues                = vkContext.getQueueInfos();
  appInfo.hasUndockableViewport = true;
  appInfo.useMenu               = !benchmarkMode;  // we hide the menu in benchmark mode

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    // right side panel container
    ImGuiID assetsID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.20F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Assets", assetsID);
    ImGuiID propertiesID = ImGui::DockBuilderSplitNode(assetsID, ImGuiDir_Down, 0.75F, nullptr, &assetsID);
    ImGui::DockBuilderDockWindow("Properties", propertiesID);

    // bottom panel container
    ImGuiID memoryID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.45F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Memory Statistics", memoryID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(memoryID, ImGuiDir_Right, 0.33F, nullptr, &memoryID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID renderingID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Down, 0.30F, nullptr, &profilerID);
    ImGui::DockBuilderDockWindow("Rendering Statistics", renderingID);
  };

  //
  gaussianSplatting->guiRegisterIniFileHandlers();

  // Initializes the application
  application.init(appInfo);

  // Add all application elements including our sample specific gaussianSplatting
  // onAttach will be invoked on elements at this stage
  application.addElement(elemSequencer);
  application.addElement(gaussianSplatting);
  application.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", "GLSL")));

  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(gaussianSplatting->cameraManip);
  application.addElement(elemCamera);

  if(benchmarkMode)
  {
    // In this mode we do not display the GUI elements
    application.setVsync(false);
  }
  else
  {
    application.addElement(std::make_shared<nvgpu_monitor::ElementGpuMonitor>());

    // setup the profiler element and view
    auto profilerViewSettings = std::make_shared<nvapp::ElementProfiler::ViewSettings>(
        nvapp::ElementProfiler::ViewSettings{.name       = "Profiler",
                                             .defaultTab = nvapp::ElementProfiler::TABLE,
                                             .pieChart   = {.cpuTotal = false, .levels = true},
                                             .lineChart  = {.cpuLine = false}});

    // setting are optional, but can be used to expose to sample code (like hiding views for benchmark)
    application.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager, profilerViewSettings));
  }

  //
  application.run();

  // Cleanup
  application.deinit();
  vkContext.deinit();

  return 0;
}
