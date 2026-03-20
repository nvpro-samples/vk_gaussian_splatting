#include "image_compare.h"
#include "image_compare_shaderio.h"

#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvslang/slang.hpp>

#include <algorithm>
#include <limits>
#include <iostream>

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// Constructor / Destructor
//--------------------------------------------------------------------------------------------------
ImageCompare::ImageCompare() {}

ImageCompare::~ImageCompare()
{
  // Resources should be released via deinit()
}

//--------------------------------------------------------------------------------------------------
// Helper: Compile Slang Shader using borrowed compiler
//--------------------------------------------------------------------------------------------------
bool ImageCompare::compileSlangShader(const std::string& filename, VkShaderModule& module)
{
  if(!m_slangCompiler)
  {
    LOGE("ImageCompare: No shader compiler provided\n");
    return false;
  }

  if(!m_slangCompiler->compileFile(filename))
  {
    return false;
  }

  if(module != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_device, module, nullptr);

  // Create the VK module
  VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = m_slangCompiler->getSpirvSize(),
                                      .pCode    = m_slangCompiler->getSpirv()};

  if(m_slangCompiler->getSpirvSize() == 0)
  {
    LOGE("Missing entry point in shader %s\n", filename.c_str());
    return false;
  }
  NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &module));
  NVVK_DBG_NAME(module);

  return true;
}

//--------------------------------------------------------------------------------------------------
// Initialization
//--------------------------------------------------------------------------------------------------
void ImageCompare::init(const Resources& res)
{
  m_device        = res.device;
  m_alloc         = res.allocator;
  m_sampler       = res.sampler;
  m_profiler      = res.profiler;
  m_slangCompiler = res.slangCompiler;
  m_params        = res.parameters;

  // Initialize descriptor sets
  initCompositeDescriptorSet();
  initMetricsDescriptorSet();

  // Compile shaders and initialize pipelines
  rebuildPipelines();
}

void ImageCompare::deinit()
{
  // Destroy pipelines (but not descriptor sets/layouts)
  deinitCompositePipeline();
  deinitMetricsPipeline();

  // Destroy composite descriptor resources
  vkDestroyDescriptorSetLayout(m_device, m_compositeDescSetLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_compositeDescPool, nullptr);
  vkDestroyShaderModule(m_device, m_compositeShader, nullptr);

  // Destroy metrics descriptor resources
  vkDestroyDescriptorSetLayout(m_device, m_metricsDescSetLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_metricsDescPool, nullptr);
  vkDestroyShaderModule(m_device, m_metricsShader, nullptr);

  // Destroy metrics buffers
  m_alloc->destroyBuffer(m_metricsResultDevice);
  m_alloc->destroyBuffer(m_metricsResultHost);

  // Release reference image if captured
  if(m_hasValidCapture)
  {
    releaseCaptureImage();
  }

  // Release composite output
  if(m_compositeOutput.image != VK_NULL_HANDLE)
  {
    m_alloc->destroyImage(m_compositeOutput);
    m_compositeOutput = {};
  }

  // Destroy current image view if created
  if(m_currentImageView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_currentImageView, nullptr);
    m_currentImageView = VK_NULL_HANDLE;
  }

  // Reset all handles to null
  m_compositePipeline       = VK_NULL_HANDLE;
  m_compositePipelineLayout = VK_NULL_HANDLE;
  m_compositeDescSetLayout  = VK_NULL_HANDLE;
  m_compositeDescSet        = VK_NULL_HANDLE;
  m_compositeDescPool       = VK_NULL_HANDLE;
  m_compositeShader         = VK_NULL_HANDLE;

  m_metricsPipeline       = VK_NULL_HANDLE;
  m_metricsPipelineLayout = VK_NULL_HANDLE;
  m_metricsDescSetLayout  = VK_NULL_HANDLE;
  m_metricsDescSet        = VK_NULL_HANDLE;
  m_metricsDescPool       = VK_NULL_HANDLE;
  m_metricsShader         = VK_NULL_HANDLE;

  // Reset resources
  m_device        = VK_NULL_HANDLE;
  m_alloc         = nullptr;
  m_sampler       = VK_NULL_HANDLE;
  m_profiler      = nullptr;
  m_slangCompiler = nullptr;
}

//--------------------------------------------------------------------------------------------------
// Per-Frame Operations
//--------------------------------------------------------------------------------------------------
void ImageCompare::capture(VkCommandBuffer cmd, const ImageInfo& imageInfo)
{
  NVVK_DBG_SCOPE(cmd);

  // Create or recreate reference image if needed
  if(m_captureImage.image == VK_NULL_HANDLE || m_captureImage.format != imageInfo.format
     || m_captureSize.width != imageInfo.size.width || m_captureSize.height != imageInfo.size.height)
  {
    // Free old image if exists
    if(m_captureImage.image != VK_NULL_HANDLE)
    {
      m_alloc->destroyImage(m_captureImage);
    }

    // Create new reference image matching source
    VkImageCreateInfo vkImageInfo{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType   = VK_IMAGE_TYPE_2D,
        .format      = imageInfo.format,
        .extent      = {imageInfo.size.width, imageInfo.size.height, 1},
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = VK_IMAGE_TILING_OPTIMAL,
        .usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkImageViewCreateInfo viewInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = imageInfo.format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };

    NVVK_CHECK(m_alloc->createImage(m_captureImage, vkImageInfo, viewInfo));
    NVVK_DBG_NAME(m_captureImage.image);

    // Store the reference image size and format
    m_captureSize   = imageInfo.size;
    m_captureFormat = imageInfo.format;
  }

  // Transition source image to transfer source layout
  nvvk::cmdImageMemoryBarrier(cmd, {imageInfo.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

  // Transition reference image to transfer destination layout
  nvvk::cmdImageMemoryBarrier(cmd, {m_captureImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

  // Copy source to reference
  VkImageCopy copyRegion{
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffset      = {0, 0, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffset      = {0, 0, 0},
      .extent         = {imageInfo.size.width, imageInfo.size.height, 1},
  };
  vkCmdCopyImage(cmd, imageInfo.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_captureImage.image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

  // Transition images back to general layout
  nvvk::cmdImageMemoryBarrier(cmd, {imageInfo.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  nvvk::cmdImageMemoryBarrier(cmd, {m_captureImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});

  m_hasValidCapture       = true;
  m_descriptorNeedsUpdate = true;

  // Reset metrics history when capturing new reference
  if(m_historySize > 0 && !m_mseHistory.empty())
  {
    std::fill(m_mseHistory.begin(), m_mseHistory.end(), 0.0f);
    std::fill(m_psnrHistory.begin(), m_psnrHistory.end(), 0.0f);
    std::fill(m_flipHistory.begin(), m_flipHistory.end(), 0.0f);
    m_historyIndex       = 0;
    m_historySampleCount = 0;
  }

  LOGD("Capture image stored for comparison\n");
}

void ImageCompare::updateCurrentImageIfNeeded(const ImageInfo& imageInfo)
{
  // Check if image actually changed - if not, just update size/format and return
  if(m_currentImage == imageInfo.image && m_currentImageView != VK_NULL_HANDLE)
  {
    m_currentFormat = imageInfo.format;
    m_currentSize   = imageInfo.size;
    return;  // No need to recreate the view
  }

  m_currentImage  = imageInfo.image;
  m_currentFormat = imageInfo.format;
  m_currentSize   = imageInfo.size;

  // Destroy old image view if it exists
  // Note: We rely on proper frame synchronization elsewhere to ensure the view is not in use
  if(m_currentImageView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_currentImageView, nullptr);
    m_currentImageView = VK_NULL_HANDLE;
  }

  // Create image view for the current image (we don't own the image, just need a view for sampling)
  if(imageInfo.image != VK_NULL_HANDLE)
  {
    VkImageViewCreateInfo viewInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image            = imageInfo.image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = imageInfo.format,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_currentImageView));
    NVVK_DBG_NAME(m_currentImageView);

    m_descriptorNeedsUpdate = true;  // Need to update descriptor set with new view
  }
}

void ImageCompare::render(VkCommandBuffer cmd, const ImageInfo& imageInfo, VkExtent2D outputSize, VkImageView outputImageView, bool skipMetricsUpdate)
{
  if(!m_hasValidCapture || !m_params || !m_params->enabled)
    return;

  // Update the current image view
  updateCurrentImageIfNeeded(imageInfo);

  // Render the composite view
  renderComposite(cmd, outputSize, outputImageView);

  // Compute metrics if requested and not skipped (skip when temporal sampling converged)
  if(m_params->computeMetrics && !skipMetricsUpdate)
  {
    computeMetrics(cmd, outputSize);
    readBackMetricsResult(cmd);
  }
}

//--------------------------------------------------------------------------------------------------
// Resource Management
//--------------------------------------------------------------------------------------------------
void ImageCompare::resize(VkExtent2D newSize)
{
  // If reference was captured and size changed, invalidate it
  if(m_hasValidCapture && (m_captureSize.width != newSize.width || m_captureSize.height != newSize.height))
  {
    releaseCaptureImage();

    // Auto-disable comparison mode when size changes
    if(m_params)
    {
      m_params->enabled = false;
    }
  }

  // Note: Composite output will be recreated on next render if needed
}

void ImageCompare::releaseCaptureImage()
{
  if(m_captureImage.image != VK_NULL_HANDLE)
  {
    // Wait for GPU to finish using the image view that's bound to descriptor sets
    // This is safe here because releaseCaptureImage() is only called from UI/resize code,
    // not during command buffer recording (unlike descriptor updates)
    vkDeviceWaitIdle(m_device);

    m_alloc->destroyImage(m_captureImage);
    m_captureImage = {};
  }

  m_hasValidCapture = false;
  m_captureSize     = {0, 0};
  m_captureFormat   = VK_FORMAT_UNDEFINED;

  // Reset metrics history when releasing capture
  if(m_historySize > 0 && !m_mseHistory.empty())
  {
    std::fill(m_mseHistory.begin(), m_mseHistory.end(), 0.0f);
    std::fill(m_psnrHistory.begin(), m_psnrHistory.end(), 0.0f);
    std::fill(m_flipHistory.begin(), m_flipHistory.end(), 0.0f);
    m_historyIndex       = 0;
    m_historySampleCount = 0;
  }

  LOGD("Capture image released\n");
}

void ImageCompare::rebuildPipelines()
{
  SCOPED_TIMER(std::string(__FUNCTION__) + "\n");

  // Deinit existing pipelines
  deinitCompositePipeline();
  deinitMetricsPipeline();

  // Compile shaders using borrowed compiler
  if(!compileSlangShader("image_compare_composite.comp.slang", m_compositeShader))
  {
    LOGE("ImageCompare: Failed to compile composite shader\n");
  }

  if(!compileSlangShader("image_compare_metric.comp.slang", m_metricsShader))
  {
    LOGE("ImageCompare: Failed to compile metrics shader\n");
  }

  // Reinitialize pipelines with newly compiled shaders
  initCompositePipeline();
  initMetricsPipeline();

  // Request descriptor set update
  m_descriptorNeedsUpdate = true;
}

//--------------------------------------------------------------------------------------------------
// State Queries
//--------------------------------------------------------------------------------------------------
// Image getters are inline in header

//--------------------------------------------------------------------------------------------------
// Internal: Pipeline Initialization
//--------------------------------------------------------------------------------------------------
void ImageCompare::initCompositePipeline()
{
  // Check if shader was compiled successfully
  if(m_compositeShader == VK_NULL_HANDLE)
  {
    LOGE("ImageCompare: Composite shader not available\n");
    return;
  }

  // Create pipeline layout
  using namespace shaderio::imcmp;

  VkPushConstantRange pcRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset     = 0,
      .size       = sizeof(PushConstantComparison),
  };

  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_compositeDescSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_compositePipelineLayout));
  NVVK_DBG_NAME(m_compositePipelineLayout);

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = m_compositeShader,
              .pName  = "main",
          },
      .layout = m_compositePipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_compositePipeline));
  NVVK_DBG_NAME(m_compositePipeline);
}

void ImageCompare::deinitCompositePipeline()
{
  vkDestroyPipeline(m_device, m_compositePipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_compositePipelineLayout, nullptr);
  // NOTE: Do NOT destroy descriptor set layout here - it's needed for rebuildPipelines()
  // Descriptor set layout is destroyed in deinit()

  m_compositePipeline       = VK_NULL_HANDLE;
  m_compositePipelineLayout = VK_NULL_HANDLE;
}

void ImageCompare::initCompositeDescriptorSet()
{
  using namespace shaderio::imcmp;

  // Use UPDATE_AFTER_BIND flag to allow descriptor updates while command buffers are in flight
  VkDescriptorBindingFlags bindingFlags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

  nvvk::DescriptorBindings bindings;
  bindings.addBinding((uint32_t)ComparisonBinding::eCaptureImage, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr, bindingFlags);
  bindings.addBinding((uint32_t)ComparisonBinding::eCurrentImage, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr, bindingFlags);
  bindings.addBinding((uint32_t)ComparisonBinding::eOutputImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr, bindingFlags);
  bindings.addBinding((uint32_t)ComparisonBinding::eSampler, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                      nullptr, bindingFlags);

  // Create descriptor set layout with UPDATE_AFTER_BIND flag
  NVVK_CHECK(bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                                &m_compositeDescSetLayout));
  NVVK_DBG_NAME(m_compositeDescSetLayout);

  // Descriptor Pool (also needs UPDATE_AFTER_BIND flag)
  std::vector<VkDescriptorPoolSize> poolSize;
  bindings.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_compositeDescPool));
  NVVK_DBG_NAME(m_compositeDescPool);

  VkDescriptorSetAllocateInfo allocInfo{
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_compositeDescPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_compositeDescSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_compositeDescSet));
  NVVK_DBG_NAME(m_compositeDescSet);
}

void ImageCompare::updateCompositeDescriptorSet(VkImageView currentImageView, VkImageView outputImageView)
{
  using namespace shaderio::imcmp;

  if(m_compositeDescSet == VK_NULL_HANDLE || !m_hasValidCapture)
    return;

  std::vector<VkDescriptorImageInfo> imageInfos = {
      {VK_NULL_HANDLE, m_captureImage.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL},  // reference
      {VK_NULL_HANDLE, currentImageView, VK_IMAGE_LAYOUT_GENERAL},                     // current
      {VK_NULL_HANDLE, outputImageView, VK_IMAGE_LAYOUT_GENERAL},                      // output
      {m_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},                          // sampler
  };

  std::vector<VkWriteDescriptorSet> writes = {
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_compositeDescSet,
       .dstBinding      = (uint32_t)ComparisonBinding::eCaptureImage,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       .pImageInfo      = &imageInfos[0]},
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_compositeDescSet,
       .dstBinding      = (uint32_t)ComparisonBinding::eCurrentImage,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       .pImageInfo      = &imageInfos[1]},
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_compositeDescSet,
       .dstBinding      = (uint32_t)ComparisonBinding::eOutputImage,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       .pImageInfo      = &imageInfos[2]},
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_compositeDescSet,
       .dstBinding      = (uint32_t)ComparisonBinding::eSampler,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER,
       .pImageInfo      = &imageInfos[3]},
  };

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  m_descriptorNeedsUpdate = false;
}

void ImageCompare::initMetricsPipeline()
{
  // Check if shader was compiled successfully
  if(m_metricsShader == VK_NULL_HANDLE)
  {
    LOGE("ImageCompare: Metrics shader not available\n");
    return;
  }

  // Create pipeline layout
  using namespace shaderio::imcmp;

  VkPushConstantRange pcRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset     = 0,
      .size       = sizeof(PushConstantMetrics),
  };

  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_metricsDescSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pcRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_metricsPipelineLayout));
  NVVK_DBG_NAME(m_metricsPipelineLayout);

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                 .module = m_metricsShader,
                 .pName  = "main"},
      .layout = m_metricsPipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_metricsPipeline));
  NVVK_DBG_NAME(m_metricsPipeline);
}

void ImageCompare::deinitMetricsPipeline()
{
  vkDestroyPipeline(m_device, m_metricsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_metricsPipelineLayout, nullptr);
  // NOTE: Do NOT destroy descriptor set layout, descriptor pool, or buffers here
  // They're needed for rebuildPipelines() and are destroyed in deinit()

  m_metricsPipeline       = VK_NULL_HANDLE;
  m_metricsPipelineLayout = VK_NULL_HANDLE;
}

void ImageCompare::initMetricsDescriptorSet()
{
  using namespace shaderio::imcmp;

  // Use UPDATE_AFTER_BIND flag to allow descriptor updates while command buffers are in flight
  VkDescriptorBindingFlags bindingFlags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

  nvvk::DescriptorBindings bindings;
  bindings.addBinding((uint32_t)MetricsBinding::eCaptureImage, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr, bindingFlags);
  bindings.addBinding((uint32_t)MetricsBinding::eCurrentImage, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr, bindingFlags);
  bindings.addBinding((uint32_t)MetricsBinding::eResultBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr, bindingFlags);
  bindings.addBinding((uint32_t)MetricsBinding::eSampler, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                      nullptr, bindingFlags);

  // Create descriptor set layout with UPDATE_AFTER_BIND flag
  NVVK_CHECK(bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                                &m_metricsDescSetLayout));
  NVVK_DBG_NAME(m_metricsDescSetLayout);

  // Descriptor Pool (also needs UPDATE_AFTER_BIND flag)
  std::vector<VkDescriptorPoolSize> poolSize;
  bindings.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      .maxSets       = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_metricsDescPool));
  NVVK_DBG_NAME(m_metricsDescPool);

  VkDescriptorSetAllocateInfo allocInfo{
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_metricsDescPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_metricsDescSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_metricsDescSet));
  NVVK_DBG_NAME(m_metricsDescSet);

  // Create device and host buffers for metrics results
  // Buffer stores: [0]=MSE sum (uint), [4]=pixel count (uint), [8]=FLIP sum (uint), [12]=reserved (uint)
  m_alloc->createBuffer(m_metricsResultDevice, 16,  // 4 x uint32
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

  m_alloc->createBuffer(m_metricsResultHost, 16,  // 4 x uint32
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

  NVVK_DBG_NAME(m_metricsResultDevice.buffer);
  NVVK_DBG_NAME(m_metricsResultHost.buffer);
}

void ImageCompare::updateMetricsDescriptorSet(VkImageView currentImageView)
{
  if(m_metricsDescSet == VK_NULL_HANDLE || !m_hasValidCapture)
    return;

  VkDescriptorBufferInfo bufferInfo{
      .buffer = m_metricsResultDevice.buffer,
      .offset = 0,
      .range  = VK_WHOLE_SIZE,
  };

  std::vector<VkDescriptorImageInfo> imageInfos = {
      {VK_NULL_HANDLE, m_captureImage.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL},  // reference
      {VK_NULL_HANDLE, currentImageView, VK_IMAGE_LAYOUT_GENERAL},                     // current
      {m_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED},                          // sampler
  };

  using namespace shaderio::imcmp;

  std::vector<VkWriteDescriptorSet> writes = {
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_metricsDescSet,
       .dstBinding      = (uint32_t)MetricsBinding::eCaptureImage,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       .pImageInfo      = &imageInfos[0]},
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_metricsDescSet,
       .dstBinding      = (uint32_t)MetricsBinding::eCurrentImage,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
       .pImageInfo      = &imageInfos[1]},
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_metricsDescSet,
       .dstBinding      = (uint32_t)MetricsBinding::eResultBuffer,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo     = &bufferInfo},
      {.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet          = m_metricsDescSet,
       .dstBinding      = (uint32_t)MetricsBinding::eSampler,
       .descriptorCount = 1,
       .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER,
       .pImageInfo      = &imageInfos[2]},
  };

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Internal: Rendering Operations
//--------------------------------------------------------------------------------------------------
void ImageCompare::renderComposite(VkCommandBuffer cmd, VkExtent2D outputSize, VkImageView outputImageView)
{
  NVVK_DBG_SCOPE(cmd);

  assert(m_profiler);

  auto timerSection = m_profiler->cmdFrameSection(cmd, "Comparison Composite");

  // Check if we have valid current image and output
  if(m_currentImageView == VK_NULL_HANDLE || outputImageView == VK_NULL_HANDLE)
  {
    LOGE("ImageCompare: No current image or output image view set\n");
    return;
  }

  // Update descriptor set if needed
  if(m_descriptorNeedsUpdate)
  {
    updateCompositeDescriptorSet(m_currentImageView, outputImageView);
    m_descriptorNeedsUpdate = false;
  }

  // Check if pipeline is valid
  if(m_compositePipeline == VK_NULL_HANDLE)
  {
    LOGE("Comparison pipeline is NULL!\n");
    return;
  }

  // Bind pipeline and descriptors
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositePipelineLayout, 0, 1, &m_compositeDescSet, 0, nullptr);

  // Push constants
  using namespace shaderio::imcmp;
  PushConstantComparison pushC{
      .splitPosition     = m_params->splitPosition,
      .outputSize        = {int(outputSize.width), int(outputSize.height)},
      .currentImgSize    = {int(m_currentSize.width), int(m_currentSize.height)},
      .captureImgSize    = {int(m_captureSize.width), int(m_captureSize.height)},
      .leftSide          = m_params->leftSide,
      .rightSide         = m_params->rightSide,
      .differenceAmplify = m_params->differenceAmplify,
  };
  vkCmdPushConstants(cmd, m_compositePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantComparison), &pushC);

  // Dispatch compute shader
  const uint32_t wgSize = 16;
  vkCmdDispatch(cmd, (outputSize.width + wgSize - 1) / wgSize, (outputSize.height + wgSize - 1) / wgSize, 1);

  // Memory barrier to ensure compute shader completes before image is displayed
  VkMemoryBarrier barrier{
      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
  };
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier,
                       0, nullptr, 0, nullptr);
}

void ImageCompare::computeMetrics(VkCommandBuffer cmd, VkExtent2D outputSize)
{
  if(!m_hasValidCapture)
  {
    LOGE("MSE: No comparison reference captured\n");
    return;
  }

  if(m_metricsPipeline == VK_NULL_HANDLE)
  {
    LOGE("MSE: Pipeline not initialized\n");
    return;
  }

  if(m_currentImageView == VK_NULL_HANDLE)
  {
    LOGE("MSE: No current image view set\n");
    return;
  }

  NVVK_DBG_SCOPE(cmd);

  assert(m_profiler);

  auto timerSection = m_profiler->cmdFrameSection(cmd, "Comparison Metrics");

  // Update descriptor set with current image view
  updateMetricsDescriptorSet(m_currentImageView);

  // Clear result buffer to zero
  vkCmdFillBuffer(cmd, m_metricsResultDevice.buffer, 0, VK_WHOLE_SIZE, 0);

  // Memory barrier after fill
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);

  // Bind pipeline and descriptors
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_metricsPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_metricsPipelineLayout, 0, 1, &m_metricsDescSet, 0, nullptr);

  // Push constants
  using namespace shaderio::imcmp;

  // Pre-calculate divider for normalization (width * height * 3)
  // This is used in shader to normalize before accumulation, preventing uint32 overflow
  float sampleDivider = float(m_captureSize.width * m_captureSize.height * 3);

  // Pixels Per Degree (PPD) - viewing distance parameter for FLIP
  // Default 67 ppd corresponds to ~0.7m viewing distance on a 24" 1920x1080 monitor
  // Formula: PPD = (resolution_pixels / screen_size_inches) × (viewing_distance_inches / 2π) × 360
  float pixelsPerDegree = 67.0f;  // TODO: Make this configurable if needed

  PushConstantMetrics pushC{
      .captureImgSize  = {int(m_captureSize.width), int(m_captureSize.height)},
      .currentImgSize  = {int(m_currentSize.width), int(m_currentSize.height)},
      .sampleDivider   = sampleDivider,
      .pixelsPerDegree = pixelsPerDegree,
      .flipMode        = m_params->flipMode,
  };
  vkCmdPushConstants(cmd, m_metricsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantMetrics), &pushC);

  // Dispatch compute shader
  const uint32_t wgSize = 16;
  vkCmdDispatch(cmd, (m_captureSize.width + wgSize - 1) / wgSize, (m_captureSize.height + wgSize - 1) / wgSize, 1);

  // Memory barrier to ensure compute completes before transfer
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
}

void ImageCompare::readBackMetricsResult(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  assert(m_profiler);

  auto timerSection = m_profiler->cmdFrameSection(cmd, "Comparison Metrics Readback");

  // Copy from device to host buffer
  VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = 16};  // 4 x uint32
  vkCmdCopyBuffer(cmd, m_metricsResultDevice.buffer, m_metricsResultHost.buffer, 1, &bc);

  m_canCollectResult = true;
}

//--------------------------------------------------------------------------------------------------
// Configure metrics history buffer size
//--------------------------------------------------------------------------------------------------
void ImageCompare::setMetricsHistorySize(int size)
{
  // Clamp to minimum 1
  size = std::max(1, size);

  if(size == m_historySize)
    return;  // No change needed

  m_historySize = size;

  // Resize all three history vectors
  m_mseHistory.resize(size, 0.0f);
  m_psnrHistory.resize(size, 0.0f);
  m_flipHistory.resize(size, 0.0f);

  // Reset index and sample count to start from beginning
  m_historyIndex       = 0;
  m_historySampleCount = 0;
}

void ImageCompare::collectMetricsResult()
{
  if(!m_canCollectResult)
    return;

  if(m_metricsResultHost.buffer == VK_NULL_HANDLE)
  {
    LOGE("MSE: Host buffer not initialized\n");
    m_canCollectResult = false;
    return;
  }

  if(!m_metricsResultHost.mapping)
  {
    LOGE("MSE: Host buffer not mapped\n");
    m_canCollectResult = false;
    return;
  }

  // Read the results from mapped memory
  uint32_t* resultData = (uint32_t*)m_metricsResultHost.mapping;
  uint32_t  mseFixed   = resultData[0];  // MSE value (fixed-point, already normalized in shader)
  // resultData[1] is reserved (pixel counting removed for optimization)
  uint32_t flipFixed = resultData[2];  // FLIP value (fixed-point, already normalized in shader)

  //==========================================================================
  // MSE and PSNR Computation
  //==========================================================================

  // Convert back from fixed-point (shader uses 1e9 multiplier)
  // Values are already normalized by dividing by (width * height * 3) in shader
  m_mseValue = (float)mseFixed / 1000000000.0f;

  // Calculate PSNR: 10 * log10(MAX^2 / MSE)
  // For normalized color values, MAX = 1.0
  // Clamp to maximum 99.99 dB to avoid infinity for near-perfect matches
  const float MSE_THRESHOLD = 1e-10f;  // Threshold for treating as zero
  const float PSNR_MAX      = 99.99f;  // Maximum PSNR value in dB

  if(m_mseValue < MSE_THRESHOLD)
  {
    m_psnrValue = PSNR_MAX;  // Clamp to max instead of infinity
  }
  else
  {
    m_psnrValue = std::min(10.0f * std::log10(1.0f / m_mseValue), PSNR_MAX);
  }

  //==========================================================================
  // FLIP Computation (Reference Implementation with Minkowski Pooling)
  //==========================================================================

  // Convert back from fixed-point (shader uses 1e9 multiplier)
  double flipSumPowered = (double)flipFixed / 1000000000.0;

  // Apply Minkowski pooling: FLIP = (Σ error^q / N)^(1/q)
  // The shader accumulated powered errors (q=3), now we apply the q-root
  const double q = 3.0;
  m_flipValue    = (float)std::pow(flipSumPowered, 1.0 / q);

  m_hasMetricsResult = true;

  // Store in history buffer (circular buffer)
  if(m_historySize > 0 && !m_mseHistory.empty())
  {
    m_mseHistory[m_historyIndex]  = m_mseValue;
    m_psnrHistory[m_historyIndex] = m_psnrValue;
    m_flipHistory[m_historyIndex] = m_flipValue;
    m_historyIndex                = (m_historyIndex + 1) % m_historySize;

    // Track number of samples written (capped at buffer size)
    if(m_historySampleCount < m_historySize)
      m_historySampleCount++;
  }

  m_canCollectResult = false;
}

}  // namespace vk_gaussian_splatting
