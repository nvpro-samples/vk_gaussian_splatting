#include <vk_radix_sort.h>

#include <utility>

#include "generated/upsweep_comp.h"
#include "generated/spine_comp.h"
#include "generated/downsweep_comp.h"
#include "generated/downsweep_key_value_comp.h"

namespace {

constexpr uint32_t RADIX = 256;
constexpr int WORKGROUP_SIZE = 512;
constexpr int PARTITION_DIVISION = 8;
constexpr int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

VkDeviceSize HistogramSize(uint32_t elementCount) {
  return (1 + 4 * RADIX + RoundUp(elementCount, PARTITION_SIZE) * RADIX) *
         sizeof(uint32_t);
}

VkDeviceSize InoutSize(uint32_t elementCount) {
  return elementCount * sizeof(uint32_t);
}

void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
             uint32_t elementCount, VkBuffer indirectBuffer,
             VkDeviceSize indirectOffset, VkBuffer buffer, VkDeviceSize offset,
             VkBuffer valueBuffer, VkDeviceSize valueOffset,
             VkBuffer storageBuffer, VkDeviceSize storageOffset,
             VkQueryPool queryPool, uint32_t query);

}  // namespace

struct VrdxSorter_T {
  VkDevice device = VK_NULL_HANDLE;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  VkPipeline upsweepPipeline = VK_NULL_HANDLE;
  VkPipeline spinePipeline = VK_NULL_HANDLE;
  VkPipeline downsweepPipeline = VK_NULL_HANDLE;
  VkPipeline downsweepKeyValuePipeline = VK_NULL_HANDLE;

  uint32_t maxWorkgroupSize = 0;
};

struct PushConstants {
  uint32_t pass;
  VkDeviceAddress elementCountReference;
  VkDeviceAddress globalHistogramReference;
  VkDeviceAddress partitionHistogramReference;
  VkDeviceAddress keysInReference;
  VkDeviceAddress keysOutReference;
  VkDeviceAddress valuesInReference;
  VkDeviceAddress valuesOutReference;
};

void vrdxCreateSorter(const VrdxSorterCreateInfo* pCreateInfo,
                      VrdxSorter* pSorter) {
  VkDevice device = pCreateInfo->device;
  VkPipelineCache pipelineCache = pCreateInfo->pipelineCache;

  // shader specialization constants and defaults
  VkPhysicalDeviceVulkan11Properties physicalDeviceVulkan11Properties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceProperties2 physicalDeviceProperties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  physicalDeviceProperties.pNext = &physicalDeviceVulkan11Properties;

  vkGetPhysicalDeviceProperties2(pCreateInfo->physicalDevice,
                                 &physicalDeviceProperties);

  // TODO: max workgroup size
  uint32_t maxWorkgroupSize =
      physicalDeviceProperties.properties.limits.maxComputeWorkGroupSize[0];
  uint32_t subgroupSize = physicalDeviceVulkan11Properties.subgroupSize;

  // pipeline layout
  VkPushConstantRange pushConstants = {};
  pushConstants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstants.offset = 0;
  pushConstants.size = sizeof(PushConstants);

  VkPipelineLayout pipelineLayout;
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstants;
  vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout);

  // pipelines
  VkPipeline upsweepPipeline;
  {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(upsweep_comp);
    shaderModuleInfo.pCode = upsweep_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL,
                             &upsweepPipeline);

    vkDestroyShaderModule(device, shaderModule, NULL);
  }

  VkPipeline spinePipeline;
  {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(spine_comp);
    shaderModuleInfo.pCode = spine_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL,
                             &spinePipeline);

    vkDestroyShaderModule(device, shaderModule, NULL);
  }

  VkPipeline downsweepPipeline;
  VkPipeline downsweepKeyValuePipeline;
  {
    VkShaderModule shaderModules[2];
    VkShaderModuleCreateInfo shaderModuleInfo = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(downsweep_comp);
    shaderModuleInfo.pCode = downsweep_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[0]);

    shaderModuleInfo.codeSize = sizeof(downsweep_key_value_comp);
    shaderModuleInfo.pCode = downsweep_key_value_comp;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[1]);

    VkComputePipelineCreateInfo pipelineInfos[2];
    pipelineInfos[0] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[0].stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfos[0].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[0].stage.module = shaderModules[0];
    pipelineInfos[0].stage.pName = "main";
    pipelineInfos[0].layout = pipelineLayout;

    pipelineInfos[1] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[1].stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfos[1].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[1].stage.module = shaderModules[1];
    pipelineInfos[1].stage.pName = "main";
    pipelineInfos[1].layout = pipelineLayout;

    VkPipeline pipelines[2];
    vkCreateComputePipelines(device, pipelineCache, 2, pipelineInfos, NULL,
                             pipelines);
    downsweepPipeline = pipelines[0];
    downsweepKeyValuePipeline = pipelines[1];

    for (auto shaderModule : shaderModules)
      vkDestroyShaderModule(device, shaderModule, NULL);
  }

  *pSorter = new VrdxSorter_T();
  (*pSorter)->device = device;
  (*pSorter)->pipelineLayout = pipelineLayout;

  (*pSorter)->upsweepPipeline = upsweepPipeline;
  (*pSorter)->spinePipeline = spinePipeline;
  (*pSorter)->downsweepPipeline = downsweepPipeline;
  (*pSorter)->downsweepKeyValuePipeline = downsweepKeyValuePipeline;

  (*pSorter)->maxWorkgroupSize = maxWorkgroupSize;
}

void vrdxDestroySorter(VrdxSorter sorter) {
  vkDestroyPipeline(sorter->device, sorter->upsweepPipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->spinePipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->downsweepPipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->downsweepKeyValuePipeline, NULL);

  vkDestroyPipelineLayout(sorter->device, sorter->pipelineLayout, NULL);
  delete sorter;
}

void vrdxGetSorterStorageRequirements(
    VrdxSorter sorter, uint32_t maxElementCount,
    VrdxSorterStorageRequirements* requirements) {
  VkDevice device = sorter->device;

  VkDeviceSize elementCountSize = sizeof(uint32_t);
  VkDeviceSize histogramSize = HistogramSize(maxElementCount);
  VkDeviceSize inoutSize = InoutSize(maxElementCount);

  VkDeviceSize histogramOffset = elementCountSize;
  VkDeviceSize inoutOffset = histogramOffset + histogramSize;
  VkDeviceSize storageSize = inoutOffset + inoutSize;

  requirements->size = storageSize;
  requirements->usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
}

void vrdxGetSorterKeyValueStorageRequirements(
    VrdxSorter sorter, uint32_t maxElementCount,
    VrdxSorterStorageRequirements* requirements) {
  VkDevice device = sorter->device;

  VkDeviceSize elementCountSize = sizeof(uint32_t);
  VkDeviceSize histogramSize = HistogramSize(maxElementCount);
  VkDeviceSize inoutSize = InoutSize(maxElementCount);

  VkDeviceSize histogramOffset = elementCountSize;
  VkDeviceSize inoutOffset = histogramOffset + histogramSize;
  // 2x for key value
  VkDeviceSize storageSize = inoutOffset + 2 * inoutSize;

  requirements->size = storageSize;
  requirements->usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
}

void vrdxCmdSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                 uint32_t elementCount, VkBuffer keysBuffer,
                 VkDeviceSize keysOffset, VkBuffer storageBuffer,
                 VkDeviceSize storageOffset, VkQueryPool queryPool,
                 uint32_t query) {
  gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset,
          NULL, 0, storageBuffer, storageOffset, queryPool, query);
}

void vrdxCmdSortIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         uint32_t maxElementCount, VkBuffer indirectBuffer,
                         VkDeviceSize indirectOffset, VkBuffer keysBuffer,
                         VkDeviceSize keysOffset, VkBuffer storageBuffer,
                         VkDeviceSize storageOffset, VkQueryPool queryPool,
                         uint32_t query) {
  gpuSort(commandBuffer, sorter, maxElementCount, indirectBuffer,
          indirectOffset, keysBuffer, keysOffset, NULL, 0, storageBuffer,
          storageOffset, queryPool, query);
}

void vrdxCmdSortKeyValue(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         uint32_t elementCount, VkBuffer keysBuffer,
                         VkDeviceSize keysOffset, VkBuffer valuesBuffer,
                         VkDeviceSize valuesOffset, VkBuffer storageBuffer,
                         VkDeviceSize storageOffset, VkQueryPool queryPool,
                         uint32_t query) {
  gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset,
          valuesBuffer, valuesOffset, storageBuffer, storageOffset, queryPool,
          query);
}

void vrdxCmdSortKeyValueIndirect(
    VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t maxElementCount,
    VkBuffer indirectBuffer, VkDeviceSize indirectOffset, VkBuffer keysBuffer,
    VkDeviceSize keysOffset, VkBuffer valuesBuffer, VkDeviceSize valuesOffset,
    VkBuffer storageBuffer, VkDeviceSize storageOffset, VkQueryPool queryPool,
    uint32_t query) {
  gpuSort(commandBuffer, sorter, maxElementCount, indirectBuffer,
          indirectOffset, keysBuffer, keysOffset, valuesBuffer, valuesOffset,
          storageBuffer, storageOffset, queryPool, query);
}

namespace {

void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
             uint32_t elementCount, VkBuffer indirectBuffer,
             VkDeviceSize indirectOffset, VkBuffer keysBuffer,
             VkDeviceSize keysOffset, VkBuffer valuesBuffer,
             VkDeviceSize valuesOffset, VkBuffer storageBuffer,
             VkDeviceSize storageOffset, VkQueryPool queryPool,
             uint32_t query) {
  VkDevice device = sorter->device;
  uint32_t partitionCount =
      RoundUp(indirectBuffer ? elementCount : elementCount, PARTITION_SIZE);

  VkDeviceSize elementCountSize = sizeof(uint32_t);
  VkDeviceSize histogramSize = HistogramSize(elementCount);
  VkDeviceSize inoutSize = InoutSize(elementCount);

  VkDeviceSize elementCountOffset = storageOffset;
  VkDeviceSize histogramOffset = elementCountOffset + elementCountSize;
  VkDeviceSize inoutOffset = histogramOffset + histogramSize;

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        queryPool, query + 0);
  }

  if (indirectBuffer) {
    // copy elementCount
    VkBufferCopy region;
    region.srcOffset = indirectOffset;
    region.dstOffset = elementCountOffset;
    region.size = sizeof(uint32_t);
    vkCmdCopyBuffer(commandBuffer, indirectBuffer, storageBuffer, 1, &region);
  } else {
    // set element count
    vkCmdUpdateBuffer(commandBuffer, storageBuffer, elementCountOffset,
                      sizeof(elementCount), &elementCount);
  }

  // reset global histogram. partition histogram is set by shader.
  vkCmdFillBuffer(commandBuffer, storageBuffer, histogramOffset,
                  4 * RADIX * sizeof(uint32_t), 0);

  VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memoryBarrier, 0, NULL, 0, NULL);

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        queryPool, query + 1);
  }

  VkBufferDeviceAddressInfo deviceAddressInfo = {
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  deviceAddressInfo.buffer = storageBuffer;
  VkDeviceAddress storageAddress =
      vkGetBufferDeviceAddress(device, &deviceAddressInfo);

  deviceAddressInfo.buffer = keysBuffer;
  VkDeviceAddress keysAddress =
      vkGetBufferDeviceAddress(device, &deviceAddressInfo);

  VkDeviceAddress valuesAddress = 0;
  if (valuesBuffer) {
    deviceAddressInfo.buffer = valuesBuffer;
    valuesAddress = vkGetBufferDeviceAddress(device, &deviceAddressInfo);
  }

  PushConstants pushConstants;
  pushConstants.elementCountReference = storageAddress + elementCountOffset;
  pushConstants.globalHistogramReference = storageAddress + histogramOffset;
  pushConstants.partitionHistogramReference =
      storageAddress + histogramOffset + sizeof(uint32_t) * 4 * RADIX;

  for (int i = 0; i < 4; ++i) {
    pushConstants.pass = i;
    pushConstants.keysInReference = keysAddress + keysOffset;
    pushConstants.keysOutReference = storageAddress + inoutOffset;
    pushConstants.valuesInReference = valuesAddress + valuesOffset;
    pushConstants.valuesOutReference = storageAddress + inoutOffset + inoutSize;

    if (i % 2 == 1) {
      std::swap(pushConstants.keysInReference, pushConstants.keysOutReference);
      std::swap(pushConstants.valuesInReference,
                pushConstants.valuesOutReference);
    }

    vkCmdPushConstants(commandBuffer, sorter->pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants),
                       &pushConstants);

    // upsweep
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      sorter->upsweepPipeline);

    vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queryPool, query + 2 + 3 * i + 0);
    }

    // spine
    memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memoryBarrier, 0, NULL, 0, NULL);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      sorter->spinePipeline);

    vkCmdDispatch(commandBuffer, RADIX, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queryPool, query + 2 + 3 * i + 1);
    }

    // downsweep
    memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memoryBarrier, 0, NULL, 0, NULL);

    if (valuesBuffer) {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sorter->downsweepKeyValuePipeline);
    } else {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sorter->downsweepPipeline);
    }

    vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          queryPool, query + 2 + 3 * i + 2);
    }

    if (i < 3) {
      memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &memoryBarrier, 0, NULL, 0, NULL);
    }
  }

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        queryPool, query + 14);
  }
}

}  // namespace
