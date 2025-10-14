#include <vk_radix_sort.h>

#include <utility>

#include "generated/upsweep_slang.h"
#include "generated/spine_slang.h"
#include "generated/downsweep_slang.h"
#include "generated/downsweep_key_value_slang.h"

namespace {

constexpr uint32_t RADIX = 256;
constexpr int WORKGROUP_SIZE = 512;
constexpr int PARTITION_DIVISION = 8;
constexpr int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
uint32_t Align(uint32_t a, uint32_t b) { return (a + b - 1) / b * b; }

VkDeviceSize HistogramSize(uint32_t elementCount) {
  return Align((4 + 4 * RADIX + RoundUp(elementCount, PARTITION_SIZE) * RADIX) * sizeof(uint32_t),
               16);
}

VkDeviceSize InoutSize(uint32_t elementCount) { return Align(elementCount * sizeof(uint32_t), 16); }

void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t elementCount,
             VkBuffer indirectBuffer, VkDeviceSize indirectOffset, VkBuffer buffer,
             VkDeviceSize offset, VkBuffer valueBuffer, VkDeviceSize valueOffset,
             VkBuffer storageBuffer, VkDeviceSize storageOffset, VkQueryPool queryPool,
             uint32_t query);

}  // namespace

struct VrdxSorter_T {
  VkDevice device = VK_NULL_HANDLE;
  PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  VkPipeline upsweepPipeline = VK_NULL_HANDLE;
  VkPipeline spinePipeline = VK_NULL_HANDLE;
  VkPipeline downsweepPipeline = VK_NULL_HANDLE;
  VkPipeline downsweepKeyValuePipeline = VK_NULL_HANDLE;
};

struct PushConstants {
  uint32_t pass;
};

void vrdxCreateSorter(const VrdxSorterCreateInfo* pCreateInfo, VrdxSorter* pSorter) {
  VkDevice device = pCreateInfo->device;
  VkPipelineCache pipelineCache = pCreateInfo->pipelineCache;

  // device extensions
  PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR =
      (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");

  // descriptor layout
  constexpr int bindingCount = 7;
  VkDescriptorSetLayoutBinding bindings[bindingCount];
  for (int i = 0; i < bindingCount; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayout descriptorSetLayout;
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  descriptorSetLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
  descriptorSetLayoutInfo.bindingCount = bindingCount;
  descriptorSetLayoutInfo.pBindings = bindings;
  vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, NULL, &descriptorSetLayout);

  // pipeline layout
  VkPushConstantRange pushConstants = {};
  pushConstants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstants.offset = 0;
  pushConstants.size = sizeof(PushConstants);

  VkPipelineLayout pipelineLayout;
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstants;
  vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout);

  // pipelines
  VkPipeline upsweepPipeline;
  {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(upsweep_slang);
    shaderModuleInfo.pCode = upsweep_slang;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL, &upsweepPipeline);

    vkDestroyShaderModule(device, shaderModule, NULL);
  }

  VkPipeline spinePipeline;
  {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(spine_slang);
    shaderModuleInfo.pCode = spine_slang;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModule);

    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, NULL, &spinePipeline);

    vkDestroyShaderModule(device, shaderModule, NULL);
  }

  VkPipeline downsweepPipeline;
  VkPipeline downsweepKeyValuePipeline;
  {
    VkShaderModule shaderModules[2];
    VkShaderModuleCreateInfo shaderModuleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderModuleInfo.codeSize = sizeof(downsweep_slang);
    shaderModuleInfo.pCode = downsweep_slang;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[0]);

    shaderModuleInfo.codeSize = sizeof(downsweep_key_value_slang);
    shaderModuleInfo.pCode = downsweep_key_value_slang;
    vkCreateShaderModule(device, &shaderModuleInfo, NULL, &shaderModules[1]);

    VkComputePipelineCreateInfo pipelineInfos[2];
    pipelineInfos[0] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[0].stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfos[0].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[0].stage.module = shaderModules[0];
    pipelineInfos[0].stage.pName = "main";
    pipelineInfos[0].layout = pipelineLayout;

    pipelineInfos[1] = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfos[1].stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfos[1].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfos[1].stage.module = shaderModules[1];
    pipelineInfos[1].stage.pName = "main";
    pipelineInfos[1].layout = pipelineLayout;

    VkPipeline pipelines[2];
    vkCreateComputePipelines(device, pipelineCache, 2, pipelineInfos, NULL, pipelines);
    downsweepPipeline = pipelines[0];
    downsweepKeyValuePipeline = pipelines[1];

    for (auto shaderModule : shaderModules) vkDestroyShaderModule(device, shaderModule, NULL);
  }

  *pSorter = new VrdxSorter_T();
  (*pSorter)->device = device;
  (*pSorter)->vkCmdPushDescriptorSetKHR = vkCmdPushDescriptorSetKHR;

  (*pSorter)->descriptorSetLayout = descriptorSetLayout;
  (*pSorter)->pipelineLayout = pipelineLayout;

  (*pSorter)->upsweepPipeline = upsweepPipeline;
  (*pSorter)->spinePipeline = spinePipeline;
  (*pSorter)->downsweepPipeline = downsweepPipeline;
  (*pSorter)->downsweepKeyValuePipeline = downsweepKeyValuePipeline;
}

void vrdxDestroySorter(VrdxSorter sorter) {
  vkDestroyPipeline(sorter->device, sorter->upsweepPipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->spinePipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->downsweepPipeline, NULL);
  vkDestroyPipeline(sorter->device, sorter->downsweepKeyValuePipeline, NULL);

  vkDestroyPipelineLayout(sorter->device, sorter->pipelineLayout, NULL);
  vkDestroyDescriptorSetLayout(sorter->device, sorter->descriptorSetLayout, NULL);
  delete sorter;
}

void vrdxGetSorterStorageRequirements(VrdxSorter sorter, uint32_t maxElementCount,
                                      VrdxSorterStorageRequirements* requirements) {
  VkDevice device = sorter->device;

  VkDeviceSize elementCountSize = Align(sizeof(uint32_t), 16);
  VkDeviceSize histogramSize = HistogramSize(maxElementCount);
  VkDeviceSize inoutSize = InoutSize(maxElementCount);

  VkDeviceSize histogramOffset = elementCountSize;
  VkDeviceSize inoutOffset = histogramOffset + histogramSize;
  VkDeviceSize storageSize = inoutOffset + inoutSize;

  requirements->size = storageSize;
  requirements->usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
}

void vrdxGetSorterKeyValueStorageRequirements(VrdxSorter sorter, uint32_t maxElementCount,
                                              VrdxSorterStorageRequirements* requirements) {
  VkDevice device = sorter->device;

  VkDeviceSize elementCountSize = Align(sizeof(uint32_t), 16);
  VkDeviceSize histogramSize = HistogramSize(maxElementCount);
  VkDeviceSize inoutSize = InoutSize(maxElementCount);

  VkDeviceSize histogramOffset = elementCountSize;
  VkDeviceSize inoutOffset = histogramOffset + histogramSize;
  // 2x for key value
  VkDeviceSize storageSize = inoutOffset + 2 * inoutSize;

  requirements->size = storageSize;
  requirements->usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
}

void vrdxCmdSort(VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t elementCount,
                 VkBuffer keysBuffer, VkDeviceSize keysOffset, VkBuffer storageBuffer,
                 VkDeviceSize storageOffset, VkQueryPool queryPool, uint32_t query) {
  gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset, NULL, 0,
          storageBuffer, storageOffset, queryPool, query);
}

void vrdxCmdSortIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t maxElementCount,
                         VkBuffer indirectBuffer, VkDeviceSize indirectOffset, VkBuffer keysBuffer,
                         VkDeviceSize keysOffset, VkBuffer storageBuffer,
                         VkDeviceSize storageOffset, VkQueryPool queryPool, uint32_t query) {
  gpuSort(commandBuffer, sorter, maxElementCount, indirectBuffer, indirectOffset, keysBuffer,
          keysOffset, NULL, 0, storageBuffer, storageOffset, queryPool, query);
}

void vrdxCmdSortKeyValue(VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t elementCount,
                         VkBuffer keysBuffer, VkDeviceSize keysOffset, VkBuffer valuesBuffer,
                         VkDeviceSize valuesOffset, VkBuffer storageBuffer,
                         VkDeviceSize storageOffset, VkQueryPool queryPool, uint32_t query) {
  gpuSort(commandBuffer, sorter, elementCount, NULL, 0, keysBuffer, keysOffset, valuesBuffer,
          valuesOffset, storageBuffer, storageOffset, queryPool, query);
}

void vrdxCmdSortKeyValueIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                                 uint32_t maxElementCount, VkBuffer indirectBuffer,
                                 VkDeviceSize indirectOffset, VkBuffer keysBuffer,
                                 VkDeviceSize keysOffset, VkBuffer valuesBuffer,
                                 VkDeviceSize valuesOffset, VkBuffer storageBuffer,
                                 VkDeviceSize storageOffset, VkQueryPool queryPool,
                                 uint32_t query) {
  gpuSort(commandBuffer, sorter, maxElementCount, indirectBuffer, indirectOffset, keysBuffer,
          keysOffset, valuesBuffer, valuesOffset, storageBuffer, storageOffset, queryPool, query);
}

namespace {

void gpuSort(VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t elementCount,
             VkBuffer indirectBuffer, VkDeviceSize indirectOffset, VkBuffer keysBuffer,
             VkDeviceSize keysOffset, VkBuffer valuesBuffer, VkDeviceSize valuesOffset,
             VkBuffer storageBuffer, VkDeviceSize storageOffset, VkQueryPool queryPool,
             uint32_t query) {
  VkDevice device = sorter->device;
  PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = sorter->vkCmdPushDescriptorSetKHR;
  VkPipelineLayout pipelineLayout = sorter->pipelineLayout;

  uint32_t partitionCount = RoundUp(elementCount, PARTITION_SIZE);

  VkDeviceSize elementCountSize = Align(sizeof(uint32_t), 16);
  VkDeviceSize histogramSize = HistogramSize(elementCount);
  VkDeviceSize inoutSize = InoutSize(elementCount);

  VkDeviceSize elementCountOffset = storageOffset;
  VkDeviceSize histogramOffset = elementCountOffset + elementCountSize;
  VkDeviceSize inoutOffset = histogramOffset + histogramSize;

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool, query + 0);
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
    vkCmdUpdateBuffer(commandBuffer, storageBuffer, elementCountOffset, sizeof(elementCount),
                      &elementCount);
  }

  // reset global histogram. partition histogram is set by shader.
  vkCmdFillBuffer(commandBuffer, storageBuffer, histogramOffset, 4 * RADIX * sizeof(uint32_t), 0);

  VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, NULL, 0,
                       NULL);

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool, query + 1);
  }

  PushConstants pushConstants;
  for (int i = 0; i < 4; ++i) {
    pushConstants.pass = i;

    int writeCount = 5;
    VkDescriptorBufferInfo buffers[7];
    buffers[0] = {storageBuffer, elementCountOffset, sizeof(elementCount)};
    buffers[1] = {storageBuffer, histogramOffset, sizeof(uint32_t) * 4 * RADIX};
    buffers[2] = {storageBuffer, histogramOffset + sizeof(uint32_t) * 4 * RADIX,
                  VK_WHOLE_SIZE};  // TODO: get exact buffer size
    buffers[3] = {keysBuffer, keysOffset, inoutSize};
    buffers[4] = {storageBuffer, inoutOffset, inoutSize};
    if (valuesBuffer) {
      writeCount = 7;
      buffers[5] = {valuesBuffer, valuesOffset, inoutSize};
      buffers[6] = {storageBuffer, inoutOffset + inoutSize, inoutSize};
    }

    // switch in->out to out->in for pass 1, pass 3
    if (i % 2 == 1) {
      std::swap(buffers[3], buffers[4]);
      if (valuesBuffer) {
        std::swap(buffers[5], buffers[6]);
      }
    }

    VkWriteDescriptorSet writes[7];
    for (int i = 0; i < writeCount; ++i) {
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = 0;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &buffers[i];
    }

    vkCmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                              writeCount, writes);

    vkCmdPushConstants(commandBuffer, sorter->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pushConstants), &pushConstants);

    // upsweep
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sorter->upsweepPipeline);

    vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool,
                          query + 2 + 3 * i + 0);
    }

    // spine
    memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, NULL, 0,
                         NULL);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sorter->spinePipeline);

    vkCmdDispatch(commandBuffer, RADIX, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool,
                          query + 2 + 3 * i + 1);
    }

    // downsweep
    memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, NULL, 0,
                         NULL);

    if (valuesBuffer) {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        sorter->downsweepKeyValuePipeline);
    } else {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sorter->downsweepPipeline);
    }

    vkCmdDispatch(commandBuffer, partitionCount, 1, 1);

    if (queryPool) {
      vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool,
                          query + 2 + 3 * i + 2);
    }

    if (i < 3) {
      memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, NULL, 0,
                           NULL);
    }
  }

  if (queryPool) {
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool, query + 14);
  }
}

}  // namespace
