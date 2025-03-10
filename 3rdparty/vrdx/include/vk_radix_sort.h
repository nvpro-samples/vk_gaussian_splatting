#ifndef VK_RADIX_SORT_H
#define VK_RADIX_SORT_H

#include <vulkan/vulkan.h>

struct VrdxSorter_T;

/**
 * VrdxSorter creates pipelines.
 */
VK_DEFINE_HANDLE(VrdxSorter)

struct VrdxSorterCreateInfo {
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkPipelineCache pipelineCache;
};

void vrdxCreateSorter(const VrdxSorterCreateInfo* pCreateInfo,
                      VrdxSorter* pSorter);

void vrdxDestroySorter(VrdxSorter sorter);

struct VrdxSorterStorageRequirements {
  VkDeviceSize size;
  VkBufferUsageFlags usage;
};

void vrdxGetSorterStorageRequirements(
    VrdxSorter sorter, uint32_t maxElementCount,
    VrdxSorterStorageRequirements* requirements);

void vrdxGetSorterKeyValueStorageRequirements(
    VrdxSorter sorter, uint32_t maxElementCount,
    VrdxSorterStorageRequirements* requirements);

/**
 * if queryPool is not VK_NULL_HANDLE, it writes timestamps to N entries
 * [query..query+N-1].
 *
 * N=15
 * query + 0: start timestamp (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
 * query + 1: transfer timestamp (VK_PIPELINE_STAGE_TRANSFER_BIT)
 * query + 2 + (3 * i) + 0: upsweep (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
 * query + 2 + (3 * i) + 1: spine (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
 * query + 2 + (3 * i) + 2: downsweep (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
 * query + 14: sort end timestamp (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
 */
void vrdxCmdSort(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                 uint32_t elementCount, VkBuffer keysBuffer,
                 VkDeviceSize keysOffset, VkBuffer storageBuffer,
                 VkDeviceSize storageOffset, VkQueryPool queryPool,
                 uint32_t query);

void vrdxCmdSortIndirect(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         uint32_t maxElementCount, VkBuffer indirectBuffer,
                         VkDeviceSize indirectOffset, VkBuffer keysBuffer,
                         VkDeviceSize keysOffset, VkBuffer storageBuffer,
                         VkDeviceSize storageOffset, VkQueryPool queryPool,
                         uint32_t query);

void vrdxCmdSortKeyValue(VkCommandBuffer commandBuffer, VrdxSorter sorter,
                         uint32_t elementCount, VkBuffer keysBuffer,
                         VkDeviceSize keysOffset, VkBuffer valuesBuffer,
                         VkDeviceSize valuesOffset, VkBuffer storageBuffer,
                         VkDeviceSize storageOffset, VkQueryPool queryPool,
                         uint32_t query);

/**
 * indirectBuffer contains elementCount.
 *
 * The sort command reads a uint32_t value from indirectBuffer at
 * indirectOffset.
 *
 * User must add barrier with second synchronization scope
 * COMPUTE_SHADER stage and SHADER_READ access.
 *
 * indirectBuffer requires TRANSFER_SRC buffer usage flag.
 */
void vrdxCmdSortKeyValueIndirect(
    VkCommandBuffer commandBuffer, VrdxSorter sorter, uint32_t maxElementCount,
    VkBuffer indirectBuffer, VkDeviceSize indirectOffset, VkBuffer keysBuffer,
    VkDeviceSize keysOffset, VkBuffer valuesBuffer, VkDeviceSize valuesOffset,
    VkBuffer storageBuffer, VkDeviceSize storageOffset, VkQueryPool queryPool,
    uint32_t query);

#endif  // VK_RADIX_SORT_H
