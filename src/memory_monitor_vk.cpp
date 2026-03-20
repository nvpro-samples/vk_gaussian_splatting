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

#include "memory_monitor_vk.h"
#include "utilities.h"

#include <nvutils/logger.hpp>

#include <iostream>
#include <fmt/format.h>

namespace vk_gaussian_splatting {

void queryVRAMInfo(VkPhysicalDevice physicalDevice)
{
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps{};
  budgetProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

  VkPhysicalDeviceMemoryProperties2 memProps2{};
  memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  memProps2.pNext = &budgetProps;

  vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProps2);

  LOGI("\n========== VRAM Memory Information ==========\n");

  // Iterate through memory heaps
  for(uint32_t i = 0; i < memProps2.memoryProperties.memoryHeapCount; i++)
  {
    VkDeviceSize heapSize = memProps2.memoryProperties.memoryHeaps[i].size;
    VkDeviceSize budget   = budgetProps.heapBudget[i];                // Available for your app
    VkDeviceSize usage    = budgetProps.heapUsage[i];                 // Currently used by your app
    VkDeviceSize free     = (budget > usage) ? (budget - usage) : 0;  // Free for your app

    // Check if this is a device-local (VRAM) heap
    bool isDeviceLocal = (memProps2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;

    // Detect Resizable BAR: device-local heap with host-visible memory types
    bool isReBAR = false;
    if(isDeviceLocal)
    {
      // Check if any memory type in this heap is host-visible (indicates ReBAR)
      for(uint32_t typeIdx = 0; typeIdx < memProps2.memoryProperties.memoryTypeCount; typeIdx++)
      {
        if(memProps2.memoryProperties.memoryTypes[typeIdx].heapIndex == i)
        {
          VkMemoryPropertyFlags flags = memProps2.memoryProperties.memoryTypes[typeIdx].propertyFlags;
          if((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
          {
            isReBAR = true;
            break;
          }
        }
      }
    }

    // Determine heap type label
    std::string heapType;
    if(isReBAR)
    {
      heapType = " (VRAM - Resizable BAR / CPU-Accessible)";
    }
    else if(isDeviceLocal)
    {
      heapType = " (VRAM - GPU Only)";
    }
    else
    {
      heapType = " (System RAM - Host Memory)";
    }

    LOGI("\nMemory Heap %u%s:\n", i, heapType.c_str());
    LOGI("  Total Size: %s\n", formatMemorySize(heapSize).c_str());
    LOGI("  Budget:     %s (available for this app)\n", formatMemorySize(budget).c_str());
    LOGI("  Used:       %s (used by this app)\n", formatMemorySize(usage).c_str());
    LOGI("  Free:       %s (free for this app)\n", formatMemorySize(free).c_str());

    // Calculate and display percentage
    if(budget > 0)
    {
      float usagePercent = (float(usage) / float(budget)) * 100.0f;
      LOGI("  Usage:      %.1f%%\n", usagePercent);
    }
  }

  LOGI("=============================================\n\n");
}

VRAMSummary queryVRAMSummary(VkPhysicalDevice physicalDevice)
{
  VRAMSummary result{};

  VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps{};
  budgetProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

  VkPhysicalDeviceMemoryProperties2 memProps2{};
  memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  memProps2.pNext = &budgetProps;

  vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProps2);

  // Find the largest device-local (non-ReBAR) heap for the primary VRAM summary
  VkDeviceSize largestBudget = 0;
  for(uint32_t i = 0; i < memProps2.memoryProperties.memoryHeapCount; i++)
  {
    bool isDeviceLocal = (memProps2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    if(!isDeviceLocal)
      continue;

    VkDeviceSize budget = budgetProps.heapBudget[i];
    VkDeviceSize usage  = budgetProps.heapUsage[i];

    if(budget > largestBudget)
    {
      largestBudget      = budget;
      result.usedBytes   = usage;
      result.budgetBytes = budget;
    }
  }

  return result;
}

}  // namespace vk_gaussian_splatting
