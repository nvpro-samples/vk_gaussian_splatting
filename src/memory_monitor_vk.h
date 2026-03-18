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

#ifndef _MEMORY_MONITOR_H_
#define _MEMORY_MONITOR_H_

#include <vulkan/vulkan_core.h>
#include <cstdint>

namespace vk_gaussian_splatting {

// Summary of VRAM usage for the primary device-local heap
struct VRAMSummary
{
  VkDeviceSize usedBytes   = 0;  // Bytes currently used by this application
  VkDeviceSize budgetBytes = 0;  // Bytes available (budget) for this application
};

// Query and print VRAM memory information using VK_EXT_memory_budget
// This function will detect and display all memory heaps including:
// - VRAM (GPU-only and Resizable BAR)
// - System RAM
void queryVRAMInfo(VkPhysicalDevice physicalDevice);

// Query VRAM usage summary for the largest device-local heap
// Returns used and budget bytes for the primary GPU-only VRAM heap
VRAMSummary queryVRAMSummary(VkPhysicalDevice physicalDevice);

}  // namespace vk_gaussian_splatting

#endif  // _MEMORY_MONITOR_H_
