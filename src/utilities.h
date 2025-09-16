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

#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <filesystem>
#include <fmt/format.h>
#include <glm/vec3.hpp>
#include <glm/gtx/transform.hpp>

#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>

#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>

// Example using the parallel loop macro
// constexpr uint32_t N = 100;
// START_PAR_LOOP( N, i)
//   std::cout << "Processing index " << i << "\n";
// END_PAR_LOOP()

#define START_PAR_LOOP(SIZE, INDEX)                                                                                    \
  {                                                                                                                    \
    nvutils::parallel_batches_pooled<8192>(           \
        SIZE, [&](uint64_t INDEX, uint64_t tidx) {

#define END_PAR_LOOP()                                                                                                 \
  }, (uint32_t)std::thread::hardware_concurrency());                                                                   \
  }

namespace vk_gaussian_splatting {

// test if file extension converted to lower case matches ext,
// ext shall be provided as lowerCase and contain '.' : Example: ".txt"
inline bool hasExtension(const std::filesystem::path& filePath, std::string ext)
{
  auto fileExt = filePath.extension().string();
  std::transform(fileExt.begin(), fileExt.end(), fileExt.begin(), ::tolower);
  return fileExt == ext;
}

inline static std::vector<std::filesystem::path> getResourcesDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  return {
      std::filesystem::absolute(exePath / TARGET_EXE_TO_ROOT_DIRECTORY / "_downloaded_resources"),
      std::filesystem::absolute(exePath / "resources"),  //
      std::filesystem::absolute(exePath)                 //
  };
}

inline static std::vector<std::filesystem::path> getShaderDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  std::filesystem::path exeName = nvutils::getExecutablePath().stem();
  return {
      std::filesystem::absolute(exePath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders"),
      std::filesystem::absolute(exePath / TARGET_NAME "_files" / "shaders"),
      std::filesystem::absolute(exePath),
  };
}

static std::string formatMemorySize(size_t sizeInBytes)
{
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1024, 1024 * 1024, 1024 * 1024 * 1024};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

static std::string formatSize(size_t sizeValue)
{
  static const std::string units[]     = {"", "K", "M", "G"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeValue < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeValue) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

static void computeTransform(glm::vec3& scale, glm::vec3& rotation, glm::vec3& translation, glm::mat4& transform, glm::mat4& transformInv)
{
  transform = glm::mat4(1.0f);  // Identity matrix

  // Apply transformations in Scale -> Rotate -> Translate order
  // 1. Apply translation
  transform = glm::translate(transform, translation);

  // 2. Apply rotations (note: glm::radians converts degrees to radians)
  transform = glm::rotate(transform, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
  transform = glm::rotate(transform, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
  transform = glm::rotate(transform, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

  // 3. Apply scaling
  transform = glm::scale(transform, scale);

  //
  transformInv = glm::inverse(transform);
}

struct PhysicalDeviceInfo
{
  VkPhysicalDeviceProperties         properties10;
  VkPhysicalDeviceVulkan11Properties properties11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceVulkan12Properties properties12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
  VkPhysicalDeviceVulkan13Properties properties13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceVulkan14Properties properties14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES};

  VkPhysicalDeviceFeatures         features10;
  VkPhysicalDeviceVulkan11Features features11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features features12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan14Features features14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};

  void init(VkPhysicalDevice physicalDevice, uint32_t apiVersion = VK_API_VERSION_1_4)
  {
    assert(apiVersion >= VK_API_VERSION_1_2);

    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext                       = &properties11;
    properties11.pNext                = &properties12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      properties12.pNext = &properties13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      properties13.pNext = &properties14;
    }
    vkGetPhysicalDeviceProperties2(physicalDevice, &props);
    properties10 = props.properties;

    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext                     = &features11;
    features11.pNext                   = &features12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      features12.pNext = &features13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      features13.pNext = &features14;
    }
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
    features10 = features.features;
  }
};


}  // namespace vk_gaussian_splatting

#endif
