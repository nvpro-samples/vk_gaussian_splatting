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

#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <filesystem>
#include <fmt/format.h>
#include <glm/vec3.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>

#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>

// Macro to check if a Vulkan handle is not null, perform a destruction operation, and reset the handle to null
#define TEST_DESTROY_AND_RESET(handle, destroyFunc)                                                                    \
  do                                                                                                                   \
  {                                                                                                                    \
    if((handle) != VK_NULL_HANDLE)                                                                                     \
    {                                                                                                                  \
      destroyFunc;                                                                                                     \
      (handle) = VK_NULL_HANDLE;                                                                                       \
    }                                                                                                                  \
  } while(0)

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
      std::filesystem::absolute(exePath / TARGET_EXE_TO_NVSHADERS_DIRECTORY),
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

// prints N/A or +Inf / -Inf
static std::string formatFloatInf(float value)
{
  if(std::isnan(value))
  {
    return "N/A";
  }
  else if(fabs(value) == FLT_MAX)
  {  // Catches both +inf and -inf
    return (value > 0.0f) ? "+Inf" : "-Inf";
  }
  else
  {
    return fmt::format("{}", value);
  }
}

// Helper to get bytes per pixel for common Vulkan color formats
static uint32_t getColorFormatBytesPerPixel(VkFormat format)
{
  switch(format)
  {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
      return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    default:
      return 8;  // Default to 64-bit
  }
}

static void computeTransform(glm::vec3& scale, glm::vec3& rotation, glm::vec3& translation, glm::mat4& transform, glm::mat4& transformInv)
{
  // Use quaternion-based rotation for consistent transform computation
  glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
  glm::mat4 R = glm::mat4_cast(glm::quat(glm::radians(rotation)));
  glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);

  transform    = T * R * S;
  transformInv = glm::inverse(transform);
}

// Overload with transformRotScaleInverse parameter (always compute all three for consistency)
static void computeTransform(glm::vec3& scale,
                             glm::vec3& rotation,
                             glm::vec3& translation,
                             glm::mat4& transform,
                             glm::mat4& transformInv,
                             glm::mat3& transformRotScaleInv)
{
  // Use quaternion-based rotation for consistent transform computation
  glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
  glm::mat4 R = glm::mat4_cast(glm::quat(glm::radians(rotation)));
  glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);

  transform    = T * R * S;
  transformInv = glm::inverse(transform);
  // For rotation-scale inverse: must use inverse() for non-uniform scaling
  // Note: transpose(RS) = inverse(RS) only holds for orthogonal matrices (pure rotation)
  transformRotScaleInv = glm::mat3(glm::inverse(R * S));
}

// Utility function to rotate a direction vector by rotation angles (in degrees)
static glm::vec3 rotateDirection(const glm::vec3& rotation, const glm::vec3& direction)
{
  // Use quaternion-based rotation (consistent with computeTransform)
  glm::mat4 R = glm::mat4_cast(glm::quat(glm::radians(rotation)));
  return glm::vec3(R * glm::vec4(direction, 0.0f));
}

// Truncate filename to max length, keeping the last characters
inline std::string truncateFilename(const std::string& filename, size_t maxLength = 25)
{
  if(filename.length() <= maxLength)
    return filename;
  return filename.substr(filename.length() - maxLength);
}

}  // namespace vk_gaussian_splatting

#endif
