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

#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>

// Example using the parallel loop macro
// constexpr uint32_t N = 100;
// START_PAR_LOOP( N, i)
//   std::cout << "Processing index " << i << "\n";
// END_PAR_LOOP()

#define START_PAR_LOOP(SIZE, INDEX)                   \
  {                                                   \
    nvutils::parallel_batches_pooled<8192>(           \
        SIZE, [&](uint64_t INDEX, uint64_t tidx) {

#define END_PAR_LOOP()                                \
  }, (uint32_t)std::thread::hardware_concurrency());  \
  }

namespace vk_gaussian_splatting {

inline static std::vector<std::filesystem::path> getResourcesDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  return {
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_ROOT_DIRECTORY) / "_downloaded_resources"),
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_ROOT_DIRECTORY) / "resources"),
      std::filesystem::absolute(exePath / "resources"),  //
      std::filesystem::absolute(exePath)                 //
  };
}

inline static std::vector<std::filesystem::path> getShaderDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  std::filesystem::path exeName = nvutils::getExecutablePath().stem();
  return {
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_SOURCE_DIRECTORY) / "shaders"),
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_NAME) / "shaders"),
      std::filesystem::absolute(exePath),
  };
}

}  // namespace vk_gaussian_splatting

#endif
