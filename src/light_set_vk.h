/*/*
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

#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <nvapp/application.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/acceleration_structures.hpp>

#include "shaderio.h"

#define MAX_LIGHTS 4

class LightSetVk
{
public:
  // number of lights
  // number of light sources will be provided to shaders through the frameInfo buffer
  uint64_t numLights = 1;
  // light sources in RAM
  // we use a vector instead of an array because m_uploader->appendBuffer does not support span of array
  std::vector<shaderio::LightSource> lights;
  // light sources in VRAM
  nvvk::Buffer lightsBuffer;

public:
  // create and upload the lightset buffer with default light
  void init(nvapp::Application* app, nvvk::ResourceAllocator* alloc, nvvk::StagingUploader* uploader);

  // free the vulkan buffer and clear the light set
  void deinit();

  // update the buffer that stores the list of lights
  // must be invoked after adding/removing light or reseting all lights
  void updateBuffer();

  // return the number of lights in the set
  uint64_t size() { return numLights; }

  // add a new light to the light set and returns its index
  uint64_t createLight()
  {
    if(numLights < MAX_LIGHTS)
    {
      lights[numLights] = shaderio::LightSource();
      ++numLights;
    }
    return numLights - 1;
  }

  // remove a light from the set
  void eraseLight(uint64_t index)
  {
    assert(index < numLights);

    if(numLights == 1)
      return;

    for(uint64_t i = index; i < numLights - 1; ++i)
    {
      lights[i] = lights[i + 1];
    }
    numLights--;
  }

  // access the light source of given index
  shaderio::LightSource& getLight(uint64_t index)
  {
    assert(index < numLights);
    return lights[index];
  }

private:
  nvapp::Application*      m_app{};
  nvvk::ResourceAllocator* m_alloc{};
  nvvk::StagingUploader*   m_uploader{};
};
