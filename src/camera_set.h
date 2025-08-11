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

#ifndef _CAMERA_SET_H_
#define _CAMERA_SET_H_

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <tinygltf/json.hpp>

#include <nvapp/application.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/acceleration_structures.hpp>

#include <nvutils/camera_manipulator.hpp>

#include "shaderio.h"

namespace vk_gaussian_splatting {

typedef nvutils::CameraManipulator::Camera Camera;

class CameraSet
{

public:
  void init(nvutils::CameraManipulator* cameraManip) { m_cameraManip = cameraManip; }

  void deinit()
  {
    // nothing to do
  }

  void setHomeCamera(const Camera& camera)
  {
    if(m_cameras.empty())
      m_cameras.resize(1);
    m_cameras[0] = camera;
  }

  // return the number of cameras in the set
  uint64_t size() { return m_cameras.size(); }

  // store the current camera parameter in a new camera entry
  // is a camera with same attributes already exists, no additional
  // camera is created and index of existing one is returned
  uint64_t storeCamera(const Camera& camera)
  {
    bool     unique = true;
    uint64_t i      = 0;
    for(; i < m_cameras.size(); ++i)
    {
      if(m_cameras[i] == camera)
      {
        unique = false;
        break;
      }
    }
    if(unique)
    {
      m_cameras.emplace_back(camera);
      return m_cameras.size() - 1;
    }
    else
    {
      return i;
    }
  }

  // store the current camera parameter in a new camera entry
  // is a camera with same attributes already exists, no additional
  // camera is created and index of existing one is returned
  uint64_t storeCurrentCamera(void) { return storeCamera(m_cameraManip->getCamera()); }

  // load a camera entry and set
  bool loadCamera(uint64_t index)
  {
    if(index >= m_cameras.size())
      return false;
    m_cameraManip->setCamera(m_cameras[index]);
    return true;
  }

  // remove a camera from the set
  // default one (index 0) is forbiden
  bool eraseCamera(uint64_t index)
  {
    if(m_cameras.size() == 1 || index == 0 || index >= m_cameras.size())
      return false;

    for(uint64_t i = index; i < m_cameras.size() - 1; ++i)
    {
      m_cameras[i] = m_cameras[i + 1];
    }
    m_cameras.resize(m_cameras.size() - 1);
    return true;
  }

  // access the camera source of given index
  Camera& getCamera(uint64_t index)
  {
    assert(index < m_cameras.size());
    return m_cameras[index];
  }

private:
  std::vector<Camera>         m_cameras;
  nvutils::CameraManipulator* m_cameraManip;
};

/////////// Utility function to omport cmareas frim INRIA json files

static bool importCamerasINRIA(std::string filename, CameraSet& cameraSet)
{
  using nlohmann::json;

  try
  {
    std::ifstream i(filename);
    if(!i.is_open())
      return false;

    // Parsing the file
    json data;
    i >> data;

    // Access and print the elements
    for(const auto& item : data)
    {
      int                             id       = item.at("id").get<int>();
      std::string                     img_name = item.at("img_name").get<std::string>();
      int                             width    = item.at("width").get<int>();
      int                             height   = item.at("height").get<int>();
      std::vector<float>              position = item.at("position").get<std::vector<float>>();
      std::vector<std::vector<float>> rotation = item.at("rotation").get<std::vector<std::vector<float>>>();
      float                           fy       = item.at("fy").get<float>();
      float                           fx       = item.at("fx").get<float>();

      glm::mat3 rotMat(rotation[0][0], rotation[1][0], rotation[2][0], rotation[0][1], rotation[1][1], rotation[2][1],
                       rotation[0][2], rotation[1][2], rotation[2][2]);

      glm::vec3 up = rotMat * glm::vec3(0.f, -1.f, 0.f);
      glm::vec3 at = rotMat * glm::vec3(0.f, 0.f, 1.f);

      Camera newCam;
      newCam.eye = {position[0], position[1], position[2]};
      newCam.ctr = at;
      newCam.up  = up;

      // add to CameraSet
      cameraSet.storeCamera(newCam);
    }
    i.close();
    return true;
  }
  catch(...)
  {
    return false;
  }
}

}  // namespace vk_gaussian_splatting

#endif