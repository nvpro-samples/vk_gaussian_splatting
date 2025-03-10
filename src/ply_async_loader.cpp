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

//
#include <fstream>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>

// 3rd party ply library
#include "miniply.h"

//
#include "ply_async_loader.h"

bool PlyAsyncLoader::loadScene(std::string filename, SplatSet& output)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status != E_READY)
  {
    return false;
  }

  // setup load info and wakeup the thread
  m_filename = filename;
  m_output   = &output;
  m_loadCV.notify_all();

  return true;
}

bool PlyAsyncLoader::initialize()
{
  // original state shall be shutdown
  std::unique_lock<std::mutex> lock(m_mutex);
  if(m_status != E_SHUTDOWN)
    return false;  // will unlock through lock destructor
  else
    lock.unlock();

  // starts the thread
  m_loader = std::thread([this]() {
    //
    std::unique_lock<std::mutex> lock(m_mutex);
    m_status = E_READY;
    lock.unlock();
    //
    while(true)
    {
      // wait to load new scene
      std::unique_lock<std::mutex> lock(m_mutex);
      m_loadCV.wait(lock, [this] { return m_shutdownRequested || m_output != nullptr; });
      bool shutdown = m_shutdownRequested;
      lock.unlock();
      // if request is not a shutdown do the job
      if(!shutdown)
      {
        // let's load
        std::unique_lock<std::mutex> lock(m_mutex);
        m_status = E_LOADING;
        lock.unlock();
        if(m_output != nullptr && innerLoad(m_filename, *m_output))
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status   = E_LOADED;
          m_output   = nullptr;
          m_filename = "";
        }
        else
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status   = E_FAILURE;
          m_output   = nullptr;
          m_filename = "";
        }
      }
      else
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status            = E_SHUTDOWN;
        m_shutdownRequested = false;
        m_output            = nullptr;
        m_filename          = "";
        return true;
      }
    }
  });

  return true;
}

void PlyAsyncLoader::cancel()
{
  // does nothing for the time beeing
}

PlyAsyncLoader::State PlyAsyncLoader::getStatus()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_status;
}

bool PlyAsyncLoader::reset()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status == E_LOADED || m_status == E_FAILURE)
  {
    m_progress = 0.0;
    m_status   = E_READY;
    return true;
  }
  else
  {
    return false;
  }
}

bool PlyAsyncLoader::innerLoad(std::string filename, SplatSet& output)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  // Open the file
  miniply::PLYReader reader(filename.c_str());
  if(!reader.valid())
  {
    std::cout << "Error: ply loader failed to open file: " << filename << std::endl;
    return false;
  }

  uint32_t indices[45];
  bool     gsFound = false;

  while(reader.has_element() && !gsFound)
  {
    if(reader.element_is(miniply::kPLYVertexElement) && reader.load_element())
    {
      const uint32_t numVerts = reader.num_rows();
      if(numVerts == 0)
      {
        std::cout << "Warning: ply loader skipping empty ply element " << std::endl;
        continue;  // move to next while iteration
      }
      output.positions.resize(numVerts * 3);
      output.scale.resize(numVerts * 3);
      output.rotation.resize(numVerts * 4);
      output.opacity.resize(numVerts);
      output.f_dc.resize(numVerts * 3);
      output.f_rest.resize(numVerts * 45);
      // load progress
      const uint32_t total  = numVerts * (3 + 3 + 4 + 1 + 3 + 45);
      uint32_t       loaded = 0;

      // put that first so the loading progress looks better
      if(reader.find_properties(indices, 45, "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3", "f_rest_4", "f_rest_5",
                                "f_rest_6", "f_rest_7", "f_rest_8", "f_rest_9", "f_rest_10", "f_rest_11", "f_rest_12",
                                "f_rest_13", "f_rest_14", "f_rest_15", "f_rest_16", "f_rest_17", "f_rest_18",
                                "f_rest_19", "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23", "f_rest_24",
                                "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29", "f_rest_30", "f_rest_31",
                                "f_rest_32", "f_rest_33", "f_rest_34", "f_rest_35", "f_rest_36", "f_rest_37", "f_rest_38",
                                "f_rest_39", "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44"))
      {
        reader.extract_properties(indices, 45, miniply::PLYPropertyType::Float, output.f_rest.data());
        loaded += numVerts * 45;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 3, "x", "y", "z"))
      {
        reader.extract_properties(indices, 3, miniply::PLYPropertyType::Float, output.positions.data());
        loaded += numVerts * 3;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 1, "opacity"))
      {
        reader.extract_properties(indices, 1, miniply::PLYPropertyType::Float, output.opacity.data());
        loaded += numVerts;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 3, "scale_0", "scale_1", "scale_2"))
      {
        reader.extract_properties(indices, 3, miniply::PLYPropertyType::Float, output.scale.data());
        loaded += numVerts * 3;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 4, "rot_0", "rot_1", "rot_2", "rot_3"))
      {
        reader.extract_properties(indices, 4, miniply::PLYPropertyType::Float, output.rotation.data());
        loaded += numVerts * 4;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 3, "f_dc_0", "f_dc_1", "f_dc_2"))
      {
        reader.extract_properties(indices, 3, miniply::PLYPropertyType::Float, output.f_dc.data());
        loaded += numVerts * 3;
        setProgress(float(loaded) / float(total));
      }


      gsFound = true;
    }

    reader.next_element();
  }

  if(gsFound)
  {
    auto      endTime  = std::chrono::high_resolution_clock::now();
    long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "File loaded in " << loadTime << "ms" << std::endl;
  }
  else
  {
    std::cout << "Error: invalid 3DGS PLY file" << std::endl;
  }

  return gsFound;
}
