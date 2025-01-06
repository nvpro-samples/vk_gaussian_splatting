/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
#include <fstream>
#include <array>
#include <chrono>
#include <filesystem>

// 3rd party ply library
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include "miniply.h"

//
#include "ply_async_loader.h"

bool PlyAsyncLoader::loadScene(std::string filename, SplatSet& output)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status != READY)
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
  if(m_status != SHUTDOWN) 
    return false; // will unlock through lock destructor
  else
    lock.unlock();

  // starts the thread
  m_loader = std::thread([this]() {
    //
    std::unique_lock<std::mutex> lock(m_mutex);
    m_status = READY;
    lock.unlock();
    //
    while(true) //!isShutdown())
    {
      // wait to load new scene
      std::unique_lock<std::mutex> lock(m_mutex);
      m_loadCV.wait(lock, [this] { return m_shutdownRequested || m_output != nullptr; });
      bool shutdown = m_shutdownRequested;
      lock.unlock();
      // if request is not a shutdown do the job
      if ( !shutdown ){
        // let's load
        std::unique_lock<std::mutex> lock(m_mutex);
        m_status = LOADING;
        lock.unlock();
        if(innerLoadMiniPly(m_filename, *m_output))
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status = LOADED;
          m_output   = nullptr;
          m_filename = "";
        }
        else
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status = FAILURE;
          m_output   = nullptr;
          m_filename = "";
        }
      }
      else
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status = SHUTDOWN;
        m_shutdownRequested = false;
        m_output            = nullptr;
        m_filename          = "";
        return;
      }
    }
  });
}

void PlyAsyncLoader::cancel() {
  // does nothing for the time beeing
}

PlyAsyncLoader::Status PlyAsyncLoader::getStatus() {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_status;
}

bool PlyAsyncLoader::reset()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status == LOADED || m_status == FAILURE)
  {
    m_progress = 0.0;
    m_status = READY;
    return true;
  }
  else
  {
    return false;
  }
}

bool PlyAsyncLoader::innerLoadMiniPly(std::string filename, SplatSet& output)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  // Open the file
  miniply::PLYReader reader(filename.c_str());
  if(!reader.valid())
  {
    std::cout << "Error: Failed to open file: " << filename << std::endl;
    return false;
  }

  uint32_t indices[45];
  bool     gsFound = false;

  while(reader.has_element() && !gsFound)
  {
    if(reader.element_is(miniply::kPLYVertexElement) && reader.load_element())
    {
      const uint32_t numVerts = reader.num_rows();
      output.positions.resize(numVerts * 3);
      output.scale.resize(numVerts * 3);
      output.rotation.resize(numVerts * 4);
      output.opacity.resize(numVerts );
      output.f_dc.resize(numVerts * 3);
      output.f_rest.resize(numVerts * 45);
      // load progress
      const uint32_t total = numVerts*(3+3+4+1+3+45);
      uint32_t loaded = 0;
      
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
        setProgress(float(loaded)/float(total));
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
    auto endTime = std::chrono::high_resolution_clock::now();
    long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "File loaded in " << loadTime << "ms" << std::endl;
  }
  else
  {
    std::cout << "Error: invalid 3DGS PLY file" << std::endl;
  }

  return gsFound;
}

bool PlyAsyncLoader::innerLoadTinyPly(std::string filename, SplatSet& output)
{
  // Open the file
  std::ifstream file(filename, std::ios::binary);
  if(!file.is_open())
  {
    std::cout << "Error: Failed to open file: " << filename << std::endl;
    return false;
  }

  // Create a tinyply::PlyFile object
  tinyply::PlyFile plyFile;

  plyFile.parse_header(file);

  std::shared_ptr<tinyply::PlyData> _vertices, _normals, _colors, _colorsRGBA, _texcoords, _faces, _tristrip;

  // The header information can be used to programmatically extract properties on elements
  // known to exist in the header prior to reading the data. For brevity of this sample, properties
  // like vertex position are hard-coded:
  try
  {
    _vertices = plyFile.request_properties_from_element("vertex", {"x", "y", "z"});
  }
  catch(const std::exception& e)
  {
  }
  try
  {
    _normals = plyFile.request_properties_from_element("vertex", {"nx", "ny", "nz"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _colorsRGBA = plyFile.request_properties_from_element("vertex", {"red", "green", "blue", "alpha"});
  }
  catch(const std::exception)
  {
  }

  if(!_colorsRGBA)
  {
    try
    {
      _colorsRGBA = plyFile.request_properties_from_element("vertex", {"r", "g", "b", "a"});
    }
    catch(const std::exception)
    {
    }
  }
  try
  {
    _colors = plyFile.request_properties_from_element("vertex", {"red", "green", "blue"});
  }
  catch(const std::exception)
  {
  }
  if(!_colors)
  {
    try
    {
      _colors = plyFile.request_properties_from_element("vertex", {"r", "g", "b"});
    }
    catch(const std::exception)
    {
    }
  }
  try
  {
    _texcoords = plyFile.request_properties_from_element("vertex", {"u", "v"});
  }
  catch(const std::exception)
  {
  }

  // 3DGS specifics
  std::shared_ptr<tinyply::PlyData> _f_dc, _f_rest, _opacity, _scale, _rotation;
  try
  {
    _f_dc = plyFile.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _f_rest = plyFile.request_properties_from_element(
        "vertex",
        {"f_rest_0",  "f_rest_1",  "f_rest_2",  "f_rest_3",  "f_rest_4",  "f_rest_5",  "f_rest_6",  "f_rest_7",
         "f_rest_8",  "f_rest_9",  "f_rest_10", "f_rest_11", "f_rest_12", "f_rest_13", "f_rest_14", "f_rest_15",
         "f_rest_16", "f_rest_17", "f_rest_18", "f_rest_19", "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23",
         "f_rest_24", "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29", "f_rest_30", "f_rest_31",
         "f_rest_32", "f_rest_33", "f_rest_34", "f_rest_35", "f_rest_36", "f_rest_37", "f_rest_38", "f_rest_39",
         "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _opacity = plyFile.request_properties_from_element("vertex", {"opacity"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _scale = plyFile.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
  }
  catch(const std::exception)
  {
  }
  try
  {
    _rotation = plyFile.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
  }
  catch(const std::exception)
  {
  }

  //
  plyFile.read(file);

  // now feed the data to the frame structure
  if(_vertices)
  {
    const size_t numVerticesBytes = _vertices->buffer.size_bytes();
    output.positions.resize(_vertices->count * 3);
    std::memcpy(output.positions.data(), _vertices->buffer.get(), numVerticesBytes);
  }
  else
  {
    std::cerr << "Error: missing vertex positions. " << std::endl;
    return false;
  }
  if(_normals)
  {
    const size_t numNormalsBytes = _normals->buffer.size_bytes();
    output.normals.resize(_normals->count * 3);
    std::memcpy(output.normals.data(), _normals->buffer.get(), numNormalsBytes);
  }

  // 3DGS per vertex infos

  if(_f_dc && _f_rest && _opacity && _scale && _rotation)
  {
    const size_t numFDcBytes      = _f_dc->buffer.size_bytes();
    const size_t numFRestBytes    = _f_rest->buffer.size_bytes();
    const size_t numOpacityBytes  = _opacity->buffer.size_bytes();
    const size_t numScaleBytes    = _scale->buffer.size_bytes();
    const size_t numRotationBytes = _rotation->buffer.size_bytes();
    output.f_dc.resize(_f_dc->count * 3);
    output.f_rest.resize(_f_rest->count * 45);
    output.opacity.resize(_opacity->count);
    output.scale.resize(_scale->count * 3);
    output.rotation.resize(_rotation->count * 4);
    std::memcpy(output.f_dc.data(), _f_dc->buffer.get(), numFDcBytes);
    std::memcpy(output.f_rest.data(), _f_rest->buffer.get(), numFRestBytes);
    std::memcpy(output.opacity.data(), _opacity->buffer.get(), numOpacityBytes);
    std::memcpy(output.scale.data(), _scale->buffer.get(), numScaleBytes);
    std::memcpy(output.rotation.data(), _rotation->buffer.get(), numRotationBytes);
  }
  else
  {
    std::cerr << "Error: missing 3DGS attributes. " << std::endl;
    return false;
  }

  return true;
}