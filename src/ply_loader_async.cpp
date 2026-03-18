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

//
#include <fstream>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cmath>  // For std::log

// 3rd party ply library
#include "miniply.h"
// 3rd party spz library
#include "load-spz.h"

//
#include "ply_loader_async.h"
#include "utilities.h"

using namespace vk_gaussian_splatting;

namespace {

// .splat file format: 32 bytes per Gaussian (no header)
// Format specification from antimatter15/splat repository
struct SplatBinaryRecord
{
  float   position[3];  // 12 bytes: xyz
  float   scale[3];     // 12 bytes: sx, sy, sz
  uint8_t color[4];     //  4 bytes: rgba
  uint8_t rotation[4];  //  4 bytes: quaternion (normalized and stored as uint8 [0, 255])
};
static_assert(sizeof(SplatBinaryRecord) == 32, "SplatBinaryRecord must be 32 bytes");

// Helper function to load ".splat" files
bool loadSplatFile(const std::filesystem::path&                          filename,
                   SplatSet&                                             output,
                   const std::chrono::high_resolution_clock::time_point& startTime,
                   std::function<void(float)>                            setProgressCallback)
{
  // Open file in binary mode
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if(!file.is_open())
  {
    std::cout << "Error: failed to open .splat file: " << filename << std::endl;
    return false;
  }

  // Get file size and calculate number of Gaussians
  const std::streamsize fileSize = file.tellg();
  if(fileSize % 32 != 0)
  {
    std::cout << "Error: invalid .splat file size (not a multiple of 32 bytes): " << filename << std::endl;
    return false;
  }

  const uint32_t numSplats = static_cast<uint32_t>(fileSize / 32);
  if(numSplats == 0)
  {
    std::cout << "Error: empty .splat file: " << filename << std::endl;
    return false;
  }

  std::cout << "Loading .splat file with " << numSplats << " Gaussians..." << std::endl;

  // Read entire file into memory
  file.seekg(0, std::ios::beg);
  std::vector<SplatBinaryRecord> records(numSplats);
  file.read(reinterpret_cast<char*>(records.data()), fileSize);
  file.close();

  if(!file)
  {
    std::cout << "Error: failed to read .splat file: " << filename << std::endl;
    return false;
  }

  setProgressCallback(0.3f);  // File read complete

  // Allocate output arrays
  output.positions.resize(numSplats * 3);
  output.scale.resize(numSplats * 3);
  output.rotation.resize(numSplats * 4);
  output.opacity.resize(numSplats);
  output.f_dc.resize(numSplats * 3);
  output.f_rest.clear();  // .splat format has no spherical harmonics

  setProgressCallback(0.4f);  // Allocation complete

  // Convert binary data to SplatSet format
  const uint32_t progressInterval = std::max(1u, numSplats / 20);  // Update 20 times during conversion

  // SH constant from PLY converter
  constexpr float SH_C0 = 0.28209479177387814f;

  for(uint32_t i = 0; i < numSplats; i++)
  {
    const SplatBinaryRecord& record    = records[i];
    const uint32_t           posOffset = i * 3;
    const uint32_t           rotOffset = i * 4;

    // Position (xyz) - Direct copy
    output.positions[posOffset + 0] = record.position[0];
    output.positions[posOffset + 1] = record.position[1];
    output.positions[posOffset + 2] = record.position[2];

    // Scale (xyz) - Inverse of: scales = exp([scale_0, scale_1, scale_2])
    // .splat stores exp(log_scale), so take log to get back to PLY format
    output.scale[posOffset + 0] = std::log(record.scale[0]);
    output.scale[posOffset + 1] = std::log(record.scale[1]);
    output.scale[posOffset + 2] = std::log(record.scale[2]);

    // Rotation quaternion - Inverse of: ((rot / norm(rot)) * 128 + 128).clip(0, 255).astype(uint8)
    // .splat stores: normalized_quat * 128 + 128 as uint8 [0, 255]
    // To recover: (uint8 - 128) / 128 → normalized quaternion
    // Note: .splat stores as [x, y, z, w], but PLY format is [w, x, y, z]
    const float qx = (static_cast<float>(record.rotation[0]) - 128.0f) / 128.0f;
    const float qy = (static_cast<float>(record.rotation[1]) - 128.0f) / 128.0f;
    const float qz = (static_cast<float>(record.rotation[2]) - 128.0f) / 128.0f;
    const float qw = (static_cast<float>(record.rotation[3]) - 128.0f) / 128.0f;

    // Store in PLY format: [w, x, y, z] (reordered from .splat's [x, y, z, w])
    output.rotation[rotOffset + 0] = qx;
    output.rotation[rotOffset + 1] = qy;
    output.rotation[rotOffset + 2] = qz;
    output.rotation[rotOffset + 3] = qw;

    // Color (RGB) - Inverse of: color = 0.5 + SH_C0 * f_dc
    // .splat stores: (0.5 + SH_C0 * f_dc) * 255 as uint8
    // To recover f_dc: (uint8 / 255 - 0.5) / SH_C0
    output.f_dc[posOffset + 0] = (record.color[0] / 255.0f - 0.5f) / SH_C0;
    output.f_dc[posOffset + 1] = (record.color[1] / 255.0f - 0.5f) / SH_C0;
    output.f_dc[posOffset + 2] = (record.color[2] / 255.0f - 0.5f) / SH_C0;

    // Opacity (Alpha) - Inverse of: sigmoid(opacity) = 1 / (1 + exp(-opacity))
    // .splat stores: sigmoid(opacity) * 255 as uint8
    // To recover opacity: -log((1 / alpha) - 1) = log(alpha / (1 - alpha))
    const float alpha = record.color[3] / 255.0f;
    // Clamp alpha to avoid log(0) or division by zero
    const float alpha_clamped = std::clamp(alpha, 1e-6f, 1.0f - 1e-6f);
    output.opacity[i]         = -std::log((1.0f / alpha_clamped) - 1.0f);

    // Update progress
    if(i % progressInterval == 0)
    {
      const float progress = 0.4f + (0.5f * static_cast<float>(i) / static_cast<float>(numSplats));
      setProgressCallback(progress);
    }
  }

  setProgressCallback(0.9f);  // Conversion complete

  // Convert coordinates from .splat format to RUB coordinate system
  // Note: .splat files typically use the same coordinate system as the original
  // 3DGS implementation (RDF), so convert to RUB like .ply files
  output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);

  setProgressCallback(1.0f);  // Complete

  // Print timing info
  auto      endTime  = std::chrono::high_resolution_clock::now();
  long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  std::cout << "Loaded " << numSplats << " splats from .splat file in " << loadTime << "ms" << std::endl;

  return true;
}

}  // anonymous namespace

bool PlyLoaderAsync::loadScene(std::filesystem::path filename, std::shared_ptr<SplatSetVk> output)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status != E_READY)
  {
    return false;
  }

  // setup load info and wakeup the thread
  m_filename = filename;
  m_output   = output;
  m_loadCV.notify_all();

  return true;
}

bool PlyLoaderAsync::initialize()
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
          m_result   = m_output;
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

void PlyLoaderAsync::cancel()
{
  // does nothing for the time beeing
}

PlyLoaderAsync::State PlyLoaderAsync::getStatus()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_status;
}

bool PlyLoaderAsync::reset()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if(m_status == E_LOADED || m_status == E_FAILURE)
  {
    m_progress = 0.0;
    m_status   = E_READY;
    m_result   = nullptr;
    return true;
  }
  else
  {
    return false;
  }
}

bool PlyLoaderAsync::innerLoad(std::filesystem::path filename, SplatSet& output)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  // Check for .splat extension first (antimatter15/splat format)
  if(hasExtension(filename, ".splat"))
  {
    // Create a lambda to capture 'this' for progress updates
    auto progressCallback = [this](float progress) { this->setProgress(progress); };
    return loadSplatFile(filename, output, startTime, progressCallback);
  }

  // we use spz library for .spz extensions
  if(hasExtension(filename, ".spz"))
  {
    // Converts to RUB coordinate system
    spz::UnpackOptions options{.to = spz::CoordinateSystem::RUB};
    // let's load
    spz::GaussianCloud cloud = spz::loadSpz(filename.string(), options);
    // convert to INRIA representation
    output.positions.swap(cloud.positions);
    output.rotation.resize(cloud.rotations.size());
    const uint32_t numSplats = uint32_t(output.positions.size() / 3);
    for(uint32_t i = 0; i < numSplats; i++)
    {
      const uint32_t offset       = i * 4;
      output.rotation[offset + 0] = cloud.rotations[offset + 3];
      output.rotation[offset + 1] = cloud.rotations[offset + 0];
      output.rotation[offset + 2] = cloud.rotations[offset + 1];
      output.rotation[offset + 3] = cloud.rotations[offset + 2];
    }
    output.scale.swap(cloud.scales);
    output.opacity.swap(cloud.alphas);
    output.f_dc = cloud.colors;
    // reorganize SH per components to match INRIA
    const size_t shCoefsCount = cloud.sh.size() / numSplats / 3;
    output.f_rest.resize(cloud.sh.size());
    for(size_t i = 0; i < numSplats; i++)
    {
      const size_t offset = i * shCoefsCount * 3;

      // Spherical harmonics: Interleave so the coefficients are the fastest-changing axis and
      // the channel (r, g, b) is slower-changing axis.
      for(size_t j = 0; j < shCoefsCount; j++)
      {
        output.f_rest[offset + j] = cloud.sh[(i * shCoefsCount + j) * 3];
      }
      for(size_t j = 0; j < shCoefsCount; j++)
      {
        output.f_rest[offset + shCoefsCount + j] = cloud.sh[(i * shCoefsCount + j) * 3 + 1];
      }
      for(size_t j = 0; j < shCoefsCount; j++)
      {
        output.f_rest[offset + shCoefsCount * 2 + j] = cloud.sh[(i * shCoefsCount + j) * 3 + 2];
      }
    }
    //
    auto      endTime  = std::chrono::high_resolution_clock::now();
    long long loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "File loaded in " << loadTime << "ms" << std::endl;
    //
    return cloud.numPoints != 0;
  }

  // We use miniply to load .ply files (binary or utf8)
  // Open the file
  miniply::PLYReader reader(filename.string().c_str());
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
        output.f_rest.resize(numVerts * 45);
        reader.extract_properties(indices, 45, miniply::PLYPropertyType::Float, output.f_rest.data());
        loaded += numVerts * 45;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 3, "x", "y", "z"))
      {
        output.positions.resize(numVerts * 3);
        reader.extract_properties(indices, 3, miniply::PLYPropertyType::Float, output.positions.data());
        loaded += numVerts * 3;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 1, "opacity"))
      {
        output.opacity.resize(numVerts);
        reader.extract_properties(indices, 1, miniply::PLYPropertyType::Float, output.opacity.data());
        loaded += numVerts;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 3, "scale_0", "scale_1", "scale_2"))
      {
        output.scale.resize(numVerts * 3);
        reader.extract_properties(indices, 3, miniply::PLYPropertyType::Float, output.scale.data());
        loaded += numVerts * 3;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 4, "rot_0", "rot_1", "rot_2", "rot_3"))
      {
        output.rotation.resize(numVerts * 4);
        reader.extract_properties(indices, 4, miniply::PLYPropertyType::Float, output.rotation.data());
        loaded += numVerts * 4;
        setProgress(float(loaded) / float(total));
      }
      if(reader.find_properties(indices, 3, "f_dc_0", "f_dc_1", "f_dc_2"))
      {
        output.f_dc.resize(numVerts * 3);
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
    // convert coordinates
    output.convertCoordinates(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);
    //
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
