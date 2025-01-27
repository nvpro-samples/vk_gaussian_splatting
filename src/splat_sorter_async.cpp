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
#include "splat_sorter_async.h"

#include <glm/vec4.hpp>

bool SplatSorterAsync::initialize()
{
  // original state shall be shutdown
  std::unique_lock<std::mutex> lock(m_mutex);
  if(m_status != SHUTDOWN)
    return false;  // will unlock through lock destructor
  else
    lock.unlock();

  // starts the thread
  m_sorter = std::thread([this]() {
    //
    std::unique_lock<std::mutex> lock(m_mutex);
    m_status = READY;
    lock.unlock();
    //
    while(true)  //!isShutdown())
    {
      // wait to load new scene
      std::unique_lock<std::mutex> lock(m_mutex);
      m_sortCV.wait(lock, [this] { return m_shutdownRequested || m_startRequested; });
      bool shutdown = m_shutdownRequested;
      lock.unlock();
      // if request is not a shutdown do the job
      if(!shutdown)
      {
        // let's load
        std::unique_lock<std::mutex> lock(m_mutex);
        m_status = SORTING;
        lock.unlock();
        if(innerSort())
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status         = SORTED;
          m_startRequested = false;
        }
        else
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status         = FAILURE;
          m_startRequested = false;
        }
      }
      else
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status         = SHUTDOWN;
        m_startRequested = false;
        return;
      }
    }
  });
}

bool SplatSorterAsync::innerSort()
{
  if(m_positions == nullptr)
    return false;

  auto startTime = std::chrono::high_resolution_clock::now();
  // we do the sorting if needed
  // find plane passing through COP and with normal dir.
  // we use distance to plane instead of distance to COP as an approximation.
  // https://mathinsight.org/distance_point_plane
  const glm::vec4 plane(sortDir[0], sortDir[1], sortDir[2],
                        -sortDir[0] * sortCop[0] - sortDir[1] * sortCop[1] - sortDir[2] * sortCop[2]);
  const float     divider = 1.0f / sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);

  const auto splatCount = m_positions->size() / 3;

  // prepare an array of pair <distance, original index>
  distArray.resize(splatCount);

  // Sequential version of compute distances
#if defined(SEQUENTIAL) || !defined(_WIN32)
  for(int i = 0; i < splatCount; ++i)
  {
    const auto pos = &(m_splatSet.positions[i * 3]);
    // distance to plane
    const float dist    = std::abs(plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3]) * divider;
    distArray[i].first  = dist;
    distArray[i].second = i;
  }
#else
  // parallel for, compute distances
  auto& tmpArray  = distArray;
  auto& positions = *m_positions;
  std::for_each(std::execution::par_unseq, tmpArray.begin(), tmpArray.end(),
                //concurrency::parallel_for_each(distArray.begin(), distArray.end(),
                [&tmpArray, &positions, &plane, &divider](std::pair<float, int> const& val) {
                  size_t     i   = &val - &tmpArray[0];
                  const auto pos = &(positions[i * 3]);
                  // distance to plane
                  const float dist = std::abs(plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3]) * divider;
                  tmpArray[i].first  = dist;
                  tmpArray[i].second = i;
                });
#endif

  auto time1 = std::chrono::high_resolution_clock::now();
  m_distTime = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - startTime).count();

  // comparison function working on the data <dist,idex>
  auto compare = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };

// Sorting the array with respect to distance keys
#if defined(SEQUENTIAL) || !defined(_WIN32)
  std::sort(distArray.begin(), distArray.end(), compare);
#else
  std::sort(std::execution::par_unseq, distArray.begin(), distArray.end(), compare);
#endif

  // create the sorted index array
  gsIndex.resize(splatCount);
  for(int i = 0; i < splatCount; ++i)
  {
    gsIndex[i] = distArray[i].second;
  }

  auto time2 = std::chrono::high_resolution_clock::now();
  m_sortTime = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();

  return true;
}
