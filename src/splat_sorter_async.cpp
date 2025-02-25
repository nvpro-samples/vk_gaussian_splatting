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

#include "splat_sorter_async.h"
#include "utilities.h"

// for parallel processing
#include <algorithm>
#include <execution>
// mathematics
#include <cmath>
#include <glm/vec4.hpp>

bool SplatSorterAsync::initialize()
{
  // original state shall be shutdown
  std::unique_lock<std::mutex> lock(m_mutex);
  if(m_status != E_SHUTDOWN)
    return false;  // will unlock through lock destructor
  else
    lock.unlock();

  // starts the thread
  m_sorter = std::thread([this]() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_status = E_READY;
    lock.unlock();

    while(true)
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
        m_status = E_SORTING;
        lock.unlock();
        if(innerSort())
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status         = E_SORTED;
          m_startRequested = false;
        }
        else
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status         = E_FAILURE;
          m_startRequested = false;
        }
      }
      else
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status         = E_SHUTDOWN;
        m_startRequested = false;
        return;
      }
    }
  });

  return true;
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
  const glm::vec4 plane(m_sortDir[0], m_sortDir[1], m_sortDir[2],
                        -m_sortDir[0] * m_sortCop[0] - m_sortDir[1] * m_sortCop[1] - m_sortDir[2] * m_sortCop[2]);
  const float     divider = 1.0f / std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);

  const auto splatCount = (uint32_t)m_positions->size() / 3;

  // prepare the arrays (noop if already sized)
  distances.resize(splatCount);
  m_indices.resize(splatCount);

  // compute distances in parallel
  START_PAR_LOOP(distances.size(), splatIdx)
  {
    const auto pos = &((*m_positions)[splatIdx * 3]);
    // distance to plane
    const float dist    = std::abs(plane[0] * pos[0] + plane[1] * pos[1] + plane[2] * pos[2] + plane[3]) * divider;
    distances[splatIdx] = dist;
    m_indices[splatIdx] = (uint32_t)splatIdx;
  }
  END_PAR_LOOP()

  auto time1 = std::chrono::high_resolution_clock::now();
  m_distTime = 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(time1 - startTime).count();

  // comparison function working on the data <dist,idex>
  auto compare = [&](size_t i, size_t j) { return distances[i] > distances[j]; };

  // Sorting the array with respect to distance keys
  std::sort(std::execution::par_unseq, m_indices.begin(), m_indices.end(), compare);

  auto time2 = std::chrono::high_resolution_clock::now();
  m_sortTime = 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();

  return true;
}
