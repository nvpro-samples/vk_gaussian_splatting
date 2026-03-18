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

#include "splat_sorter_async.h"
#include "utilities.h"

// for parallel processing
#include <algorithm>
#include <execution>
// mathematics
#include <cmath>
#include <glm/vec4.hpp>

using namespace vk_gaussian_splatting;

bool SplatSorterAsync::initialize(nvutils::ProfilerTimeline* profiler)
{
  m_profiler = profiler;

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
          m_sortCV.notify_all();
        }
        else
        {
          std::lock_guard<std::mutex> lock(m_mutex);
          m_status         = E_FAILURE;
          m_startRequested = false;
          m_sortCV.notify_all();
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
  assert(m_profiler);

  if(m_instances.empty())
    return false;

  auto timer = m_profiler->asyncBeginSection("CPU Dist");

  // Distance to camera plane approximation
  // https://mathinsight.org/distance_point_plane
  const glm::vec4 plane(m_sortDir[0], m_sortDir[1], m_sortDir[2],
                        -m_sortDir[0] * m_sortCop[0] - m_sortDir[1] * m_sortCop[1] - m_sortDir[2] * m_sortCop[2]);
  const float     divider = 1.0f / std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);

  distances.resize(m_totalSplatCount);
  m_indices.resize(m_totalSplatCount);

  for(const auto& inst : m_instances)
  {
    if(!inst.positions)
      continue;
    const uint32_t   offset = inst.globalOffset;
    const glm::mat4& xform  = inst.transform;
    const auto&      pos    = *inst.positions;

    START_PAR_LOOP(inst.splatCount, splatIdx)
    {
      const glm::vec4 p = xform * glm::vec4(pos[splatIdx * 3], pos[splatIdx * 3 + 1], pos[splatIdx * 3 + 2], 1.0f);
      const float     dist              = std::abs(plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]) * divider;
      distances[offset + splatIdx]      = dist;
      m_indices[offset + splatIdx]      = offset + (uint32_t)splatIdx;
    }
    END_PAR_LOOP()
  }

  m_profiler->asyncEndSection(timer);

  timer = m_profiler->asyncBeginSection("CPU Sort");

  auto compare = [&](size_t i, size_t j) {
    return m_frontToBack ? (distances[i] < distances[j]) : (distances[i] > distances[j]);
  };

  std::sort(std::execution::par_unseq, m_indices.begin(), m_indices.end(), compare);

  m_profiler->asyncEndSection(timer);

  return true;
}
