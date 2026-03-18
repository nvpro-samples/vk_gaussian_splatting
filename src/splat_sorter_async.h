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

#ifndef _SPLAT_SORTER_ASYNC_H_
#define _SPLAT_SORTER_ASYNC_H_

#include <string>
// threading
#include <thread>
#include <condition_variable>
#include <mutex>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <nvvk/profiler_vk.hpp>

#include "splat_set.h"

namespace vk_gaussian_splatting {

class SplatSorterAsync
{
public:
  enum State
  {
    E_SHUTDOWN,  // must be initialized (loading thread is not started)
    E_READY,     // ready to sort a set of points, call startSorting
    E_SORTING,   // currently sorting the point set
    E_SORTED,    // the result of a sort is available, consume before another load
    E_FAILURE    // an error eccured. call consume before another load.
  };

public:
  // starts the loader thread
  bool initialize(nvutils::ProfilerTimeline*);
  // stops the loader thread, cannot be re-used afterward
  inline void shutdown()
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_shutdownRequested = true;
    m_sortCV.notify_all();
    lock.unlock();
    // wait for thread termination
    m_sorter.join();
  }
  // return loader status
  inline State getStatus()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_status;
  }
  struct InstanceSortInput
  {
    std::vector<float>* positions;
    glm::mat4           transform;
    uint32_t            globalOffset;
    uint32_t            splatCount;
  };

  // triggers a new sort, only if viewpoint's
  // position or orientation did change since last run
  // return false if sorter not in READY state or if camera did not move
  // positions must not be accessed while sorting
  // if lazy is set, a new sort will be started only if viewpoint changed,
  // otherwise a new sort is systematically started if sorter is ready
  // if frontToBack is set, splats are sorted nearest-first instead of farthest-first
  inline bool sortAsync(const glm::vec3&                      camDir,
                        const glm::vec3&                      camCop,
                        const std::vector<InstanceSortInput>& instances,
                        uint32_t                              totalSplatCount,
                        bool                                  lazy        = true,
                        bool                                  frontToBack = false)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if(m_status != E_READY)
    {
      return false;
    }
    if(lazy && m_sortDir == camDir && m_sortCop == camCop && m_frontToBack == frontToBack)
    {
      bool same = (m_instances.size() == instances.size());
      for(size_t i = 0; same && i < instances.size(); ++i)
        same = (m_instances[i].transform == instances[i].transform);
      if(same)
        return false;
    }
    m_sortDir         = camDir;
    m_sortCop         = camCop;
    m_startRequested  = true;
    m_instances       = instances;
    m_totalSplatCount = totalSplatCount;
    m_frontToBack     = frontToBack;
    m_sortCV.notify_all();

    return true;
  }
  // Fill indices with sorted values (call std::swap) and stats
  inline bool consume(std::vector<uint32_t>& indices)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if(m_status == E_SORTED || m_status == E_FAILURE)
    {
      m_status = E_READY;
      indices.swap(m_indices);
      return true;
    }
    else
    {
      return false;
    }
  }
  // Wait for any in-flight sort to complete and clear stale data.
  // Must be called before deleting splat sets whose positions may be referenced.
  inline void waitUntilIdleAndReset()
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_sortCV.wait(lock, [this] { return m_status != E_SORTING; });
    if(m_status == E_SORTED)
      m_status = E_READY;
    m_instances.clear();
    m_totalSplatCount = 0;
    m_indices.clear();
  }

private:
  bool innerSort();

private:
  State       m_status = E_SHUTDOWN;
  std::thread m_sorter;
  // asked for shutdown
  bool m_shutdownRequested = false;
  // asked for start
  bool m_startRequested = false;
  // protects the condition variables and other attributes
  std::mutex m_mutex;
  // corted wakeup condition
  std::condition_variable m_sortCV;
  // profiling utility
  nvutils::ProfilerTimeline* m_profiler;

  // input parameters
  glm::vec3                      m_sortDir         = {0.0f, 0.0f, 0.0f};  // camera direction
  glm::vec3                      m_sortCop         = {0.0f, 0.0f, 0.0f};  // camera position
  std::vector<InstanceSortInput> m_instances;                              // per-instance sort inputs
  uint32_t                       m_totalSplatCount = 0;                    // total splats across all instances
  bool                           m_frontToBack     = false;                // front-to-back sorting order

  std::vector<float> distances;  // points distances, internal buffer

  std::vector<uint32_t> m_indices;       // sorted indices result
  double                m_distTime = 0;  // distance update timer
  double                m_sortTime = 0;  // distance sorting timer
};

}  // namespace vk_gaussian_splatting

#endif