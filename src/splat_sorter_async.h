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

#ifndef _SPLAT_SORTER_ASYNC_H_
#define _SPLAT_SORTER_ASYNC_H_

#include <string>
// for parallel processing
#ifdef _WIN32
#include <ppl.h>
#include <execution>
#include <algorithm>
#endif
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
//
#include <glm/vec3.hpp>
//
#include "splat_set.h"

//
class SplatSorterAsync
{
public:
  enum Status
  {
    SHUTDOWN,  // must be initialized (loading thread is not started)
    READY,     // ready to sort a set of points, call startSorting
    SORTING,   // currently sorting the point set
    SORTED,    // the result of a sort is available, consume then call reset before another load
    FAILURE    // an error eccured. call reset before another load.
  };

public:
  // starts the loader thread
  bool initialize();
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
  Status getStatus()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_status;
  }
  // triggers a new sort, only if viewpoint 
  // position/orientation did change since last run
  // return false if sorter not in idled state or if camera did not move
  // positions must not be accessed while sorting
  inline bool sortAsync(const glm::vec3& camDir, const glm::vec3& camCop, std::vector<float>& positions)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if(m_status != READY)
    {
      return false;
    }
    if(sortDir == camDir && sortCop == camCop)
    {
      return false;
    }
    sortDir          = camDir;
    sortCop          = camCop;
    m_startRequested = true;
    m_positions = &positions;
    // wakeup the thread
    m_sortCV.notify_all();

    return true;
  }
  //
  bool reset()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if(m_status == SORTED || m_status == FAILURE)
    {
      m_status   = READY;
      return true;
    }
    else
    {
      return false;
    }
  }

private:

  bool innerSort();

public:

  // those attributues shall not be accessed if status==SORTING
  std::vector<uint32_t> gsIndex;
  float                 m_distTime = 0.0f;  // distance update timer
  float                 m_sortTime = 0.0f;  // distance sorting timer

private:
  Status      m_status = SHUTDOWN;
  std::thread m_sorter;
  // asked for shutdown
  bool m_shutdownRequested = false;
  // asked for start
  bool m_startRequested = false;
  // protects the condition variables and other attributes
  std::mutex m_mutex;
  // loader wakeup condition
  std::condition_variable m_sortCV;
  //
  glm::vec3 sortDir = {1.0f, 0.0f, 0.0f};
  glm::vec3 sortCop = {0.0f, 0.0f, 0.0f};
  std::vector<float>* m_positions = nullptr;

  std::vector<std::pair<float, int>> distArray;  // splat - <dist, index>

};

#endif