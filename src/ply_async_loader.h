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

#ifndef _PLY_ASYNC_LOADER_H_
#define _PLY_ASYNC_LOADER_H_

#include <string>
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
//
#include "splat_set.h"

//
class PlyAsyncLoader
{
public:
  enum State
  {
    E_SHUTDOWN,  // loader must be initialized (loading thread is not started)
    E_READY,     // loader ready to load a new model
    E_LOADING,   // loader is currently loading
    E_LOADED,    // loader has finished loading, model is available. call reset before another load.
    E_FAILURE    // an error eccured. call reset before another load.
  };

public:
  // starts the loader thread
  bool initialize();
  // stops the loader thread, cannot be re-used afterward
  inline void shutdown()
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_shutdownRequested = true;
    m_loadCV.notify_all();
    lock.unlock();
    // wait for thread termination
    m_loader.join();
  }
  // triggers the load of a new scene
  // return false if loader not in idled state
  // output must not be accessed if status is not LOADED or READY (after reset)
  bool loadScene(std::string filename, SplatSet& output);
  // cancel scene loading if possible
  // non blocking, may have no effect
  void cancel();
  // return loader status
  State getStatus();
  // Resets the loader to READY after LOADED or FAILURE
  // used to ack that the consumer has consumed the loaded model
  // loader must be reset to be able to launch a new load
  // thread safe
  bool reset();
  // return the filename currently beeing loaded, "" otherwise
  [[nodiscard]] inline std::string getFilename()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_filename;
  }
  // return percentage in {0,1} of progress
  // do not rely on progress to find loader status
  // use getStatus(). progress is just an indication
  // for UI display.
  [[nodiscard]] inline float getProgress()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_progress;
  }

private:
  // actually loads the scene
  bool innerLoad(std::string filename, SplatSet& output);

  // in {0.0,1.0}
  void setProgress(float progress)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_progress = progress;
  }

private:
  // loading thread
  std::thread m_loader;
  // loader status
  State m_status = E_SHUTDOWN;
  // ask to cancel a load
  bool m_cancelRequested = false;
  // ask for loader shutdown before destruction
  bool m_shutdownRequested = false;
  // protects the condition variables and other attributes
  mutable std::mutex m_mutex;
  // loader wakeup condition
  mutable std::condition_variable m_loadCV;

  // the ply pathname
  std::string m_filename = "";
  // the output data storage
  SplatSet* m_output = nullptr;
  // the loading percentage
  float m_progress = 0.0f;
};

#endif