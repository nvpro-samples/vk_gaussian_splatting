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

#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <thread>
#include <execution>
#include <ranges>

// Example using the parallel loop macro
// constexpr uint32_t N = 100;
// START_PAR_LOOP( N, i)
//   std::cout << "Processing index " << i << "\n";
// END_PAR_LOOP()

#if !defined(_WIN32)

// Macro to start a parallel loop with a thread ID
#define START_PAR_LOOP(SIZE, INDEX)                                                                                      \
  {                                                                                                                      \
    assert(THREAD_COUNT > 0);                                                                                            \
    std::vector<std::thread> _par_threads;                                                                               \
    uint32_t                 _par_start = 0, _par_end = 0;                                                               \
    uint32_t                 _par_num_threads = std::min((uint32_t)std::thread::hardware_concurrency(), (uint32_t)SIZE); \
    uint32_t                 _par_chunk_size  = (SIZE) / _par_num_threads;                                               \
    uint32_t                 _par_remainder   = (SIZE) % _par_num_threads;                                               \
    for(uint32_t _par_tid = 0; _par_tid < _par_num_threads; ++_par_tid)                                                  \
    {                                                                                                                    \
      _par_end = _par_start + _par_chunk_size + (_par_tid < _par_remainder ? 1 : 0);                                     \
        _par_threads.emplace_back([&, _start=_par_start, _end=_par_end]() {                       \
            for (uint32_t INDEX = _start; INDEX < _end; ++INDEX) {

// Macro to end the parallel loop
#define END_PAR_LOOP()                                                                                                 \
  }                                                                                                                    \
  });                                                                                                                  \
  _par_start = _par_end;                                                                                               \
  }                                                                                                                    \
  for(auto& _par_thread : _par_threads)                                                                                \
  {                                                                                                                    \
    if(_par_thread.joinable())                                                                                         \
    {                                                                                                                  \
      _par_thread.join();                                                                                              \
    }                                                                                                                  \
  }                                                                                                                    \
  }

#else

#define START_PAR_LOOP(SIZE, INDEX)                                                                                    \
  {                                                                                                                    \
    auto __range_ = std::views::iota(0, (int)SIZE);                                                                    \
    auto __begin_ = std::ranges::begin(__range_);                                                                      \
    auto __end_   = std::ranges::end(__range_);                                                                        \
    std::for_each(std::execution::par, __begin_, __end_, [&](int INDEX) {

#define END_PAR_LOOP()                                                                                                 \
  });                                                                                                                \
  }

#endif

#endif
