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

#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <nvh/parallel_work.hpp>

// Example using the parallel loop macro
// constexpr uint32_t N = 100;
// START_PAR_LOOP( N, i)
//   std::cout << "Processing index " << i << "\n";
// END_PAR_LOOP()

#define START_PAR_LOOP(SIZE, INDEX)                                                                                    \
  {                                                                                                                    \
    nvh::parallel_batches_indexed<8192>(                                                                               \
        SIZE, [&](int INDEX, int tidx) {

#define END_PAR_LOOP()                                                                                                 \
  }, (uint32_t)std::thread::hardware_concurrency());                                                                   \
  }

#endif
