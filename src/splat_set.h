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

#ifndef _SPLAT_SET_H_
#define _SPLAT_SET_H_

#include <vector>

// Storage for a 3D gaussian splatting (3DGS) model loaded from PLY file
struct SplatSet
{
  // standard poiont cloud attributes
  std::vector<float> positions; // point positions (x,y,z)
  // specific data fields introduced by INRIA for 3DGS
  std::vector<float> f_dc;      // 3 components per point (f_dc_0, f_dc_1, f_dc_2 in ply file)
  std::vector<float> f_rest;    // 45 components per point (f_rest_0 to f_rest_44 in ply file), SH coeficients
  std::vector<float> opacity;   // 1 value per point in ply file
  std::vector<float> scale;     // 3 components per point in ply file
  std::vector<float> rotation;  // 4 components per point in ply file - a quaternion

  // returns the number of splate in the set
  inline size_t size() const { return positions.size() / 3; }
};

#endif
