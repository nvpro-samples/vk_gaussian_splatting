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

#ifndef _SPLAT_SET_H_
#define _SPLAT_SET_H_

#include <vector>
#include <cassert>

// 3rd party spz library, used here for coordinate system convertions
#include "splat-types.h"

namespace vk_gaussian_splatting {

// Storage for a 3D gaussian splatting (3DGS) model loaded from PLY file
struct SplatSet
{
  // standard poiont cloud attributes
  std::vector<float> positions = {};  // point positions (x,y,z)
  // specific data fields introduced by INRIA for 3DGS
  std::vector<float> f_dc     = {};  // 3 components per point (f_dc_0, f_dc_1, f_dc_2 in ply file)
  std::vector<float> f_rest   = {};  // 45 components per point (f_rest_0 to f_rest_44 in ply file), SH coeficients
  std::vector<float> opacity  = {};  // 1 value per point in ply file
  std::vector<float> scale    = {};  // 3 components per point in ply file
  std::vector<float> rotation = {};  // 4 components per point in ply file - a quaternion

  // returns the number of splats in the set
  inline size_t size() const { return positions.size() / 3; }

  // returns the maximumSH degree of splat in the set
  // returns -1 if splat set is invalid
  inline int32_t maxShDegree() const
  {
    const uint32_t splatCount = size();
    if(splatCount == 0)
      return -1;
    const uint32_t totalSphericalHarmonicsComponentCount    = (uint32_t)f_rest.size() / splatCount;
    const uint32_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    int sphericalHarmonicsDegree = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
    {
      sphericalHarmonicsDegree = 1;
    }
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
    {
      sphericalHarmonicsDegree = 2;
    }
    if(sphericalHarmonicsCoefficientsPerChannel == 15)
    {
      sphericalHarmonicsDegree = 3;
    }
    return sphericalHarmonicsDegree;
  }

  // Convert between two coordinate systems
  // This is performed in-place.
  void convertCoordinates(spz::CoordinateSystem from, spz::CoordinateSystem to)
  {
    spz::CoordinateConverter c = coordinateConverter(from, to);

    const auto numPoints = size();

    for(size_t i = 0; i < positions.size(); i += 3)
    {
      positions[i + 0] *= c.flipP[0];
      positions[i + 1] *= c.flipP[1];
      positions[i + 2] *= c.flipP[2];
    }
    for(size_t i = 0; i < rotation.size(); i += 4)
    {
      // Don't modify the scalar component (index 0)
      rotation[i + 1] *= c.flipQ[0];
      rotation[i + 2] *= c.flipQ[1];
      rotation[i + 3] *= c.flipQ[2];
    }
    // Rotate spherical harmonics by inverting coefficients that reference the y and z axes, for
    // each RGB channel. See spherical_harmonics_kernel_impl.h for spherical harmonics formulas.
    const size_t numCoeffs         = f_rest.size() / 3;
    const size_t numCoeffsPerPoint = numCoeffs / numPoints;
    size_t       idx               = 0;
    for(size_t i = 0; i < numPoints; ++i)
    {
      // Process R, G, and B coefficients for each point
      for(size_t j = 0; j < numCoeffsPerPoint; ++j)
      {
        const auto flip = c.flipSh[j];
        f_rest[idx + j] *= flip;                          // R
        f_rest[idx + numCoeffsPerPoint + j] *= flip;      // G
        f_rest[idx + numCoeffsPerPoint * 2 + j] *= flip;  // B
      }
      idx += 3 * numCoeffsPerPoint;
    }
  }
};

}  // namespace vk_gaussian_splatting

#endif
