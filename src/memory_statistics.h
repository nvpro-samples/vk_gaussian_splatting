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

#pragma once

#include <cstdint>

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// Memory statistics for model data (per splat set)
//
struct ModelMemoryStats
{
  // Memory footprint on source memory (RAM)
  uint64_t hostAll       = 0;  // RAM bytes used for all the data of source model
  uint64_t hostCenters   = 0;  // RAM bytes used for splat centers of source model
  uint64_t hostScales    = 0;  // RAM bytes used for scales
  uint64_t hostRotations = 0;  // RAM bytes used for rotations
  uint64_t hostCov       = 0;  // RAM bytes used for covariances
  uint64_t hostShAll     = 0;  // RAM bytes used for all the SH coefs of source model
  uint64_t hostSh0       = 0;  // RAM bytes used for SH degree 0 of source model
  uint64_t hostShOther   = 0;  // RAM bytes used for SH degree 1,2,3 of source model

  // Memory footprint on device memory (allocated)
  uint64_t deviceAllocAll       = 0;  // GRAM bytes used for all the data of source model
  uint64_t deviceAllocCenters   = 0;  // GRAM bytes used for splat centers of source model
  uint64_t deviceAllocScales    = 0;  // GRAM bytes used for scales
  uint64_t deviceAllocRotations = 0;  // GRAM bytes used for rotations
  uint64_t deviceAllocCov       = 0;  // GRAM bytes used for covariances
  uint64_t deviceAllocShAll     = 0;  // GRAM bytes used for all the SH coefs of source model
  uint64_t deviceAllocSh0       = 0;  // GRAM bytes used for SH degree 0 of source model
  uint64_t deviceAllocShOther   = 0;  // GRAM bytes used for SH degree 1,2,3 of source model

  // Actual data size within textures (a.k.a. mem footprint minus padding and eventual unused components)
  uint64_t deviceUsedAll       = 0;  // GRAM bytes used for all the data of source model
  uint64_t deviceUsedCenters   = 0;  // GRAM bytes used for splat centers of source model
  uint64_t deviceUsedScales    = 0;  // GRAM bytes used for scales
  uint64_t deviceUsedRotations = 0;  // GRAM bytes used for rotations
  uint64_t deviceUsedCov       = 0;  // GRAM bytes used for covariances
  uint64_t deviceUsedShAll     = 0;  // GRAM bytes used for all the SH coefs of source model
  uint64_t deviceUsedSh0       = 0;  // GRAM bytes used for SH degree 0 of source model
  uint64_t deviceUsedShOther   = 0;  // GRAM bytes used for SH degree 1,2,3 of source model

  // Index tables (for multi-instance support)
  uint64_t globalIndexTableBuffer   = 0;  // Global splat index table
  uint64_t splatSetIndexTableBuffer = 0;  // Splat set global index table
  uint64_t descriptorBuffer         = 0;  // GPU descriptor array (SplatSetDesc)

  // RTX splat model buffers (per splat set, tracked locally, accumulated to memRaytracing)
  uint64_t rtxVertexBuffer = 0;  // Vertex buffer for splat geometry (triangles/AABBs)
  uint64_t rtxIndexBuffer  = 0;  // Index buffer for splat geometry
  uint64_t rtxAabbBuffer   = 0;  // AABB buffer for splat geometry

  // Subtotals (calculated by updateMemoryStatistics())
  uint64_t hostTotal        = 0;  // Total host memory (hostAll)
  uint64_t deviceUsedTotal  = 0;  // Total device memory used (deviceUsedAll + index tables + descriptor)
  uint64_t deviceAllocTotal = 0;  // Total device memory allocated (deviceAllocAll + index tables + descriptor)
};

//--------------------------------------------------------------------------------------------------
// Memory statistics for rasterization (sorting + indirect draw)
//
struct RasterizationMemoryStats
{
  // Sorting buffers
  uint64_t hostAllocDistances      = 0;  // Host buffer for distances (allocated = used)
  uint64_t hostAllocIndices        = 0;  // Host buffer for indices (allocated = used)
  uint64_t deviceAllocIndices      = 0;  // Device indices buffer (allocated)
  uint64_t DeviceUsedIndices       = 0;  // Device indices buffer (used)
  uint64_t deviceAllocDistances    = 0;  // Device distances buffer (allocated)
  uint64_t deviceUsedDistances     = 0;  // Device distances buffer (used)
  uint64_t deviceAllocVrdxInternal = 0;  // VRDX internal buffer (allocated = used)

  // Indirect draw parameters
  uint64_t usedIndirect = 0;  // Indirect parameters buffer (used = alloc)

  // Subtotals
  uint64_t hostTotal        = 0;
  uint64_t deviceUsedTotal  = 0;
  uint64_t deviceAllocTotal = 0;
};

//--------------------------------------------------------------------------------------------------
// Memory statistics for ray tracing
//
struct RaytracingMemoryStats
{
  // Acceleration structures
  uint64_t usedTlas = 0;  // TLAS (used = alloc)
  uint64_t usedBlas = 0;  // BLAS (used = alloc)

  // TLAS support buffers
  uint64_t tlasAddressBuffer = 0;  // TLAS device addresses buffer
  uint64_t tlasOffsetBuffer  = 0;  // TLAS instance offsets buffer

  // Scratch and working buffers
  uint64_t blasScratchBuffer    = 0;  // Scratch buffer for BLAS building
  uint64_t tlasInstancesBuffers = 0;  // Instance buffers (sum of all TLAS)
  uint64_t tlasScratchBuffers   = 0;  // Scratch buffers for TLAS building (sum of all TLAS)

  // RTX splat model buffers (per splat set)
  uint64_t vertexBuffer      = 0;  // Vertex buffer for splat geometry (triangles/AABBs)
  uint64_t indexBuffer       = 0;  // Index buffer for splat geometry
  uint64_t aabbBuffer        = 0;  // AABB buffer for splat geometry
  uint64_t vertexBufferAlloc = 0;  // Vertex buffer (allocated)
  uint64_t indexBufferAlloc  = 0;  // Index buffer (allocated)
  uint64_t aabbBufferAlloc   = 0;  // AABB buffer (allocated)

  // Subtotals
  uint64_t hostTotal        = 0;
  uint64_t deviceUsedTotal  = 0;
  uint64_t deviceAllocTotal = 0;
};

//--------------------------------------------------------------------------------------------------
// Memory statistics for renderer commons
//
struct RenderMemoryStats
{
  // Renderer commons
  uint64_t usedUboFrameInfo = 0;  // Frame info UBO (used = alloc)
  uint64_t quadVertices     = 0;  // Quad vertices buffer
  uint64_t quadIndices      = 0;  // Quad indices buffer

  // GBuffers (main rendering targets)
  uint64_t gBuffersColor = 0;  // All color attachments (MAIN + AUX1 + COMPARISON_OUTPUT)
  uint64_t gBuffersDepth = 0;  // Depth attachment

  // Helper GBuffers (for helper overlay rendering)
  uint64_t helperGBuffersColor = 0;  // Helper color attachment
  uint64_t helperGBuffersDepth = 0;  // Helper depth attachment

  // Subtotals
  uint64_t hostTotal        = 0;
  uint64_t deviceUsedTotal  = 0;
  uint64_t deviceAllocTotal = 0;
};

//--------------------------------------------------------------------------------------------------
// Grand totals across all sections
//
struct TotalMemoryStats
{
  uint64_t hostTotal        = 0;  // Total host memory used
  uint64_t deviceUsedTotal  = 0;  // Total device memory used
  uint64_t deviceAllocTotal = 0;  // Total device memory allocated
};

//--------------------------------------------------------------------------------------------------
// Global memory statistics instances
//
extern ModelMemoryStats         memModels;         // Consolidated model data statistics (sum of all splat sets)
extern RasterizationMemoryStats memRasterization;  // Rasterization (sorting + indirect draw) statistics
extern RaytracingMemoryStats    memRaytracing;     // Ray tracing statistics
extern RenderMemoryStats        memRender;         // Renderer commons (UBO, Quad, GBuffers)
extern TotalMemoryStats         memTotal;          // Grand totals across all sections

//--------------------------------------------------------------------------------------------------
// Statistics calculation and UI functions
//
void updateMemoryStatistics();                   // Calculate and update all memory statistics (call after data changes)
void guiDrawMemoryStatisticsWindow(bool* show);  // Display memory statistics in ImGui window

}  // namespace vk_gaussian_splatting
