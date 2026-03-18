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

#include "memory_statistics.h"
#include "parameters.h"
#include "utilities.h"
#include <imgui/imgui.h>
#include <nvgui/IconsMaterialSymbols.h>

namespace vk_gaussian_splatting {

//--------------------------------------------------------------------------------------------------
// Global memory statistics instances
//
ModelMemoryStats         memModels        = {};
RasterizationMemoryStats memRasterization = {};
RaytracingMemoryStats    memRaytracing    = {};
RenderMemoryStats        memRender        = {};
TotalMemoryStats         memTotal         = {};

//--------------------------------------------------------------------------------------------------
// Calculate all memory statistics subtotals and grand totals
// Call this after any data changes (e.g., after processUpdateRequests())
//
void updateMemoryStatistics()
{
  // =========================================================================
  // MODEL DATA SUBTOTALS
  // =========================================================================
  memModels.hostTotal       = memModels.hostAll;
  memModels.deviceUsedTotal = memModels.deviceUsedAll + memModels.globalIndexTableBuffer
                              + memModels.splatSetIndexTableBuffer + memModels.descriptorBuffer;
  memModels.deviceAllocTotal = memModels.deviceAllocAll + memModels.globalIndexTableBuffer
                               + memModels.splatSetIndexTableBuffer + memModels.descriptorBuffer;

  // =========================================================================
  // RASTERIZATION SUBTOTALS
  // =========================================================================
  memRasterization.hostTotal       = memRasterization.hostAllocIndices + memRasterization.hostAllocDistances;
  memRasterization.deviceUsedTotal = memRasterization.usedIndirect + memRasterization.DeviceUsedIndices
                                     + memRasterization.deviceUsedDistances + memRasterization.deviceAllocVrdxInternal;
  memRasterization.deviceAllocTotal = memRasterization.usedIndirect + memRasterization.deviceAllocIndices
                                      + memRasterization.deviceAllocDistances + memRasterization.deviceAllocVrdxInternal;

  // =========================================================================
  // RAY TRACING SUBTOTALS
  // =========================================================================
  memRaytracing.hostTotal       = 0;  // No host-side buffers for RT
  memRaytracing.deviceUsedTotal = memRaytracing.usedTlas + memRaytracing.usedBlas + memRaytracing.vertexBuffer
                                  + memRaytracing.indexBuffer + memRaytracing.aabbBuffer + memRaytracing.tlasAddressBuffer
                                  + memRaytracing.tlasOffsetBuffer + memRaytracing.blasScratchBuffer
                                  + memRaytracing.tlasInstancesBuffers + memRaytracing.tlasScratchBuffers;
  memRaytracing.deviceAllocTotal =
      memRaytracing.usedTlas + memRaytracing.usedBlas + memRaytracing.vertexBufferAlloc + memRaytracing.indexBufferAlloc
      + memRaytracing.aabbBufferAlloc + memRaytracing.tlasAddressBuffer + memRaytracing.tlasOffsetBuffer
      + memRaytracing.blasScratchBuffer + memRaytracing.tlasInstancesBuffers + memRaytracing.tlasScratchBuffers;

  // =========================================================================
  // RENDERER COMMONS SUBTOTALS
  // =========================================================================
  memRender.hostTotal = memRender.usedUboFrameInfo;
  memRender.deviceUsedTotal = memRender.usedUboFrameInfo + memRender.quadVertices + memRender.quadIndices + memRender.gBuffersColor
                              + memRender.gBuffersDepth + memRender.helperGBuffersColor + memRender.helperGBuffersDepth;
  memRender.deviceAllocTotal = memRender.deviceUsedTotal;  // No overallocation for renderer commons

  // =========================================================================
  // GRAND TOTALS (Model Data + Rasterization + Ray Tracing + Renderer Commons)
  // =========================================================================
  memTotal.hostTotal = memModels.hostTotal + memRasterization.hostTotal + memRaytracing.hostTotal + memRender.hostTotal;
  memTotal.deviceUsedTotal = memModels.deviceUsedTotal + memRasterization.deviceUsedTotal
                             + memRaytracing.deviceUsedTotal + memRender.deviceUsedTotal;
  memTotal.deviceAllocTotal = memModels.deviceAllocTotal + memRasterization.deviceAllocTotal
                              + memRaytracing.deviceAllocTotal + memRender.deviceAllocTotal;
}

//--------------------------------------------------------------------------------------------------
// Helper function to draw right-aligned text in current table column
//
static void drawRightAlignedText(const char* text)
{
  float columnWidth = ImGui::GetContentRegionAvail().x;
  float textWidth   = ImGui::CalcTextSize(text).x;
  if(textWidth < columnWidth)
  {
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + columnWidth - textWidth);
  }
  ImGui::Text("%s", text);
}

//--------------------------------------------------------------------------------------------------
// Draw the memory statistics window
//
void guiDrawMemoryStatisticsWindow(bool* show)
{
  if(!show || !*show)
    return;

  if(ImGui::Begin("Memory Statistics", show))
  {
    ImGuiTableFlags commonFlags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_SpanAllColumns;
    ImGuiTableFlags itemFlags   = commonFlags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    ImGuiTableFlags totalFlags  = commonFlags;  // not open by default | ImGuiTreeNodeFlags_DefaultOpen;

    int expandCollapseAction = 0;  // 0=none, 1=expand all, 2=collapse all

    if(ImGui::BeginTable("Scene stats", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Host used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device used", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Device allocated", ImGuiTableColumnFlags_WidthStretch);

      // Custom header row with buttons in first column
      ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
      ImGui::TableSetColumnIndex(0);

      // Expand all button
      if(ImGui::SmallButton(ICON_MS_EXPAND_MORE))
      {
        expandCollapseAction = 1;  // Expand all
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("Expand all sections");
      }

      ImGui::SameLine();

      // Collapse all button
      if(ImGui::SmallButton(ICON_MS_EXPAND_LESS))
      {
        expandCollapseAction = 2;  // Collapse all
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("Collapse all sections");
      }

      // Other column headers
      ImGui::TableSetColumnIndex(1);
      ImGui::TableHeader("Host used");
      ImGui::TableSetColumnIndex(2);
      ImGui::TableHeader("Device used");
      ImGui::TableSetColumnIndex(3);
      ImGui::TableHeader("Device allocated");

      // ===========================================
      // MODEL DATA
      // ===========================================
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      if(expandCollapseAction == 1)
        ImGui::SetNextItemOpen(true);
      else if(expandCollapseAction == 2)
        ImGui::SetNextItemOpen(false);

      bool open = ImGui::TreeNodeEx("Model data", totalFlags);
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memModels.hostTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memModels.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memModels.deviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("Centers", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.hostCenters).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceUsedCenters).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceAllocCenters).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Scales", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.hostScales).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceUsedScales).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceAllocScales).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Rotations", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.hostRotations).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceUsedRotations).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceAllocRotations).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Covariances", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.hostCov).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceUsedCov).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceAllocCov).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH degree 0", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.hostSh0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceUsedSh0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceAllocSh0).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH degree 1,2,3", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.hostShOther).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceUsedShOther).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceAllocShOther).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("SH total", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.hostShAll).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceUsedShAll).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.deviceAllocShAll).c_str());

        // RTX model buffers
        uint64_t totalRtxBuffers = memModels.rtxVertexBuffer + memModels.rtxIndexBuffer + memModels.rtxAabbBuffer;
        if(totalRtxBuffers > 0)
        {
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::TreeNodeEx("RTX Geometry", itemFlags);
          ImGui::TableNextColumn();
          drawRightAlignedText(formatMemorySize(0).c_str());
          ImGui::TableNextColumn();
          drawRightAlignedText(formatMemorySize(totalRtxBuffers).c_str());
          ImGui::TableNextColumn();
          drawRightAlignedText(formatMemorySize(totalRtxBuffers).c_str());
        }

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Global index table", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.globalIndexTableBuffer).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.globalIndexTableBuffer).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Splat set index table", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.splatSetIndexTableBuffer).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.splatSetIndexTableBuffer).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Descriptor buffer", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.descriptorBuffer).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memModels.descriptorBuffer).c_str());

        ImGui::TreePop();
      }

      // ===========================================
      // RASTERIZATION
      // ===========================================
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      if(expandCollapseAction == 1)
        ImGui::SetNextItemOpen(true);
      else if(expandCollapseAction == 2)
        ImGui::SetNextItemOpen(false);

      open = ImGui::TreeNodeEx("Rasterization", totalFlags);
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRasterization.hostTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRasterization.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRasterization.deviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("Indirect params", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.usedIndirect).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.usedIndirect).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Sort distances", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.hostAllocDistances).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.deviceUsedDistances).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.deviceAllocDistances).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Sort indices", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.hostAllocIndices).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.DeviceUsedIndices).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.deviceAllocIndices).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("GPU sort", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.deviceAllocVrdxInternal).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRasterization.deviceAllocVrdxInternal).c_str());

        ImGui::TreePop();
      }

      // ===========================================
      // RAY TRACING
      // ===========================================
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      if(expandCollapseAction == 1)
        ImGui::SetNextItemOpen(true);
      else if(expandCollapseAction == 2)
        ImGui::SetNextItemOpen(false);

      open = ImGui::TreeNodeEx("Ray tracing", totalFlags);
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRaytracing.hostTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRaytracing.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRaytracing.deviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("TLAS", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.usedTlas).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.usedTlas).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("BLAS", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.usedBlas).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.usedBlas).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("RTX Geometry", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(
            formatMemorySize(memRaytracing.vertexBuffer + memRaytracing.indexBuffer + memRaytracing.aabbBuffer).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(
            formatMemorySize(memRaytracing.vertexBufferAlloc + memRaytracing.indexBufferAlloc + memRaytracing.aabbBufferAlloc)
                .c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("TLAS address buffer", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasAddressBuffer).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasAddressBuffer).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("TLAS offset buffer", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasOffsetBuffer).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasOffsetBuffer).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("BLAS scratch buffer", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.blasScratchBuffer).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.blasScratchBuffer).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("TLAS instances buffers", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasInstancesBuffers).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasInstancesBuffers).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("TLAS scratch buffers", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasScratchBuffers).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRaytracing.tlasScratchBuffers).c_str());

        ImGui::TreePop();
      }

      // ===========================================
      // RENDERER COMMONS
      // ===========================================
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      if(expandCollapseAction == 1)
        ImGui::SetNextItemOpen(true);
      else if(expandCollapseAction == 2)
        ImGui::SetNextItemOpen(false);

      open = ImGui::TreeNodeEx("Renderer commons", totalFlags);
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRender.hostTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRender.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memRender.deviceAllocTotal).c_str());
      if(open)
      {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Separator();
        ImGui::TreeNodeEx("UBO frame info", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.usedUboFrameInfo).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.usedUboFrameInfo).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.usedUboFrameInfo).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Quad buffers", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.quadVertices + memRender.quadIndices).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.quadVertices + memRender.quadIndices).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("GBuffers color", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.gBuffersColor).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.gBuffersColor).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("GBuffers depth", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.gBuffersDepth).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.gBuffersDepth).c_str());

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TreeNodeEx("Helper GBuffers", itemFlags);
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(0).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.helperGBuffersColor + memRender.helperGBuffersDepth).c_str());
        ImGui::TableNextColumn();
        drawRightAlignedText(formatMemorySize(memRender.helperGBuffersColor + memRender.helperGBuffersDepth).c_str());

        ImGui::TreePop();
      }

      // ===========================================
      // TOTAL
      // ===========================================
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Separator();
      ImGui::Text("Total");
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memTotal.hostTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memTotal.deviceUsedTotal).c_str());
      ImGui::TableNextColumn();
      drawRightAlignedText(formatMemorySize(memTotal.deviceAllocTotal).c_str());

      ImGui::EndTable();
    }
  }
  ImGui::End();
}

}  // namespace vk_gaussian_splatting
