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

#include "image_compare.h"
#include <string>

// Forward declarations for ImGui types
struct ImVec2;
struct ImDrawList;

namespace vk_gaussian_splatting {

// Forward declaration
class ImageCompare;

//==============================================================================
// ImageCompareUI: Handles all UI overlay rendering for ImageCompare
//==============================================================================
class ImageCompareUI
{
public:
  // Constructor takes pointer to the ImageCompare instance to visualize
  explicit ImageCompareUI(ImageCompare* imageCompare);

  ~ImageCompareUI() = default;

  // ============================================================================
  // UI: Overlay Rendering
  // ============================================================================

  // Render overlay UI on viewport (combo boxes, sliders, buttons, metrics window, split divider)
  // Returns true if capture was requested (via camera button)
  bool renderOverlay(ImVec2 imagePos, ImVec2 imageSize, bool viewportClicked, bool viewportScrolled);

  // Update display titles (called by main application before renderOverlay)
  void setCaptureViewTitle(const std::string& title);
  void setCurrentViewTitle(const std::string& title);

  // Temporal sampling status (used for displaying "since enabled" info in the metrics window)
  void setTemporalSamplingState(bool active, int frameSampleId, int frameSampleMax);

  // Query UI state - true if user is dragging the split divider (disables camera)
  bool isDraggingSplitDivider() const { return m_isDraggingSplitDivider; }

private:
  // ============================================================================
  // UI: Internal Drawing Methods
  // ============================================================================

  void drawMetricsToggle(ImVec2 imagePos, ImVec2 imageSize);
  void drawMetricsDisplay(ImVec2 imagePos, ImVec2 imageSize, ImDrawList* drawList);
  void renderMetricsCharts(ImVec2 fixedSize);  // Render charts in subplots (3 stacked plots: MSE, PSNR, FLIP)
  void drawOverlayWidgets(ImVec2 imagePos, ImVec2 imageSize, ImDrawList* drawList, bool& captureRequested);
  void drawComparisonSide(float posMultiplier, ImageCompare::Mode& mode, const char* comboId, ImVec2 imagePos, ImVec2 imageSize, bool& captureRequested);
  void drawBottomInfo(ImVec2 imagePos, ImVec2 imageSize, ImDrawList* drawList);
  void handleSplitDivider(ImVec2 imagePos, ImVec2 imageSize, bool viewportClicked, bool viewportScrolled);
  bool isMouseOverWidgets(ImVec2 imagePos, ImVec2 imageSize);

  // ============================================================================
  // Member Variables
  // ============================================================================

  ImageCompare* m_imageCompare = nullptr;  // Pointer to the core comparison instance (not owned)

  // UI State
  std::string m_captureViewTitle;                // Display title for reference view
  std::string m_currentViewTitle;                // Display title for current view
  bool        m_isDraggingSplitDivider = false;  // User is dragging split divider
  bool        m_showBars               = false;  // Toggle between line charts and bar charts
  bool        m_showDetailedMetrics    = false;  // Toggle between simple and detailed metrics view

  // Temporal sampling tracking for UI display
  bool   m_temporalSamplingActive              = false;
  bool   m_hasTemporalSamplingStart            = false;
  double m_temporalSamplingStartTimeSec        = 0.0;
  int    m_temporalSamplingStartDisplayFrame   = 0;
  int    m_temporalSamplingCurrentDisplayFrame = 0;
  int    m_temporalSamplingFrameMax            = 0;
  bool   m_temporalSamplingFrozen              = false;
  double m_temporalSamplingFrozenElapsedMs     = 0.0;
  int    m_temporalSamplingFrozenFrameCount    = 0;
};

}  // namespace vk_gaussian_splatting
