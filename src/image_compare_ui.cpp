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

#include "image_compare_ui.h"

// ImGui includes
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvgui/IconsMaterialSymbols.h>
#include <nvgui/property_editor.hpp>

#include <algorithm>
#include <cmath>

namespace vk_gaussian_splatting {

//==============================================================================
// UI: Display Names
//==============================================================================

// Static table of display names indexed by Mode enum value
static const char* MODE_DISPLAY_NAMES[] = {
    "Frame Capture",             // Mode::eCapture = 0
    "Current Render",            // Mode::eCurrent = 1
    "Difference (Raw)",          // Mode::eDifferenceRaw = 2
    "Difference (Red on Gray)",  // Mode::eDifferenceRedGray = 3
    "Difference (Red Only)",     // Mode::eDifferenceRedOnly = 4
    "FLIP Error Map"             // Mode::eFLIPError = 5
};

static const char* getModeDisplayName(ImageCompare::Mode mode)
{
  int index = static_cast<int>(mode);

  // Bounds check for safety
  if(index >= 0 && index < (int)(sizeof(MODE_DISPLAY_NAMES) / sizeof(MODE_DISPLAY_NAMES[0])))
  {
    return MODE_DISPLAY_NAMES[index];
  }

  return "Unknown";
}

//==============================================================================
// Constructor
//==============================================================================

ImageCompareUI::ImageCompareUI(ImageCompare* imageCompare)
    : m_imageCompare(imageCompare)
{
}

//==============================================================================
// UI: Public Methods
//==============================================================================

void ImageCompareUI::setCaptureViewTitle(const std::string& title)
{
  m_captureViewTitle = title;
}

void ImageCompareUI::setCurrentViewTitle(const std::string& title)
{
  m_currentViewTitle = title;
}

void ImageCompareUI::setTemporalSamplingState(bool active, int frameSampleId, int frameSampleMax)
{
  // Clamp inputs to sane ranges
  frameSampleMax = std::max(1, frameSampleMax);
  frameSampleId  = std::max(0, frameSampleId);

  // Displayed progress is 1-based and clamped to max (matches SPP progress bar behavior)
  const int displayFrame                = std::min(frameSampleId + 1, frameSampleMax);
  const int prevDisplayFrame            = m_temporalSamplingCurrentDisplayFrame;
  m_temporalSamplingCurrentDisplayFrame = displayFrame;

  // Track rising edge (and handle "already active on first call")
  if(active && (!m_temporalSamplingActive || !m_hasTemporalSamplingStart))
  {
    m_temporalSamplingStartTimeSec      = ImGui::GetTime();
    m_temporalSamplingStartDisplayFrame = displayFrame;
    m_hasTemporalSamplingStart          = true;
    m_temporalSamplingFrozen            = false;
    m_temporalSamplingFrozenElapsedMs   = 0.0;
    m_temporalSamplingFrozenFrameCount  = 0;
  }
  // If temporal sampling stays enabled but accumulation restarts (frame counter reset),
  // restart the timing/counting session as well.
  else if(active && m_hasTemporalSamplingStart && prevDisplayFrame > 0 && displayFrame < prevDisplayFrame)
  {
    m_temporalSamplingStartTimeSec      = ImGui::GetTime();
    m_temporalSamplingStartDisplayFrame = displayFrame;
    m_temporalSamplingFrozen            = false;
    m_temporalSamplingFrozenElapsedMs   = 0.0;
    m_temporalSamplingFrozenFrameCount  = 0;
  }
  else if(!active)
  {
    // Reset so next enable restarts the counters
    m_hasTemporalSamplingStart = false;
    m_temporalSamplingFrozen   = false;
  }

  m_temporalSamplingActive   = active;
  m_temporalSamplingFrameMax = frameSampleMax;

  // Freeze the UI counters once we reach the target sample count.
  // This mirrors the behavior where metrics/charts stop updating when temporal sampling converges.
  if(m_temporalSamplingActive && m_hasTemporalSamplingStart && !m_temporalSamplingFrozen && displayFrame >= m_temporalSamplingFrameMax)
  {
    m_temporalSamplingFrozenElapsedMs = (ImGui::GetTime() - m_temporalSamplingStartTimeSec) * 1000.0;
    // Inclusive count: if we start at 1 and reach 200, that's 200 frames.
    m_temporalSamplingFrozenFrameCount = displayFrame - m_temporalSamplingStartDisplayFrame + 1;
    m_temporalSamplingFrozen           = true;
  }
}

bool ImageCompareUI::renderOverlay(ImVec2 imagePos, ImVec2 imageSize, bool viewportClicked, bool viewportScrolled)
{
  if(!m_imageCompare->m_params->enabled || !m_imageCompare->hasValidCaptureImage())
    return false;

  bool captureRequested = false;

  ImDrawList* drawList = ImGui::GetWindowDrawList();

  // Draw metrics toggle button at top left
  drawMetricsToggle(imagePos, imageSize);

  // Draw overlay widgets (combos, sliders, buttons)
  drawOverlayWidgets(imagePos, imageSize, drawList, captureRequested);

  // Draw metrics display if enabled
  if(m_imageCompare->m_params->computeMetrics && m_imageCompare->hasMetricsResult())
  {
    drawMetricsDisplay(imagePos, imageSize, drawList);
  }

  // Draw info text at bottom
  drawBottomInfo(imagePos, imageSize, drawList);

  // Handle split divider dragging
  handleSplitDivider(imagePos, imageSize, viewportClicked, viewportScrolled);

  return captureRequested;
}

//==============================================================================
// UI: Internal Drawing Methods
//==============================================================================

void ImageCompareUI::drawMetricsToggle(ImVec2 imagePos, ImVec2 imageSize)
{
  const float buttonSize = 32.0f;
  const float margin     = 10.0f;
  ImVec2      buttonPos  = ImVec2(imagePos.x + margin, imagePos.y + margin);

  ImGui::SetCursorScreenPos(buttonPos);

  // Style the button with semi-transparent background
  ImU32 bgColor    = m_imageCompare->m_params->computeMetrics ? IM_COL32(60, 120, 60, 200) : IM_COL32(0, 0, 0, 180);
  ImU32 hoverColor = m_imageCompare->m_params->computeMetrics ? IM_COL32(80, 160, 80, 230) : IM_COL32(50, 100, 50, 210);
  ImU32 activeColor = IM_COL32(100, 200, 100, 250);

  ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoverColor);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, activeColor);

  // Use analytics icon for metrics
  if(ImGui::Button(ICON_MS_DIFFERENCE "##MetricsToggle", ImVec2(buttonSize, buttonSize)))
  {
    m_imageCompare->m_params->computeMetrics = !m_imageCompare->m_params->computeMetrics;
  }

  ImGui::PopStyleColor(3);

  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip(
        "Continuously computes Mean Squared Error (MSE)\n"
        "and Peak Signal-to-Noise Ratio (PSNR)\n"
        "between the capture and current images in real-time");
  }
}

void ImageCompareUI::drawMetricsDisplay(ImVec2 imagePos, ImVec2 imageSize, ImDrawList* drawList)
{
  const float leftMargin        = 15.0f;
  const float metricsButtonSize = 32.0f;
  const float metricsMargin     = 10.0f;
  const float spacing           = 10.0f;
  const float padding           = 8.0f;

  // Position window just below the top-left metrics toggle button
  float  windowPosY = imagePos.y + metricsMargin + metricsButtonSize + spacing;
  ImVec2 windowPos  = ImVec2(imagePos.x + leftMargin, windowPosY);

  // === STEP 1: Calculate FIXED window size (with viewport clamping) ===
  const float windowPaddingPerSide   = 10.0f;
  const float totalVerticalPadding   = windowPaddingPerSide * 2.0f;
  const float totalHorizontalPadding = windowPaddingPerSide * 2.0f;

  // Calculate IDEAL (fixed) window dimensions based on mode
  const float buttonSize    = 32.0f;
  const char* titleText     = "Capture vs Current";
  ImVec2      titleTextSize = ImGui::CalcTextSize(titleText);

  float idealWindowHeight = titleTextSize.y + spacing + buttonSize + spacing + totalVerticalPadding;
  float idealWindowWidth;

  if(!m_showDetailedMetrics)
  {
    // Simple mode: fixed dimensions for text display
    // Width: button + combo + margins
    const float comboWidth = 160.0f;
    idealWindowWidth       = buttonSize + spacing + comboWidth + padding * 2 + totalHorizontalPadding;

    // Height: space for 3 metric lines
    char mseText[64];
    snprintf(mseText, sizeof(mseText), "MSE:  0.000000");
    ImVec2 metricTextSize = ImGui::CalcTextSize(mseText);
    idealWindowHeight += (metricTextSize.y + 6.0f) * 3 + padding * 2;
  }
  else
  {
    // Detailed mode: fixed dimensions for charts
    const float idealChartsWidth  = 400.0f;  // Fixed ideal width for charts
    const float idealChartsHeight = 500.0f;  // Fixed ideal height for charts

    // Width: charts + show bars button + margins
    idealWindowWidth = idealChartsWidth + buttonSize + spacing + totalHorizontalPadding + 20.0f;

    // Height: charts area
    if(m_imageCompare->m_historySize > 1)
    {
      idealWindowHeight += idealChartsHeight;
    }
  }

  // Clamp window dimensions to viewport (leaving margins)
  const float maxWindowHeight = imageSize.y - 40.0f;  // Leave 40px margin
  const float maxWindowWidth  = imageSize.x - 40.0f;  // Leave 40px margin
  float       windowHeight    = std::min(idealWindowHeight, maxWindowHeight);
  float       windowWidth     = std::min(idealWindowWidth, maxWindowWidth);

  // Set window position and size
  ImGui::SetNextWindowPos(windowPos, ImGuiCond_FirstUseEver);  // Only set position on first use
  ImGui::SetNextWindowSize(ImVec2(windowWidth, windowHeight), ImGuiCond_Always);

  // Set window background alpha for semi-transparency
  ImGui::SetNextWindowBgAlpha(0.75f);  // 75% opaque = 25% transparent

  // Set window padding for proper margins on all sides (MUST be before Begin)
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f));

  // Window flags: no resize, no collapse, no docking (but allow move)
  // Allow scrollbar if content doesn't fit due to viewport clamping
  ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking;

  // Begin window with title
  if(!ImGui::Begin("Capture vs Current", nullptr, windowFlags))
  {
    ImGui::PopStyleVar();  // Pop even on early exit
    ImGui::End();
    return;
  }

  // Clamp window position to keep ENTIRE window (including title bar) within viewport bounds
  // BUT: Only clamp AFTER the drag is complete (not during the drag)
  if(!ImGui::IsMouseDragging(ImGuiMouseButton_Left))
  {
    ImVec2 currentWindowPos  = ImGui::GetWindowPos();
    ImVec2 currentWindowSize = ImGui::GetWindowSize();

    // Get the full window rect including title bar and decorations
    ImGuiWindow* window         = ImGui::GetCurrentWindow();
    ImVec2       windowMin      = window->Rect().Min;  // Top-left including title bar
    ImVec2       windowMax      = window->Rect().Max;  // Bottom-right including borders
    ImVec2       windowFullSize = ImVec2(windowMax.x - windowMin.x, windowMax.y - windowMin.y);

    // Calculate viewport bounds with small margin
    const float margin = 5.0f;
    float       minX   = imagePos.x + margin;
    float       minY   = imagePos.y + margin;
    float       maxX   = imagePos.x + imageSize.x - windowFullSize.x - margin;
    float       maxY   = imagePos.y + imageSize.y - windowFullSize.y - margin;

    // Clamp position (handle case where window is larger than viewport)
    float clampedX = std::clamp(windowMin.x, minX, std::max(minX, maxX));
    float clampedY = std::clamp(windowMin.y, minY, std::max(minY, maxY));

    // Apply clamped position if it changed (adjust for difference between window pos and rect min)
    ImVec2 offset = ImVec2(currentWindowPos.x - windowMin.x, currentWindowPos.y - windowMin.y);
    if(windowMin.x != clampedX || windowMin.y != clampedY)
    {
      ImGui::SetWindowPos(ImVec2(clampedX + offset.x, clampedY + offset.y));
    }
  }

  // Scale font slightly larger for better readability
  ImGui::SetWindowFontScale(1.1f);

  // === STEP 2: Calculate inner content sizes relative to WINDOW ===
  // Get available content region (window size minus padding)
  ImVec2 availRegion = ImGui::GetContentRegionAvail();

  // Button and combo sizes relative to available window space
  const float innerButtonSize = 32.0f;
  const float innerComboWidth = 160.0f;

  // === Always visible controls: Toggle button + FLIP mode combo + Show Bars button ===
  const char* toggleText = m_showDetailedMetrics ? ICON_MS_SHORT_TEXT : ICON_MS_MONITORING;
  if(ImGui::Button(toggleText, ImVec2(innerButtonSize, innerButtonSize)))
  {
    m_showDetailedMetrics = !m_showDetailedMetrics;
  }

  // FLIP mode combo box (same line)
  ImGui::SameLine();
  ImGui::SetNextItemWidth(innerComboWidth);
  const char* flipModeNames[] = {"FLIP: Off", "FLIP: Approx", "FLIP: Ref"};
  int         currentMode     = static_cast<int>(m_imageCompare->m_params->flipMode);

  if(ImGui::Combo("##FLIPModeOverlay", &currentMode, flipModeNames, IM_ARRAYSIZE(flipModeNames)))
  {
    m_imageCompare->m_params->flipMode = static_cast<ImageCompare::FLIPMode>(currentMode);
  }

  // Show Bars button (same line, right of FLIP combo) - only visible in detailed mode
  if(m_showDetailedMetrics)
  {
    ImGui::SameLine();
    if(ImGui::Button(m_showBars ? ICON_MS_BAR_CHART_OFF : ICON_MS_BAR_CHART, ImVec2(innerButtonSize, innerButtonSize)))
    {
      m_showBars = !m_showBars;
    }

    // When temporal sampling is active, show elapsed time + frame count since activation.
    if(m_temporalSamplingActive && m_hasTemporalSamplingStart)
    {
      const double elapsedMs  = m_temporalSamplingFrozen ? m_temporalSamplingFrozenElapsedMs :
                                                           (ImGui::GetTime() - m_temporalSamplingStartTimeSec) * 1000.0;
      const int    frameCount = m_temporalSamplingFrozen ?
                                    m_temporalSamplingFrozenFrameCount :
                                    std::max(1, m_temporalSamplingCurrentDisplayFrame - m_temporalSamplingStartDisplayFrame + 1);

      ImGui::SameLine(0.0f, 12.0f);
      ImGui::AlignTextToFramePadding();
      ImGui::Text("%.1f ms | %d frames", elapsedMs, frameCount);
    }
  }

  ImGui::Spacing();
  // === Conditional rendering based on toggle ===
  if(!m_showDetailedMetrics)
  {
    // Simple metrics display using PropertyEditor for consistent UI style
    namespace PE = nvgui::PropertyEditor;

    if(PE::begin("##MetricsTable"))
    {
      // Display metrics as read-only text entries with default colors
      char mseText[64];
      char psnrText[64];
      char flipText[64];
      snprintf(mseText, sizeof(mseText), "%.6f", m_imageCompare->getMSE());
      snprintf(psnrText, sizeof(psnrText), "%.2f dB", m_imageCompare->getPSNR());
      snprintf(flipText, sizeof(flipText), "%.4f", m_imageCompare->getFLIP());

      PE::Text("MSE", mseText);
      PE::Text("PSNR", psnrText);
      PE::Text("FLIP", flipText);

      PE::end();
    }
  }
  else
  {
    // Detailed charts view
    if(m_imageCompare->m_historySize > 1)
    {
      // Chart height is fixed, but width uses available window space
      const float idealChartsHeight = 500.0f;

      // Get available width from the window content region
      ImVec2 availRegion = ImGui::GetContentRegionAvail();
      float  chartsWidth = availRegion.x;  // Use full available width

      // Render charts directly - let window scrollbar appear if needed
      renderMetricsCharts(ImVec2(chartsWidth, idealChartsHeight));
    }
    else
    {
      ImGui::TextDisabled("Insufficient history data for charts");
    }
  }

  // Pop window padding style
  ImGui::PopStyleVar();

  ImGui::SetWindowFontScale(1.0f);
  ImGui::End();
}

void ImageCompareUI::renderMetricsCharts(ImVec2 fixedSize)
{
  // Render charts directly in current ImGui context (no window wrapper)
  // Caller is responsible for checking if metrics are enabled and valid
  // Show Bars button is now in the main window controls

  // Use the provided fixed size for chart dimensions
  const float graphWidth  = fixedSize.x;
  const float totalHeight = fixedSize.y;
  const float plotSpacing = ImGui::GetStyle().ItemSpacing.y;
  // Total spacing = 2 gaps between 3 plots
  const float miniGraphHeight = (totalHeight - plotSpacing * 2.0f) / 3.0f;

  // Get history data (direct access via friend class)
  const std::vector<float>& mseHistory   = m_imageCompare->m_mseHistory;
  const std::vector<float>& psnrHistory  = m_imageCompare->m_psnrHistory;
  const std::vector<float>& flipHistory  = m_imageCompare->m_flipHistory;
  int                       historySize  = m_imageCompare->m_historySize;
  int                       historyIndex = m_imageCompare->m_historyIndex;
  int                       sampleCount  = m_imageCompare->m_historySampleCount;

  if(historySize < 2 || sampleCount < 2)
    return;

  // Prepare data arrays (oldest → newest, aligned to right edge)
  std::vector<float> mseData(historySize);
  std::vector<float> psnrData(historySize);
  std::vector<float> flipData(historySize);

  auto round1 = [](float v) { return std::round(v * 10.0f) / 10.0f; };

  if(sampleCount < historySize)
  {
    // Buffer filling: align data to RIGHT edge
    int offset = historySize - sampleCount;
    for(int i = 0; i < historySize; i++)
    {
      if(i < offset)
      {
        mseData[i]  = mseHistory[0];
        psnrData[i] = round1(psnrHistory[0]);
        flipData[i] = flipHistory[0];
      }
      else
      {
        int dataIdx = i - offset;
        mseData[i]  = mseHistory[dataIdx];
        psnrData[i] = round1(psnrHistory[dataIdx]);
        flipData[i] = flipHistory[dataIdx];
      }
    }
  }
  else
  {
    // Buffer full: circular read
    for(int i = 0; i < historySize; i++)
    {
      int idx     = (historyIndex + i) % historySize;
      mseData[i]  = mseHistory[idx];
      psnrData[i] = round1(psnrHistory[idx]);
      flipData[i] = flipHistory[idx];
    }
  }

  // Calculate adaptive Y-axis ranges (with temporal smoothing like ElemProfiler)
  static float s_maxMSE           = 0.0f;
  static float s_minPSNR          = 0.0f;
  static float s_maxPSNR          = 0.0f;
  static float s_maxFLIP          = 0.0f;
  const float  TEMPORAL_SMOOTHING = 20.0f;

  // Find current min/max in visible data
  float curMaxMSE  = *std::max_element(mseData.begin(), mseData.end());
  float curMinPSNR = *std::min_element(psnrData.begin(), psnrData.end());
  float curMaxPSNR = *std::max_element(psnrData.begin(), psnrData.end());
  float curMaxFLIP = *std::max_element(flipData.begin(), flipData.end());

  // Apply temporal smoothing for MSE (starts from 0)
  if(s_maxMSE == 0.0f)
    s_maxMSE = curMaxMSE;
  else
    s_maxMSE = (TEMPORAL_SMOOTHING * s_maxMSE + curMaxMSE) / (TEMPORAL_SMOOTHING + 1.0f);

  // Apply temporal smoothing for PSNR (adaptive min/max)
  if(s_minPSNR == 0.0f)
    s_minPSNR = curMinPSNR;
  else
    s_minPSNR = (TEMPORAL_SMOOTHING * s_minPSNR + curMinPSNR) / (TEMPORAL_SMOOTHING + 1.0f);

  if(s_maxPSNR == 0.0f)
    s_maxPSNR = curMaxPSNR;
  else
    s_maxPSNR = (TEMPORAL_SMOOTHING * s_maxPSNR + curMaxPSNR) / (TEMPORAL_SMOOTHING + 1.0f);

  // Apply temporal smoothing for FLIP (starts from 0)
  if(s_maxFLIP == 0.0f)
    s_maxFLIP = curMaxFLIP;
  else
    s_maxFLIP = (TEMPORAL_SMOOTHING * s_maxFLIP + curMaxFLIP) / (TEMPORAL_SMOOTHING + 1.0f);

  // Get current values for titles
  float currentMSE  = m_imageCompare->getMSE();
  float currentPSNR = round1(m_imageCompare->getPSNR());
  float currentFLIP = m_imageCompare->getFLIP();

  // ImPlot setup - common flags for all plots
  static const ImPlotFlags plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs | ImPlotFlags_NoLegend;
  static const ImPlotAxisFlags xAxisFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels;
  static const ImPlotAxisFlags yAxisFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_Opposite;

  // Make chart backgrounds fully transparent
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));  // Fully transparent background
  //ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));  // Transparent plot background
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));  // Transparent frame background

  // Use BeginSubplots to automatically align all plot areas (Y-axis labels will have consistent width)
  // LinkCols ensures all plots in the column have the same width (aligned Y-axis labels)
  static const ImPlotSubplotFlags subplotFlags = ImPlotSubplotFlags_NoTitle | ImPlotSubplotFlags_NoResize | ImPlotSubplotFlags_LinkCols;

  if(ImPlot::BeginSubplots("##MetricsSubplots", 3, 1, ImVec2(graphWidth, totalHeight), subplotFlags, nullptr, nullptr))
  {
    // Plot 1: MSE (red) - top
    char mseTitleBuf[128];
    snprintf(mseTitleBuf, sizeof(mseTitleBuf), "MSE: %.6f", currentMSE);
    if(ImPlot::BeginPlot(mseTitleBuf, ImVec2(-1, -1), plotFlags))
    {
      if(m_showBars)
      {
        // Bar mode - sliding bars like line mode
        ImPlot::SetupAxis(ImAxis_X1, nullptr, xAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, historySize, ImPlotCond_Always);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, yAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, s_maxMSE * 1.2f, ImPlotCond_Always);

        ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.4f, 0.4f, 0.8f));
        ImPlot::PlotBars("MSE", mseData.data(), historySize, 1.0);  // Bar width = 1.0
      }
      else
      {
        // Line mode
        ImPlot::SetupAxis(ImAxis_X1, nullptr, xAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, historySize, ImPlotCond_Always);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, yAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, s_maxMSE * 1.2f, ImPlotCond_Always);

        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), 2.0f);
        ImPlot::PlotLine("MSE", mseData.data(), historySize);
      }

      ImPlot::EndPlot();
    }

    // Plot 2: PSNR (green) - middle
    char psnrTitleBuf[128];
    snprintf(psnrTitleBuf, sizeof(psnrTitleBuf), "PSNR: %.2f dB", currentPSNR);
    if(ImPlot::BeginPlot(psnrTitleBuf, ImVec2(-1, -1), plotFlags))
    {
      if(m_showBars)
      {
        // Bar mode - sliding bars like line mode
        ImPlot::SetupAxis(ImAxis_X1, nullptr, xAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, historySize, ImPlotCond_Always);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, yAxisFlags);

        // Adaptive range for PSNR (uses min and max with 10% padding)
        float psnrRange   = s_maxPSNR - s_minPSNR;
        float psnrPadding = psnrRange * 0.1f;
        ImPlot::SetupAxisLimits(ImAxis_Y1, s_minPSNR - psnrPadding, s_maxPSNR + psnrPadding, ImPlotCond_Always);

        ImPlot::SetNextFillStyle(ImVec4(0.4f, 1.0f, 0.4f, 0.8f));
        ImPlot::PlotBars("PSNR", psnrData.data(), historySize, 1.0);  // Bar width = 1.0
      }
      else
      {
        // Line mode
        ImPlot::SetupAxis(ImAxis_X1, nullptr, xAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, historySize, ImPlotCond_Always);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, yAxisFlags);

        // Adaptive range for PSNR (uses min and max with 10% padding)
        float psnrRange   = s_maxPSNR - s_minPSNR;
        float psnrPadding = psnrRange * 0.1f;
        ImPlot::SetupAxisLimits(ImAxis_Y1, s_minPSNR - psnrPadding, s_maxPSNR + psnrPadding, ImPlotCond_Always);

        ImPlot::SetNextLineStyle(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), 2.0f);
        ImPlot::PlotLine("PSNR", psnrData.data(), historySize);
      }

      ImPlot::EndPlot();
    }

    // Plot 3: FLIP (yellow) - bottom
    char flipTitleBuf[128];
    snprintf(flipTitleBuf, sizeof(flipTitleBuf), "FLIP: %.4f", currentFLIP);
    if(ImPlot::BeginPlot(flipTitleBuf, ImVec2(-1, -1), plotFlags))
    {
      if(m_showBars)
      {
        // Bar mode - sliding bars like line mode
        ImPlot::SetupAxis(ImAxis_X1, nullptr, xAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, historySize, ImPlotCond_Always);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, yAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, s_maxFLIP * 1.2f, ImPlotCond_Always);

        ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.8f, 0.4f, 0.8f));
        ImPlot::PlotBars("FLIP", flipData.data(), historySize, 1.0);  // Bar width = 1.0
      }
      else
      {
        // Line mode
        ImPlot::SetupAxis(ImAxis_X1, nullptr, xAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, historySize, ImPlotCond_Always);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, yAxisFlags);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, s_maxFLIP * 1.2f, ImPlotCond_Always);

        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), 2.0f);
        ImPlot::PlotLine("FLIP", flipData.data(), historySize);
      }

      ImPlot::EndPlot();
    }

    ImPlot::EndSubplots();
  }

  // Pop transparent background styles
  ImPlot::PopStyleColor(1);  // Pop 1 ImPlot colors (FrameBg)
  ImGui::PopStyleColor(1);   // Pop 1 ImGui color (FrameBg)
}

void ImageCompareUI::drawOverlayWidgets(ImVec2 imagePos, ImVec2 imageSize, ImDrawList* drawList, bool& captureRequested)
{
  // Calculate divider position for overlay
  float splitPosX = imagePos.x + imageSize.x * m_imageCompare->m_params->splitPosition;

  // Draw left and right combo boxes
  drawComparisonSide(0.25f, m_imageCompare->m_params->leftSide, "##LeftOverlay", imagePos, imageSize, captureRequested);
  drawComparisonSide(0.75f, m_imageCompare->m_params->rightSide, "##RightOverlay", imagePos, imageSize, captureRequested);

  // Draw split position percentage
  char splitText[32];
  snprintf(splitText, sizeof(splitText), "%.0f%%", m_imageCompare->m_params->splitPosition * 100.0f);
  ImVec2 splitTextSize = ImGui::CalcTextSize(splitText);
  ImVec2 splitTextPos  = ImVec2(splitPosX - splitTextSize.x * 0.5f, imagePos.y + imageSize.y - 30.0f);

  drawList->AddRectFilled(ImVec2(splitTextPos.x - 5.0f, splitTextPos.y - 2.0f),
                          ImVec2(splitTextPos.x + splitTextSize.x + 5.0f, splitTextPos.y + splitTextSize.y + 2.0f),
                          IM_COL32(0, 0, 0, 180));
  drawList->AddText(splitTextPos, IM_COL32(255, 255, 255, 255), splitText);
}

void ImageCompareUI::drawComparisonSide(float posMultiplier, ImageCompare::Mode& mode, const char* comboId, ImVec2 imagePos, ImVec2 imageSize, bool& captureRequested)
{
  const float comboWidth = 260.0f;
  ImVec2      comboPos   = ImVec2(imagePos.x + imageSize.x * posMultiplier - comboWidth * 0.5f, imagePos.y + 10.0f);

  ImGui::SetCursorScreenPos(comboPos);

  // Push semi-transparent styling with green hover
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
  ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.3f, 0.6f, 0.3f, 0.8f));
  ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.4f, 0.8f, 0.4f, 0.9f));
  ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.1f, 0.1f, 0.1f, 0.95f));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 4.0f));

  ImGui::PushItemWidth(comboWidth);

  // Get current mode name and prepend arrow icon
  const char* modeName = getModeDisplayName(mode);
  char        currentModeWithArrow[128];
  snprintf(currentModeWithArrow, sizeof(currentModeWithArrow), ICON_MS_KEYBOARD_ARROW_DOWN " %s", modeName);

  // Use BeginCombo with arrow in preview, no default arrow button
  if(ImGui::BeginCombo(comboId, currentModeWithArrow, ImGuiComboFlags_NoArrowButton))
  {
    // Enumerate all available modes
    for(int i = 0; i < (int)(sizeof(MODE_DISPLAY_NAMES) / sizeof(MODE_DISPLAY_NAMES[0])); i++)
    {
      ImageCompare::Mode modeValue  = static_cast<ImageCompare::Mode>(i);
      bool               isSelected = ((int)mode == i);
      if(ImGui::Selectable(MODE_DISPLAY_NAMES[i], isSelected))
      {
        mode = modeValue;
      }
      if(isSelected)
      {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  ImGui::PopItemWidth();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(4);

  // Add camera capture button to the right of combo box when showing current render
  if((int)mode != (int)ImageCompare::Mode::eCapture)
  {
    // Match combo box height for perfect alignment
    const float buttonSize = ImGui::GetFrameHeight() + 2.0f;
    ImVec2      buttonPos  = ImVec2(comboPos.x + comboWidth + 5.0f, comboPos.y);

    ImGui::SetCursorScreenPos(buttonPos);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.3f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.8f, 0.4f, 0.9f));

    // Create unique button label with icon and ID (icon##uniqueID format)
    char buttonLabel[64];
    snprintf(buttonLabel, sizeof(buttonLabel), ICON_MS_ADD_A_PHOTO "%s", comboId);
    if(ImGui::Button(buttonLabel, ImVec2(buttonSize, buttonSize)))
    {
      // Signal capture request to main application
      captureRequested = true;
    }
    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Capture current frame");
    }

    ImGui::PopStyleColor(3);
  }

  // Add amplification slider below combo box for difference modes only (not for FLIP - it's already perceptually calibrated)
  bool isDifferenceMode = ((int)mode == (int)ImageCompare::Mode::eDifferenceRaw || (int)mode == (int)ImageCompare::Mode::eDifferenceRedGray
                           || (int)mode == (int)ImageCompare::Mode::eDifferenceRedOnly);

  if(isDifferenceMode)
  {
    const float sliderWidth = comboWidth - 20.0f;  // Slightly narrower than combo
    const float comboHeight = ImGui::GetFrameHeight() + 2.0f;
    ImVec2      sliderPos   = ImVec2(comboPos.x + (comboWidth - sliderWidth) * 0.5f, comboPos.y + comboHeight + 10.0f);

    ImGui::SetCursorScreenPos(sliderPos);
    ImGui::PushItemWidth(sliderWidth);

    // Apply slider styling to match overlay aesthetic
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.3f, 0.6f, 0.3f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.4f, 0.8f, 0.4f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(0.4f, 0.8f, 0.4f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 3.0f));

    // Create unique slider ID by appending "Slider" to combo ID
    char sliderId[64];
    snprintf(sliderId, sizeof(sliderId), "%sSlider", comboId);
    ImGui::SliderFloat(sliderId, &m_imageCompare->m_params->differenceAmplify, 1.0f, 128.0f, "Amplify: %.1fx", ImGuiSliderFlags_None);

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(5);
    ImGui::PopItemWidth();
  }
}

void ImageCompareUI::drawBottomInfo(ImVec2 imagePos, ImVec2 imageSize, ImDrawList* drawList)
{
  ImGui::SetWindowFontScale(0.85f);

  const float comboWidth  = 260.0f;
  const float baseBottomY = imagePos.y + imageSize.y - 30.0f;  // Base position at bottom

  // Lambda to draw info text at bottom for each side
  auto drawBottomInfoSide = [&](float posMultiplier, ImageCompare::Mode mode) {
    ImVec2 centerPos = ImVec2(imagePos.x + imageSize.x * posMultiplier, baseBottomY);

    bool isDifferenceMode = ((int)mode == (int)ImageCompare::Mode::eDifferenceRaw || (int)mode == (int)ImageCompare::Mode::eDifferenceRedGray
                             || (int)mode == (int)ImageCompare::Mode::eDifferenceRedOnly);
    bool isFLIPMode = ((int)mode == (int)ImageCompare::Mode::eFLIPError);

    if((int)mode == (int)ImageCompare::Mode::eCapture)
    {
      // Single line: capture title
      ImVec2 textSize = ImGui::CalcTextSize(m_captureViewTitle.c_str());
      ImVec2 textPos  = ImVec2(centerPos.x - textSize.x * 0.5f, centerPos.y);

      // Draw background
      drawList->AddRectFilled(ImVec2(textPos.x - 5.0f, textPos.y - 2.0f),
                              ImVec2(textPos.x + textSize.x + 5.0f, textPos.y + textSize.y + 2.0f), IM_COL32(0, 0, 0, 180));
      // Draw text
      drawList->AddText(textPos, IM_COL32(255, 200, 0, 255), m_captureViewTitle.c_str());
    }
    else if((int)mode == (int)ImageCompare::Mode::eCurrent)
    {
      // Single line: current title
      ImVec2 textSize = ImGui::CalcTextSize(m_currentViewTitle.c_str());
      ImVec2 textPos  = ImVec2(centerPos.x - textSize.x * 0.5f, centerPos.y);

      // Draw background
      drawList->AddRectFilled(ImVec2(textPos.x - 5.0f, textPos.y - 2.0f),
                              ImVec2(textPos.x + textSize.x + 5.0f, textPos.y + textSize.y + 2.0f), IM_COL32(0, 0, 0, 180));
      // Draw text
      drawList->AddText(textPos, IM_COL32(255, 200, 0, 255), m_currentViewTitle.c_str());
    }
    else  // Difference modes and FLIP mode
    {
      // Two lines - center each line
      std::string refInfo = "Capture: " + m_captureViewTitle;
      std::string curInfo = "Current: " + m_currentViewTitle;

      float lineHeight = ImGui::GetTextLineHeight();

      ImVec2 refTextSize = ImGui::CalcTextSize(refInfo.c_str());
      ImVec2 refTextPos  = ImVec2(centerPos.x - refTextSize.x * 0.5f, centerPos.y - lineHeight - 3.0f);

      // Draw background for reference line
      drawList->AddRectFilled(ImVec2(refTextPos.x - 5.0f, refTextPos.y - 2.0f),
                              ImVec2(refTextPos.x + refTextSize.x + 5.0f, refTextPos.y + refTextSize.y + 2.0f),
                              IM_COL32(0, 0, 0, 180));
      // Draw reference text
      drawList->AddText(refTextPos, IM_COL32(255, 200, 0, 255), refInfo.c_str());

      ImVec2 curTextSize = ImGui::CalcTextSize(curInfo.c_str());
      ImVec2 curTextPos  = ImVec2(centerPos.x - curTextSize.x * 0.5f, centerPos.y);

      // Draw background for current line
      drawList->AddRectFilled(ImVec2(curTextPos.x - 5.0f, curTextPos.y - 2.0f),
                              ImVec2(curTextPos.x + curTextSize.x + 5.0f, curTextPos.y + curTextSize.y + 2.0f),
                              IM_COL32(0, 0, 0, 180));
      // Draw current text
      drawList->AddText(curTextPos, IM_COL32(255, 200, 0, 255), curInfo.c_str());
    }
  };

  // Draw bottom info for left and right sides
  drawBottomInfoSide(0.25f, m_imageCompare->m_params->leftSide);
  drawBottomInfoSide(0.75f, m_imageCompare->m_params->rightSide);

  ImGui::SetWindowFontScale(1.0f);
}

void ImageCompareUI::handleSplitDivider(ImVec2 imagePos, ImVec2 imageSize, bool viewportClicked, bool viewportScrolled)
{
  // Calculate divider position in screen space
  ImVec2 mousePos  = ImGui::GetMousePos();
  float  splitPosX = imagePos.x + imageSize.x * m_imageCompare->m_params->splitPosition;

  // Check if mouse is hovering near the divider
  float dividerThreshold = 10.0f;  // Pixels for hover detection
  bool  isNearDivider    = (mousePos.x >= splitPosX - dividerThreshold && mousePos.x <= splitPosX + dividerThreshold
                        && mousePos.y >= imagePos.y && mousePos.y <= imagePos.y + imageSize.y);

  // If dragging, create an invisible button overlay to capture all mouse input
  if(m_isDraggingSplitDivider)
  {
    ImGui::SetCursorScreenPos(imagePos);
    ImGui::InvisibleButton("##divider_drag_overlay", imageSize);

    // Update split position based on mouse X position
    float newSplitPos                       = (mousePos.x - imagePos.x) / imageSize.x;
    m_imageCompare->m_params->splitPosition = std::clamp(newSplitPos, 0.0f, 1.0f);

    // Set cursor
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

    // Check if we should stop dragging
    if(!ImGui::IsMouseDown(ImGuiMouseButton_Left))
    {
      m_isDraggingSplitDivider = false;
    }
  }
  else
  {
    // Check if we should start dragging (but not if clicking on a widget)
    if(isNearDivider && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsAnyItemHovered())
    {
      m_isDraggingSplitDivider = true;
    }
    else if(isNearDivider && !ImGui::IsAnyItemHovered())
    {
      // Just hovering near divider (and not over any widget), show resize cursor
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    else if((viewportClicked || viewportScrolled) && !isMouseOverWidgets(imagePos, imageSize)
            && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopup))
    {
      // User clicked or scrolled in viewport outside widget zone - disable comparison mode
      // But not if we're over widgets or if any popup is open
      m_imageCompare->releaseCaptureImage();
      m_imageCompare->m_params->enabled = false;
      m_isDraggingSplitDivider          = false;
      // Note: Logging would require access to logging macros, omitted for now
    }
  }
}

bool ImageCompareUI::isMouseOverWidgets(ImVec2 imagePos, ImVec2 imageSize)
{
  ImVec2      mousePos     = ImGui::GetMousePos();
  const float comboWidth   = 260.0f;
  const float comboHeight  = ImGui::GetFrameHeight() + 2.0f;  // Combo box height
  const float buttonSize   = comboHeight;                     // Camera button size (square)
  const float totalWidth   = comboWidth + 5.0f + buttonSize;  // Include camera button area
  const float sliderHeight = ImGui::GetFrameHeight();         // Slider height

  // Check metrics toggle button at top left
  const float metricsButtonSize = 32.0f;
  const float metricsMargin     = 10.0f;
  ImVec2      metricsButtonPos  = ImVec2(imagePos.x + metricsMargin, imagePos.y + metricsMargin);
  if(mousePos.x >= metricsButtonPos.x && mousePos.x <= metricsButtonPos.x + metricsButtonSize
     && mousePos.y >= metricsButtonPos.y && mousePos.y <= metricsButtonPos.y + metricsButtonSize)
  {
    return true;
  }

  // Check metrics display window (if metrics are active)
  // The window handles its own mouse interactions, we just need to check if mouse is over it
  if(m_imageCompare->m_params->computeMetrics && m_imageCompare->hasMetricsResult())
  {
    // Window will capture mouse automatically via ImGui, but we check for safety
    // This prevents camera movement when interacting with the metrics window
    if(ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
    {
      // Check if it's specifically our metrics window
      ImGuiWindow* window = ImGui::FindWindowByName("Capture vs Current");
      if(window
         && ImGui::IsMouseHoveringRect(window->Pos, ImVec2(window->Pos.x + window->Size.x, window->Pos.y + window->Size.y)))
      {
        return true;
      }
    }
  }

  // Check if any side is showing a difference mode (for slider detection - FLIP has no slider)
  bool isLeftDifference  = (m_imageCompare->m_params->leftSide == ImageCompare::Mode::eDifferenceRaw
                           || m_imageCompare->m_params->leftSide == ImageCompare::Mode::eDifferenceRedGray
                           || m_imageCompare->m_params->leftSide == ImageCompare::Mode::eDifferenceRedOnly);
  bool isRightDifference = (m_imageCompare->m_params->rightSide == ImageCompare::Mode::eDifferenceRaw
                            || m_imageCompare->m_params->rightSide == ImageCompare::Mode::eDifferenceRedGray
                            || m_imageCompare->m_params->rightSide == ImageCompare::Mode::eDifferenceRedOnly);

  // Left combo box area (including camera button when not showing reference)
  ImVec2 leftComboPos = ImVec2(imagePos.x + imageSize.x * 0.25f - comboWidth * 0.5f, imagePos.y + 10.0f);
  // Camera button is shown for all modes except eCapture
  bool  leftHasCameraButton = ((int)m_imageCompare->m_params->leftSide != (int)ImageCompare::Mode::eCapture);
  float leftWidth           = leftHasCameraButton ? totalWidth : comboWidth;
  float leftHeight          = comboHeight;

  // Extend height for difference modes to include slider (slider is at comboHeight + 10.0f spacing)
  if(isLeftDifference)
  {
    leftHeight = comboHeight + 10.0f + sliderHeight;
  }

  if(mousePos.x >= leftComboPos.x && mousePos.x <= leftComboPos.x + leftWidth && mousePos.y >= leftComboPos.y
     && mousePos.y <= leftComboPos.y + leftHeight)
  {
    return true;
  }

  // Right combo box area (including camera button when not showing reference)
  ImVec2 rightComboPos = ImVec2(imagePos.x + imageSize.x * 0.75f - comboWidth * 0.5f, imagePos.y + 10.0f);
  // Camera button is shown for all modes except eCapture
  bool  rightHasCameraButton = ((int)m_imageCompare->m_params->rightSide != (int)ImageCompare::Mode::eCapture);
  float rightWidth           = rightHasCameraButton ? totalWidth : comboWidth;
  float rightHeight          = comboHeight;

  // Extend height for difference modes to include slider (slider is at comboHeight + 10.0f spacing)
  if(isRightDifference)
  {
    rightHeight = comboHeight + 10.0f + sliderHeight;
  }

  if(mousePos.x >= rightComboPos.x && mousePos.x <= rightComboPos.x + rightWidth && mousePos.y >= rightComboPos.y
     && mousePos.y <= rightComboPos.y + rightHeight)
  {
    return true;
  }

  return false;
}

}  // namespace vk_gaussian_splatting
