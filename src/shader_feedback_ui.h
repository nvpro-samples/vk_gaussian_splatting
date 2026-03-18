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

#include <vector>

// Forward declare the shared data types (avoid pulling in heavy headers)
namespace shaderio {
struct IndirectParams;
}

namespace vk_gaussian_splatting {

//==============================================================================
// ShaderFeedbackUI: Handles the "Shader feedback" window
//==============================================================================
class ShaderFeedbackUI
{
public:
  ShaderFeedbackUI();

  // Draw the "Shader feedback" window.
  //  show                    - window visibility flag (read/write by ImGui::Begin)
  //  showCursorTargetOverlay - true when the persistent target overlay is active
  //  readback                - GPU readback data (IndirectParams)
  //  requestUpdateShaders    - set to true when a shader rebuild is needed
  void drawWindow(bool& show, bool showCursorTargetOverlay, const shaderio::IndirectParams& readback, bool& requestUpdateShaders);

  // Clear cached data (call on scene reset / load)
  void reset();

  // Number of available graph types
  static constexpr int kGraphCount = 14;

private:
  // Per-plot display state
  struct PlotState
  {
    int  graphIdx  = 0;      // index into kGraphDescs
    bool showBars  = true;   // show stem/bar plot
    bool showLines = false;  // show connected line plot
    bool showIso   = true;   // show iso-threshold vertical line
  };

  std::vector<PlotState> m_plots;  // active plots with display state

  std::vector<float> m_xs;  // shared X values (distance)
  std::vector<float> m_ys;  // per-graph Y values (reused)
};

}  // namespace vk_gaussian_splatting
