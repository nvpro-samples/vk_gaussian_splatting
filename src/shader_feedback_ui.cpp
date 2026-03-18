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

#include "shader_feedback_ui.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <imgui/imgui.h>
#include <implot/implot.h>

#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

#include <nvgui/fonts.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/tooltip.hpp>

#include "shaderio.h"
#include "parameters.h"
#include "utilities.h"

namespace vk_gaussian_splatting {

// ============================================================================
// Graph type descriptors (static, shared across all instances)
// ============================================================================
struct GraphDesc
{
  const char* label;
  bool        isNormalized;
  bool        defaultShowLines;  // default state for the line plot toggle
  ImU32       color;
};

static const GraphDesc kGraphDescs[ShaderFeedbackUI::kGraphCount] = {
    {"Integrated transmittance R", true, true, IM_COL32(255, 220, 80, 255)},
    {"Integrated transmittance G", true, true, IM_COL32(160, 255, 160, 255)},
    {"Integrated transmittance B", true, true, IM_COL32(80, 120, 255, 255)},
    {"Integrated opacity R", true, true, IM_COL32(255, 160, 50, 255)},
    {"Integrated opacity G", true, true, IM_COL32(120, 220, 120, 255)},
    {"Integrated opacity B", true, true, IM_COL32(100, 160, 255, 255)},
    {"Integrated luminance", false, true, IM_COL32(200, 200, 200, 255)},
    {"Integrated radiance R", false, true, IM_COL32(255, 80, 80, 255)},
    {"Integrated radiance G", false, true, IM_COL32(80, 255, 80, 255)},
    {"Integrated radiance B", false, true, IM_COL32(80, 120, 255, 255)},
    {"Particle hit response R", false, false, IM_COL32(255, 80, 80, 255)},
    {"Particle hit response G", false, false, IM_COL32(80, 255, 80, 255)},
    {"Particle hit response B", false, false, IM_COL32(80, 120, 255, 255)},
    {"Particle hit response Alpha", true, false, IM_COL32(200, 200, 200, 255)},
};

// Extract the Y value for a given graph type from a HitProfile entry
static float extractGraphValue(int graphIdx, const shaderio::HitProfile& h)
{
  switch(graphIdx)
  {
    case 0:
      return h.transmittance.x;
    case 1:
      return h.transmittance.y;
    case 2:
      return h.transmittance.z;
    case 3:
      return 1.0f - h.transmittance.x;
    case 4:
      return 1.0f - h.transmittance.y;
    case 5:
      return 1.0f - h.transmittance.z;
    case 6:
      return 0.2126f * h.integratedRadiance.x + 0.7152f * h.integratedRadiance.y + 0.0722f * h.integratedRadiance.z;
    case 7:
      return h.integratedRadiance.x;
    case 8:
      return h.integratedRadiance.y;
    case 9:
      return h.integratedRadiance.z;
    case 10:
      return h.color.x;
    case 11:
      return h.color.y;
    case 12:
      return h.color.z;
    case 13:
      return h.alpha;
    default:
      return 0.0f;
  }
}

// ============================================================================
// Constructor - start with a single "Transmittance R" graph
// ============================================================================
ShaderFeedbackUI::ShaderFeedbackUI()
{
  m_plots.push_back({0, true, kGraphDescs[0].defaultShowLines, true});  // Transmittance R
}

// ============================================================================
void ShaderFeedbackUI::reset()
{
  m_xs.clear();
  m_ys.clear();
}

// ============================================================================
// Shader Feedback Window
// ============================================================================
void ShaderFeedbackUI::drawWindow(bool& show, bool showCursorTargetOverlay, const shaderio::IndirectParams& readback, bool& requestUpdateShaders)
{
  if(!show)
    return;

  if(ImGui::Begin("Shader feedback", &show))
  {
    namespace PE = nvgui::PropertyEditor;

    if(PE::begin())
    {
      PE::Text(showCursorTargetOverlay ? "Target" : "Mouse", fmt::format("{} {}", prmFrame.cursor.x, prmFrame.cursor.y));
      PE::end();
    }

    if(ImGui::CollapsingHeader("Raygen - particles", ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(PE::begin())
      {
        PE::Text("Splat Id (Global)", fmt::format("{}", readback.particleGlobalId));
        PE::Text("Splat Set Index", fmt::format("{}", readback.splatSetId));
        PE::Text("Local Splat Index", fmt::format("{}", readback.particleId));
        PE::Text("Splat # Hits", fmt::format("{}", readback.particleHitCount));

        PE::Text("Splat Dist closest", formatFloatInf(readback.particleDist));
        PE::Text("Splat Dist Iso surface", fmt::format("{}", readback.particleIntegratedDist));

        glm::vec3 nrm = readback.particleNormal;
        PE::Text("Splat Norm closest", fmt::format("{:.3f} {:.3f} {:.3f}", nrm.x, nrm.y, nrm.z));
        glm::vec3 nrmInt = readback.particleIntegratedNormal;
        PE::Text("Splat Norm integrated", fmt::format("{:.3f} {:.3f} {:.3f}", nrmInt.x, nrmInt.y, nrmInt.z));
        PE::Text("Splat length(Norm int)", fmt::format("{:.4f}", glm::length(nrmInt)));

        float alpha = readback.closestParticleAlpha;
        PE::Text("Splat Alpha closest", fmt::format("{:.3f}", alpha));
        glm::vec3 cpw = readback.closestParticleWeight;
        PE::Text("Splat Weight closest", fmt::format("{:.3f} {:.3f} {:.3f}", cpw.x, cpw.y, cpw.z));
        glm::dvec3 cpt = readback.closestParticleTransmittance;
        PE::Text("Transmittance after closest", fmt::format("{:.5f} {:.5f} {:.5f}", cpt.x, cpt.y, cpt.z));
        glm::dvec3 trnsm = readback.particleTransmittance;
        PE::Text("Transmittance after all", fmt::format("{:.3f} {:.3f} {:.3f}", trnsm.x, trnsm.y, trnsm.z));

        PE::end();
      }
    }

    if(ImGui::CollapsingHeader("Trace profile", ImGuiTreeNodeFlags_DefaultOpen))
    {
      // Toggle drives compile-time macro TRACE_PROFILE (shader rebuild required)
      if(ImGui::Checkbox("Enable##TraceProfileShaderFeedback", &prmRtx.traceProfile))
      {
        requestUpdateShaders = true;
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::SetTooltip("%s", "Recompile shaders with TRACE_PROFILE and collect hit samples.");
      }

      if(prmRtx.traceProfile)
      {
        const uint32_t hitCountTotal = readback.traceProfileHitCount;
        const uint32_t hitCount      = std::min(hitCountTotal, uint32_t(200));

        ImGui::Text("%s", fmt::format("Hits recorded: {} (showing {}){}", hitCountTotal, hitCount,
                                      (hitCountTotal > 200) ? " [overflow]" : "")
                              .c_str());

        if(hitCount > 0)
        {
          // Build shared X-axis data (distance) once per frame
          m_xs.resize(hitCount);
          for(uint32_t i = 0; i < hitCount; ++i)
            m_xs[i] = readback.traceProfileHits[i].dist;

          // Iso-surface distance for red marker
          const double isoX = (double)readback.particleIntegratedDist;

          m_ys.resize(hitCount);

          // --- Draw each active graph node ---
          int deleteIdx = -1;  // deferred deletion (avoid iterator invalidation)

          // Toggle icon helper: green background when active, gray when inactive
          // (matches the menu bar icon button style)
          auto toggleIcon = [](const char* icon, const char* id, bool& flag, const char* tooltip) {
            ImGui::SameLine();
            if(flag)
            {
              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.5f, 0.1f, 1.0f));
            }
            else
            {
              ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.4f, 0.4f, 1.0f));
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
            }
            if(ImGui::SmallButton(fmt::format("{}##{}", icon, id).c_str()))
              flag = !flag;
            ImGui::PopStyleColor(3);
            if(ImGui::IsItemHovered())
              ImGui::SetTooltip("%s", tooltip);
          };

          for(int n = 0; n < (int)m_plots.size(); ++n)
          {
            PlotState&       plot = m_plots[n];
            const GraphDesc& desc = kGraphDescs[plot.graphIdx];

            ImGui::PushID(n);  // unique ID scope per node position

            ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_DefaultOpen;

            bool nodeOpen = ImGui::TreeNodeEx(desc.label, nodeFlags);

            // --- Toggle icons (after name, not right-aligned) ---
            toggleIcon(ICON_MS_BAR_CHART, "bars", plot.showBars, "Toggle bars");
            toggleIcon(ICON_MS_SHOW_CHART, "lines", plot.showLines, "Toggle lines");
            toggleIcon(ICON_MS_LAST_PAGE, "iso", plot.showIso, "Toggle iso-threshold line");

            // Delete button on the same line (right-aligned)
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 10);
            if(ImGui::SmallButton(ICON_MS_DELETE))
            {
              deleteIdx = n;
            }
            if(ImGui::IsItemHovered())
            {
              ImGui::SetTooltip("Remove this graph");
            }

            if(nodeOpen)
            {
              // Extract Y values
              for(uint32_t i = 0; i < hitCount; ++i)
                m_ys[i] = extractGraphValue(plot.graphIdx, readback.traceProfileHits[i]);

              const std::string plotId = fmt::format("##TraceProfile_{}", n);
              if(ImPlot::BeginPlot(plotId.c_str(), ImVec2(-1, 150), ImPlotFlags_NoLegend | ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText))
              {
                ImPlot::SetupAxes("Distance", nullptr, ImPlotAxisFlags_AutoFit, 0);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Once);

                // Iso-surface depth marker (red vertical line) - plotted first so it appears behind the data
                if(plot.showIso && isoX > 0.0 && isoX < DBL_MAX)
                {
                  const float isoXVal = (float)isoX;
                  ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.24f, 0.24f, 1.0f));
                  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                  ImPlot::PlotInfLines("##iso", &isoXVal, 1);
                  ImPlot::PopStyleVar();
                  ImPlot::PopStyleColor();
                }

                const ImVec4 plotColor = ImGui::ColorConvertU32ToFloat4(desc.color);

                if(plot.showBars)
                {
                  ImPlot::PushStyleColor(ImPlotCol_Line, plotColor);
                  ImPlot::PlotStems("##hits", m_xs.data(), m_ys.data(), (int)hitCount, 0.0);
                  ImPlot::PopStyleColor();
                }

                if(plot.showLines)
                {
                  ImPlot::PushStyleColor(ImPlotCol_Line, plotColor);
                  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
                  ImPlot::PlotLine("##line", m_xs.data(), m_ys.data(), (int)hitCount);
                  ImPlot::PopStyleVar();
                  ImPlot::PopStyleColor();
                }

                // Hover tooltip: find nearest sample by X distance and show its value
                if(ImPlot::IsPlotHovered())
                {
                  ImPlotPoint mouse   = ImPlot::GetPlotMousePos();
                  float       bestDx  = FLT_MAX;
                  int         bestIdx = -1;
                  for(uint32_t i = 0; i < hitCount; ++i)
                  {
                    float dx = std::abs(m_xs[i] - (float)mouse.x);
                    if(dx < bestDx)
                    {
                      bestDx  = dx;
                      bestIdx = (int)i;
                    }
                  }
                  if(bestIdx >= 0)
                  {
                    ImGui::BeginTooltip();
                    ImGui::Text("Hit #%d", bestIdx);
                    ImGui::Text("dist = %.5f", m_xs[bestIdx]);
                    ImGui::Text("%s = %.5f", desc.label, m_ys[bestIdx]);
                    ImGui::EndTooltip();
                  }
                }

                ImPlot::EndPlot();
              }

              ImGui::TreePop();
            }

            ImGui::PopID();
          }

          // Apply deferred deletion
          if(deleteIdx >= 0)
          {
            m_plots.erase(m_plots.begin() + deleteIdx);
          }

          if(ImGui::SmallButton("Add plot"))
          {
            ImGui::OpenPopup("##AddPlotPopup");
          }
          if(ImGui::BeginPopup("##AddPlotPopup"))
          {
            for(int g = 0; g < kGraphCount; ++g)
            {
              if(ImGui::MenuItem(kGraphDescs[g].label))
              {
                m_plots.push_back({g, true, kGraphDescs[g].defaultShowLines, true});
              }
            }
            ImGui::EndPopup();
          }
        }
        else
        {
          ImGui::TextUnformatted("No hits recorded (move cursor or target overlay\nover splats and ensure RTX is active).");
        }
      }
    }
#ifndef NDEBUG
    if(ImGui::CollapsingHeader("Debugging"))
    {
      if(PE::begin())
      {
        PE::Text("val1", std::to_string(readback.val1));
        PE::Text("val2", std::to_string(readback.val2));
        PE::Text("val3", std::to_string(readback.val3));
        PE::Text("val4", std::to_string(readback.val4));
        PE::Text("val5", std::to_string(readback.val5));
        PE::Text("val6", std::to_string(readback.val6));
        PE::Text("val7", std::to_string(readback.val7));

        PE::end();
      }
    }
#endif
  }
  ImGui::End();
}

}  // namespace vk_gaussian_splatting
