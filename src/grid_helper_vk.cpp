/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "grid_helper_vk.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <glm/gtc/matrix_transform.hpp>

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

//-----------------------------------------------------------------------------
// Initialization
//-----------------------------------------------------------------------------

void GridHelperVk::init(const Resources& res)
{
  if(m_initialized)
  {
    deinit();
  }

  m_app                       = res.app;
  m_alloc                     = res.alloc;
  m_uploader                  = res.uploader;
  m_device                    = res.device;
  m_slangCompiler             = res.slangCompiler;
  m_colorFormat               = res.colorFormat;
  m_depthFormat               = res.depthFormat;
  m_helperDescriptorSetLayout = res.helperDescriptorSetLayout;  // Use shared layout

  createPipeline();

  m_initialized = true;
}

void GridHelperVk::deinit()
{
  if(!m_initialized)
    return;

  vkDeviceWaitIdle(m_device);

  // Destroy buffers
  if(m_vertexBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_vertexBuffer);
    m_vertexBuffer = {};
  }
  if(m_indexBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_indexBuffer);
    m_indexBuffer = {};
  }
  m_vertexCapacity    = 0;
  m_indexCapacity     = 0;
  m_geometryGenerated = false;  // Reset so geometry is regenerated on next render

  // Destroy pipeline
  if(m_pipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_pipeline = VK_NULL_HANDLE;
  }
  if(m_pipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_pipelineLayout = VK_NULL_HANDLE;
  }
  // Note: m_helperDescriptorSetLayout is shared/owned by TransformHelper, don't destroy
  m_helperDescriptorSetLayout = VK_NULL_HANDLE;

  // Destroy shaders
  if(m_vertexShader != VK_NULL_HANDLE)
  {
    vkDestroyShaderModule(m_device, m_vertexShader, nullptr);
    m_vertexShader = VK_NULL_HANDLE;
  }
  if(m_fragmentShader != VK_NULL_HANDLE)
  {
    vkDestroyShaderModule(m_device, m_fragmentShader, nullptr);
    m_fragmentShader = VK_NULL_HANDLE;
  }

  m_initialized = false;
}

//-----------------------------------------------------------------------------
// Pipeline Creation
//-----------------------------------------------------------------------------

bool GridHelperVk::compileSlangShader(const char* filename, VkShaderModule& outModule)
{
  if(!m_slangCompiler)
  {
    LOGE("GridHelperVk: No shader compiler provided\n");
    return false;
  }

  // Note: compileFile() will compile all entry points in the file
  // The entry point is specified later in addShader()
  if(!m_slangCompiler->compileFile(filename))
  {
    LOGE("GridHelperVk: Failed to compile shader %s\n", filename);
    return false;
  }

  if(outModule != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_device, outModule, nullptr);

  VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = m_slangCompiler->getSpirvSize(),
                                      .pCode    = m_slangCompiler->getSpirv()};

  if(m_slangCompiler->getSpirvSize() == 0)
  {
    LOGE("GridHelperVk: Missing entry point in shader %s\n", filename);
    return false;
  }

  NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &outModule));
  NVVK_DBG_NAME(outModule);

  return true;
}

void GridHelperVk::createPipeline()
{
  // Compile unified visual helpers shaders
  if(!compileSlangShader("visual_helpers.slang", m_vertexShader))
  {
    LOGE("GridHelperVk: Failed to compile vertex shader\n");
    return;
  }

  if(!compileSlangShader("visual_helpers.slang", m_fragmentShader))
  {
    LOGE("GridHelperVk: Failed to compile fragment shader\n");
    return;
  }

  // Create pipeline layout with push constants and descriptor set
  VkPushConstantRange pushConstantRange = {};
  pushConstantRange.stageFlags          = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset              = 0;
  pushConstantRange.size                = sizeof(PushConstants);

  // Two descriptor set layouts: Set 0 empty, Set 1 for scene depth
  VkDescriptorSetLayoutCreateInfo emptyLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  emptyLayoutInfo.bindingCount                    = 0;
  emptyLayoutInfo.pBindings                       = nullptr;

  VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &emptyLayoutInfo, nullptr, &emptyLayout));

  VkDescriptorSetLayout layouts[2] = {emptyLayout, m_helperDescriptorSetLayout};

  VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount             = 2;
  layoutInfo.pSetLayouts                = layouts;
  layoutInfo.pushConstantRangeCount     = 1;
  layoutInfo.pPushConstantRanges        = &pushConstantRange;

  NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  // Destroy temporary empty layout
  vkDestroyDescriptorSetLayout(m_device, emptyLayout, nullptr);

  // Create graphics pipeline
  nvvk::GraphicsPipelineState pipelineState;

  // Vertex input
  pipelineState.vertexBindings = {{
      .binding = 0,
      .stride  = sizeof(GridVertex),
  }};

  pipelineState.vertexAttributes = {
      {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(GridVertex, position)},
      {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(GridVertex, color)},
  };

  // Line list topology for simple line rendering
  pipelineState.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

  // Rasterization
  pipelineState.rasterizationState.cullMode  = VK_CULL_MODE_NONE;
  pipelineState.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  pipelineState.rasterizationState.lineWidth = 3.0f;  // Line width (requires wideLines feature)

  // Enable smooth line rasterization (anti-aliased lines - Vulkan 1.4 core feature)
  // nvvk::GraphicsPipelineState has rasterizationLineState which gets auto-chained to pNext
  pipelineState.rasterizationLineState.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH;

  // Depth: test enabled, write disabled (for proper alpha blending)
  pipelineState.depthStencilState.depthTestEnable  = VK_TRUE;
  pipelineState.depthStencilState.depthWriteEnable = VK_FALSE;  // No depth write for blended lines
  pipelineState.depthStencilState.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

  // Enable alpha blending for smooth LOD transitions
  pipelineState.colorBlendEnables[0]                       = VK_TRUE;
  pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
  pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;

  // Create pipeline with dynamic rendering
  nvvk::GraphicsPipelineCreator creator;
  creator.pipelineInfo.layout                  = m_pipelineLayout;
  creator.colorFormats                         = {m_colorFormat};
  creator.renderingState.depthAttachmentFormat = m_depthFormat;

  // Add dynamic line width state (for wider lines)
  creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);

  creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertmain", m_vertexShader);
  creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmain", m_fragmentShader);

  creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_pipeline);
  NVVK_DBG_NAME(m_pipeline);
}

//-----------------------------------------------------------------------------
// Grid Scale Calculation
//-----------------------------------------------------------------------------

GridHelperVk::ScaleLevelInfo GridHelperVk::calculateScaleLevels(const glm::vec3& cameraPos)
{
  // Use camera distance to origin to determine scale
  // This is rotation-independent - only changes with zoom
  float distanceToOrigin = glm::length(cameraPos);

  // Prevent log of zero
  if(distanceToOrigin < 0.001f)
    distanceToOrigin = 0.001f;

  // Calculate which scale "decade" we're in using log10
  // Reference distance 3m corresponds to scale 1 (20m grid)
  float logDist = std::log10(distanceToOrigin / 3.0f);

  // Base scale is 10^floor(logDist)
  float baseExponent = std::floor(logDist);
  float baseScale    = std::pow(10.0f, baseExponent);

  // Clamp to reasonable range
  baseScale = std::clamp(baseScale, 0.001f, 100000.0f);

  // Fine scale is one decade smaller, coarse scale is one decade larger
  float fineScale   = baseScale / 10.0f;
  float coarseScale = baseScale * 10.0f;

  // Fractional part tells us where we are within the decade [0, 1)
  float fractional = logDist - baseExponent;

  // Fine grid blend: visible when fractional is low (just zoomed in), fades as we zoom out
  // At fractional=0: fineBlend=1 (fully visible)
  // At fractional=0.5: fineBlend=0 (invisible)
  float fineBlend = std::clamp(1.0f - fractional * 2.0f, 0.0f, 1.0f);

  // Coarse grid blend: appears early when zooming out for better anticipation
  // At fractional=0.2: coarseBlend=0 (invisible)
  // At fractional=0.8: coarseBlend=1 (fully visible)
  // This gives more overlap with the base grid for smoother transitions
  float coarseBlend = std::clamp((fractional - 0.2f) / 0.6f, 0.0f, 1.0f);

  // Apply smoothstep for nicer transitions
  auto smoothstep = [](float t) { return t * t * (3.0f - 2.0f * t); };
  fineBlend       = smoothstep(fineBlend);
  coarseBlend     = smoothstep(coarseBlend);

  ScaleLevelInfo info;
  info.baseScale   = baseScale;
  info.fineScale   = fineScale;
  info.coarseScale = coarseScale;
  info.fineBlend   = fineBlend;
  info.coarseBlend = coarseBlend;

  return info;
}

//-----------------------------------------------------------------------------
// Grid Geometry Generation (generated once at scale=1)
//-----------------------------------------------------------------------------

void GridHelperVk::generateGridGeometry()
{
  m_vertices.clear();
  m_indices.clear();

  // Generate grid at scale=1 (base level)
  // Runtime scaling is done via model matrix in push constants
  // Grid dimensions at base scale:
  // - Grid size: 20m (±10m from center)
  // - Minor lines: every 1m
  // - Major lines: every 5m
  const float gridHalfSize = 10.0f;  // Half of grid (±10m from center)
  const float minorSpacing = 1.0f;   // Minor line spacing

  // Colors (dark for every unit, light for every 5 units)
  glm::vec3 minorColor(0.25f, 0.25f, 0.25f);  // Dark gray for minor lines (every 1 unit)
  glm::vec3 majorColor(0.5f, 0.5f, 0.5f);     // Light gray for major lines (every 5 units)
  glm::vec3 axisColorX(0.9f, 0.2f, 0.2f);     // Red for X axis
  glm::vec3 axisColorY(0.2f, 0.9f, 0.2f);     // Green for Y axis
  glm::vec3 axisColorZ(0.2f, 0.2f, 0.9f);     // Blue for Z axis

  // Helper lambda to add a line segment (2 vertices)
  auto addLine = [this](const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& color) {
    uint32_t baseIndex = static_cast<uint32_t>(m_vertices.size());
    m_vertices.push_back({p0, color});
    m_vertices.push_back({p1, color});
    m_indices.push_back(baseIndex);
    m_indices.push_back(baseIndex + 1);
  };

  // Generate grid lines centered at origin
  // Minor lines every minorSpacing, major lines every majorSpacing
  int numMinorLines = static_cast<int>(gridHalfSize / minorSpacing);

  for(int i = -numMinorLines; i <= numMinorLines; ++i)
  {
    float pos = i * minorSpacing;

    // Skip origin (will be drawn as colored axis)
    if(i == 0)
      continue;

    // Check if this is a major line (every 5 units at base scale)
    bool      isMajor = (i % 5 == 0);
    glm::vec3 color   = isMajor ? majorColor : minorColor;

    // X-parallel line (at Z = pos)
    addLine(glm::vec3(-gridHalfSize, 0, pos), glm::vec3(gridHalfSize, 0, pos), color);

    // Z-parallel line (at X = pos)
    addLine(glm::vec3(pos, 0, -gridHalfSize), glm::vec3(pos, 0, gridHalfSize), color);
  }

  // Add colored axis lines (positive direction only, starting at origin)
  float axisExtent = gridHalfSize * 1.2f;

  // X axis (red) - positive X direction
  addLine(glm::vec3(0, 0, 0), glm::vec3(axisExtent, 0, 0), axisColorX);

  // Z axis (blue) - positive Z direction
  addLine(glm::vec3(0, 0, 0), glm::vec3(0, 0, axisExtent), axisColorZ);

  // Y axis (green) - positive Y direction (vertical)
  addLine(glm::vec3(0, 0, 0), glm::vec3(0, axisExtent, 0), axisColorY);

  // Add gray lines at origin for negative directions (replacing colored axes)
  // X axis negative (gray)
  addLine(glm::vec3(-gridHalfSize, 0, 0), glm::vec3(0, 0, 0), majorColor);

  // Z axis negative (gray)
  addLine(glm::vec3(0, 0, -gridHalfSize), glm::vec3(0, 0, 0), majorColor);

  m_indexCount        = static_cast<uint32_t>(m_indices.size());
  m_geometryGenerated = true;
}

//-----------------------------------------------------------------------------
// Geometry Upload
//-----------------------------------------------------------------------------

void GridHelperVk::uploadGeometry()
{
  if(m_vertices.empty() || m_indices.empty())
    return;

  VkDeviceSize vertexSize = m_vertices.size() * sizeof(GridVertex);
  VkDeviceSize indexSize  = m_indices.size() * sizeof(uint32_t);

  // Recreate vertex buffer if needed
  if(m_vertexBuffer.buffer == VK_NULL_HANDLE || m_vertexCapacity < m_vertices.size())
  {
    if(m_vertexBuffer.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(m_vertexBuffer);
    }

    m_vertexCapacity = static_cast<uint32_t>(m_vertices.size() * 2);  // Double capacity for growth
    NVVK_CHECK(m_alloc->createBuffer(m_vertexBuffer, m_vertexCapacity * sizeof(GridVertex),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
    NVVK_DBG_NAME(m_vertexBuffer.buffer);
  }

  // Recreate index buffer if needed
  if(m_indexBuffer.buffer == VK_NULL_HANDLE || m_indexCapacity < m_indices.size())
  {
    if(m_indexBuffer.buffer != VK_NULL_HANDLE)
    {
      m_alloc->destroyBuffer(m_indexBuffer);
    }

    m_indexCapacity = static_cast<uint32_t>(m_indices.size() * 2);
    NVVK_CHECK(m_alloc->createBuffer(m_indexBuffer, m_indexCapacity * sizeof(uint32_t),
                                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
    NVVK_DBG_NAME(m_indexBuffer.buffer);
  }

  // Upload data
  VkCommandBuffer uploadCmd = m_app->createTempCmdBuffer();
  m_uploader->appendBuffer(m_vertexBuffer, 0, std::span<const GridVertex>(m_vertices));
  m_uploader->appendBuffer(m_indexBuffer, 0, std::span<const uint32_t>(m_indices));
  m_uploader->cmdUploadAppended(uploadCmd);
  m_app->submitAndWaitTempCmdBuffer(uploadCmd);
}

//-----------------------------------------------------------------------------
// Rendering
//-----------------------------------------------------------------------------

void GridHelperVk::renderRaster(VkCommandBuffer  cmd,
                                VkDescriptorSet  helperDescriptorSet,
                                const glm::mat4& viewMatrix,
                                const glm::mat4& projMatrix,
                                const glm::vec2& viewportSize,
                                const glm::vec2& depthBufferSize)
{
  if(!m_visible || !m_initialized || m_pipeline == VK_NULL_HANDLE)
    return;

  // Generate grid geometry once (at scale=1)
  if(!m_geometryGenerated)
  {
    generateGridGeometry();
    uploadGeometry();
  }

  if(m_indexCount == 0)
    return;

  // Calculate scale levels and blend factor for smooth LOD transitions
  glm::mat4      invView   = glm::inverse(viewMatrix);
  glm::vec3      cameraPos = glm::vec3(invView[3]);
  ScaleLevelInfo scaleInfo = calculateScaleLevels(cameraPos);
  m_lastGridScale          = scaleInfo.baseScale;

  // Bind pipeline once
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

  // Set line width (requires wideLines feature, falls back to 1.0 if not supported)
  vkCmdSetLineWidth(cmd, 2.0f);

  // Bind descriptor set (scene depth for occlusion)
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 1, 1, &helperDescriptorSet, 0, nullptr);

  // Bind vertex and index buffers once
  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, &offset);
  vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

  // Helper to draw a grid at a given scale and alpha
  auto drawGrid = [&](float scale, float alpha) {
    if(alpha < 0.01f || scale < 0.0001f || scale > 1000000.0f)
      return;

    glm::mat4 modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(scale));

    PushConstants pc;
    pc.mvp             = projMatrix * viewMatrix * modelMatrix;
    pc.color           = glm::vec4(1.0f, 1.0f, 1.0f, alpha);  // Grid alpha in .w component
    pc.viewportSize    = viewportSize;
    pc.depthBufferSize = depthBufferSize;
    pc.mode            = shaderio::visual_helpers::HelperMode::eGrid;
    pc.componentID     = 0;  // Unused for grid
    pc.padding         = glm::vec2(0.0f);

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstants), &pc);

    vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
  };

  // Draw coarse grid (fading in as we zoom out)
  drawGrid(scaleInfo.coarseScale, scaleInfo.coarseBlend);

  // Draw base grid (always fully visible)
  drawGrid(scaleInfo.baseScale, 1.0f);

  // Draw fine grid (fading out as we zoom out)
  drawGrid(scaleInfo.fineScale, scaleInfo.fineBlend);
}

}  // namespace vk_gaussian_splatting
