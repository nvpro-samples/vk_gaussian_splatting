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

#include "transform_helper_vk.h"
#include "utilities.h"

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvutils/primitives.hpp>
#include <nvutils/logger.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cfloat>  // For FLT_MAX
#include <glm/gtx/matrix_decompose.hpp>

#include <iostream>

namespace vk_gaussian_splatting {

//-----------------------------------------------------------------------------
// Helper: Upload nvutils primitive mesh to GPU
//-----------------------------------------------------------------------------

static void uploadPrimitiveMesh(const nvutils::PrimitiveMesh&                  primMesh,
                                const glm::mat4&                               transform,
                                const glm::vec3&                               color,
                                TransformHelperVk::GizmoComponent              component,
                                TransformHelperVk::GeometryType                type,
                                nvapp::Application*                            app,
                                nvvk::ResourceAllocator*                       alloc,
                                nvvk::StagingUploader*                         uploader,
                                std::vector<TransformHelperVk::GizmoGeometry>& outGeometry)
{
  std::vector<TransformHelperVk::GizmoVertex> vertices;
  std::vector<uint32_t>                       indices;

  // Initialize bounds
  glm::vec3 boundsMin(FLT_MAX);
  glm::vec3 boundsMax(-FLT_MAX);

  // Calculate normal matrix - for transforms with non-uniform scale, use inverse transpose
  // For rotation+uniform scale, the 3x3 part works directly
  glm::mat3 transform3x3 = glm::mat3(transform);
  glm::mat3 normalMatrix = glm::transpose(glm::inverse(transform3x3));

  // Convert nvutils vertices to gizmo vertices and apply transform
  vertices.reserve(primMesh.vertices.size());
  for(const auto& v : primMesh.vertices)
  {
    glm::vec3 transformedPos = glm::vec3(transform * glm::vec4(v.pos, 1.0f));
    glm::vec3 transformedNrm = glm::normalize(normalMatrix * v.nrm);
    vertices.push_back({transformedPos, transformedNrm});

    // Update bounds
    boundsMin = glm::min(boundsMin, transformedPos);
    boundsMax = glm::max(boundsMax, transformedPos);
  }

  // Convert triangles to indices
  indices.reserve(primMesh.triangles.size() * 3);
  for(const auto& tri : primMesh.triangles)
  {
    indices.push_back(tri.indices[0]);
    indices.push_back(tri.indices[1]);
    indices.push_back(tri.indices[2]);
  }

  // Expand bounds for translate/scale shafts to make picking easier
  if(component == TransformHelperVk::GizmoComponent::eTranslateX || component == TransformHelperVk::GizmoComponent::eTranslateY
     || component == TransformHelperVk::GizmoComponent::eTranslateZ || component == TransformHelperVk::GizmoComponent::eScaleX
     || component == TransformHelperVk::GizmoComponent::eScaleY || component == TransformHelperVk::GizmoComponent::eScaleZ
     || component == TransformHelperVk::GizmoComponent::eScaleUniform)
  {
    // Add padding to make picking area larger (visual is thin, but picking volume is wide)
    glm::vec3 padding(0.05f);  // Generous padding for easier selection
    boundsMin -= padding;
    boundsMax += padding;
  }

  // Create GPU buffers
  TransformHelperVk::GizmoGeometry geom;
  geom.color      = color;
  geom.component  = component;
  geom.type       = type;
  geom.indexCount = static_cast<uint32_t>(indices.size());
  geom.boundsMin  = boundsMin;
  geom.boundsMax  = boundsMax;

  VkCommandBuffer cmd = app->createTempCmdBuffer();

  // Add ACCELERATION_STRUCTURE_BUILD_INPUT usage for RT picking
  NVVK_CHECK(alloc->createBuffer(geom.vertexBuffer, vertices.size() * sizeof(TransformHelperVk::GizmoVertex),
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
  NVVK_CHECK(alloc->createBuffer(geom.indexBuffer, indices.size() * sizeof(uint32_t),
                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));

  uploader->appendBuffer(geom.vertexBuffer, 0, std::span(vertices));
  uploader->appendBuffer(geom.indexBuffer, 0, std::span(indices));
  uploader->cmdUploadAppended(cmd);

  app->submitAndWaitTempCmdBuffer(cmd);
  uploader->releaseStaging();

  NVVK_DBG_NAME(geom.vertexBuffer.buffer);
  NVVK_DBG_NAME(geom.indexBuffer.buffer);

  outGeometry.push_back(geom);
}

//-----------------------------------------------------------------------------
// Lifecycle
//-----------------------------------------------------------------------------

void TransformHelperVk::init(const Resources& res)
{
  m_app           = res.app;
  m_alloc         = res.alloc;
  m_uploader      = res.uploader;
  m_device        = res.device;
  m_sampler       = res.sampler;
  m_slangCompiler = res.slangCompiler;
  m_colorFormat   = res.colorFormat;
  m_depthFormat   = res.depthFormat;

  // Create gizmo geometry
  createGizmoGeometry();

  // Create raster rendering pipeline layout
  createRasterPipeline();

  // Compile shaders and create graphics pipeline
  rebuildPipelines();
}

void TransformHelperVk::deinit()
{
  // Clear attachment
  clearAttachment();

  // Destroy raster pipeline
  destroyRasterPipeline();

  // Destroy geometry
  destroyGizmoGeometry();

  // Clear pointers
  m_app      = nullptr;
  m_alloc    = nullptr;
  m_uploader = nullptr;
  m_device   = VK_NULL_HANDLE;
  m_sampler  = VK_NULL_HANDLE;
}

//-----------------------------------------------------------------------------
// Transform Attachment API (Entity-Agnostic)
//-----------------------------------------------------------------------------

void TransformHelperVk::attachTransform(glm::vec3* position, glm::vec3* rotation, glm::vec3* scale, uint32_t visibilityFlags)
{
  if(!position || !rotation || !scale)
  {
    clearAttachment();
    return;
  }

  m_attachedPosition = position;
  m_attachedRotation = rotation;
  m_attachedScale    = scale;
  m_visibilityFlags  = visibilityFlags;
  m_isDragging       = false;
  m_hoveredComponent = GizmoComponent::eNone;
  m_draggedComponent = GizmoComponent::eNone;
}

bool TransformHelperVk::isAttached() const
{
  return m_attachedPosition != nullptr && m_attachedRotation != nullptr && m_attachedScale != nullptr;
}

void TransformHelperVk::clearAttachment()
{
  m_attachedPosition = nullptr;
  m_attachedRotation = nullptr;
  m_attachedScale    = nullptr;
  m_visibilityFlags  = ShowAll;
  m_isDragging       = false;
  m_hoveredComponent = GizmoComponent::eNone;
  m_draggedComponent = GizmoComponent::eNone;
}

//-----------------------------------------------------------------------------
// Configuration
//-----------------------------------------------------------------------------

void TransformHelperVk::setTransformMode(TransformMode mode)
{
  if(m_mode != mode)
  {
    m_mode = mode;
    if(m_isDragging)
    {
      endDrag();
    }
  }
}

void TransformHelperVk::setTransformSpace(TransformSpace space)
{
  if(m_space != space)
  {
    m_space = space;
    if(m_isDragging)
    {
      endDrag();
    }
  }
}

void TransformHelperVk::setSnapValues(float translate, float rotate, float scale)
{
  m_snapTranslate = translate;
  m_snapRotate    = rotate;
  m_snapScale     = scale;
}

//-----------------------------------------------------------------------------
// Interaction
//-----------------------------------------------------------------------------

bool TransformHelperVk::processInput(const glm::vec2& mousePos,
                                     const glm::vec2& mouseDelta,
                                     bool             mouseDown,
                                     bool             mousePressed,
                                     bool             mouseReleased,
                                     const glm::mat4& viewMatrix,
                                     const glm::mat4& projMatrix,
                                     const glm::vec2& viewport)
{
  if(!isAttached())
    return false;

  // Handle drag release
  if(mouseReleased && m_isDragging)
  {
    endDrag();
    return false;
  }

  // Handle drag update
  if(mouseDown && m_isDragging)
  {
    updateDrag(mousePos, mouseDelta, viewMatrix, projMatrix, viewport);
    return true;
  }

  // Handle drag start
  if(mousePressed)
  {
    GizmoComponent component = pickGizmoComponent(mousePos, viewMatrix, projMatrix, viewport);
    if(component != GizmoComponent::eNone)
    {
      startDrag(component, mousePos, viewMatrix, projMatrix, viewport);
      return true;
    }
  }

  // Handle hover
  if(!mouseDown)
  {
    m_hoveredComponent = pickGizmoComponent(mousePos, viewMatrix, projMatrix, viewport);
  }

  return false;
}

//-----------------------------------------------------------------------------
// Rendering
//-----------------------------------------------------------------------------

void TransformHelperVk::renderRaster(VkCommandBuffer  cmd,
                                     VkDescriptorSet  sceneDescriptorSet,
                                     VkDescriptorSet  helperDescriptorSet,
                                     const glm::mat4& viewMatrix,
                                     const glm::mat4& projMatrix,
                                     const glm::vec2& viewportSize,
                                     const glm::vec2& depthBufferSize)
{
  if(!isAttached())
    return;

  if(m_rasterPipeline == VK_NULL_HANDLE)
    return;  // Pipeline not ready

  // Get gizmo transform and apply screen-space scaling
  glm::mat4 gizmoTransform = getGizmoTransform();
  glm::vec3 gizmoPosition  = getAttachedPosition();
  float     scale          = getScreenSpaceScale(gizmoPosition, viewMatrix, projMatrix, viewportSize);
  glm::mat4 scaleMatrix    = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
  glm::mat4 modelMatrix    = gizmoTransform * scaleMatrix;
  glm::mat4 mvp            = projMatrix * viewMatrix * modelMatrix;

  // Bind graphics pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipeline);

  // Bind helper descriptor set (set 1: scene depth texture)
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipelineLayout, 1, 1, &helperDescriptorSet, 0, nullptr);

  // Render all gizmo geometries (unified gizmo - all modes visible)
  for(const auto& geom : m_gizmoGeometry)
  {
    // Check if this geometry should be rendered (based on hover state)
    if(!shouldRenderGeometry(geom))
      continue;

    // Set unified push constants
    shaderio::visual_helpers::PushConstantVisualHelpers pc;
    pc.mvp             = mvp;
    pc.viewportSize    = viewportSize;                                      // Native viewport resolution
    pc.depthBufferSize = depthBufferSize;                                   // DLSS render size (or viewport if no DLSS)
    pc.mode            = shaderio::visual_helpers::HelperMode::eTransform;  // Transform helper mode

    // Simple hover feedback: highlight if this is the hovered component
    bool      isHovered = (geom.component == m_hoveredComponent);
    glm::vec3 color     = getColorFromComponent(geom.component, isHovered);
    pc.color            = glm::vec4(color, 1.0f);  // Fully opaque
    pc.componentID      = static_cast<uint32_t>(geom.component);
    pc.padding          = glm::vec2(0.0f);  // Padding for alignment

    vkCmdPushConstants(cmd, m_rasterPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(shaderio::visual_helpers::PushConstantVisualHelpers), &pc);

    // Bind vertex and index buffers
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &geom.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, geom.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Draw
    vkCmdDrawIndexed(cmd, geom.indexCount, 1, 0, 0, 0);
  }
}

//-----------------------------------------------------------------------------
// Update Notifications
//-----------------------------------------------------------------------------

void TransformHelperVk::notifyExternalTransformChange()
{
  if(m_isDragging)
  {
    endDrag();
  }
}

//-----------------------------------------------------------------------------
// Geometry Generation
//-----------------------------------------------------------------------------

void TransformHelperVk::createGizmoGeometry()
{
  generateTranslateGizmo();
  generateRotateGizmo();
  generateScaleGizmo();
}

void TransformHelperVk::destroyGizmoGeometry()
{
  for(auto& geom : m_gizmoGeometry)
  {
    if(geom.vertexBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(geom.vertexBuffer);
    if(geom.indexBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(geom.indexBuffer);
  }
  m_gizmoGeometry.clear();
}

void TransformHelperVk::generateTranslateGizmo()
{
  // Translation gizmo using nvutils primitives:
  // - 3 axis shafts for X/Y/Z translation
  // - 3 plane handles for XY/XZ/YZ manipulation

  const float shaftRadius = 0.01f;  // Thin shaft for visual clarity
  const float shaftLength = 1.5f;   // Length to scale box placement

  // Use elongated cube as shaft (no cylinder primitive available)
  nvutils::PrimitiveMesh shaftMesh = nvutils::createCube(1.0f, 1.0f, 1.0f);

  // X axis (Red) - Shaft only
  glm::mat4 xShaftTransform = glm::translate(glm::mat4(1.0f), glm::vec3(shaftLength * 0.5f, 0, 0))
                              * glm::scale(glm::mat4(1.0f), glm::vec3(shaftLength, shaftRadius * 2.0f, shaftRadius * 2.0f));
  uploadPrimitiveMesh(shaftMesh, xShaftTransform, glm::vec3(1, 0, 0), GizmoComponent::eTranslateX,
                      TransformHelperVk::GeometryType::eShaft, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Y axis (Green) - Shaft only
  glm::mat4 yShaftTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, shaftLength * 0.5f, 0))
                              * glm::scale(glm::mat4(1.0f), glm::vec3(shaftRadius * 2.0f, shaftLength, shaftRadius * 2.0f));
  uploadPrimitiveMesh(shaftMesh, yShaftTransform, glm::vec3(0, 1, 0), GizmoComponent::eTranslateY,
                      TransformHelperVk::GeometryType::eShaft, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Z axis (Blue) - Shaft only
  glm::mat4 zShaftTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, shaftLength * 0.5f))
                              * glm::scale(glm::mat4(1.0f), glm::vec3(shaftRadius * 2.0f, shaftRadius * 2.0f, shaftLength));
  uploadPrimitiveMesh(shaftMesh, zShaftTransform, glm::vec3(0, 0, 1), GizmoComponent::eTranslateZ,
                      TransformHelperVk::GeometryType::eShaft, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Create plane handles using nvutils::createPlane
  // Make planes 2x larger than center box (0.15 * 2 = 0.3)
  const float            planeSize2x = 0.3f;
  const float            planeOffset = 0.6f;  // Position quads further from origin for better visibility
  nvutils::PrimitiveMesh planeMesh   = nvutils::createPlane(1, planeSize2x, planeSize2x);

  // XY plane (Cyan) - Rotate 90° around X to lie flat in XY, offset to positive quadrant
  glm::mat4 xyTransform = glm::translate(glm::mat4(1.0f), glm::vec3(planeOffset, planeOffset, 0))
                          * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
  uploadPrimitiveMesh(planeMesh, xyTransform, glm::vec3(0, 1, 1), GizmoComponent::eTranslateXY,
                      TransformHelperVk::GeometryType::ePlane, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // XZ plane (Magenta) - No rotation (lies flat horizontally in XZ), offset to positive quadrant
  glm::mat4 xzTransform = glm::translate(glm::mat4(1.0f), glm::vec3(planeOffset, 0, planeOffset));
  uploadPrimitiveMesh(planeMesh, xzTransform, glm::vec3(1, 0, 1), GizmoComponent::eTranslateXZ,
                      TransformHelperVk::GeometryType::ePlane, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // YZ plane (Yellow) - Rotate 90° around Z to lie flat in YZ, offset to positive quadrant
  glm::mat4 yzTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, planeOffset, planeOffset))
                          * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
  uploadPrimitiveMesh(planeMesh, yzTransform, glm::vec3(1, 1, 0), GizmoComponent::eTranslateYZ,
                      TransformHelperVk::GeometryType::ePlane, m_app, m_alloc, m_uploader, m_gizmoGeometry);
}

void TransformHelperVk::generateRotateGizmo()
{
  // Rotation gizmo using nvutils::createTorusMesh:
  // - 3 torus rings for rotation around X/Y/Z axes

  const float ringRadius = 1.3f;     // Larger radius for better visibility
  const float tubeRadius = 0.0135f;  // Thin tube for visual clarity

  // Create torus primitive (lies in XZ plane by default, around Y axis)
  nvutils::PrimitiveMesh torusMesh = nvutils::createTorusMesh(ringRadius, tubeRadius, 64, 16);

  // X ring (Red) - Rotate 90° around Z to lie in YZ plane (perpendicular to X axis)
  glm::mat4 xTransform = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
  uploadPrimitiveMesh(torusMesh, xTransform, glm::vec3(1, 0, 0), GizmoComponent::eRotateX,
                      TransformHelperVk::GeometryType::eTorus, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Y ring (Green) - Identity, already in XZ plane (perpendicular to Y axis)
  glm::mat4 yTransform = glm::mat4(1.0f);
  uploadPrimitiveMesh(torusMesh, yTransform, glm::vec3(0, 1, 0), GizmoComponent::eRotateY,
                      TransformHelperVk::GeometryType::eTorus, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Z ring (Blue) - Rotate 90° around X to lie in XY plane (perpendicular to Z axis)
  glm::mat4 zTransform = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
  uploadPrimitiveMesh(torusMesh, zTransform, glm::vec3(0, 0, 1), GizmoComponent::eRotateZ,
                      TransformHelperVk::GeometryType::eTorus, m_app, m_alloc, m_uploader, m_gizmoGeometry);
}

void TransformHelperVk::generateScaleGizmo()
{
  // Scale gizmo:
  // - 3 cubes at end of X/Y/Z axes for per-axis scaling
  // - 1 center cube for uniform scaling
  // Note: Shafts are shared with translate mode

  const float axisLength = 1.67f;  // Three times closer to shaft end (was 2.0, shaft is 1.5)
  const float cubeSize   = 0.15f;

  // Create primitives
  nvutils::PrimitiveMesh cubeMesh = nvutils::createCube(cubeSize, cubeSize, cubeSize);

  // Only create cubes at the ends - shafts are shared with translate mode
  // X axis (Red) - Cube at end
  glm::mat4 xCubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(axisLength, 0, 0));
  uploadPrimitiveMesh(cubeMesh, xCubeTransform, glm::vec3(1, 0, 0), GizmoComponent::eScaleX,
                      TransformHelperVk::GeometryType::eBox, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Y axis (Green) - Cube at end
  glm::mat4 yCubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, axisLength, 0));
  uploadPrimitiveMesh(cubeMesh, yCubeTransform, glm::vec3(0, 1, 0), GizmoComponent::eScaleY,
                      TransformHelperVk::GeometryType::eBox, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Z axis (Blue) - Cube at end
  glm::mat4 zCubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, axisLength));
  uploadPrimitiveMesh(cubeMesh, zCubeTransform, glm::vec3(0, 0, 1), GizmoComponent::eScaleZ,
                      TransformHelperVk::GeometryType::eBox, m_app, m_alloc, m_uploader, m_gizmoGeometry);

  // Center cube (Light gray) - For uniform scaling, larger
  glm::mat4 centerTransform = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
  uploadPrimitiveMesh(cubeMesh, centerTransform, glm::vec3(1, 1, 1), GizmoComponent::eScaleUniform,
                      TransformHelperVk::GeometryType::eCenterBox, m_app, m_alloc, m_uploader, m_gizmoGeometry);
}

//-----------------------------------------------------------------------------
// Rendering Pipelines
//-----------------------------------------------------------------------------

void TransformHelperVk::createRasterPipeline()
{
  // Create Set 0: Empty (for future scene bindings if needed)
  VkDescriptorSetLayoutCreateInfo emptyLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  emptyLayoutInfo.bindingCount      = 0;
  emptyLayoutInfo.pBindings         = nullptr;
  VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &emptyLayoutInfo, nullptr, &emptyLayout));

  // Create Set 1: Scene depth texture (for occlusion testing)
  VkDescriptorSetLayoutBinding bindings[2];

  // Binding 0: Scene depth texture
  bindings[0].binding            = 0;
  bindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  bindings[0].descriptorCount    = 1;
  bindings[0].stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[0].pImmutableSamplers = nullptr;

  // Binding 1: Depth sampler
  bindings[1].binding            = 1;
  bindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_SAMPLER;
  bindings[1].descriptorCount    = 1;
  bindings[1].stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
  bindings[1].pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = 2;
  layoutInfo.pBindings    = bindings;
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  // Create pipeline layout with both sets
  VkDescriptorSetLayout setLayouts[2] = {emptyLayout, m_descriptorSetLayout};

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset     = 0;
  pushConstantRange.size       = sizeof(shaderio::visual_helpers::PushConstantVisualHelpers);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount         = 2;  // Set 0 and Set 1
  pipelineLayoutInfo.pSetLayouts            = setLayouts;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pushConstantRange;

  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_rasterPipelineLayout));
  NVVK_DBG_NAME(m_rasterPipelineLayout);

  // Clean up temporary empty layout
  vkDestroyDescriptorSetLayout(m_device, emptyLayout, nullptr);

  // Note: Actual pipeline creation deferred until shaders are compiled
}

void TransformHelperVk::destroyRasterPipeline()
{
  if(m_rasterPipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_rasterPipeline, nullptr);
    m_rasterPipeline = VK_NULL_HANDLE;
  }

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

  if(m_rasterPipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_rasterPipelineLayout, nullptr);
    m_rasterPipelineLayout = VK_NULL_HANDLE;
  }

  if(m_descriptorSet != VK_NULL_HANDLE)
  {
    vkFreeDescriptorSets(m_device, m_descriptorPool, 1, &m_descriptorSet);
    m_descriptorSet = VK_NULL_HANDLE;
  }

  if(m_descriptorPool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    m_descriptorPool = VK_NULL_HANDLE;
  }

  if(m_descriptorSetLayout != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    m_descriptorSetLayout = VK_NULL_HANDLE;
  }
}

//-----------------------------------------------------------------------------
// Picking and Interaction
//-----------------------------------------------------------------------------

TransformHelperVk::GizmoComponent TransformHelperVk::pickGizmoComponent(const glm::vec2& mousePos,
                                                                        const glm::mat4& viewMatrix,
                                                                        const glm::mat4& projMatrix,
                                                                        const glm::vec2& viewport)
{
  if(!isAttached())
    return GizmoComponent::eNone;

  // Create ray from mouse position
  glm::vec3 rayOrigin, rayDir;
  createRayFromMouse(mousePos, viewMatrix, projMatrix, viewport, rayOrigin, rayDir);

  // Get gizmo transform
  glm::mat4 gizmoTransform = getGizmoTransform();
  glm::vec3 gizmoPosition  = *m_attachedPosition;

  // Compute screen-space scale
  float     scale       = getScreenSpaceScale(gizmoPosition, viewMatrix, projMatrix, viewport);
  glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
  glm::mat4 modelMatrix = gizmoTransform * scaleMatrix;

  // Test against each gizmo component geometry - pick the closest one
  float          closestDist      = FLT_MAX;
  GizmoComponent closestComponent = GizmoComponent::eNone;

  for(const auto& geom : m_gizmoGeometry)
  {
    // Skip geometries that aren't currently visible
    if(!shouldRenderGeometry(geom))
      continue;

    // Transform ray to local space of this component
    glm::mat4 invModel       = glm::inverse(modelMatrix);
    glm::vec3 localRayOrigin = glm::vec3(invModel * glm::vec4(rayOrigin, 1.0f));
    glm::vec3 localRayDir    = glm::normalize(glm::vec3(invModel * glm::vec4(rayDir, 0.0f)));

    // Simple bounding box test first (early rejection)
    if(!rayIntersectsBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax))
      continue;

    // Calculate distance to this specific geometry
    float dist          = 0.0f;
    float pickThreshold = 0.6f;  // Default threshold

    switch(geom.component)
    {
      case GizmoComponent::eTranslateX:
      case GizmoComponent::eTranslateY:
      case GizmoComponent::eTranslateZ:
      case GizmoComponent::eScaleX:
      case GizmoComponent::eScaleY:
      case GizmoComponent::eScaleZ:
      case GizmoComponent::eScaleUniform:
        // Use per-geometry bounds distance
        dist          = rayDistanceToBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax);
        pickThreshold = 0.6f;  // Much larger for easier axis shaft selection
        break;

      case GizmoComponent::eRotateX:
      case GizmoComponent::eRotateY:
      case GizmoComponent::eRotateZ:
        // Use analytical circle distance
        dist          = rayDistanceToComponent(localRayOrigin, localRayDir, geom.component);
        pickThreshold = 0.2f;  // Larger for easier selection despite thinner tube
        break;

      case GizmoComponent::eTranslateXY:
      case GizmoComponent::eTranslateXZ:
      case GizmoComponent::eTranslateYZ:
        // Plane handles
        dist          = rayDistanceToBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax);
        pickThreshold = 0.2f;  // Larger for easier selection
        break;

      default:
        dist          = rayDistanceToBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax);
        pickThreshold = 0.6f;
        break;
    }

    // Simple closest-hit selection
    if(dist < pickThreshold && dist < closestDist)
    {
      closestDist      = dist;
      closestComponent = geom.component;
    }
  }

  return closestComponent;
}

void TransformHelperVk::startDrag(GizmoComponent   component,
                                  const glm::vec2& mousePos,
                                  const glm::mat4& viewMatrix,
                                  const glm::mat4& projMatrix,
                                  const glm::vec2& viewport)
{
  m_isDragging        = true;
  m_draggedComponent  = component;
  m_dragStartPosMouse = mousePos;

  // Capture current transform components
  if(isAttached())
  {
    m_dragStartPosition = *m_attachedPosition;
    m_dragStartRotation = *m_attachedRotation;
    m_dragStartScale    = *m_attachedScale;
  }

  // Calculate initial hit point for relative dragging
  glm::vec3 rayOrigin, rayDir;
  createRayFromMouse(mousePos, viewMatrix, projMatrix, viewport, rayOrigin, rayDir);

  // Compute the 3D hit point based on component type
  switch(component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eTranslateZ: {
      glm::vec3 axis     = getAxisFromComponent(component);
      glm::vec3 gizmoPos = m_dragStartPosition;

      // Create a plane perpendicular to the camera view, containing the axis
      glm::vec3 viewDir     = glm::normalize(glm::vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]));
      glm::vec3 planeNormal = glm::normalize(glm::cross(axis, glm::cross(viewDir, axis)));

      // Intersect ray with plane
      float t;
      if(rayIntersectPlane(rayOrigin, rayDir, gizmoPos, planeNormal, t))
      {
        m_dragStartHitPoint = rayOrigin + rayDir * t;
      }
      else
      {
        m_dragStartHitPoint = gizmoPos;  // Fallback
      }
      break;
    }

    case GizmoComponent::eTranslateXY:
    case GizmoComponent::eTranslateXZ:
    case GizmoComponent::eTranslateYZ: {
      glm::vec3 planeNormal = getAxisFromComponent(component);
      glm::vec3 gizmoPos    = m_dragStartPosition;

      // Intersect ray with plane
      float t;
      if(rayIntersectPlane(rayOrigin, rayDir, gizmoPos, planeNormal, t))
      {
        m_dragStartHitPoint = rayOrigin + rayDir * t;
      }
      else
      {
        m_dragStartHitPoint = gizmoPos;  // Fallback
      }
      break;
    }

    default:
      m_dragStartHitPoint = m_dragStartPosition;
      break;
  }

  if(m_onTransformBegin)
    m_onTransformBegin();
}

void TransformHelperVk::updateDrag(const glm::vec2& mousePos,
                                   const glm::vec2& mouseDelta,
                                   const glm::mat4& viewMatrix,
                                   const glm::mat4& projMatrix,
                                   const glm::vec2& viewport)
{
  if(!m_isDragging || !isAttached())
    return;

  // Create ray from current mouse position
  glm::vec3 rayOrigin, rayDir;
  createRayFromMouse(mousePos, viewMatrix, projMatrix, viewport, rayOrigin, rayDir);

  // Handle different component types
  switch(m_draggedComponent)
  {
    // Translation along single axis
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eTranslateZ: {
      glm::vec3 axis     = getAxisFromComponent(m_draggedComponent);
      glm::vec3 gizmoPos = m_dragStartPosition;

      // Create a plane perpendicular to the camera view, containing the axis
      glm::vec3 viewDir     = glm::normalize(glm::vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]));
      glm::vec3 planeNormal = glm::normalize(glm::cross(axis, glm::cross(viewDir, axis)));

      // Intersect ray with plane
      float t;
      if(rayIntersectPlane(rayOrigin, rayDir, gizmoPos, planeNormal, t))
      {
        glm::vec3 currentHitPoint = rayOrigin + rayDir * t;

        // Calculate delta from initial hit point to current hit point (relative dragging)
        glm::vec3 dragDelta = currentHitPoint - m_dragStartHitPoint;

        // Project delta onto axis to get movement along axis only
        float projection = glm::dot(dragDelta, axis);

        // Apply relative translation
        *m_attachedPosition = m_dragStartPosition + axis * projection;

        // Notify callback
        if(m_onTransformChange)
          m_onTransformChange();
      }
      break;
    }

    // Translation constrained to plane (XY, XZ, YZ)
    case GizmoComponent::eTranslateXY:
    case GizmoComponent::eTranslateXZ:
    case GizmoComponent::eTranslateYZ: {
      glm::vec3 planeNormal = getAxisFromComponent(m_draggedComponent);
      glm::vec3 gizmoPos    = m_dragStartPosition;

      // Intersect ray with plane
      float t;
      if(rayIntersectPlane(rayOrigin, rayDir, gizmoPos, planeNormal, t))
      {
        glm::vec3 currentHitPoint = rayOrigin + rayDir * t;

        // Calculate delta from initial hit point to current hit point (relative dragging)
        glm::vec3 dragDelta = currentHitPoint - m_dragStartHitPoint;

        // Apply full delta (movement is already constrained to plane by the plane intersection)
        *m_attachedPosition = m_dragStartPosition + dragDelta;

        // Notify callback
        if(m_onTransformChange)
          m_onTransformChange();
      }
      break;
    }

    // Rotation around single axis
    case GizmoComponent::eRotateX:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eRotateZ: {
      glm::vec3 axis     = getAxisFromComponent(m_draggedComponent);
      glm::vec3 gizmoPos = m_dragStartPosition;

      // Intersect ray with plane perpendicular to rotation axis
      float t;
      if(rayIntersectPlane(rayOrigin, rayDir, gizmoPos, axis, t))
      {
        glm::vec3 hitPoint = rayOrigin + rayDir * t;
        glm::vec3 toHit    = hitPoint - gizmoPos;

        // Get start ray intersection
        glm::vec3 startRayOrigin, startRayDir;
        createRayFromMouse(m_dragStartPosMouse, viewMatrix, projMatrix, viewport, startRayOrigin, startRayDir);

        float startT;
        if(rayIntersectPlane(startRayOrigin, startRayDir, gizmoPos, axis, startT))
        {
          glm::vec3 startHitPoint = startRayOrigin + startRayDir * startT;
          glm::vec3 toStartHit    = startHitPoint - gizmoPos;

          // Compute angle between start and current
          float cosAngle = glm::dot(glm::normalize(toStartHit), glm::normalize(toHit));
          cosAngle       = glm::clamp(cosAngle, -1.0f, 1.0f);
          float angleDeg = glm::degrees(std::acos(cosAngle));

          // Determine sign using cross product
          glm::vec3 cross = glm::cross(toStartHit, toHit);
          if(glm::dot(cross, axis) < 0.0f)
            angleDeg = -angleDeg;

          // Apply rotation
          *m_attachedRotation = m_dragStartRotation + axis * angleDeg;

          // Notify callback
          if(m_onTransformChange)
            m_onTransformChange();
        }
      }
      break;
    }

    // Scale along single axis
    case GizmoComponent::eScaleX:
    case GizmoComponent::eScaleY:
    case GizmoComponent::eScaleZ: {
      // Calculate scale based on total mouse movement from drag start
      glm::vec2 totalDelta  = mousePos - m_dragStartPosMouse;
      float     scaleFactor = 1.0f + totalDelta.y * 0.005f;  // 0.5% per pixel
      scaleFactor           = glm::max(0.01f, scaleFactor);  // Clamp to prevent negative/zero scale

      int axisIndex = (m_draggedComponent == GizmoComponent::eScaleX) ? 0 :
                      (m_draggedComponent == GizmoComponent::eScaleY) ? 1 :
                                                                        2;

      // Apply scale only to the selected axis
      *m_attachedScale              = m_dragStartScale;
      (*m_attachedScale)[axisIndex] = m_dragStartScale[axisIndex] * scaleFactor;

      // Notify callback
      if(m_onTransformChange)
        m_onTransformChange();
      break;
    }

    // Uniform scale
    case GizmoComponent::eScaleUniform: {
      // Calculate uniform scale based on total mouse movement from drag start
      glm::vec2 totalDelta  = mousePos - m_dragStartPosMouse;
      float     scaleFactor = 1.0f + totalDelta.y * 0.005f;  // 0.5% per pixel
      scaleFactor           = glm::max(0.01f, scaleFactor);  // Clamp to prevent negative/zero scale

      *m_attachedScale = m_dragStartScale * scaleFactor;

      // Notify callback
      if(m_onTransformChange)
        m_onTransformChange();
      break;
    }

    default:
      break;
  }
}

void TransformHelperVk::endDrag()
{
  m_isDragging       = false;
  m_draggedComponent = GizmoComponent::eNone;

  if(m_onTransformEnd)
    m_onTransformEnd();
}

//-----------------------------------------------------------------------------
// Picking Helper Functions
//-----------------------------------------------------------------------------

void TransformHelperVk::createRayFromMouse(const glm::vec2& mousePos,
                                           const glm::mat4& viewMatrix,
                                           const glm::mat4& projMatrix,
                                           const glm::vec2& viewport,
                                           glm::vec3&       rayOrigin,
                                           glm::vec3&       rayDir) const
{
  // Convert mouse position to normalized device coordinates
  // Mouse Y is top-down (0 at top), need to flip for NDC
  // Vulkan NDC: X [-1,1] left to right, Y [-1,1] top to bottom, Z [0,1] near to far
  float ndcX = (2.0f * mousePos.x) / viewport.x - 1.0f;
  float ndcY = (2.0f * mousePos.y) / viewport.y - 1.0f;  // Vulkan Y convention (top = -1, bottom = 1)

  // Unproject to world space
  // GPU uses: X=right, Y=up, Z=front (toward camera)
  glm::mat4 invView = glm::inverse(viewMatrix);
  glm::mat4 invProj = glm::inverse(projMatrix);

  // Near and far points in NDC
  glm::vec4 rayStartNDC(ndcX, ndcY, 0.0f, 1.0f);  // Near plane (Z=0 in Vulkan)
  glm::vec4 rayEndNDC(ndcX, ndcY, 1.0f, 1.0f);    // Far plane (Z=1 in Vulkan)

  // Unproject through inverse projection (NDC -> View space)
  glm::vec4 rayStartView = invProj * rayStartNDC;
  glm::vec4 rayEndView   = invProj * rayEndNDC;

  rayStartView /= rayStartView.w;
  rayEndView /= rayEndView.w;

  // Transform to world space
  glm::vec4 rayStartWorld = invView * rayStartView;
  glm::vec4 rayEndWorld   = invView * rayEndView;

  rayOrigin = glm::vec3(rayStartWorld);
  rayDir    = glm::normalize(glm::vec3(rayEndWorld - rayStartWorld));
}

bool TransformHelperVk::rayIntersectsBounds(const glm::vec3& rayOrigin,
                                            const glm::vec3& rayDir,
                                            const glm::vec3& boundsMin,
                                            const glm::vec3& boundsMax) const
{
  // AABB ray intersection using slab method
  glm::vec3 invDir = glm::vec3(1.0f) / rayDir;
  glm::vec3 t0s    = (boundsMin - rayOrigin) * invDir;
  glm::vec3 t1s    = (boundsMax - rayOrigin) * invDir;

  glm::vec3 tsmaller = glm::min(t0s, t1s);
  glm::vec3 tbigger  = glm::max(t0s, t1s);

  float tmin = glm::max(glm::max(tsmaller.x, tsmaller.y), tsmaller.z);
  float tmax = glm::min(glm::min(tbigger.x, tbigger.y), tbigger.z);

  return tmax >= tmin && tmax >= 0.0f;
}

float TransformHelperVk::rayDistanceToBounds(const glm::vec3& rayOrigin,
                                             const glm::vec3& rayDir,
                                             const glm::vec3& boundsMin,
                                             const glm::vec3& boundsMax) const
{
  // Calculate distance from ray to AABB
  // This is more accurate for thin elongated boxes (axis shafts)

  glm::vec3 center  = (boundsMin + boundsMax) * 0.5f;
  glm::vec3 extents = (boundsMax - boundsMin) * 0.5f;

  // Find closest point on ray to box center
  glm::vec3 toCenter     = center - rayOrigin;
  float     rayT         = glm::dot(toCenter, rayDir);
  rayT                   = glm::max(0.0f, rayT);  // Clamp to forward direction
  glm::vec3 closestOnRay = rayOrigin + rayDir * rayT;

  // Clamp to AABB
  glm::vec3 closestOnBox = glm::clamp(closestOnRay, boundsMin, boundsMax);

  // Distance between ray point and box point
  return glm::length(closestOnRay - closestOnBox);
}

float TransformHelperVk::rayDistanceToComponent(const glm::vec3& rayOrigin, const glm::vec3& rayDir, GizmoComponent component) const
{
  // Compute minimum distance from ray to the gizmo component
  // For arrows/cones: distance to axis line segment
  // For circles/toruses: distance to circle in 3D
  // For cubes: distance to box surface

  switch(component)
  {
    // Translation arrows (distance to axis line segment)
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eTranslateZ: {
      glm::vec3 axis = getAxisFromComponent(component);
      // Match geometry: shaft goes from 0 to 1.6, cone from 1.6 to 2.0
      const float totalLength = 2.0f;
      glm::vec3   p1          = glm::vec3(0.0f);
      glm::vec3   p2          = axis * totalLength;

      // Distance from ray to line segment
      glm::vec3 v = p2 - p1;
      glm::vec3 w = rayOrigin - p1;

      float c1 = glm::dot(w, v);
      float c2 = glm::dot(v, v);

      if(c2 == 0.0f)
        return glm::length(glm::cross(rayDir, w));

      float     t              = glm::clamp(c1 / c2, 0.0f, 1.0f);
      glm::vec3 pointOnSegment = p1 + t * v;

      // Distance from ray to point on segment
      glm::vec3 w2           = pointOnSegment - rayOrigin;
      float     dotRD        = glm::dot(w2, rayDir);
      glm::vec3 closestOnRay = rayOrigin + rayDir * dotRD;

      return glm::length(closestOnRay - pointOnSegment);
    }

    // Rotation circles (distance to circle in 3D space)
    case GizmoComponent::eRotateX:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eRotateZ: {
      glm::vec3 axis   = getAxisFromComponent(component);
      glm::vec3 center = glm::vec3(0.0f);
      float     radius = 1.3f;  // Match geometry generation radius

      // Find intersection with plane containing the circle
      float t;
      if(!rayIntersectPlane(rayOrigin, rayDir, center, axis, t))
        return FLT_MAX;

      glm::vec3 hitPoint       = rayOrigin + rayDir * t;
      float     distFromCenter = glm::length(hitPoint - center);

      // Distance from circle radius (with some tolerance for the tube thickness)
      return glm::abs(distFromCenter - radius);
    }

    // Scale cubes (distance to box surface)
    case GizmoComponent::eScaleX:
    case GizmoComponent::eScaleY:
    case GizmoComponent::eScaleZ: {
      glm::vec3   axis       = getAxisFromComponent(component);
      const float axisLength = 2.0f;               // Match geometry
      glm::vec3   cubeCenter = axis * axisLength;  // Cube at end of axis

      // Distance from ray to cube center
      glm::vec3 w            = cubeCenter - rayOrigin;
      float     dotRD        = glm::dot(w, rayDir);
      glm::vec3 closestOnRay = rayOrigin + rayDir * dotRD;

      return glm::length(closestOnRay - cubeCenter);
    }

    // Uniform scale (center cube)
    case GizmoComponent::eScaleUniform: {
      glm::vec3 center       = glm::vec3(0.0f);
      glm::vec3 w            = center - rayOrigin;
      float     dotRD        = glm::dot(w, rayDir);
      glm::vec3 closestOnRay = rayOrigin + rayDir * dotRD;

      return glm::length(closestOnRay - center);
    }

    default:
      return FLT_MAX;
  }
}

//-----------------------------------------------------------------------------
// Transform Application
//-----------------------------------------------------------------------------

void TransformHelperVk::applyTranslation(const glm::vec3& delta)
{
  if(!isAttached())
    return;

  // Apply snapping if enabled
  glm::vec3 snappedDelta = delta;
  if(m_enableSnapping)
  {
    snappedDelta = glm::round(delta / m_snapTranslate) * m_snapTranslate;
  }

  // Translate in world or local space
  glm::vec3 newPosition = m_dragStartPosition;
  if(m_space == TransformSpace::eWorld)
  {
    newPosition += snappedDelta;
  }
  else
  {
    // Rotate delta by object's current rotation
    glm::quat rotation   = getAttachedRotation();
    glm::vec3 localDelta = rotation * snappedDelta;
    newPosition += localDelta;
  }

  *m_attachedPosition = newPosition;

  if(m_onTransformChange)
    m_onTransformChange();
}

void TransformHelperVk::applyRotation(const glm::vec3& axis, float angleDeg)
{
  if(!isAttached())
    return;

  // Apply snapping if enabled
  float snappedAngle = angleDeg;
  if(m_enableSnapping)
  {
    snappedAngle = glm::round(angleDeg / m_snapRotate) * m_snapRotate;
  }

  glm::vec3 newRotation = m_dragStartRotation;

  // Apply rotation based on axis (assume axis is one of X/Y/Z unit vectors)
  if(glm::abs(axis.x) > 0.5f)  // X axis
    newRotation.x += snappedAngle;
  else if(glm::abs(axis.y) > 0.5f)  // Y axis
    newRotation.y += snappedAngle;
  else if(glm::abs(axis.z) > 0.5f)  // Z axis
    newRotation.z += snappedAngle;

  *m_attachedRotation = newRotation;

  if(m_onTransformChange)
    m_onTransformChange();
}

void TransformHelperVk::applyScale(const glm::vec3& scaleFactor)
{
  if(!isAttached())
    return;

  // Apply snapping if enabled
  glm::vec3 snappedFactor = scaleFactor;
  if(m_enableSnapping)
  {
    snappedFactor = glm::round(scaleFactor / m_snapScale) * m_snapScale;
  }

  // Apply scale factor to starting scale
  glm::vec3 newScale = m_dragStartScale * snappedFactor;

  *m_attachedScale = newScale;

  if(m_onTransformChange)
    m_onTransformChange();
}

//-----------------------------------------------------------------------------
// Utility Functions
//-----------------------------------------------------------------------------

glm::mat4 TransformHelperVk::getGizmoTransform() const
{
  if(!isAttached())
    return glm::mat4(1.0f);

  glm::vec3 position = getAttachedPosition();
  glm::quat rotation = getAttachedRotation();

  if(m_space == TransformSpace::eWorld)
  {
    return glm::translate(glm::mat4(1.0f), position);
  }
  else
  {
    return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);
  }
}

glm::vec3 TransformHelperVk::getAttachedPosition() const
{
  if(isAttached())
    return *m_attachedPosition;
  return glm::vec3(0.0f);
}

glm::quat TransformHelperVk::getAttachedRotation() const
{
  if(!isAttached())
    return glm::quat(1, 0, 0, 0);

  // Convert Euler angles (degrees) to quaternion
  glm::vec3 radians = glm::radians(*m_attachedRotation);
  return glm::quat(radians);
}

glm::vec3 TransformHelperVk::getAttachedScale() const
{
  if(isAttached())
    return *m_attachedScale;
  return glm::vec3(1.0f);
}

float TransformHelperVk::getScreenSpaceScale(const glm::vec3& worldPos,
                                             const glm::mat4& viewMatrix,
                                             const glm::mat4& projMatrix,
                                             const glm::vec2& viewport) const
{
  glm::vec4 clipPos = projMatrix * viewMatrix * glm::vec4(worldPos, 1.0f);
  float     depth   = clipPos.w;
  return (m_gizmoSize / viewport.y) * depth;
}

bool TransformHelperVk::rayIntersectPlane(const glm::vec3& rayOrigin,
                                          const glm::vec3& rayDir,
                                          const glm::vec3& planePoint,
                                          const glm::vec3& planeNormal,
                                          float&           t) const
{
  float denom = glm::dot(rayDir, planeNormal);
  if(glm::abs(denom) < 1e-6f)
    return false;

  t = glm::dot(planePoint - rayOrigin, planeNormal) / denom;
  return t >= 0.0f;
}

bool TransformHelperVk::rayIntersectSphere(const glm::vec3& rayOrigin,
                                           const glm::vec3& rayDir,
                                           const glm::vec3& sphereCenter,
                                           float            sphereRadius,
                                           float&           t) const
{
  glm::vec3 oc   = rayOrigin - sphereCenter;
  float     a    = glm::dot(rayDir, rayDir);
  float     b    = 2.0f * glm::dot(oc, rayDir);
  float     c    = glm::dot(oc, oc) - sphereRadius * sphereRadius;
  float     disc = b * b - 4 * a * c;

  if(disc < 0)
    return false;

  t = (-b - glm::sqrt(disc)) / (2.0f * a);
  return t >= 0.0f;
}

glm::vec3 TransformHelperVk::getAxisFromComponent(GizmoComponent component) const
{
  switch(component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eRotateX:
    case GizmoComponent::eScaleX:
      return glm::vec3(1, 0, 0);
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eScaleY:
      return glm::vec3(0, 1, 0);
    case GizmoComponent::eTranslateZ:
    case GizmoComponent::eRotateZ:
    case GizmoComponent::eScaleZ:
      return glm::vec3(0, 0, 1);
    // Plane handles - return plane normal (perpendicular axis)
    case GizmoComponent::eTranslateXY:
      return glm::vec3(0, 0, 1);  // XY plane, Z normal
    case GizmoComponent::eTranslateXZ:
      return glm::vec3(0, 1, 0);  // XZ plane, Y normal
    case GizmoComponent::eTranslateYZ:
      return glm::vec3(1, 0, 0);  // YZ plane, X normal
    default:
      return glm::vec3(0, 0, 0);
  }
}

bool TransformHelperVk::shouldRenderGeometry(const GizmoGeometry& geom) const
{
  // Unified gizmo - visibility controlled by flags

  // Check if component is valid
  if(geom.component == GizmoComponent::eNone || geom.component == GizmoComponent::eRotateScreen)
    return false;

  // Check visibility flags based on component type
  switch(geom.component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eTranslateZ:
    case GizmoComponent::eTranslateXY:
    case GizmoComponent::eTranslateXZ:
    case GizmoComponent::eTranslateYZ:
      return (m_visibilityFlags & ShowTranslation) != 0;

    case GizmoComponent::eRotateX:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eRotateZ:
      return (m_visibilityFlags & ShowRotation) != 0;

    case GizmoComponent::eScaleX:
    case GizmoComponent::eScaleY:
    case GizmoComponent::eScaleZ:
    case GizmoComponent::eScaleUniform:
      return (m_visibilityFlags & ShowScale) != 0;

    default:
      return true;
  }
}

glm::vec3 TransformHelperVk::getColorFromComponent(GizmoComponent component, bool hovered) const
{
  glm::vec3 color;

  switch(component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eRotateX:
    case GizmoComponent::eScaleX:
      color = glm::vec3(1, 0, 0);
      break;
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eScaleY:
      color = glm::vec3(0, 1, 0);
      break;
    case GizmoComponent::eTranslateZ:
    case GizmoComponent::eRotateZ:
    case GizmoComponent::eScaleZ:
      color = glm::vec3(0, 0, 1);
      break;
    case GizmoComponent::eTranslateXY:
      color = glm::vec3(0, 1, 1);
      break;
    case GizmoComponent::eTranslateXZ:
      color = glm::vec3(1, 0, 1);
      break;
    case GizmoComponent::eTranslateYZ:
      color = glm::vec3(1, 1, 0);
      break;
    case GizmoComponent::eRotateScreen:
      color = glm::vec3(1, 1, 1);
      break;
    case GizmoComponent::eScaleUniform:
      color = glm::vec3(0.6f, 0.6f, 0.6f);  // Light gray, becomes white on hover
      break;
    default:
      color = glm::vec3(0.5f);
      break;
  }

  // Make hovered component much brighter (almost white)
  if(hovered)
    color = glm::mix(color, glm::vec3(1.0f), 0.7f);

  return color;
}

//-----------------------------------------------------------------------------
// Shader Compilation
//-----------------------------------------------------------------------------

bool TransformHelperVk::compileSlangShader(const std::string& filename, VkShaderModule& module)
{
  if(!m_slangCompiler)
  {
    LOGE("TransformHelperVk: No shader compiler provided\n");
    return false;
  }

  if(!m_slangCompiler->compileFile(filename))
  {
    return false;
  }

  if(module != VK_NULL_HANDLE)
    vkDestroyShaderModule(m_device, module, nullptr);

  // Create the VK module
  VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = m_slangCompiler->getSpirvSize(),
                                      .pCode    = m_slangCompiler->getSpirv()};

  if(m_slangCompiler->getSpirvSize() == 0)
  {
    LOGE("TransformHelperVk: Missing entry point in shader %s\n", filename.c_str());
    return false;
  }

  NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &module));
  NVVK_DBG_NAME(module);

  return true;
}

void TransformHelperVk::rebuildPipelines()
{
  // Destroy existing pipeline (keeps layout and descriptor sets)
  if(m_rasterPipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_rasterPipeline, nullptr);
    m_rasterPipeline = VK_NULL_HANDLE;
  }

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

  // Compile unified visual helpers shaders
  if(!compileSlangShader("visual_helpers.slang", m_vertexShader))
  {
    LOGE("TransformHelperVk: Failed to compile vertex shader\n");
    return;
  }

  if(!compileSlangShader("visual_helpers.slang", m_fragmentShader))
  {
    LOGE("TransformHelperVk: Failed to compile fragment shader\n");
    return;
  }

  // Create graphics pipeline
  nvvk::GraphicsPipelineState pipelineState;

  // Vertex input: position (vec3) + normal (vec3)
  pipelineState.vertexBindings = {{
      .binding = 0,
      .stride  = sizeof(GizmoVertex),
  }};

  pipelineState.vertexAttributes = {
      {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(GizmoVertex, position)},
      {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(GizmoVertex, normal)},
  };

  // Rasterization: disable culling so gizmo is visible from all angles
  // (rotation transforms can flip winding order, causing some rings to disappear)
  pipelineState.rasterizationState.cullMode  = VK_CULL_MODE_NONE;
  pipelineState.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  // Depth: test and write enabled for proper occlusion (gizmo parts occlude each other)
  pipelineState.depthStencilState.depthTestEnable  = VK_TRUE;
  pipelineState.depthStencilState.depthWriteEnable = VK_TRUE;
  pipelineState.depthStencilState.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

  // No alpha blending - render opaque
  pipelineState.colorBlendEnables[0] = VK_FALSE;

  // Create pipeline using nvvk helper (dynamic rendering)
  nvvk::GraphicsPipelineCreator creator;
  creator.pipelineInfo.layout                  = m_rasterPipelineLayout;
  creator.colorFormats                         = {m_colorFormat};
  creator.renderingState.depthAttachmentFormat = m_depthFormat;

  creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertmain", m_vertexShader);
  creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmain", m_fragmentShader);

  creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_rasterPipeline);
  NVVK_DBG_NAME(m_rasterPipeline);
}

}  // namespace vk_gaussian_splatting
