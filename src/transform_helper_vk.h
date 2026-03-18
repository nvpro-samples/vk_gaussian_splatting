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

#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/staging.hpp>
#include <nvapp/application.hpp>
#include <nvslang/slang.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <memory>
#include <vector>
#include <functional>

#include "visual_helpers_shaderio.h.slang"

namespace vk_gaussian_splatting {

// Forward declaration
class TransformHelperVk;

//-----------------------------------------------------------------------------
// Transform Helper for 3D Gizmo Manipulation and Rasterization
// Provides 3D Helpers for transforming objects.
// Supports translate, rotate, and scale modes in world space.
//-----------------------------------------------------------------------------

class TransformHelperVk
{
public:
  //-----------------------------------------------------------------------------
  // Enums
  //-----------------------------------------------------------------------------

  enum class TransformMode
  {
    eTranslate,
    eRotate,
    eScale
  };

  enum class TransformSpace
  {
    eWorld,
    // eLocal // Not supported yet
  };

  enum class GizmoComponent
  {
    eNone = 0,
    // Translation
    eTranslateX,
    eTranslateY,
    eTranslateZ,
    eTranslateXY,
    eTranslateXZ,
    eTranslateYZ,
    // Rotation
    eRotateX,
    eRotateY,
    eRotateZ,
    eRotateScreen,
    // Scale
    eScaleX,
    eScaleY,
    eScaleZ,
    eScaleUniform
  };

  enum class GeometryType
  {
    eShaft,     // Elongated axis shaft (cylinder/box)
    eCone,      // Cone tip for translation arrows
    eBox,       // Box at axis end for scale
    eTorus,     // Rotation ring
    ePlane,     // Translation plane handle
    eCenterBox  // Center uniform scale box
  };

  // Flags for controlling which gizmo operations are visible
  enum GizmoVisibilityFlags : uint32_t
  {
    ShowTranslation = 1 << 0,  // Show translation shafts and planes
    ShowRotation    = 1 << 1,  // Show rotation rings
    ShowScale       = 1 << 2,  // Show scale boxes
    ShowAll         = ShowTranslation | ShowRotation | ShowScale
  };

  //-----------------------------------------------------------------------------
  // Structures
  //-----------------------------------------------------------------------------

  struct GizmoVertex
  {
    glm::vec3 position;
    glm::vec3 normal;
  };

  // Import unified push constants from shared shader I/O header
  using PushConstantGizmo = shaderio::visual_helpers::PushConstantVisualHelpers;

  struct GizmoGeometry
  {
    nvvk::Buffer   vertexBuffer;
    nvvk::Buffer   indexBuffer;
    uint32_t       indexCount = 0;
    glm::vec3      color      = glm::vec3(1.0f);
    GizmoComponent component  = GizmoComponent::eNone;
    GeometryType   type       = GeometryType::eShaft;
    glm::vec3      boundsMin  = glm::vec3(0.0f);
    glm::vec3      boundsMax  = glm::vec3(0.0f);
  };

  // Resources needed for initialization
  struct Resources
  {
    nvapp::Application*      app           = nullptr;
    nvvk::ResourceAllocator* alloc         = nullptr;
    nvvk::StagingUploader*   uploader      = nullptr;
    VkDevice                 device        = VK_NULL_HANDLE;
    VkSampler                sampler       = VK_NULL_HANDLE;
    nvslang::SlangCompiler*  slangCompiler = nullptr;                        // Shared shader compiler
    VkFormat                 colorFormat   = VK_FORMAT_R16G16B16A16_SFLOAT;  // Render target format
    VkFormat                 depthFormat   = VK_FORMAT_D32_SFLOAT;           // Depth buffer format
  };

  //-----------------------------------------------------------------------------
  // Lifecycle
  //-----------------------------------------------------------------------------

  void init(const Resources& res);
  void deinit();

  //-----------------------------------------------------------------------------
  // Transform Attachment API (Entity-Agnostic)
  //-----------------------------------------------------------------------------

  // Attach to position/rotation/scale for manipulation
  // visibilityFlags: controls which gizmo parts are visible (default: ShowAll)
  void attachTransform(glm::vec3* position, glm::vec3* rotation, glm::vec3* scale, uint32_t visibilityFlags = ShowAll);

  // Check if currently attached
  bool isAttached() const;

  // Clear current attachment
  void clearAttachment();

  //-----------------------------------------------------------------------------
  // Configuration
  //-----------------------------------------------------------------------------

  void setTransformMode(TransformMode mode);
  void setTransformSpace(TransformSpace space);
  void setSnapEnabled(bool enabled) { m_enableSnapping = enabled; }
  void setSnapValues(float translate, float rotate, float scale);
  void setGizmoSize(float sizePixels) { m_gizmoSize = sizePixels; }

  TransformMode  getTransformMode() const { return m_mode; }
  TransformSpace getTransformSpace() const { return m_space; }
  bool           isSnapEnabled() const { return m_enableSnapping; }

  //-----------------------------------------------------------------------------
  // Interaction
  //-----------------------------------------------------------------------------

  // Process input and return true if gizmo is being manipulated
  bool processInput(const glm::vec2& mousePos,
                    const glm::vec2& mouseDelta,
                    bool             mouseDown,
                    bool             mousePressed,
                    bool             mouseReleased,
                    const glm::mat4& viewMatrix,
                    const glm::mat4& projMatrix,
                    const glm::vec2& viewport);

  //-----------------------------------------------------------------------------
  // Rendering
  //-----------------------------------------------------------------------------

  void renderRaster(VkCommandBuffer  cmd,
                    VkDescriptorSet  sceneDescriptorSet,
                    VkDescriptorSet  helperDescriptorSet,  // Set 1: scene depth texture
                    const glm::mat4& viewMatrix,
                    const glm::mat4& projMatrix,
                    const glm::vec2& viewportSize,
                    const glm::vec2& depthBufferSize);  // DLSS render size or viewport size

  // Get descriptor set layout (for creating descriptor sets in host app)
  VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }

  //-----------------------------------------------------------------------------
  // Callbacks
  //-----------------------------------------------------------------------------

  void setOnTransformBegin(std::function<void()> callback) { m_onTransformBegin = callback; }
  void setOnTransformChange(std::function<void()> callback) { m_onTransformChange = callback; }
  void setOnTransformEnd(std::function<void()> callback) { m_onTransformEnd = callback; }

  //-----------------------------------------------------------------------------
  // Mode and Space Control
  //-----------------------------------------------------------------------------

  // Unified gizmo - no mode selection needed (operation determined by component selection)
  // Mode is kept internally for compatibility but not used for UI

  void           setSpace(TransformSpace space) { m_space = space; }
  TransformSpace getSpace() const { return m_space; }

  // Check if currently dragging (for blocking camera input)
  bool isDragging() const { return m_isDragging; }

  //-----------------------------------------------------------------------------
  // Update Notifications
  //-----------------------------------------------------------------------------

  // Call this when an external system modifies the selected object's transform
  void notifyExternalTransformChange();

  //-----------------------------------------------------------------------------
  // Shader Management
  //-----------------------------------------------------------------------------

  // Rebuild pipelines after shader recompilation (compiles shaders internally)
  void rebuildPipelines();

private:
  //-----------------------------------------------------------------------------
  // Shader Compilation
  //-----------------------------------------------------------------------------

  // Shader compilation helper
  bool compileSlangShader(const std::string& filename, VkShaderModule& module);

  //-----------------------------------------------------------------------------
  // Geometry Generation (using nvutils primitives)
  //-----------------------------------------------------------------------------

  void createGizmoGeometry();
  void destroyGizmoGeometry();
  void generateTranslateGizmo();
  void generateRotateGizmo();
  void generateScaleGizmo();

  //-----------------------------------------------------------------------------
  // Rendering
  //-----------------------------------------------------------------------------

  void createRasterPipeline();
  void destroyRasterPipeline();

  //-----------------------------------------------------------------------------
  // Picking and Interaction
  //-----------------------------------------------------------------------------

  GizmoComponent pickGizmoComponent(const glm::vec2& mousePos,
                                    const glm::mat4& viewMatrix,
                                    const glm::mat4& projMatrix,
                                    const glm::vec2& viewport);

  void startDrag(GizmoComponent   component,
                 const glm::vec2& mousePos,
                 const glm::mat4& viewMatrix,
                 const glm::mat4& projMatrix,
                 const glm::vec2& viewport);

  void updateDrag(const glm::vec2& mousePos,
                  const glm::vec2& mouseDelta,
                  const glm::mat4& viewMatrix,
                  const glm::mat4& projMatrix,
                  const glm::vec2& viewport);

  void endDrag();

  //-----------------------------------------------------------------------------
  // Transform Application
  //-----------------------------------------------------------------------------

  void applyTranslation(const glm::vec3& delta);
  void applyRotation(const glm::vec3& axis, float angleDeg);
  void applyScale(const glm::vec3& scale);

  //-----------------------------------------------------------------------------
  // Utility
  //-----------------------------------------------------------------------------

  glm::mat4 getGizmoTransform() const;
  glm::vec3 getAttachedPosition() const;
  glm::quat getAttachedRotation() const;
  glm::vec3 getAttachedScale() const;

  float getScreenSpaceScale(const glm::vec3& worldPos, const glm::mat4& viewMatrix, const glm::mat4& projMatrix, const glm::vec2& viewport) const;

  bool rayIntersectPlane(const glm::vec3& rayOrigin,
                         const glm::vec3& rayDir,
                         const glm::vec3& planePoint,
                         const glm::vec3& planeNormal,
                         float&           t) const;

  bool rayIntersectSphere(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& sphereCenter, float sphereRadius, float& t) const;

  void createRayFromMouse(const glm::vec2& mousePos,
                          const glm::mat4& viewMatrix,
                          const glm::mat4& projMatrix,
                          const glm::vec2& viewport,
                          glm::vec3&       rayOrigin,
                          glm::vec3&       rayDir) const;

  bool rayIntersectsBounds(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& boundsMin, const glm::vec3& boundsMax) const;

  float rayDistanceToBounds(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& boundsMin, const glm::vec3& boundsMax) const;

  float rayDistanceToComponent(const glm::vec3& rayOrigin, const glm::vec3& rayDir, GizmoComponent component) const;

  glm::vec3 getAxisFromComponent(GizmoComponent component) const;
  glm::vec3 getColorFromComponent(GizmoComponent component, bool hovered = false) const;
  bool      shouldRenderGeometry(const GizmoGeometry& geom) const;

  //-----------------------------------------------------------------------------
  // Member Variables
  //-----------------------------------------------------------------------------

  // Context
  nvapp::Application*      m_app           = nullptr;
  nvvk::ResourceAllocator* m_alloc         = nullptr;
  nvvk::StagingUploader*   m_uploader      = nullptr;
  VkDevice                 m_device        = VK_NULL_HANDLE;
  VkSampler                m_sampler       = VK_NULL_HANDLE;
  nvslang::SlangCompiler*  m_slangCompiler = nullptr;
  VkFormat                 m_colorFormat   = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkFormat                 m_depthFormat   = VK_FORMAT_D32_SFLOAT;

  // Attached Transform Components (Entity-Agnostic)
  glm::vec3* m_attachedPosition = nullptr;
  glm::vec3* m_attachedRotation = nullptr;  // Euler angles in degrees
  glm::vec3* m_attachedScale    = nullptr;
  uint32_t   m_visibilityFlags  = ShowAll;  // Controls which gizmo parts are visible

  // Configuration
  TransformMode  m_mode           = TransformMode::eTranslate;
  TransformSpace m_space          = TransformSpace::eWorld;
  bool           m_enableSnapping = false;
  float          m_snapTranslate  = 0.25f;
  float          m_snapRotate     = 15.0f;
  float          m_snapScale      = 0.1f;
  float          m_gizmoSize      = 100.0f;  // Size in pixels

  // Interaction State
  bool           m_isDragging        = false;
  GizmoComponent m_hoveredComponent  = GizmoComponent::eNone;
  GizmoComponent m_draggedComponent  = GizmoComponent::eNone;
  glm::vec2      m_dragStartPosMouse = glm::vec2(0.0f);
  glm::vec3      m_dragStartPosition = glm::vec3(0.0f);
  glm::vec3      m_dragStartRotation = glm::vec3(0.0f);
  glm::vec3      m_dragStartScale    = glm::vec3(1.0f);
  glm::vec3      m_dragStartHitPoint = glm::vec3(0.0f);  // Initial 3D hit point for relative dragging

  // Geometry
  std::vector<GizmoGeometry> m_gizmoGeometry;

  // Raster Rendering
  VkPipeline            m_rasterPipeline       = VK_NULL_HANDLE;
  VkPipelineLayout      m_rasterPipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descriptorSetLayout  = VK_NULL_HANDLE;
  VkDescriptorPool      m_descriptorPool       = VK_NULL_HANDLE;
  VkDescriptorSet       m_descriptorSet        = VK_NULL_HANDLE;
  VkShaderModule        m_vertexShader         = VK_NULL_HANDLE;
  VkShaderModule        m_fragmentShader       = VK_NULL_HANDLE;

  // Callbacks
  std::function<void()> m_onTransformBegin;
  std::function<void()> m_onTransformChange;
  std::function<void()> m_onTransformEnd;
};

}  // namespace vk_gaussian_splatting
