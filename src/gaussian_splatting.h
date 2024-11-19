/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//////////////////////////////////////////////////////////////////////////
/*

 This sample creates a 3D cube and render using the builtin camera

*/
//////////////////////////////////////////////////////////////////////////

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }

// clang-format on
#include <string>
#include <array>
#include <vulkan/vulkan_core.h>
// ply loader
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
// mathematics
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
// for parallel processing
#ifdef _WIN32
#include <ppl.h>
#include <execution>
#endif
// threading
#include <thread>
#include <condition_variable>
#include <mutex>
// time
#include<chrono>


//
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "imgui/imgui_axis.hpp"
//
#include "nvh/primitives.hpp"
//
#include "nvvk/context_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/extensions_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"
//
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#if USE_HLSL
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint8_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint8_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#elif USE_SLANG
#include "_autogen/raster_slang.h"
#else
#include "_autogen/raster.frag.glsl.h"
#include "_autogen/raster.vert.glsl.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert_glsl), std::end(raster_vert_glsl)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag_glsl), std::end(raster_frag_glsl)};
#endif  // USE_HLSL

// 
struct SampleTexture {
private:
  VkDevice          m_device{};
  uint32_t          m_queueIndex{ 0 };
  VkExtent2D        m_size{ 0, 0 };
  nvvk::Texture     m_texture;
  nvvkhl::AllocVma* m_alloc{ nullptr };

public:

  SampleTexture(VkDevice device, uint32_t queueIndex, nvvkhl::AllocVma* a)
    : m_device(device)
    , m_queueIndex(queueIndex)
    , m_alloc(a)
  {
  }

  ~SampleTexture(){
    destroy();
  }

  // Create the image, the sampler and the image view + generate the mipmap level for all
  void create(uint32_t width, uint32_t height, uint32_t bufsize, void* data, VkFormat format)
  {
    const VkSamplerCreateInfo sampler_info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    m_size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    const VkImageCreateInfo   create_info = nvvk::makeImage2DCreateInfo(m_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

    nvvk::CommandPool cpool(m_device, m_queueIndex);
    VkCommandBuffer   cmd = cpool.createCommandBuffer();
    m_texture = m_alloc->createTexture(cmd, bufsize, data, create_info, sampler_info);
    // no need nvvk::cmdGenerateMipmaps(cmd, m_texture.image, format, m_size, create_info.mipLevels);
    cpool.submitAndWait(cmd);
  }

  void destroy() 
  {
    // Destroying in next frame, avoid deleting while using
    nvvkhl::Application::submitResourceFree( [tex = m_texture, a = m_alloc]() { a->destroy(const_cast<nvvk::Texture&>(tex)); });
  }

  void               setSampler(const VkSampler& sampler) { m_texture.descriptor.sampler = sampler; }
  [[nodiscard]] bool isValid() const { return m_texture.image != nullptr; }
  [[nodiscard]] const VkDescriptorImageInfo& descriptor() const { return m_texture.descriptor; }
  [[nodiscard]] const VkExtent2D& getSize() const { return m_size; }
  [[nodiscard]] float getAspect() const { return static_cast<float>(m_size.width) / static_cast<float>(m_size.height); }


};


// TODO: for parallel sort, can be defined as a lambda
inline bool compare(const std::pair<float, int>& a, const std::pair<float, int>& b)
{
  return a.first > b.first;
}

// TODO: class documentation
class GaussianSplatting : public nvvkhl::IAppElement
{
public: // Methods specializing IAppElement

  GaussianSplatting():
    sortingThread([this] { this->sortingThreadFunc(); })  // starts the splat sorting thread
  {};
  
  ~GaussianSplatting() override
  {  
    // all threads must be stoped, 
    // work done in onDetach(), 
    // could be done here, same result
  };

  void onAttach(nvvkhl::Application* app) override;
  
  void onDetach() override;

  void onResize(uint32_t width, uint32_t height) override;
  
  void onRender(VkCommandBuffer cmd) override;

  void onUIRender() override;
    
  void onUIMenu() override {}

  void onFileDrop(const char* filename) override {} 

private: // structures and types

  // Storage for a 3D gaussian splatting (3DGS) model loaded from PLY file
  struct SplatSet {
    // standard poiont cloud attributes
    std::vector<float>       positions; // point positions (x,y,z)
    std::vector<float>       normals;   // point normals (x,y,z) - not used but stored in file
    // specific data fields introduced by INRIA for 3DGS
    std::vector<float>       f_dc;         // 3 components per point (f_dc_0, f_dc_1, f_dc_2 in ply file)
    std::vector<float>       f_rest;       // 45 components per point (f_rest_0 to f_rest_44 in ply file), SH coeficients
    std::vector<float>       opacity;      // 1 value per point in ply file
    std::vector<float>       scale;        // 3 components per point in ply file 
    std::vector<float>       rotation;     // 4 components per point in ply file - a quaternion
  };

private: // Methods

  // main loop of the sorting thread for gaussians
  // the thread is started by the class constructor 
  // then wait for triggers
  void sortingThreadFunc(void);

  void createScene();

  void createPipeline();

  void createGbuffers(const glm::vec2& size);

  void createVkBuffers();

  void destroyResources();

  // reset the attributes of the frameInformation that are 
  // modified by the user interface
  inline void resetFrameInfo() {
    frameInfo.splatScale = 1.0f;              // in [0.1,2.0]
    frameInfo.orthoZoom = 1.0f;               // in ?
    frameInfo.orthographicMode = 0;           // disabled, in {0,1}
    frameInfo.pointCloudModeEnabled = 0;      // disabled, in {0,1}
    frameInfo.sphericalHarmonicsDegree = 2;   // in {0,1,2}
    frameInfo.sphericalHarmonics8BitMode = 0; // disabled, in {0,1}
    frameInfo.showShOnly = 0;                 // disabled, in {0,1}
  }

  // Find the 3D position under the mouse cursor and set the camera interest to this position
  void rasterPicking();

  // Read the depth buffer at the X,Y coordinates
  // Note: depth format is VK_FORMAT_D32_SFLOAT
  float getDepth(int x, int y);

  // Utility function to compute the texture size according to the size of the data to be stored
  // TODO: doc of parameters
  glm::ivec2 computeDataTextureSize(int elementsPerTexel, int elementsPerSplat, int maxSplatCount);

  // create the texture maps on the device and upload the splat set data from host to device
  void create3dgsTextures(void);

  // to be placed at a better location
  bool loadPly(std::string filename, SplatSet& output);

private: // Attributes

  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::shared_ptr<nvvkhl::AllocVma> m_alloc;

  glm::vec2                        m_viewSize    = {0, 0};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.0F, 0.0F, 0.0F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                            // Descriptor set

  struct Vertex
  {
    glm::vec3 pos;
  };

  // Resources
  nvvk::Buffer m_frameInfo;
  nvvk::Buffer m_pixelBuffer;

  nvvk::Buffer  m_splatIndicesHost;   // Buffer of splat indices on host for transfers
  nvvk::Buffer  m_splatIndicesDevice; // Buffer of splat indices on device

  nvvk::Buffer  m_vertices; // Buffer of the vertices for the splat quad
  nvvk::Buffer  m_indices;  // Buffer of the indices for the splat quad
  VkSampler     m_sampler;  // texture sampler

  // Data and setting
  SplatSet m_splatSet;

  std::shared_ptr<SampleTexture> m_centersMap;
  std::shared_ptr<SampleTexture> m_colorsMap;
  std::shared_ptr<SampleTexture> m_covariancesMap;
  std::shared_ptr<SampleTexture> m_sphericalHarmonicsMap;

  glm::vec2 centersMapSize = { 0,0 };
  glm::vec2 colorsMapSize = { 0,0 };
  glm::vec2 covariancesMapSize = { 0,0 };
  glm::vec2 sphericalHarmonicsMapSize = { 0,0 };

  // threaded sorting
  std::vector<std::pair<float, int>> distArray;
  std::thread sortingThread;
  std::mutex mutex;
  std::condition_variable cond_var;
  bool sortStart = false;
  bool sortDone = false;
  bool sortExit = false;
  glm::vec3 sortDir;
  glm::vec3 sortCop;
  std::vector<uint32_t> gsIndex;
  std::vector<uint32_t> sortGsIndex;
  int m_sortTime = 0;

  // Pipeline
  DH::PushConstant m_pushConst{};                        // Information sent to the shader
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  DH::FrameInfo    frameInfo{}; // frame parameters, sent to device using a uniform buffer
};
