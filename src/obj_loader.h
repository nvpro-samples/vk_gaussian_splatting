/*/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>
#include <filesystem>
#include <glm/glm.hpp>
#include <tinyobjloader/tiny_obj_loader.h>

// Structure holding the material
struct ObjMaterial
{
  glm::vec3 ambient       = glm::vec3(0.1f, 0.1f, 0.1f);
  glm::vec3 diffuse       = glm::vec3(0.7f, 0.7f, 0.7f);
  glm::vec3 specular      = glm::vec3(1.0f, 1.0f, 1.0f);
  glm::vec3 transmittance = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 emission      = glm::vec3(0.0f, 0.0f, 0.10);
  float     shininess     = 0.f;
  float     ior           = 1.0f;  // index of refraction
  float     dissolve      = 1.f;   // not used
  int       illum         = 0;     // 0 opaque, 1 reflective, 2 refractive
  int       textureID     = -1;
};
// OBJ representation of a vertex
// NOTE: BLAS builder depends on pos being the first member
struct ObjVertex
{
  glm::vec3 pos;
  glm::vec3 nrm;
  //glm::vec3 color;
  //glm::vec2 texCoord;
};


struct ObjShape
{
  uint32_t offset;
  uint32_t nbIndex;
  uint32_t matIndex;
};

class ObjLoader
{
public:
  bool load(const std::filesystem::path& filename);

  void reset(void)
  {
    m_vertices.clear();
    m_indices.clear();
    m_materials.clear();
    m_textures.clear();
    m_matIndices.clear();
  }

  bool isValid(void) const { return !m_vertices.empty() && !m_indices.empty(); }

  std::filesystem::path    filename;
  std::vector<ObjVertex>   m_vertices;    // triangle vertices
  std::vector<uint32_t>    m_indices;     // triangle vertex indices
  std::vector<int32_t>     m_matIndices;  // per face material indices
  std::vector<ObjMaterial> m_materials;   // materials
  std::vector<std::string> m_matNames;    // material names
  std::vector<std::string> m_textures;    // texture names
};
