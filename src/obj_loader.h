/*/*
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
#include <filesystem>
#include <glm/glm.hpp>
#include <tinyobjloader/tiny_obj_loader.h>

#include "shaderio.h"

// Use unified ObjMaterial from shaderio (shared with GPU shaders)
using ObjMaterial = shaderio::ObjMaterial;

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
  std::vector<uint32_t>    m_matIndices;  // per face material indices (unsigned for consistency)
  std::vector<ObjMaterial> m_materials;   // materials
  std::vector<std::string> m_matNames;    // material names
  std::vector<std::string> m_textures;    // texture names
};
