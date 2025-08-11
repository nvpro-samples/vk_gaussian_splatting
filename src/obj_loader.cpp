/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#define TINYOBJLOADER_IMPLEMENTATION
#include "obj_loader.h"
#include <nvutils/logger.hpp>

bool ObjLoader::load(const std::filesystem::path& filename)
{
  // reset the mesh batch
  reset();

  //
  tinyobj::ObjReader reader;
  reader.ParseFromFile(filename.string());
  if(!reader.Valid())
  {
    LOGE("Cannot load %s: %s", filename.string().c_str(), reader.Error().c_str());
    return false;
  }

  this->filename = filename;

  // Collecting the material in the scene
  for(const auto& material : reader.GetMaterials())
  {
    ObjMaterial m;
    m.ambient       = glm::vec3(material.ambient[0], material.ambient[1], material.ambient[2]);
    m.diffuse       = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
    m.specular      = glm::vec3(material.specular[0], material.specular[1], material.specular[2]);
    m.emission      = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
    m.transmittance = glm::vec3(material.transmittance[0], material.transmittance[1], material.transmittance[2]);
    m.dissolve      = material.dissolve;
    m.ior           = material.ior;
    m.shininess     = material.shininess;
    if(m.transmittance.x != 0.0 || m.transmittance.y != 0.0 || m.transmittance.x != 0.0)
      m.illum = 2;  // refractive
    else if(material.illum > 1)
      m.illum = 1;  // reflective
    else
      m.illum = 0;

    if(!material.diffuse_texname.empty())
    {
      m_textures.push_back(material.diffuse_texname);
      m.textureID = static_cast<int>(m_textures.size()) - 1;
    }

    m_materials.emplace_back(m);
    m_matNames.emplace_back(material.name);
  }

  // If there were none, add a default
  if(m_materials.empty())
  {
    m_materials.emplace_back(ObjMaterial());
    m_matNames.emplace_back("Default");
  }

  const tinyobj::attrib_t& attrib = reader.GetAttrib();

  // storage to generate some normal vectors if needed
  std::vector<uint8_t>   visited;
  std::vector<glm::vec3> normals;

  for(const auto& shape : reader.GetShapes())
  {
    m_vertices.reserve(shape.mesh.indices.size() + m_vertices.size());
    m_indices.reserve(shape.mesh.indices.size() + m_indices.size());
    m_matIndices.insert(m_matIndices.end(), shape.mesh.material_ids.begin(), shape.mesh.material_ids.end());

    // If we do not have normal vectors we generate some
    if(true)  //attrib.normals.empty())
    {
      // Compute per vertex normal when no normal were provided.
      // no smoothing groups or crease angle. avarages per face
      // normals to compute per vertex normals
      visited.resize(visited.size() + shape.mesh.indices.size(), 0);
      normals.resize(normals.size() + shape.mesh.indices.size());

      // iterate over the faces
      for(size_t i = 0; i < shape.mesh.indices.size(); i += 3)
      {
        const auto& index0 = shape.mesh.indices[i + 0].vertex_index;
        const auto& index1 = shape.mesh.indices[i + 1].vertex_index;
        const auto& index2 = shape.mesh.indices[i + 2].vertex_index;
        const auto  vp0 =
            glm::vec3(attrib.vertices[3 * index0], attrib.vertices[3 * index0 + 1], attrib.vertices[3 * index0 + 2]);
        const auto vp1 =
            glm::vec3(attrib.vertices[3 * index1], attrib.vertices[3 * index1 + 1], attrib.vertices[3 * index1 + 2]);
        const auto vp2 =
            glm::vec3(attrib.vertices[3 * index2], attrib.vertices[3 * index2 + 1], attrib.vertices[3 * index2 + 2]);

        glm::vec3 n = glm::normalize(glm::cross((vp1 - vp0), (vp2 - vp0)));

        glm::vec3& nrm0 = normals[index0];
        glm::vec3& nrm1 = normals[index1];
        glm::vec3& nrm2 = normals[index2];

        // dispatch the normal to each of the face vertex
        if(visited[index0])
        {
          nrm0 = glm::mix(nrm0, n, 0.5);
        }
        else
        {
          nrm0            = n;
          visited[index0] = 1;
        }
        if(visited[index1])
        {
          nrm1 = glm::mix(nrm1, n, 0.5);
        }
        else
        {
          nrm1            = n;
          visited[index1] = 1;
        }
        if(visited[index2])
        {
          nrm2 = glm::mix(nrm2, n, 0.5);
        }
        else
        {
          nrm2            = n;
          visited[index2] = 1;
        }
      }
    }

    // prepare output so that all attributes uses
    // same primary index for rendering
    for(const auto& index : shape.mesh.indices)
    {
      ObjVertex    vertex = {};
      const float* vp     = &attrib.vertices[3 * index.vertex_index];
      vertex.pos          = {*(vp + 0), *(vp + 1), *(vp + 2)};

      if(!attrib.normals.empty() && index.normal_index >= 0)
      {
        const float* np = &attrib.normals[3 * index.normal_index];
        vertex.nrm      = {*(np + 0), *(np + 1), *(np + 2)};
      }
      else
      {
        vertex.nrm = normals[index.vertex_index];
      }

      /*
      if(!attrib.texcoords.empty() && index.texcoord_index >= 0)
      {
        const float* tp = &attrib.texcoords[2 * index.texcoord_index + 0];
        vertex.texCoord = {*tp, 1.0f - *(tp + 1)};
      }

      if(!attrib.colors.empty())
      {
        const float* vc = &attrib.colors[3 * index.vertex_index];
        vertex.color    = {*(vc + 0), *(vc + 1), *(vc + 2)};
      }
      */
      m_vertices.push_back(vertex);
      m_indices.push_back(static_cast<int>(m_indices.size()));
    }
  }

  // Fixing material indices
  for(auto& mi : m_matIndices)
  {
    if(mi < 0 || mi > m_materials.size())
      mi = 0;
  }

  if(!isValid())
  {
    LOGE("Invalid Obj file %s \n", filename.c_str());
    return false;
  }

  return true;
}
