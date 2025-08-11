/*
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

#include "parameters.h"

namespace vk_gaussian_splatting {

// no reset function on purpose
extern SceneParameters prmScene{};

extern VramDataParameters    prmData{};
extern RtxVramDataParameters prmRtxData{};

// no reset function on purpose
extern uint32_t            prmSelectedPipeline = PIPELINE_MESH;
extern shaderio::FrameInfo prmFrame{};
extern RenderParameters    prmRender{};
extern RasterParameters    prmRaster{};
extern RtxParameters       prmRtx{};

// Storage for respective default values

static VramDataParameters    prmDataDefault{};
static RtxVramDataParameters prmRtxDataDefault{};

static shaderio::FrameInfo prmFrameDefault{};
static RenderParameters    prmRenderDefault{};
static RasterParameters    prmRasterDefault{};
static RtxParameters       prmRtxDefault{};

void storeDefaultParameters()
{
  prmDataDefault    = prmData;
  prmRtxDataDefault = prmRtxData;

  prmFrameDefault  = prmFrame;
  prmRenderDefault = prmRender;
  prmRasterDefault = prmRaster;
  prmRtxDefault    = prmRtx;
}

void resetDataParameters()
{
  prmData = prmDataDefault;
}
void resetRtxDataParameters()
{
  prmRtxData = prmRtxDataDefault;
}
void resetFrameParameters()
{
  prmFrame = prmFrameDefault;
}
void resetRenderParameters()
{
  prmRender = prmRenderDefault;
}
void resetRasterParameters()
{
  prmRaster = prmRasterDefault;
}
void resetRtxParameters()
{
  prmRtx = prmRtxDefault;
}

void registerCommandLineParameters(nvutils::ParameterRegistry* parameterRegistry)
{
  // Scene
  parameterRegistry->add({"inputFile", "load a ply file"}, {".ply"}, &prmScene.sceneToLoadFilename);
#ifdef WITH_DEFAULT_SCENE_FEATURE
  parameterRegistry->add({"loadDefaultScene", "0=disable the load of a default scene when no ply file is provided"},
                         &prmScene.enableDefaultScene);
#endif
  // Projects
  parameterRegistry->add({"inputProject", "load a vkgs project file"}, {".vkgs"}, &prmScene.projectToLoadFilename);

  // Data
  parameterRegistry->add({"shformat", "0=fp32 1=fp16 2=uint8"}, &prmData.shFormat);
  parameterRegistry->add({"useAABBs", "0(Default)=use icosahedron 3D mesh and built-in triangle/ray intersection, 1=use AABBs and parametric intersection shader. DO NOT COMBINE with useTlasInstances=0."},
                         &prmRtxData.useAABBs);
  parameterRegistry->add({"useTlasInstances", "1(default)=use one TLAS instance per particle and a small unit particle BLAS. 0=use one TLAS entry and a large BLAS."},
                         &prmRtxData.useTlasInstances);
  parameterRegistry->add({"compressBlas", "1(default)=compress BLAS. 0=diabled."}, &prmRtxData.compressBlas);

  // Pipelines
  parameterRegistry->add({"pipeline", "0=vert 1(default)=mesh 2=raytracing 3=hybrid"}, &prmSelectedPipeline);
  parameterRegistry->add({"maxShDegree", "max sh degree used for rendering in [0,1,2,3]"}, &prmRender.maxShDegree);
}

}  // namespace vk_gaussian_splatting
