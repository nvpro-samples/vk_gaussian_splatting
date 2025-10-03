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

#extension GL_EXT_shader_explicit_arithmetic_types : require

#ifndef _GUT_DEFINITIONS_H_
#define _GUT_DEFINITIONS_H_

#include "shaderio.h"

// File translated from github-grut\threedgut_tracer\include\3dgut\threedgut.cuh

/* from setup_3dgut.py
ut_d = 3
ut_alpha = conf.render.splat.ut_alpha
ut_beta = conf.render.splat.ut_beta
ut_kappa = conf.render.splat.ut_kappa
ut_delta = math.sqrt(ut_alpha*ut_alpha*(ut_d+ut_kappa))
*/

/* from 3dgut.yaml
n_rolling_shutter_iterations : 5 
ut_alpha : 1.0 
ut_beta : 2.0 
ut_kappa : 0.0 
ut_in_image_margin_factor : 0.1 
ut_require_all_sigma_points_valid : false
*/

#define GUT_N_ROLLING_SHUTTER_ITERATIONS 5
#define GUT_D 3
#define GUT_ALPHA 1.0
#define GUT_BETA 2.0
#define GUT_KAPPA 0.0
#define GUT_DELTA 1.73205080757  // sqrt(GAUSSIAN_UT_ALPHA * GAUSSIAN_UT_ALPHA * (D + GAUSSIAN_UT_KAPPA));
#define GUT_LAMBDA 0.0           // UTParams.Alpha * UTParams.Alpha * (UTParams.D + UTParams.Kappa) - UTParams.D;
#define GUT_IN_IMAGE_MARGIN_FACTOR 0.1
#define GUT_REQUIRE_ALL_SIGMA_POINTS_VALID false

#define GUT_COVARIANCE_DILATION 0.3
#define GUT_ALPHA_THRESHOLD 0.01
#define GUT_TIGHT_OPACITY_BOUNDING true
#define GUT_RECT_BOUNDING true

#endif
