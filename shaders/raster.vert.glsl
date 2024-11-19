/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#include "device_host.h"

precision highp float;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in uint32_t inSplatIndex;

layout(location = 0) out vec2 outFragPos;
layout(location = 1) out vec4 outFragCol;

// we could write to manage alignment automatically
// layout(set = 0, binding = 0, scalar) uniform FrameInfo_
// but it may be less performant than aligning 
// attribute in the struct (see device_host.h comment)
layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

layout(set = 0, binding = 1) uniform sampler2D centersTexture;
layout(set = 0, binding = 2) uniform sampler2D colorsTexture;
layout(set = 0, binding = 3) uniform sampler2D covariancesTexture;
layout(set = 0, binding = 4) uniform sampler2D sphericalHarmonicsTexture;

//float splatIndex = float(gl_InstanceIndex);
uint32_t splatIndex = inSplatIndex;

const float sqrt8 = sqrt(8.0);
const float minAlpha = 1.0 / 255.0;

const vec4 encodeNorm4 = vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0);
const uvec4 mask4 = uvec4(uint(0x000000FF), uint(0x0000FF00), uint(0x00FF0000), uint(0xFF000000));
const uvec4 shift4 = uvec4(0, 8, 16, 24);

vec4 uintToRGBAVec(uint u) {
    uvec4 urgba = mask4 & u;
    urgba = urgba >> shift4;
    vec4 rgba = vec4(urgba) * encodeNorm4;
    return rgba;
}

vec2 getDataUV(in int stride, in int offset, in vec2 dimensions) {
    vec2 samplerUV = vec2(0.0, 0.0);
    float d = float(splatIndex * uint(stride) + uint(offset)) / dimensions.x;
    samplerUV.y = float(floor(d)) / dimensions.y;
    samplerUV.x = fract(d);
    return samplerUV;
}

vec2 getDataUVF(in uint sIndex, in float stride, in uint offset, in vec2 dimensions) {
    vec2 samplerUV = vec2(0.0, 0.0);
    float d = float(uint(float(sIndex) * stride) + offset) / dimensions.x;
    samplerUV.y = float(floor(d)) / dimensions.y;
    samplerUV.x = fract(d);
    return samplerUV;
}

const float SH_C1 = 0.4886025119029199f;
const float[5] SH_C2 = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);

const float SphericalHarmonics8BitCompressionRange = 3.0;
const float SphericalHarmonics8BitCompressionHalfRange = SphericalHarmonics8BitCompressionRange / 2.0;
const vec3 vec8BitSHShift = vec3(SphericalHarmonics8BitCompressionHalfRange);

//#define TEST
#ifdef TEST
void main()
{
  vec3 splatCenter = vec3(texture(centersTexture, getDataUV(1, 0, textureSize(centersTexture,0))));
  vec3 scaled = inPosition.xyz * vec3(0.01);           // manual scaling for debug test
  vec4 pos    = vec4(scaled + splatCenter, 1.0);
  gl_Position = frameInfo.projectionMatrix * frameInfo.viewMatrix * pos;
  outFragPos = pos.xy;
  //outFragCol = texture(colorsTexture, getDataUV(1, 0, textureSize(colorsTexture,0)));
  outFragCol = texture(covariancesTexture, getDataUV(1, 0, textureSize(covariancesTexture,0))) * 10.0;
}
#else

void main() {

    uint oddOffset = uint(splatIndex) & uint(0x00000001);
    uint doubleOddOffset = oddOffset * uint(2);
    bool isEven = oddOffset == uint(0);
    uint nearestEvenIndex = uint(splatIndex) - oddOffset;
    float fOddOffset = float(oddOffset);
    
    vec3 splatCenter = vec3(texture(centersTexture, getDataUV(1, 0, textureSize(centersTexture,0))));

    mat4 transformModelViewMatrix = frameInfo.viewMatrix;
    vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);

    vec4 clipCenter = frameInfo.projectionMatrix * viewCenter;

    float clip = 1.2 * clipCenter.w;
    if (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip || clipCenter.y < -clip || clipCenter.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    outFragPos = inPosition.xy;
    // vColor = uintToRGBAVec(sampledCenterColor.r);
    outFragCol = texture(colorsTexture, getDataUV(1, 0, textureSize(colorsTexture,0)));
    if (frameInfo.showShOnly == 1) {
       outFragCol.r = 0.5;
       outFragCol.g = 0.5;
       outFragCol.b = 0.5;
    }
    
    if (frameInfo.sphericalHarmonicsDegree >= 1) {

        vec3 worldViewDir = normalize(splatCenter - frameInfo.cameraPosition);
        // worldViewDir.y = -worldViewDir.y; // left handed (mm) to right handed (3.js) coordinate system

        vec4 sampledSH0123   = texture(sphericalHarmonicsTexture, getDataUV(6, 0, textureSize(sphericalHarmonicsTexture,0)));
        vec4 sampledSH4567   = texture(sphericalHarmonicsTexture, getDataUV(6, 1, textureSize(sphericalHarmonicsTexture,0)));
        vec4 sampledSH891011 = texture(sphericalHarmonicsTexture, getDataUV(6, 2, textureSize(sphericalHarmonicsTexture,0)));
        vec3 sh1 = sampledSH0123.rgb;
        vec3 sh2 = vec3(sampledSH0123.a, sampledSH4567.rg);
        vec3 sh3 = vec3(sampledSH4567.ba, sampledSH891011.r);

        
        //if (sphericalHarmonics8BitMode == 1) {
        //    sh1 = sh1 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //    sh2 = sh2 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //    sh3 = sh3 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
        //}
        
        float x = worldViewDir.x;
        float y = worldViewDir.y;
        float z = worldViewDir.z;
        outFragCol.rgb += SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);

        if (frameInfo.sphericalHarmonicsDegree >= 2) {

            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;

            vec4 sampledSH12131415 = texture(sphericalHarmonicsTexture, getDataUV(6, 3, textureSize(sphericalHarmonicsTexture,0)));
            vec4 sampledSH16171819 = texture(sphericalHarmonicsTexture, getDataUV(6, 4, textureSize(sphericalHarmonicsTexture,0)));
            vec4 sampledSH20212223 = texture(sphericalHarmonicsTexture, getDataUV(6, 5, textureSize(sphericalHarmonicsTexture,0)));

            vec3 sh4 = sampledSH891011.gba;
            vec3 sh5 = sampledSH12131415.rgb;
            vec3 sh6 = vec3(sampledSH12131415.a, sampledSH16171819.rg);
            vec3 sh7 = vec3(sampledSH16171819.ba, sampledSH20212223.r);
            vec3 sh8 = sampledSH20212223.gba;

            //if (sphericalHarmonics8BitMode == 1) {
            //    sh4 = sh4 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
            //    sh5 = sh5 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
            //    sh6 = sh6 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
            //    sh7 = sh7 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
            //    sh8 = sh8 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
            //}

            outFragCol.rgb +=
                (SH_C2[0] * xy) * sh4 +
                (SH_C2[1] * yz) * sh5 +
                (SH_C2[2] * (2.0 * zz - xx - yy)) * sh6 +
                (SH_C2[3] * xz) * sh7 +
                (SH_C2[4] * (xx - yy)) * sh8;
        }
    }
    
    // Original code to use RGBA texture map (is is faster ? to check)
    vec4 sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset, textureSize(covariancesTexture,0)));
    vec4 sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1), textureSize(covariancesTexture,0)));

    vec3 cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) + vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
    vec3 cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) + vec3(sampledCovarianceB.gba) * fOddOffset;
    
    // Construct the 3D covariance matrix
    mat3 Vrk = mat3(
        cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
        cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
        cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
    );

    mat3 J;
    if (frameInfo.orthographicMode == 1) {
        // Since the projection is linear, we don't need an approximation
        J = transpose(mat3(frameInfo.orthoZoom, 0.0, 0.0,
            0.0, frameInfo.orthoZoom, 0.0,
            0.0, 0.0, 0.0));
    }
    else {
        // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
        // 3D covariance matrix instead of using the actual projection matrix because that transformation would
        // require a non-linear component (perspective division) which would yield a non-gaussian result.
        float s = 1.0 / (viewCenter.z * viewCenter.z);
        J = mat3(
            frameInfo.focal.x / viewCenter.z, 0., -(frameInfo.focal.x * viewCenter.x) * s,
            0., frameInfo.focal.y / viewCenter.z, -(frameInfo.focal.y * viewCenter.y) * s,
            0., 0., 0.
        );
    }

    // Concatenate the projection approximation with the model-view transformation
    mat3 W = transpose(mat3(transformModelViewMatrix));
    mat3 T = W * J;

    // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
    mat3 cov2Dm = transpose(T) * Vrk * T;

    float compensation = 1.0;

    bool antialiased = false;
    if (antialiased) {
        float detOrig = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
        cov2Dm[0][0] += 0.3;
        cov2Dm[1][1] += 0.3;
        float detBlur = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
        compensation = sqrt(max(detOrig / detBlur, 0.0));
    }
    else {
        cov2Dm[0][0] += 0.3;
        cov2Dm[1][1] += 0.3;
        compensation = 1.0;
    }

    outFragCol.a *= compensation;

    // from original code
    if (outFragCol.a < minAlpha) {
        // JEM added gl_position set for proper discard
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
        }

    // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
    // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
    // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
    // need cov2Dm[1][0] because it is a symetric matrix.
    vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

    vec3 ndcCenter = clipCenter.xyz / clipCenter.w;

    // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
    // so that we can determine the 2D basis for the splat. This is done using the method described
    // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
    // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * eigen-value), which is
    // equal to scaling them by sqrt(8) standard deviations.
    //
    // This is a different approach than in the original work at INRIA. In that work they compute the
    // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
    // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
    // times the maximum eigen-value, or 3 standard deviations. They then use the inverse 2D covariance
    // matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by calculating the
    // full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
    float a = cov2Dv.x;
    float d = cov2Dv.z;
    float b = cov2Dv.y;
    float D = a * d - b * b;
    float trace = a + d;
    float traceOver2 = 0.5 * trace;
    float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
    float eigenValue1 = traceOver2 + term2;
    float eigenValue2 = traceOver2 - term2;

    if (frameInfo.pointCloudModeEnabled == 1) 
    {
        eigenValue1 = eigenValue2 = 0.2;
    }

    // from original code
    if (eigenValue2 <= 0.0) {
        // JEM added gl_position set for proper discard
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
    // since the eigen vectors are orthogonal, we derive the second one from the first
    vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

    // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
    vec2 basisVector1 = eigenVector1 * frameInfo.splatScale * min(sqrt8 * sqrt(eigenValue1), 2048.0);
    vec2 basisVector2 = eigenVector2 * frameInfo.splatScale * min(sqrt8 * sqrt(eigenValue2), 2048.0);

    vec2 ndcOffset = vec2(outFragPos.x * basisVector1 + outFragPos.y * basisVector2) *  frameInfo.basisViewport * 2.0 * frameInfo.inverseFocalAdjustment;

    vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
    gl_Position = quadPos;
    
    // Scale the position data we send to the fragment shader
    outFragPos *= sqrt8;
}
#endif