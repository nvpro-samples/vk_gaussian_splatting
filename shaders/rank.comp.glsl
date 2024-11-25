#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#include "device_host.h"

// we could write to manage alignment automatically
// layout(set = 0, binding = 0, scalar) uniform FrameInfo_
// but it may be less performant than aligning
// attribute in the struct (see device_host.h comment)
layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(local_size_x = 256) in;

layout(set = 0, binding = 1) uniform sampler2D centersTexture;

// Key contains nbSplat keys + nbSplatSamples + nbSplats
layout(std430, set = 0, binding = 3) writeonly buffer InstanceKey
{
  uint32_t key[];
};

vec2 getDataUV(in int index, in int stride, in int offset, in vec2 dimensions)
{
  vec2  samplerUV = vec2(0.0, 0.0);
  float d         = float(index * uint(stride) + uint(offset)) / dimensions.x;
  samplerUV.y     = float(floor(d)) / dimensions.y;
  samplerUV.x     = fract(d);
  return samplerUV;
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if(id >= frameInfo.splatCount)
    return;
  
  vec4 pos = vec3(texture(centersTexture, getDataUV(id, 1, 0, textureSize(centersTexture, 0))),1.0f);
  //pos = projection * view * model * pos;
  pos         = frameInfo.projectionMatrix * frameInfo.viewMatrix * pos;
  pos = pos / pos.w;
  float depth = pos.z;

  // valid only when center is inside NDC clip space.
  if (abs(pos.x) <= 1.f && abs(pos.y) <= 1.f && pos.z >= 0.f && pos.z <= 1.f) {
    uint instance_index = atomicAdd(visible_point_count, 1);
    key[instance_index] = floatBitsToUint(1.f - depth);
    index[instance_index] = id;
  }
  */
}
