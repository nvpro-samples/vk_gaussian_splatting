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
layout(std430, set = 0, binding = 5) buffer InstanceKey
{
  uint32_t key[];
};

layout(std430, set = 0, binding = 6) buffer IndirectParams
{
  uint32_t indirect[];
};

vec2 getDataUV(in uint index, in int stride, in int offset, in vec2 dimensions)
{
  vec2  samplerUV = vec2(0.0, 0.0);
  float d         = float(index * uint(stride) + uint(offset)) / dimensions.x;
  samplerUV.y     = float(floor(d)) / dimensions.y;
  samplerUV.x     = fract(d);
  return samplerUV;
}

uint encodeMinMaxFp32(float val)
{
  uint bits = floatBitsToUint(val);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  return bits;
}

void main() {
  uint id = gl_GlobalInvocationID.x;
  if(id >= frameInfo.splatCount)
    return;
  
  vec4 pos = texture(centersTexture, getDataUV(id, 1, 0, textureSize(centersTexture, 0)));
  pos.w    = 1.0;
  //pos = projection * view * model * pos;
  pos         = frameInfo.projectionMatrix * frameInfo.viewMatrix * pos;
  pos = pos / pos.w;
  float depth = pos.z;

  // valid only when center is inside NDC clip space.
  // Note: when culling between x=[-1,1] y=[-1,1], which is NDC extent,
  // the culling is not good since we only take into account 
  // the center of each splat instead of its extent.
  // for the time being we just add 0.1 to the NDC as a margin which 
  // make the job with most models
  if (abs(pos.x) <= 1.1f && abs(pos.y) <= 1.1f && pos.z >= 0.f && pos.z <= 1.f) {
    // increments the visible splat counter in the indirect buffer (second entry of the array)
    uint instance_index = atomicAdd(indirect[1], 1);
    // stores the key
    key[instance_index] = encodeMinMaxFp32(- depth);
    // stores the value
    key[frameInfo.splatCount + instance_index ] = id;
  }
}
