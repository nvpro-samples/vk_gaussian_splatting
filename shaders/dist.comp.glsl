#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#include "device_host.h"
#include "common.glsl"

// we could write to manage alignment automatically
// layout(set = 0, binding = 0, scalar) uniform FrameInfo_
// but it may be less performant than aligning
// attribute in the struct (see device_host.h comment)
layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(local_size_x = 256) in;

layout(set = 0, binding = 5, scalar) writeonly buffer _distances
{
  uint32_t distances[];
};
layout(std430, set = 0, binding = 6, scalar) writeonly buffer _indices
{
  uint32_t indices[];
};
layout(std430, set = 0, binding = 7, scalar) writeonly buffer _indirect
{
  IndirectParams indirect;
};

// encodes an fp32 into a uint32 that can be ordered
uint encodeMinMaxFp32(float val)
{
  uint bits = floatBitsToUint(val);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  return bits;
}

void main()
{
  const uint id = gl_GlobalInvocationID.x;
  if(id >= frameInfo.splatCount)
    return;

  vec4 pos = vec4(fetchCenter(id), 1.0);
  pos      = frameInfo.projectionMatrix * frameInfo.viewMatrix * pos;
  pos      = pos / pos.w;
  const float depth = pos.z;

  // valid only when center is inside NDC clip space.
  // Note: when culling between x=[-1,1] y=[-1,1], which is NDC extent,
  // the culling is not good since we only take into account
  // the center of each splat instead of its extent.
#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_DIST
  const float clip = 1.0f + frameInfo.frustumDilation;
  if(abs(pos.x) > clip || abs(pos.y) > clip || pos.z < 0.f - frameInfo.frustumDilation || pos.z > 1.0)
    return;
#endif
  
  // increments the visible splat counter in the indirect buffer (second entry of the array)
  const uint instance_index = atomicAdd(indirect.instanceCount, 1);
  // stores the distance
  distances[instance_index] = encodeMinMaxFp32(-depth);
  // stores the base index
  indices[instance_index] = id;
  // set the workgroup count for the mesh shading pipeline
  if(instance_index % 32 == 0)
  {
    atomicAdd(indirect.groupCountX, 1);
  }
}
