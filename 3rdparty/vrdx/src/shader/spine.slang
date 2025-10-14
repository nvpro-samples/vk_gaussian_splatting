import constants;

StructuredBuffer<uint> elementCounts : register(t0, space0);
RWStructuredBuffer<uint> globalHistogram : register(u1, space0);
RWStructuredBuffer<uint> partitionHistogram : register(u2, space0);

groupshared uint reduction;
groupshared uint intermediate[MAX_SUBGROUP_SIZE];

// dispatch this shader (RADIX, 1, 1), so that gl_WorkGroupID.x is radix
[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID,
          uint groupIndex: SV_GroupIndex, uniform int pass) {
  uint elementCount = elementCounts[0];

  uint laneIndex = WaveGetLaneIndex();  // 0..31
  uint laneCount = WaveGetLaneCount();  // 32
  uint waveIndex = groupIndex / laneCount;
  uint waveCount = WORKGROUP_SIZE / laneCount;
  uint index = waveIndex * laneCount + laneIndex;

  uint radix = groupId.x;

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;

  if (index == 0) {
    reduction = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = 0; WORKGROUP_SIZE * i < partitionCount; ++i) {
    uint partitionIndex = WORKGROUP_SIZE * i + index;
    uint value =
        partitionIndex < partitionCount ? partitionHistogram[RADIX * partitionIndex + radix] : 0;
    uint excl = WavePrefixSum(value) + reduction;
    uint sum = WaveActiveSum(value);

    if (WaveIsFirstLane()) {
      intermediate[waveIndex] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (index < waveCount) {
      uint excl = WavePrefixSum(intermediate[index]);
      uint sum = WaveActiveSum(intermediate[index]);
      intermediate[index] = excl;

      if (index == 0) {
        reduction += sum;
      }
    }
    GroupMemoryBarrierWithGroupSync();

    if (partitionIndex < partitionCount) {
      excl += intermediate[waveIndex];
      partitionHistogram[RADIX * partitionIndex + radix] = excl;
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (radix == 0) {
    // one workgroup is responsible for global histogram prefix sum
    if (index < RADIX) {
      uint value = globalHistogram[RADIX * pass + index];
      uint excl = WavePrefixSum(value);
      uint sum = WaveActiveSum(value);

      if (WaveIsFirstLane()) {
        intermediate[waveIndex] = sum;
      }
      GroupMemoryBarrierWithGroupSync();

      if (index < RADIX / laneCount) {
        uint excl = WavePrefixSum(intermediate[index]);
        intermediate[index] = excl;
      }
      GroupMemoryBarrierWithGroupSync();

      excl += intermediate[waveIndex];
      globalHistogram[RADIX * pass + index] = excl;
    }
  }
}
