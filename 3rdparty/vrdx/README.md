# vulkan_radix_sort

Vulkan implementation of radix sort.

Reduce-then-scan GPU radix sort algorithm is implemented (Onesweep is abandoned.)


## Requirements
- `VulkanSDK>=1.2`
  - Download from https://vulkan.lunarg.com/ and follow install instruction.
  - Requires several features available in `1.2`.
  - Must support `VK_KHR_buffer_device_address`:
    - Run `vulkaninfo` and check if `VK_KHR_buffer_device_address` device extension is available.
- `cmake>=3.15`


## Build
```bash
$ cmake . -B build
$ cmake --build build --config Release -j
```

## Test
```bash
$ ./build/Release/bench.exe <N> <type>  # Windows
$ ./build/bench <N> <type>              # Linux
$ ./build/bench 10000000 vulkan
```
- N = number of elements to sort
- type = one of cpu,vulkan,cuda


### Test Environment
- Windows, NVIDIA GeForce RTX 4090.


### Benchmark Result
- Not precisely benchmarked, but the speed is competitive compare to CUB radix sort.
- 32-bit key-only: my implementation is 10% slower when sorting 33M (2^25) elements.
- 32-bit Key-value: my implementation is 15-25% faster when sorting 33M (2^25) key-value pairs.
- Note that CUB radix sort is not in-place operation. It may require an additional copy operation, or double storage.
- vulkan
  ```bash
  > .\build\Release\bench.exe 33554432 vulkan
  vk_radix_sort benchmark
  ================ sort ================
  total time: 2.67571ms (12.5404 GItems/s)
  ================ sort key value ================
  total time: 3.42221ms (9.80491 GItems/s)
  ================ sort key value speed ================
  [0] total time: 3.41706ms (9.81969 GItems/s)
  [1] total time: 3.43142ms (9.77857 GItems/s)
  [2] total time: 3.42298ms (9.80271 GItems/s)
  [3] total time: 3.46208ms (9.69199 GItems/s)
  [4] total time: 3.42426ms (9.79904 GItems/s)
  [5] total time: 3.43725ms (9.762 GItems/s)
  [6] total time: 3.42016ms (9.81078 GItems/s)
  [7] total time: 3.42016ms (9.81078 GItems/s)
  [8] total time: 3.42099ms (9.80839 GItems/s)
  [9] total time: 3.41606ms (9.82254 GItems/s)
  ...
  ```
- CUDA Version 12.6 CUB
  ```bash
  > .\build\Release\bench.exe 33554432 cuda
  vk_radix_sort benchmark
  ================ sort ================
  total time: 2.5047ms (13.3966 GItems/s)
  ================ sort key value ================
  total time: 4.19226ms (8.00391 GItems/s)
  ================ sort key value speed ================
  [0] total time: 4.20352ms (7.98246 GItems/s)
  [1] total time: 4.50355ms (7.45066 GItems/s)
  [2] total time: 4.21376ms (7.96306 GItems/s)
  [3] total time: 4.22298ms (7.94568 GItems/s)
  [4] total time: 4.22208ms (7.94737 GItems/s)
  [5] total time: 4.2199ms (7.95147 GItems/s)
  [6] total time: 4.21274ms (7.965 GItems/s)
  [7] total time: 4.20352ms (7.98246 GItems/s)
  [8] total time: 4.21376ms (7.96306 GItems/s)
  [9] total time: 4.21478ms (7.96113 GItems/s)
  ...
  ```

## Use as a Library with CMake
- Add subdirectory `vulkan_radix_sort`
    ```cmake
    add_subdirectory(path/to/vulkan_radix_sort)
    ```

- Link to `vk_radix_sort` in your project (library, binary)
    ```cmake
    target_link_libraries(my_project PRIVATE Vulkan::Vulkan VulkanMemoryAllocator vk_radix_sort)
    ```

## Usage
1. When creating `VkDevice`, enable `VkPhysicalDeviceBufferAddressFeatures`.

1. When creating `VmaAllocator`, enable `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT` flag.

1. Create `VkBuffer` for keys and values, with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` and `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`.

1. Create `VrdxSorter`

    It creates shared resources: pipeline layouts, pipelines, etc.

    ```c++
    VrdxSorter sorter = VK_NULL_HANDLE;
    VrdxSorterCreateInfo sorterInfo = {};
    sorterInfo.physicalDevice = physicalDevice;
    sorterInfo.device = device;
    sorterInfo.pipelineCache = pipelineCache;
    vrdxCreateSorter(&sorterInfo, &sorter);
    ```

1. Create a temporary storage buffer for sort.

    ```c++
    // request storage buffer request
    VrdxSorterStorageRequirements requirements;
    // for key-only
    vrdxGetSorterStorageRequirements(sorter, elementCount, &requirements);
    // for key-value
    vrdxGetSorterKeyValueStorageRequirements(sorter, elementCount, &requirements);

    // create or reuse buffer
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = requirements.size;
    bufferInfo.usage = requirements.usage;
    // ...
    ```

1. Record sort commands.

    This command binds pipeline, pipeline layout, and push constants internally.

    So, users must not expect previously bound targets retain after the sort command.

    Users must add proper **execution barriers**.

    One can use buffer memory barrier, but in general, global barriers are more efficient than per-resource, according to [official synchronization examples](https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples#three-dispatches-first-dispatch-writes-to-one-storage-buffer-second-dispatch-writes-to-a-different-storage-buffer-third-dispatch-reads-both):

    > ... global memory barrier covers all resources. Generally considered more efficient to do a global memory barrier than per-resource barriers, per-resource barriers should usually be used for queue ownership transfers and image layout transitions - otherwise use global barriers.

    The sort command will read from key/value buffers (and elementCount buffer for indirect sort) in compute shader stage, and write to output key/value buffers in later compute shader stage.

    The second synchronization scope **before** sort command must include `COMPUTE_SHADER` stage (and `TRANSFER` for indirect sort) and `SHADER_READ` access (and `TRANSFER_READ` for indirect sort).

    The first synchronization scope **after** sort command must include `COMPUTE_SHADER` stage and `SHADER_WRITE` access.

    ```c++
    VkQueryPool queryPool;  // VK_NULL_HANDLE, or a valid timestamp query pool with size at least 8.

    // sort keys
    vrdxCmdSort(commandBuffer, sorter, elementCount,
                keysBuffer, 0,
                storageBuffer, 0,
                queryPool, 0);

    // sort keys with values
    vrdxCmdSortKeyValue(commandBuffer, sorter, elementCount,
                        keysBuffer, 0,
                        valuesBuffer, 0,
                        storageBuffer, 0,
                        queryPool, 0);

    // indirectBuffer contains elementCount, a single uint entry in GPU buffer.
    // maxElementCount is required for storage buffer offsets.
    // element count in the indirect buffer must not be greater than maxElementCount. Otherwise, undefined behavior.
    vrdxCmdSortKeyValueIndirect(commandBuffer, sorter, maxElementCount,
                                indirectBuffer, 0,
                                keysBuffer, 0,
                                valuesBuffer, 0,
                                storageBuffer, 0,
                                queryPool, 0);
    ```


## TODO
- [x] Use `VkPhysicalDeviceLimits` to get compute shader-related limits, such as `maxComputeWorkGroupSize` or `maxComputeSharedMemorySize`.
- [x] Increase allowed `maxElementCount` by allocating buffers properly.
- [x] Compare with CUB radix sort
- [ ] Compare with VkRadixSort
- [ ] Compare with Fuchsia radix sort
- [ ] Find best `WORKGROUP_SIZE` and `PARTITION_DIVISION` for different devices.
- [x] Support for SubgroupSize=64.


## References
- https://github.com/b0nes164/GPUSorting : their CUDA kernel codes were very helpful when trying to catch the idea of how the algorithm works.


## Troubleshooting
- (NVIDIA GPU, Windows) Slow runtime after a few seconds.
  - Reason: NVidia driver adjust GPU/Memory clock.
    Open Performance Overlay (Alt+R), then you will see GPU/Memory Clock gets down.
  - Solution: change performance mode in control panel.
    ![](media/performance_mode.jpg)
