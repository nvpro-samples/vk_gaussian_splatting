# Vulkan Gaussian Splatting  Sample

## Building and Running

```
# Clone the repository
git clone https://github.com/nvpro-samples/vk_gaussian_splatting
cd vk_gaussian_splatting

# Configure and build
cmake -S . -B build
cmake --build build --config Release

# Running
../bin_x64/Release/vk_gaussian_splatting.exe

# Running the benchmark defined in benchmark.txt 
mkdir _benchmark
cd _benchmark
../bin_x64/Release/vk_gaussian_splatting.exe -benchmark ../benchmark.txt <path_to_3dgs_dataset>/bicycle/point_cloud/iteration_30000/point_cloud.ply

```

## The pipelines

![image showing gaussian splatting rasterization pipelines](doc/pipelines.png)

## Sorting methods

![image showing gaussian splatting sorting methods](doc/sorting.png)


## License
Apache-2.0

