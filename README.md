# Vulkan Gaussian Splatting

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

![image showing the rendering modes on the train 3DGS model](doc/rendering_modes.jpg)

We envision this project as a **testbed** to explore and compare various approaches to real-time visualization of **3D Gaussian Splatting (3DGS) [[Kerbl2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)]** and related evolutions. By evaluating various techniques and optimizations, we aim to provide valuable insights into **performance, quality, and implementation trade-offs** when using the **Vulkan API**.

This new version of the sample introduces two new pipelines, a **ray tracing** and a **hybrid** one in addition to the original **rasterization** one. Since the documentation is getting more elaborated, we reorganized and split the documentation as described hereafter. *For a good general understanding we strongly recommend to read those pages in order*.

1. The present **readme.md**, will get you **up and running** with the software, point out where to **get sample scenes** and how to **open your first scene**. It also centralizes the bibliographic references for all the sub pages.
2. The [Vulkan Gaussian Splatting Overview](./doc/overview_of_vk_gaussian_splatting.md) page will drive you through the different elements of the user interface and the functionalities of the software. *Some important points are introduced in this page*.

    We then describe our Vulkan implementation of the three rendering approaches aforementioned:
3. [VK3DGSR: 3D Gaussian Splatting (3DGS) [Kerbl2023] using Vulkan Rasterization](./doc/rasterization_of_3d_gaussian_splatting.md)
4. [VK3DGRT: 3D Gaussian Ray Tracing (3DGRT) [Moënne-Loccoz2024] using Vulkan RTX](./doc/ray_tracing_3d_gaussians.md)
5. [VK3DGHR: 3D Gaussian Hybrid Rendering Using Vulkan RTX and Rasterization](./doc/hybrid_rendering_3d_gaussians.md)

## News

- [2025/08] Added raytracing (3DGRT) and hybrid rendering (3DGS/3DGRT) pipelines + 3DGRT dataset.
- [2025/08] Added compositing with meshes and project save/load functionality.
- [2025/06] Ported to new NVIDIA DesignWorks [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2).
- [2025/03] Added new models from [3ds-scan.de](https://3ds-scan.de/). Thanks to Christian Rochner.
- [2025/03] First release with 3DGS rasterization.

## Requirements

- 64-bit Windows or Linux
- [Vulkan 1.4 SDK](https://vulkan.lunarg.com/sdk/home)  
- [CMake 3.22 or higher](https://cmake.org/download/)
- Compiler supporting basic C++20 features
  - MSVC 2019 on Windows
  - GCC 10.5 or Clang on Linux
- Additional Libraries on Linux
    - `sudo apt install libx11-dev libxcb1-dev libxcb-keysyms1-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev libxxf86vm-dev libtbb-dev`
- [CUDA v12.6](https://developer.nvidia.com/cuda-downloads) is **optional** and can be used to activate **NVML GPU monitoring** in the sample. 
- NVIDIA DesignWorks [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2) will be automatically downloaded if not found next to the sample directory.

## Building and Running

``` sh
# Clone the repository (use --recursive instead for git verison < 2.13)
git clone --recurse-submodules https://github.com/nvpro-samples/vk_gaussian_splatting
cd vk_gaussian_splatting

# Configure 
cmake -S . -B build

# Alternatively, configure as follows to disable the default "bouquet of flowers"  
# scene download by CMake and prevent auto-loading in the sample
cmake -S . -B build -DDISABLE_DEFAULT_SCENE=ON

# Build
cmake --build build --config Release

# Run
./_bin/Release/vk_gaussian_splatting.exe [path_to_ply]

```

## Opening 3DGS PLY Files

By default the application opens a 3DGS model representing a bouquet of flowers unless you disabled it at CMake stage. The sample application supports PLY files in the format introduced by INRIA [[Kerbl2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)]. PLY files can be opened using any of the following methods:
* **Command Line** – Provide the file path as last argument when launching the application.
* **File Menu** – Use File > Open to browse and load a PLY file.
* **Drag and Drop** – Simply drag the PLY file into the viewport.

**Compatibility**
* [Jawset Postshot](https://www.jawset.com/) and [3DGRUT](https://github.com/nv-tlabs/3dgrut) output ply files are compatible with the INRIA format and can be opened directly.
* Other reconstruction software's ply outputs such as [NerfStudio](https://docs.nerf.studio/nerfology/methods/splat.html) may work but have not been tested.

## 3DGS Datasets

The INRIA dataset of pre-trained models is available for [download](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) from the INRIA server.

* To visualize 3DGS reconstruction output, **open PLY files located in the point_cloud subfolders**, corresponding to 7,000 or 30,000 iterations.
* Attention: The **input.ply files cannot be loaded**, as they represent raw point clouds (not 3DGS) generated by Structure From Motion (SfM) during model reconstruction.

Additionally, you can download the [3D gaussian model of a place with a fountain in France](http://developer.download.nvidia.com/ProGraphics/nvpro-samples/fountain_place.zip) and [the 3DGS model of fountain Sindelfingen](http://developer.download.nvidia.com/ProGraphics/nvpro-samples/fountain_sindelfingen.zip), both by [3ds-scan (Christian Rochner)](https://3ds-scan.de/) and licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

The Bouquet of Flowers scene, which is automatically downloaded by CMake, can also be [downloaded here](http://developer.download.nvidia.com/ProGraphics/nvpro-samples/flowers_1.zip). The model is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## 3DGRT Datasets

We provide the reconstruction of the MipNerf 360 image set v1, using 3DGRT parametric particles intersections and MCCM/Adam, providing 1M splats for each model. [Download (1.5GB)](http://developer.download.nvidia.com/ProGraphics/nvpro-samples/3dgrt-mipnerf360-parametric-1M-v1.zip). 

Note that even if not generated by 3DGRT, the 3DGS bouquet of flowers and the two 3DGS fountain scenes from 3ds-scan will provide acceptable visual results when rendered using 3DGRT.

## Profiling and Benchmarking

To properly assess the performance of the pipelines, you should **deactivate vertical synchronization (V-Sync)**. This can be done from the **View > V-Sync** menu, the **Renderer Properties** panel, or the **Profiler** window. If V-Sync is not deactivated, the system does not run at optimal performance, and the reported timings in the **Window Top Bar** and **Profiler** window are generally higher (and fps lower) than the achievable performance. The V-Sync option is enabled by default to save energy.

The system also provides a means to run automatic benchmarks, which are detailed in the **Performance Results** sections of the documentation sub-pages. In Benchmarking mode, V-Sync is automatically disabled.

## Continue Reading

1. [Vulkan Gaussian Splatting Overview](./doc/overview_of_vk_gaussian_splatting.md) 
2. [VK3DGSR: 3D Gaussian Splatting (3DGS) [Kerbl2023] using Vulkan Rasterization](./doc/rasterization_of_3d_gaussian_splatting.md)
3. [VK3DGRT: 3D Gaussian Ray Tracing (3DGRT) [Moënne-Loccoz2024] using Vulkan RTX](./doc/ray_tracing_3d_gaussians.md)
4. [VK3DGHR: 3D Gaussian Hybrid Rendering Using Vulkan RTX and Rasterization](./doc/hybrid_rendering_3d_gaussians.md)

## References

[[Zwicker2002](https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf)]. **EWA Splatting**. E., Zwicker, M., Pfister, H., Van Baar, J., Gross, M.H., Zwicker, M., Pfister, H., Van Baar, J., & Gross, M.H. (2002). IEEE Transactions on Visualization and Computer Graphics.

[[Kerbl2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)] **3D Gaussian Splatting for Real-Time Radiance Field Rendering**. Kerbl, B., Kopanas, G., Leimkuehler, T., & Drettakis, G. (2023). ACM Transactions on Graphics (TOG), 42, 1 - 14.

[[Radl2024](https://r4dl.github.io/StopThePop/)] **StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering**. Radl, L., Steiner, M., Parger, M., Weinrauch, A., Kerbl, B., & Steinberger, M. (2024). ACM Trans. Graph., 43, 64:1-64:17.

[[Moënne-Loccoz2024](https://gaussiantracer.github.io/)] **3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes**. Moënne-Loccoz, N., Mirzaei, A., Perel, O., Lutio, R.D., Esturo, J.M., State, G., Fidler, S., Sharp, N., & Gojcic, Z. (2024).  ACM Trans. Graph., 43, 232:1-232:19.

[[Hou2024](https://arxiv.org/abs/2410.18931)] **Sort-free Gaussian Splatting via Weighted Sum Rendering**. Hou, Q., Rauwendaal, R., Li, Z., Le, H., Farhadzadeh, F., Porikli, F.M., Bourd, A., & Said, A. (2024). ArXiv, abs/2410.18931.

[[Morgenstern2024](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/)] **Compact 3D Scene Representation via Self-Organizing Gaussian Grids**. Wieland Morgenstern, Florian Barthel, Anna Hilsmann, Peter Eisert. ECCV 2024.

[[Wu2024](https://research.nvidia.com/labs/toronto-ai/3DGUT/)] **3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting**. Wu, Q., Esturo, J.M., Mirzaei, A., Moënne-Loccoz, N., & Gojcic, Z. (2024). ArXiv, abs/2412.12507. CVPR 2025.

[[3DGRUT](https://github.com/nv-tlabs/3dgrut)] This repository provides the official implementations of 3D Gaussian Ray Tracing (3DGRT)[Moënne-Loccoz2024] and 3D Gaussian Unscented Transform (3DGUT)[Wu2024]. 

## 3rd-Party Licenses

| Library | URL | License |
|--------------|---------|--|
| **miniply** | https://github.com/vilya/miniply | [MIT](https://github.com/vilya/miniply/blob/master/LICENSE.md) |
| **vrdx** | https://github.com/jaesung-cs/vulkan_radix_sort | [MIT](https://github.com/jaesung-cs/vulkan_radix_sort/blob/master/LICENSE) |

Some parts of the current implementation are strongly inspired by, and in some cases incorporate, source code and comments from the following third-party projects:

| Project | URL | License |
|--------------|---------|--|
| **vkgs** | https://github.com/jaesung-cs/vkgs | [MIT](https://github.com/jaesung-cs/vkgs/blob/master/LICENSE) |
| **GaussianSplats3D** | https://github.com/mkkellogg/GaussianSplats3D | [MIT](https://github.com/mkkellogg/GaussianSplats3D/blob/main/LICENSE) |


Additional 3rd-Party sofwtares are listed in [nvpro_core2/third_party](https://gitlab-master.nvidia.com/devtechproviz/nvpro-samples/nvpro_core2/-/tree/main/third_party).

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Contributing

Merge requests to vk_gaussian_splatting are welcome, and use the Developer Certificate of Origin (https://developercertificate.org included in [CONTRIBUTING](CONTRIBUTING)).

When committing, please certify that your contribution adheres to the DCO and use `git commit --sign-off`. Thank you!

## Support

- For bug reports and feature requests, please use the [GitHub Issues](https://github.com/nvpro-samples/vk_gaussian_splatting/issues) page.
- For general questions and discussions, please use the [GitHub Discussions](https://github.com/nvpro-samples/vk_gaussian_splatting/discussions) page.
