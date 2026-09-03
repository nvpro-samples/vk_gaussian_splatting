# Vulkan Gaussian Splatting

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Download](https://img.shields.io/badge/Download-Releases-brightgreen)](https://github.com/nvpro-samples/vk_gaussian_splatting/releases)
[![Documentation](https://img.shields.io/badge/Documentation-Website-brightgreen)](https://nvpro-samples.github.io/vk_gaussian_splatting/)

![image showing the rendering modes on the train 3DGS model](docs/images/rendering_modes.jpg)

This project is a **testbed** to explore and compare various approaches to **real-time visualization** of **3D Gaussian Splatting (3DGS) [[Kerbl2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)]** and related evolutions. 

This project is not a 3DGS reconstruction tool — it is a viewer. It does not generate or train models, only tries to visualize them at the highest possible speed.

By evaluating various techniques and optimizations, we aim to provide valuable insights into **performance, quality, and implementation trade-offs** when using the **Vulkan API**.

This project implements several **rendering pipelines** based on **rasterization**, **ray tracing**, and **hybrid** approaches. It can be used as a lab for developers or as a day-to-day viewer for your splats. 

## Quick Start

Requirements: NVIDIA GPU or Full Vulkan 1.4 enabled GPU, 64-bit Windows or Linux.

1. [Download the latest release](https://github.com/nvpro-samples/vk_gaussian_splatting/releases) for your OS
    * decompress the archive
    * run the executable vk_gaussian_splatting[.exe]
2. [Download this small 3DGS flowers model](http://developer.download.nvidia.com/ProGraphics/nvpro-samples/flowers_1.zip) 
    * decompress the archive
    * drag and drop the file flowers_1.ply into the app viewport
3. Happy splatting!

## Documentation

Please consult the [documentation](https://nvpro-samples.github.io/vk_gaussian_splatting/) to browse:

* [Release Notes](https://nvpro-samples.github.io/vk_gaussian_splatting/release-notes) — with feature videos for novelties
* [Getting Started](https://nvpro-samples.github.io/vk_gaussian_splatting/getting-started/) — get **up and running**, download pre-builds or build from source, and open your first scene
* 🔥[Sample Projects](https://nvpro-samples.github.io/vk_gaussian_splatting/sample-projects/) — ready-made scenes showcasing the renderer, with image comparisons
* [Datasets](https://nvpro-samples.github.io/vk_gaussian_splatting/datasets/) — download links and recommended settings for additional sample models
* [User Guides](https://nvpro-samples.github.io/vk_gaussian_splatting/user-guides/overview/) — walkthrough of the UI, rendering settings, and tools
* [Deep Dives](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/rasterization_of_3d_gaussian_splatting/) — in-depth technical presentation of the rendering implementations and features:
  * [VK3DGSR — Rasterization](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/rasterization_of_3d_gaussian_splatting/)
  * [VK3DGRT — Ray Tracing](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/ray_tracing_3d_gaussians/)
  * [VK3DGUT — Unscented Transform](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/rasterization_of_3dgut/)
  * [VK3DGHR — Hybrid Rendering](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/hybrid_rendering_3d_gaussians/)
  * [Stochastic Transparency](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/stochastic_transparency/)
  * [Lighting and Shadows](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/lighting_and_shadows/)
  * 🔥[Billboard Ray Tracing](https://nvpro-samples.github.io/vk_gaussian_splatting/deep-dives/billboard_ray_tracing/)
* [References](https://nvpro-samples.github.io/vk_gaussian_splatting/references/) — bibliography

If you are working offline, you can browse the documentation markdown pages in the [docs](./docs) folder.

![Implemented pipelines features](docs/images/overview_of_implemented_pipelines.png)

## News

- [2026/09] Release 2026.2.9
  - **Faster ray tracing** with reduced per-ray payload and improved stochastic transparency
  - **Fixed splat materials** being taken from the wrong splat set in raster and hybrid with lighting enabled
  - **Cross-platform asset paths** for reliable project saving and loading across Windows and Linux
  - **Updated sample projects** and dataset listings featuring winter garden content (🔥new [video](https://developer.download.nvidia.com/ProGraphics/nvpro-samples/vkgs/vkgs_v2026_2_lighting_from_radiance_field.mp4))
  - Fixed the build with `-DUSE_DLSS=OFF`, and a crash when saving images before the first frame
- [2026/06] 🔥Release 2026.2
  - **GGX PBR materials** (metallic-roughness) replacing Phong, with indirect lighting for splat sets
  - **glTF/GLB mesh loading** with full PBR textures, emissive, and scene-graph flattening
  - **Physically-based lighting**: IBL environment maps, physical sky background, and NEE + BRDF path tracing with MIS
  - **Tone mapping and auto-exposure** integrated from nvpro_core2
  - **Billboard ray tracing** path with intersection shaders and selectable TLAS bounding modes
  - **Sphere primitive mode** via `VK_NV_ray_tracing_linear_swept_spheres`
  - **Shared TLAS** for per-particle RTX with reduced VRAM and fewer acceleration-structure rebuilds
  - **DLSS improvements**: motion vectors for meshes and splat sets, hardware depth, firefly clamp
  - **Renderer UI rework** with Denoising / Rasterization / Raytracing tabs
  - **Save / Save As projects** including renderer and tonemapping settings
  - **Samples folder** with an asset-download build script and new sample projects
  - **Billboard benchmark suite** and **headless mode** with an expanded CLI
  - **Platform**: HardwareSupport capability registry, Intel Arc and Linux build fixes, upgraded nvpro_core2

For earlier releases and full details, see the [Release Notes](https://nvpro-samples.github.io/vk_gaussian_splatting/release-notes) page.

## Support and Feedback

For bug reports and feature requests, please use the [GitHub Issues](https://github.com/nvpro-samples/vk_gaussian_splatting/issues) page.

For general questions and discussions, please use the [GitHub Q&A Discussions](https://github.com/nvpro-samples/vk_gaussian_splatting/discussions/categories/q-a) page.

We would be very happy to get feedback from you, to know whether you use the software to learn, teach, apply to your own renderer, or just use it as a viewer. Please do not hesitate to give feedback and send screenshots in the [GitHub Show and Tell Discussions](https://github.com/nvpro-samples/vk_gaussian_splatting/discussions/categories/show-and-tell) page.

## Contributing

Merge requests to vk_gaussian_splatting are welcome, and use the Developer Certificate of Origin (https://developercertificate.org included in [CONTRIBUTING](CONTRIBUTING)).

When committing, please certify that your contribution adheres to the DCO and use `git commit --sign-off`. Thank you!

## References

[[Zwicker2002](https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf)]. **EWA Splatting**. E., Zwicker, M., Pfister, H., Van Baar, J., Gross, M.H., Zwicker, M., Pfister, H., Van Baar, J., & Gross, M.H. (2002). IEEE Transactions on Visualization and Computer Graphics.

[[Enderton2010](https://luebke.us/publications/StochasticTransparency_I3D2010.pdf)]. **Stochastic transparency**. Eric Enderton, Erik Sintorn, Peter Shirley, and David Luebke. In Proc. Symposium on Interactive 3D Graphics and Games (I3D), pages 157–164, 2010.

[[Kerbl2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)] **3D Gaussian Splatting for Real-Time Radiance Field Rendering**. Kerbl, B., Kopanas, G., Leimkuehler, T., & Drettakis, G. (2023). ACM Transactions on Graphics (TOG), 42, 1 - 14.

[[Yu2023](https://niujinshuchong.github.io/mip-splatting/)] **Mip-Splatting: Alias-Free 3D Gaussian Splatting.**. Yu, Z., Chen, A., Huang, B., Sattler, T., & Geiger, A. (2023).  2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19447-19456.

[[Radl2024](https://r4dl.github.io/StopThePop/)] **StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering**. Radl, L., Steiner, M., Parger, M., Weinrauch, A., Kerbl, B., & Steinberger, M. (2024). ACM Trans. Graph., 43, 64:1-64:17.

[[Moënne-Loccoz2024](https://gaussiantracer.github.io/)] **3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes**. Moënne-Loccoz, N., Mirzaei, A., Perel, O., Lutio, R.D., Esturo, J.M., State, G., Fidler, S., Sharp, N., & Gojcic, Z. (2024).  ACM Trans. Graph., 43, 232:1-232:19.

[[Hou2024](https://arxiv.org/abs/2410.18931)] **Sort-free Gaussian Splatting via Weighted Sum Rendering**. Hou, Q., Rauwendaal, R., Li, Z., Le, H., Farhadzadeh, F., Porikli, F.M., Bourd, A., & Said, A. (2024). ArXiv, abs/2410.18931.

[[Morgenstern2024](https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/)] **Compact 3D Scene Representation via Self-Organizing Gaussian Grids**. Wieland Morgenstern, Florian Barthel, Anna Hilsmann, Peter Eisert. ECCV 2024.

[[Wu2024](https://research.nvidia.com/labs/toronto-ai/3DGUT/)] **3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting**. Wu, Q., Esturo, J.M., Mirzaei, A., Moënne-Loccoz, N., & Gojcic, Z. (2024). ArXiv, abs/2412.12507. CVPR 2025.

[[3DGRUT](https://github.com/nv-tlabs/3dgrut)] This repository provides the official implementations of 3D Gaussian Ray Tracing (3DGRT)[Moënne-Loccoz2024] and 3D Gaussian Unscented Transform (3DGUT)[Wu2024]. 

[[Kheradmand2025](https://ubc-vision.github.io/stochasticsplats/)] **“StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting.”**. Kheradmand Shakiba, Delio Vicini, George Kopanas, Dmitry Lagun, Kwang Moo Yi, Mark J. Matthews and Andrea Tagliasacchi. ICCV 2025.

## 3rd-Party Licenses

| Library | URL | License |
|--------------|---------|--|
| **miniply** | https://github.com/vilya/miniply | [MIT](https://github.com/vilya/miniply/blob/master/LICENSE.md) |
| **vrdx** | https://github.com/jaesung-cs/vulkan_radix_sort | [MIT](https://github.com/jaesung-cs/vulkan_radix_sort/blob/master/LICENSE) |
| **spz** | https://github.com/nianticlabs/spz | [MIT](https://github.com/nianticlabs/spz?tab=MIT-1-ov-file#readme) |

Some parts of the current implementation are strongly inspired by, and in some cases incorporate, source code and comments from the following third-party projects:

| Project | URL | License |
|--------------|---------|--|
| **vkgs** | https://github.com/jaesung-cs/vkgs | [MIT](https://github.com/jaesung-cs/vkgs/blob/master/LICENSE) |
| **GaussianSplats3D** | https://github.com/mkkellogg/GaussianSplats3D | [MIT](https://github.com/mkkellogg/GaussianSplats3D/blob/main/LICENSE) |


Additional 3rd-Party software is listed in [nvpro_core2/third_party](https://gitlab-master.nvidia.com/devtechproviz/nvpro-samples/nvpro_core2/-/tree/main/third_party).

