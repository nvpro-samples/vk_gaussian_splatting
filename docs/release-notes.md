# Release Notes

## Unreleased

### Projects & assets

- **Portable asset paths in project files** (project file version 8): splat set, mesh, and HDR environment paths are now stored UTF-8 encoded with forward slashes, relative to the project file when possible and absolute when the resource lives on a different drive or network share. Projects saved on Windows previously stored backslashes, which made them fail to load on Linux — including the shipped sample projects as soon as they were re-saved. Projects up to version 7 are still read, with their native separators converted on load. Also fixes relative paths being emitted as a nonsensical `..` chain when the asset was on a different drive than the project.

### Ray tracing — particles & acceleration

- **Single-splat-set payload reduction**: `RTX_HAS_PARTICLES` now encodes the clamped RTX splat descriptor count (0 = no particle code, 1 = single set, 2 = multi-set). With a single splat set (one instance, no BLAS chunk split), the per-sample `splatSetIdx[]` array and `currentSplatSetInstance` are removed from the ray payload (descriptor index 0 implied), reducing payload size by `PARTICLES_SPP + 1` dwords and lowering register pressure across `TraceRay()`. Shader recompilation triggers automatically when the mode changes (e.g. a second set is added).
- **Stochastic any-hit payload/ALU reduction**: the any-hit now evaluates opacity only (`threedgrtProcessHit` gained a `kRadiance` template flag); the SH radiance fetch and the normal (surface-info gated, previously always computed) run once per ray in the raygen for the surviving sample instead of once per candidate hit. The per-sample `color[]`/`normal[]` payload fields are removed (7 dwords), and the stochastic shadow path simplifies to binary occlusion (semantically identical to the previous constant-alpha averaging).

## Version 2026.2

### Rendering & materials (PBR / meshes / lighting)

- Migrated materials from Phong to **GGX PBR** (metallic-roughness).
- Added **glTF/GLB mesh loading** (tinygltf) with full PBR textures, emissive strength, sRGB formats, `KHR_texture_transform`, multi-UV sets, and scene-graph flattening at import.
- Integrated **tone mapping and auto-exposure** from nvpro_core2; CLIP tonemapping default; tonemapping parameters in project save/load.
- Added **IBL environment maps** and physical **sky background** with equirectangular bake; sky specular/reflection; NEE + BRDF path tracing with MIS; removed legacy illumination modes (always path trace with per-material max bounces).
- **Indirect lighting** for splat sets; splat radiance converted from sRGB to linear for PBR.
- Extended glTF shading: specular/clearcoat, legacy specular-glossiness workflow.
- **Particle emissive ambient occlusion** for mesh proximity attenuation.

### Ray tracing — particles & acceleration

- **Shared TLAS** for per-particle RTX; VRAM lifecycle improvements (release covariance in pure RTX, skip global index table, fewer BLAS/TLAS rebuilds).
- **Particle depth modes** for ray-traced Gaussians, including **billboard depth**.
- **Sphere primitive mode** via `VK_NV_ray_tracing_linear_swept_spheres` (plus silhouette wireframe).
- **Billboard frustum culling**; quantizable mesh hit payload; `RTX_HAS_MESHES` / `RTX_HAS_PARTICLES` compile-time shader guards; any-hit refactor and stochastic early-out fixes (billboard holes fix).
- Other: **Clay** visualization mode; configurable secondary ray offset; in-memory **VkPipelineCache**.

### Billboard ray tracing

- Full **billboard RT** path: intersection shaders, AABB geometry, and [billboard ray tracing deep dive](deep-dives/billboard_ray_tracing.md).
- **Bounding modes** for TLAS instance scale: Uniform (½, ⅓, ¼), Fitted, Optimal (per-particle), AABB max opacity, icosahedron ellipsoid; default bounding mode is now Uniform ½ (adaptive bounding modes were removed after evaluation).
- **RTX_SHORTEN_RAY** option for stochastic any-hit early termination; billboard depth available for all trace strategies.
- Other RTX Fixes: multi–splat-set compositing, light leak, depth incompatibilities, large BLAS build path for bounding modes.

### DLSS / denoising

- DLSS guide improvements: motion vectors (meshes and splat sets), hardware depth, firefly clamp, min radiance, `DLSS_ENABLED` compile-time macro.
- Skip DLSS for debug visualization modes; fixes for clay mode, splat roughness, and negative radiance clamp on DLSS input.

### Rasterization & hybrid

- Pipeline-aware temporal accumulation.
- Hybrid unlit albedo fix.
- MipSplatting: decouple 2D covariance dilation from opacity compensation.

### User interface

- Renderer UI rework: **Denoising**, **Rasterization**, and **Raytracing** tabs, collapsible groups, tooltips, pipeline selector in menu bar.
- **View** menu: fullscreen, footer bar, window visibility; playback button (replaces keyboard shortcuts); navigation toolbar icons.
- DLSS controls moved into denoising tab; tone mapping UI rework.
- Per-instance **show/hide** for splat sets and meshes; profiler and NVML monitor owned by main UI.

### Projects, assets & samples

- **Save / Save As** projects; renderer settings and tonemapping in project files; project name in title bar.
- Asset tree: **Settings**, **Recent Meshes**, mesh import modal.
- **Samples** folder with build script for downloading external assets; new and updated [sample projects](sample-projects/index.md) (large city with sky, chessboard winter house, greenhouse on stove scene, and others).
- Mesh asset download moved from CMake to the samples build script.

### Documentation & CI

- Documentation website (this site), getting started and datasets pages.
- [Sample projects](sample-projects/index.md) documentation with image comparison sliders; deep dive links on home page; billboard benchmark documentation.

### Benchmarking & headless

- Billboard benchmark suite: HDR/PNG capture, PSNR, multi-camera, charts, `--skip-exec`, `--saveImage`, G-buffer and DLSS guide export.
- **Headless mode** for benchmarks; expanded CLI (camera presets, `colorBufferFormat`, `lightingEnabled`, and more). See [benchmarking](benchmarking.md).

### Platform & compatibility

- **HardwareSupport** capability registry; optional ray tracing, mesh shader, and LSS sphere extensions with guarded code paths.
- **Intel Arc** support fixes; Linux build fixes.
- Upgraded **nvpro_core2**.

### Stability & bug fixes

- RTX build failure → raster fallback: device lost, shader rebuild, mesh-shader fallback stuck state.
- Texture mode in RTX: device lost fixes.
- Vulkan validation: pipeline / G-buffer color format mismatch.
- Image comparator format mismatch when tonemapping and visual helpers are both active.

## Version 2026.1.7

- **Pre-built binaries** available on GitHub Releases page
- **New online [documentation](https://nvpro-samples.github.io/vk_gaussian_splatting/) website** — This site.
- **Simplified root github repo README.md**
- **Rendering pipeline selector** added to the menu bar
- **Navigation mode icons** in toolbar with improved camera controls
- **Fix profiler reporting** during camera drag in raster/hybrid pipelines
- **Fix 32-bit addressing overflow** in sorting buffers by converting to LargeBuffer (supports larger models)

## Version 2026.1

- Multi-instance splat set architecture with global index tables and unified sorting
- Centralized bindless asset management with a root bindless scene assets buffer
- Multi-light lighting system (point / spot / directional, ray traced hard / soft shadows)
- Front-to-back rasterization with depth consolidation for lighting
- Deferred shading for rasterized Gaussian splats
- GPU-built particle acceleration structures with multi-TLAS / multi-BLAS chunking and VRAM budget pre-checking
- Ray tracing now supports very large models and updates faster
- Stochastic splat sorting and Monte Carlo trace strategy for enhanced interactivity
- DLSS Ray Reconstruction for anti-aliasing, upscaling and denoising
- Expanded visualization modes in ray tracing and hybrid pipelines (normals, depth, DLSS buffers, splat ID)
- UI / UX enhancements (summary overlay, shader feedback and ray hit profile, editing mode)
- 3D transform and infinite grid visual helpers
- Runtime image comparison tool with MSE / PSNR / FLIP metrics
- Added **Winter house** and **Large city** models from <a href="https://teleportour.com/" target="_blank">teleportour.com</a> to [datasets page](datasets.md#models-from-teleportourcom). Thanks to <a href="https://www.linkedin.com/in/andrii-shramko" target="_blank">Andrii Shramko</a>.

## Older Versions

- [2025/10] Migration from GLSL to SLANG.
- [2025/09] Added Unscented Transform (3DGUT) and hybrid rendering (3DGUT/3DGRT) pipelines, with support for fisheye and depth of field.
- [2025/09] Added import of .spz file format from [nianticlabs](https://github.com/nianticlabs/spz).
- [2025/08] Added depth of field to raytracing (3DGRT) pipeline.
- [2025/08] Added raytracing (3DGRT) and hybrid rendering (3DGS/3DGRT) pipelines + 3DGRT dataset.
- [2025/08] Added compositing with meshes and project save/load functionality.
- [2025/06] Ported to new NVIDIA DesignWorks [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2).
- [2025/03] Added new models from [3ds-scan.de](https://3ds-scan.de/). Thanks to Christian Rochner.
- [2025/03] First release with 3DGS rasterization.
