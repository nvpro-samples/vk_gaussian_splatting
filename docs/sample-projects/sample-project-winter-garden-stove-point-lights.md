# Splat Set and Meshes With Point Lights and Shadows

<div><img-comparison-slider>
  <figure slot="first" class="before">
    <img src="../../images/wintergarden-point-lights-off.jpg" />
    <figcaption>Lights Off</figcaption>
  </figure>
  <figure slot="second" class="after">
    <img src="../../images/wintergarden-point-lights.jpg" />
    <figcaption>Point Lights</figcaption>
  </figure>
</img-comparison-slider></div>

**File:** `3dgs_winter_garden_objects_on_stove_lighting.vkgs`

This project places mesh objects on a stove inside a winter garden captured as a Gaussian Splatting radiance field. Point lights with ray-traced shadows illuminate the scene. It demonstrates:

- **Hybrid pipeline** (3DGS raster + 3DGRT ray tracing) with full path-traced lighting
- OBJ mesh assets (horse, teapot) composited into the radiance field
- Multiple colored **point lights** lighting the splat set and the meshes
- **Ray-traced shadows** cast by mesh objects and splats including self-shadowing

## Interacting with the scene

- Switch cameras by pressing the space bar
- Switch DLSS on and experiment with the different upscaling modes (MAX, OPTIMAL, MIN)
- Switch the rendering pipeline to pure RTX and compare performance
- Select a light and move it using the visual helpers to see the impact on shading and shadows
- Play with the visualization modes
- Explore the scene and the UI!

!!! note
    If RTX is not supported, the system will fall back to 3DGS Raster. Shadows and advanced mesh materials will not show up in this mode.

## Assets

| Type | Source |
|------|--------|
| Splat model | Winter-Garden-view2 (teleportour.com, by Andrii Shramko) |
| Mesh models | horse.obj, teapot.obj (common 3D test models) |
