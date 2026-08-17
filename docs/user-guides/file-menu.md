# File Menu and Project Management

The **File** menu provides the following operations:

*   **New project** – Resets the scene to a clean state.
*   **Open project** / **Recent projects** – Loads a previously saved project file (`.vkgs`), restoring all splat sets, meshes, camera presets, and renderer settings.
*   **Save project** – Saves the current scene configuration to a `.vkgs` project file. Splat sets and meshes are not embedded in the project file; only paths to these resources are stored, along with their transforms, materials, and renderer settings. Paths are written relative to the project file (and with forward slashes) so a project folder can be moved, archived, or shared between Windows and Linux. A resource that lives on another drive or network share cannot be expressed relatively and is stored as an absolute path instead.
*   **Open Splat Set** / **Recent Splat Sets** – Imports a radiance field from a `.ply`, `.spz`, or `.splat` file.
*   **Open Mesh** – Imports a 3D mesh from an `.obj` file.
*   **Exit** – Closes the application (`Ctrl+Q`).

Files can also be loaded by **drag and drop** onto the viewport. Supported formats are `.ply`, `.spz`, and `.splat` (splat sets), `.obj` (meshes), and `.vkgs` (project files). Multiple files can be dropped at once.
