cmake_minimum_required(VERSION 3.6...3.31)

# get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(PROJECT_NAME "vk_gaussian_splatting")
Project(${PROJECT_NAME} LANGUAGES C CXX)

# Load version from VERSION file and expose to CMake and C++
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VKGS_VERSION_RAW)
string(STRIP "${VKGS_VERSION_RAW}" VKGS_VERSION)
message(STATUS "VKGS_VERSION: ${VKGS_VERSION}")

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

# set with -DDISABLE_DEFAULT_SCENE=ON on command line or use cmake-gui to set the parameter
# if default scene is not disabled at build using this option, one can still disable it by 
# providing "-loadDefaultScene 0" to the sample command line
option(DISABLE_DEFAULT_SCENE "If set to true the default scene is not downloaded by CMake, and the sample does not try to open it when starting up with no scene as parameter" OFF)

# DLSS Ray Reconstruction support
option(USE_DLSS "Use DLSS-RR for denoising" ON)

# Download the default scene
if (NOT DISABLE_DEFAULT_SCENE)
  add_definitions(-DWITH_DEFAULT_SCENE_FEATURE)
  download_files(FILENAMES flowers_1.zip EXTRACT)
endif()

#####################################################################################
# additions of 3rdparty

#####
# Include Vulkan Radix Sort
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx)

set_property(TARGET vk_radix_sort PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET spine_slang PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET downsweep_slang PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET downsweep_key_value_slang PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET upsweep_slang PROPERTY FOLDER "ThirdParty/vrdx")

#####
# Include spz (uses zlib from nvpro_core2)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spz)
set_property(TARGET spz PROPERTY FOLDER "ThirdParty")

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.*)
# DLSS sources (src/dlss_*.cpp/.hpp) require the NVIDIA NGX SDK and are added
# explicitly in the USE_DLSS block below. Exclude them from the glob so the
# project builds with -DUSE_DLSS=OFF on platforms without NGX (e.g. AMD/RADV).
list(FILTER SOURCE_FILES EXCLUDE REGEX "/dlss_")
file(GLOB SHADER_FILES shaders/*.slang shaders/*.h)
file(GLOB EXTERN_FILES 3rdparty/miniply/*.*)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src 
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders 
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/miniply
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spz/src/cc)

#####################################################################################
# Executable
#

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${SHADER_FILES} ${EXTERN_FILES})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_DEFINE_MATH_OPERATORS )
target_compile_definitions(${PROJECT_NAME} PRIVATE VKGS_VERSION="${VKGS_VERSION}")

#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group("Shader Files" FILES ${SHADER_FILES})
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Extern Files" FILES ${EXTERN_FILES})

#####################################################################################
# Linkage
#

# Linking with other libraries
# Link libraries and include directories
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvvkglsl
  nvpro2::nvslang
  nvpro2::nvshaders_host
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvgpu_monitor
  vk_radix_sort
  spz::spz
  tinygltf
)

add_project_definitions(${PROJECT_NAME})

#####################################################################################
# Adding DLSS if selected
if(USE_DLSS)
  set(DLSS_VERSION "310.5.0")
  find_package(NGX REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DLSS)
  target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_denoiser.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_denoiser.hpp
  )
endif()

#####################################################################################
# copies files
#

copy_to_runtime_and_install(${PROJECT_NAME}
  DIRECTORIES "${NVSHADERS_DIR}/nvshaders"
  FILES ${Slang_GLSLANG} ${DLSS_DLLS}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
  AUTO
)