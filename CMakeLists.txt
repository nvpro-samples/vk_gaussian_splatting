cmake_minimum_required(VERSION 3.6...3.31)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)

get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME} LANGUAGES C CXX)
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet

# which nvprocore tag or branch to download if repo not found
set(NVPRO_GIT_TAG main)
# Where to decompress nvprocore source code if repo not found
set(NVPRO_TGT_SRC_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps)

if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_SOURCE_DIR}/external
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  set(OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin_x64)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  # nvpro_core not found, will try to download.
  # first find where the current sample comes from
  execute_process( 
    COMMAND git config --get remote.origin.url 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
    OUTPUT_VARIABLE GIT_REPO_URL OUTPUT_STRIP_TRAILING_WHITESPACE 
  ) 
  # Check if "github.com" is in URL
  string(FIND "${GIT_REPO_URL}" "github.com" FOUND_INDEX)
  if (FOUND_INDEX GREATER -1)
    # Use regex to extract everything up to and including "github.com"
    string(REGEX MATCH ".*github\\.com" GIT_BASE_URL "${GIT_REPO_URL}")
    # construct URL
    string(FIND "${GIT_REPO_URL}" "git@" SSH_FOUND_INDEX)
    if (SSH_FOUND_INDEX GREATER -1) # ssh
      set(NVPRO_GIT_URL ${GIT_BASE_URL}:nvpro-samples/nvpro_core.git)
    else() # https
      set(NVPRO_GIT_URL ${GIT_BASE_URL}/nvpro-samples/nvpro_core.git)
    endif()
    if("${NVPRO_GIT_TAG}" STREQUAL "main" )
      set(NVPRO_GIT_TAG master)
    endif()
    message("Sample comes from github , nvprocore is at " ${NVPRO_GIT_URL} )
  else ()
    # reconstruct the path to nvpro_core, preserving the protocol
    string(REGEX MATCH "^[^/]+//[^/]+/" GIT_BASE_URL "${GIT_REPO_URL}")
    # construct URL
    set(NVPRO_GIT_URL ${GIT_BASE_URL}devtechproviz/nvpro-samples/nvpro_core.git)
    # message("Sample comes from prod server, nvprocore is at " ${NVPRO_GIT_URL})
  endif()
  # let's clone the commit we need, depth to 1 so that we do not download the full history
  execute_process( 
    COMMAND git clone --depth 1 --branch ${NVPRO_GIT_TAG} ${NVPRO_GIT_URL} ${CMAKE_CURRENT_BINARY_DIR}/_deps/nvpro_core
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
  ) 
  # do the search again with downloaded version, use find to be sure everyting runs ok
  find_path(BASE_DIRECTORY
    NAMES nvpro_core
    PATHS ${CMAKE_CURRENT_BINARY_DIR}/_deps
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
  # invoke the setup
  if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
    set(OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin_x64)
    include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  else()
    message(FATAL_ERROR "could not find base directory or download nvpro_core, please set BASE_DIRECTORY to folder containing nvpro_core")
  endif()
endif()
set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)

_add_project_definitions(${PROJNAME})

# Download the default scene
# download_files(FILENAMES flowers_1.zip EXTRACT)

# Set the install diectory in the project directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/_install" CACHE PATH "folder in which INSTALL will put everything needed to run the binaries" FORCE)

#####################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES

_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_IMGUI()
_add_package_NVML()

#####
# Include Vulkan Radix Sort
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx)

set_property(TARGET vk_radix_sort PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET spine_comp PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET downsweep_comp PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET downsweep_key_value_comp PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET upsweep_comp PROPERTY FOLDER "ThirdParty/vrdx")

#####################################################################################
# process the rest of some cmake code that needs to be done *after* the packages add
_add_nvpro_core_lib()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.*)
file(GLOB SHADER_FILES shaders/*.glsl shaders/*.h)
file(GLOB EXTERN_FILES 3rdparty/miniply/*.*)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src 
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders 
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/miniply
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx)

#####################################################################################
# Executable
#

if(WIN32 AND NOT GLUT_FOUND)
  add_definitions(/wd4996) #remove printf warning
  add_definitions(/wd4244) #remove double to float conversion warning
  add_definitions(/wd4305) #remove double to float truncation warning
else()
  add_definitions(-fpermissive)
endif()
add_executable(${PROJNAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${SHADER_FILES} ${EXTERN_FILES})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJNAME})

target_compile_definitions(${PROJNAME} PRIVATE NVPRO_CORE_DIR="${NVPRO_CORE_DIR}")

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

if(UNIX)
  set(UNIXLINKLIBS dl pthread)
else()
  set(UNIXLINKLIBS)
endif()

#####################################################################################
# Linkage
#

target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} nvpro_core Vulkan::Vulkan vk_radix_sort)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#

_finalize_target( ${PROJNAME} )

install(FILES ${SHADER_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${PROJNAME}")
install(FILES ${SHADER_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${PROJNAME}")

