
# -----------------------------------------------------------------------------
# Finding DXC and the runtime library

include(${CMAKE_CURRENT_LIST_DIR}/find_entrypoints.cmake)


# Find dxc using the Vulkan include paths. This doesn't use COMPONENTS DXC so
# that it is compatible with CMake < 3.25.
find_package(Vulkan QUIET REQUIRED)
get_filename_component(_VULKAN_LIB_DIR ${Vulkan_LIBRARY} DIRECTORY)
find_file(Vulkan_dxc_EXECUTABLE
  NAMES dxc${CMAKE_EXECUTABLE_SUFFIX}
  PATHS ${_VULKAN_LIB_DIR}/../Bin)
if(Vulkan_dxc_EXECUTABLE)
  message(STATUS "--> using DXC from: ${Vulkan_dxc_EXECUTABLE}")
else()
  message(STATUS "--> Could not find DXC")
endif()
mark_as_advanced(Vulkan_dxc_EXECUTABLE)


function(get_target stage profile_version)
    set(target "")
    if(${stage} STREQUAL "vertex")
        set(target "vs_")
    elseif(${stage} STREQUAL "pixel")
        set(target "ps_")
    elseif(${stage} STREQUAL "compute")
        set(target "cs_")
    elseif(${stage} STREQUAL "geometry")
        set(target "gs_")
    elseif(${stage} STREQUAL "raygeneration" OR
           ${stage} STREQUAL "intersection" OR
           ${stage} STREQUAL "anyhit" OR
           ${stage} STREQUAL "miss" OR
           ${stage} STREQUAL "closesthit")
        set(target "lib_")
    endif()

    set(_TARGET "${target}${profile_version}" PARENT_SCOPE)
    return()
endfunction()


# -----------------------------------------------------------------------------
# Function to compile a HLSL file
#
# Example:
# compile_hlsl_file SOURCE_FILE foo.hlsl
#   FLAGS "-I ${SAMPLES_COMMON_DIR}/shaders"
# )
# target_sources(${PROJECT_NAME} PRIVATE ${HLSL_OUTPUT_FILES})
#
# Inputs:
# SOURCE_FILE (string): The file to compile.
# FLAGS (list of strings): Optional flags to add to the dxc command line.
#
# Outputs (all in the parent scope):
# HLSL_OUTPUT_FILES: Files that will be generated. Set new each time, not appended to.
#
# Flags:
#  
function(compile_hlsl_file)
    set(options DEBUG)
    set(oneValueArgs SOURCE_FILE PROFILE_VERSION)
    set(multiValueArgs FLAGS)
    cmake_parse_arguments(COMPILE  "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    
    set(HLSL_OUTPUT_FILES "")
    
    # Do nothing if HLSL isn't used
    if(NOT USE_HLSL)       
        return()
    endif()

    # Trying to compile a HLSL file without a compiler is a fatal error.
    # This prevents silent failure at compile-time.
    if(NOT Vulkan_dxc_EXECUTABLE)
      message(FATAL_ERROR "compile_hlsl_file called, but Vulkan_dxc_EXECUTABLE (value: ${Vulkan_dxc_EXECUTABLE}) was not found or specified!")
    endif()
    
    if(NOT COMPILE_SOURCE_FILE)
      message(FATAL_ERROR "compile_hlsl_file called without a source file!")
    endif()

    # Set the default target
    if(NOT COMPILE_PROFILE_VERSION)
      set(COMPILE_PROFILE_VERSION "6_7")
    endif()
    
    # Get only the filename without path or extension
    get_filename_component(_FILE_STEM ${COMPILE_SOURCE_FILE} NAME_WE)
    
    # Create the output directory if it does not exist
    set(_OUT_DIR ${SAMPLE_FOLDER}/_autogen)
    file(MAKE_DIRECTORY ${_OUT_DIR})
    
    # This will contain a list of COMMAND arguments.
    set(_HLSL_COMMANDS )
    
     # Call the function to find entrypoints
    find_entrypoints(${COMPILE_SOURCE_FILE} ENTRYPOINTS_LIST STAGES_LIST)

    # Get the length of the list
    list(LENGTH ENTRYPOINTS_LIST ENTRYPOINTS_LIST_LENGTH)
    math(EXPR ENTRYPOINTS_LIST_LENGTH "${ENTRYPOINTS_LIST_LENGTH} - 1")

    # Iterate over the indices
    foreach(INDEX RANGE ${ENTRYPOINTS_LIST_LENGTH})
      # Access the element at the current index
      list(GET ENTRYPOINTS_LIST ${INDEX} _ENTRY_NAME)
      list(GET STAGES_LIST ${INDEX} _STAGE)
   
      message(STATUS "  - Found entrypoint: ${_ENTRY_NAME} (${_STAGE})")

      set(_VAR_NAME "${_FILE_STEM}_${_ENTRY_NAME}") # Variable name in header files
      set(_OUT_STEM "${_OUT_DIR}/${_VAR_NAME}") # Path to the output without extensions
      
      get_target(${_STAGE} ${COMPILE_PROFILE_VERSION}) # Get the target for the stage

      # Command line flags
      set(_HLSL_FLAGS 
          -spirv
          -Zi
          -fspv-target-env=vulkan1.3
          -D__hlsl
          -HV 2021
          -fvk-use-scalar-layout
          -fspv-extension=KHR
          -fspv-extension=SPV_EXT_descriptor_indexing
          # -fspv-debug=vulkan-with-source
          -E ${_ENTRY_NAME} 
          -T ${_TARGET}
          -Vn ${_VAR_NAME}
          -Fh "${_OUT_STEM}.spirv.h"
          -Fo "${_OUT_STEM}.spv"
      )

      # Adding extra command argument flags
      list(APPEND _HLSL_FLAGS ${COMPILE_FLAGS})
      
      set(_OUT_FILE "${_OUT_STEM}.spirv.h")

      list(APPEND _HLSL_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ${Vulkan_dxc_EXECUTABLE} ${_HLSL_FLAGS} ${COMPILE_SOURCE_FILE}
        COMMAND ${Vulkan_dxc_EXECUTABLE} ${_HLSL_FLAGS} ${COMPILE_SOURCE_FILE}
      )
      list(APPEND HLSL_OUTPUT_FILES ${_OUT_FILE})
    endforeach()
    
    # We emit a single large add_custom_command so that it's possible to
    # right-click on a .hlsl file in Visual Studio and build all its outputs.
    # nbickford imagines doing this using add_custom_command(... APPEND) one day,
    # but CMake doesn't support that yet according to its 3.27.1 docs.
    # Note that DEPENDS isn't required, because MAIN_DEPENDENCY is automatically
    # a dependency.
    add_custom_command(
        OUTPUT ${HLSL_OUTPUT_FILES}
        ${_HLSL_COMMANDS}
        MAIN_DEPENDENCY ${COMPILE_SOURCE_FILE}
        VERBATIM COMMAND_EXPAND_LISTS
      )
    set(HLSL_OUTPUT_FILES ${HLSL_OUTPUT_FILES} PARENT_SCOPE)
endfunction()

