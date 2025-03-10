
#
# CMAKE to deal with Slang shader files
#

include(${CMAKE_CURRENT_LIST_DIR}/find_entrypoints.cmake)

set(CMAKE_PATH ${CMAKE_CURRENT_LIST_DIR})

# -----------------------------------------------------------------------------
# Function to compile a Slang file
#
# Example:
# compile_slang_file SOURCE_FILE foo.slang
#   FLAGS "-I ${SAMPLES_COMMON_DIR}/shaders"
# )
# target_sources(${PROJECT_NAME} PRIVATE ${SLANG_OUTPUT_FILES})
#
# Inputs:
# SOURCE_FILE (string): The file to compile.
# FLAGS (list of strings): Optional flags to add to the slangc command line.
#
# Outputs (all in the parent scope):
# SLANG_OUTPUT_FILES: Files that will be generated. Set new each time, not appended to.
#
# Flags:
# DEBUG: If specified, will generate .glsl instead.
function(compile_slang_file)
    set(options DEBUG)
    set(oneValueArgs SOURCE_FILE)
    set(multiValueArgs FLAGS)
    cmake_parse_arguments(COMPILE  "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    
    set(SLANG_OUTPUT_FILES "")
    
    # Do nothing if Slang isn't used
    if(NOT USE_SLANG)       
        return()
    endif()
    
    if(NOT COMPILE_SOURCE_FILE)
      message(FATAL_ERROR "compile_slang_file called without a source file!")
    endif()
    
    # Get only the filename without path or extension
    get_filename_component(_FILE_STEM ${COMPILE_SOURCE_FILE} NAME_WE)
    
    # Create the output directory if it does not exist
    set(_OUT_DIR ${SAMPLE_FOLDER}/_autogen)
    file(MAKE_DIRECTORY ${_OUT_DIR})
    
    # This will contain a list of COMMAND arguments.
    set(_SLANG_COMMANDS )
    
    # !! Compiling all entry points in a single compilation
    set(_OUT_ARG "${_OUT_DIR}/${_FILE_STEM}_slang.h")
    set(_SLANG_FLAGS
        -profile sm_6_6+spirv_1_6
        -capability spvInt64Atomics+spvShaderInvocationReorderNV+spvShaderClockKHR+spvRayTracingMotionBlurNV
        -target spirv
        -emit-spirv-directly
        -force-glsl-scalar-layout
        -fvk-use-entrypoint-name
        -g3
        -source-embed-style text 
        -source-embed-name ${_FILE_STEM}Slang
        -o ${_OUT_ARG}
    )
    list(APPEND _SLANG_FLAGS ${COMPILE_FLAGS})

    list(APPEND _SLANG_COMMANDS
         COMMAND ${CMAKE_COMMAND} -E echo ${Slang_slangc_EXE} ${_SLANG_FLAGS} ${COMPILE_SOURCE_FILE}  
         COMMAND ${Slang_slangc_EXE} ${_SLANG_FLAGS} ${COMPILE_SOURCE_FILE}
    )
    # list(APPEND SLANG_OUTPUT_FILES ${_OUT_ARG})
    set(SLANG_OUTPUT_FILES ${_OUT_ARG})

    # We emit a single large add_custom_command so that it's possible to
    # right-click on a .slang file in Visual Studio and build all its outputs.
    # nbickford imagines doing this using add_custom_command(... APPEND) one day,
    # but CMake doesn't support that yet according to its 3.27.1 docs.
    add_custom_command(
      OUTPUT ${SLANG_OUTPUT_FILES}
      ${_SLANG_COMMANDS}
      MAIN_DEPENDENCY ${COMPILE_SOURCE_FILE}
      VERBATIM COMMAND_EXPAND_LISTS
    )
    set(SLANG_OUTPUT_FILES ${SLANG_OUTPUT_FILES} PARENT_SCOPE)

endfunction()

