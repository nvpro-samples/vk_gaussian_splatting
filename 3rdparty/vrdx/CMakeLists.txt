cmake_minimum_required(VERSION 3.15)

project(vk_radix_sort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED)

# We use SLANG from external resolution in ${Slang_SLANGC_EXECUTABLE}

# adds -fPIC, works for linux, when building shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# build_shader(TARGET SHADER OUTPUT DEFINE...)
function(build_shader)
  list(POP_FRONT ARGV TARGET SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  set(OUTPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT})

  if(UNIX)
     # Workaround: the Vulkan SDK sets LD_LIBRARY_PATH and
     # slangc may find a libslang.so there instead of its own
     set(_SLANGC env LD_LIBRARY_PATH= ${Slang_SLANGC_EXECUTABLE})
  else()
     set(_SLANGC ${Slang_SLANGC_EXECUTABLE})
  endif()

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  set(_COMMAND_H ${_SLANGC}
     ${DEFINES}
     -target spirv             # Target SPIR-V
     -fvk-t-shift 0 0
     -fvk-u-shift 0 0
     -source-embed-name ${OUTPUT}
     -source-embed-style text
     -depfile "${OUTPUT_FILE}.dep"
     -o "${OUTPUT_FILE}.h" ${SHADER}
  )
  add_custom_command(
     OUTPUT ${OUTPUT_FILE}.h ${OUTPUT_FILE}.spv
     COMMAND echo ${_COMMAND_H}
     COMMAND ${_COMMAND_H}
     MAIN_DEPENDENCY ${SHADER}
     DEPFILE "${OUTPUT_FILE}.dep"
     COMMENT "Compiling Slang shader ${SHADER_NAME}"
  )

  add_custom_target(${OUTPUT} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h)
  add_dependencies(${TARGET} ${OUTPUT})
endfunction()

# library
add_library(vk_radix_sort STATIC
  src/vk_radix_sort.cc
)

target_include_directories(vk_radix_sort
  PUBLIC include
  PRIVATE src
)

target_link_libraries(vk_radix_sort
  PUBLIC volk
)

build_shader(vk_radix_sort src/shader/upsweep.slang upsweep_slang)
build_shader(vk_radix_sort src/shader/spine.slang spine_slang)
build_shader(vk_radix_sort src/shader/downsweep.slang downsweep_slang)
build_shader(vk_radix_sort src/shader/downsweep.slang downsweep_key_value_slang KEY_VALUE)

