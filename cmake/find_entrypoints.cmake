
function(find_entrypoints COMPILE_SOURCE_FILE entrypoints_list stages_list)
  # Read the input file and search through it to find entrypoints.
  file(READ ${COMPILE_SOURCE_FILE} _CONTENT)
  
  # This regex has 3 groups: 1 is the shader stage, and 3 contains the name of
  # the entrypoint, using [=[ ... ]=] to avoid CMake escaping
  # For instance, in
  # ```
  # [shader("compute")]
  # [numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
  # void computeMain(uint3 threadIdx : SV_DispatchThreadID)
  # ```
  # group 0 is `compute` and group 1 is `computeMain`.
  #
  # To do this, it searches for strings that look like this:
  # [shader("group_1_text"...
  # [text_we_ignore] (0 or more times)
  # line starting declarations group_3_text(   
  set(_REGEX [=[\[shader\(\"(\w+)\"[^\n]*[\n](\[[^\n]*\][\n])*[\w+[ \t]+]*(\w+)\(]=] )
  
  # CMake has its own regex engine quirks, so replace accordingly
  string(REPLACE [=[\n]=] "\r\n" _REGEX "${_REGEX}")
  string(REPLACE [=[\t]=] "\t" _REGEX "${_REGEX}")
  string(REPLACE [=[\w]=] "[a-zA-Z0-9_]" _REGEX "${_REGEX}")
  
  set(entrypoint_list)
  set(stage_list)
  
  while(TRUE)
    string(REGEX MATCH "${_REGEX}" _REGEX_MATCH ${_CONTENT})
    if(NOT _REGEX_MATCH)
      break()
    endif()
    
    if(NOT CMAKE_MATCH_1)
      message(FATAL_ERROR "Could not find shader stage!")
    endif()
    
    if(NOT CMAKE_MATCH_3)
      message(FATAL_ERROR "Could not find shader entrypoint entry name!")
    endif()
    
    set(_STAGE ${CMAKE_MATCH_1})
    set(_ENTRY_NAME ${CMAKE_MATCH_3})
    
    list(APPEND entrypoint_list ${_ENTRY_NAME})
    list(APPEND stage_list ${_STAGE})
    
    # Remove the part of the string up to and including the point where the
    # entrypoint declaration was found
    string(FIND "${_CONTENT}" "${_REGEX_MATCH}" _ENTRYPOINT_DECL_START)
    string(LENGTH "${_REGEX_MATCH}" _ENTRYPOINT_DECL_LENGTH)
    math(EXPR _SUBSTR_START ${_ENTRYPOINT_DECL_START}+${_ENTRYPOINT_DECL_LENGTH})
    string(SUBSTRING "${_CONTENT}" "${_SUBSTR_START}" -1 _CONTENT)
  endwhile()

  set(${entrypoints_list} ${entrypoint_list} PARENT_SCOPE)
  set(${stages_list} ${stage_list} PARENT_SCOPE)
endfunction()
