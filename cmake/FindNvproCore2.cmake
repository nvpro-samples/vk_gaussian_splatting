# FindNvproCore2.cmake

if(NOT NvproCore2_FOUND)
    set(_Print_info TRUE)
endif()


# Try to find local installation first
find_path(NvproCore2_ROOT
    NAMES nvpro_core2/cmake/Setup.cmake
    PATHS
    ${CMAKE_BINARY_DIR}/_deps
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/../..
)

if(NvproCore2_ROOT)
    set(NvproCore2_FOUND TRUE)
else()
    # Option to allow downloading if not found
    option(NVPROCORE2_DOWNLOAD "Download nvpro_core2 if not found" ON)

    if(NVPROCORE2_DOWNLOAD)
        include(FetchContent)

        # Set default git tag/branch
        set(NVPRO_GIT_TAG "main" CACHE STRING "Git tag/branch for nvpro_core2")

        # Try to determine nvpro_core location from git remote
        execute_process(
            COMMAND git config --get remote.origin.url
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_REPO_URL
            RESULT_VARIABLE GIT_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NOT GIT_RESULT EQUAL 0)
            message(WARNING "Failed to get git remote URL. Defaulting to GitHub URL.")
            set(GIT_REPO_URL "https://github.com/nvpro-samples/nvpro_core2.git")
            set(GIT_BASE_URL "https://github.com")
            set(NVPRO_GIT_URL ${GIT_REPO_URL})
        else()
            # Check if this is a GitHub repository
            string(FIND "${GIT_REPO_URL}" "github.com" FOUND_INDEX)
            if(FOUND_INDEX GREATER -1)
                # Extract base URL up to github.com
                string(REGEX MATCH ".*github\\.com" GIT_BASE_URL "${GIT_REPO_URL}")
                if(NOT GIT_BASE_URL)
                    message(FATAL_ERROR "Failed to extract GitHub base URL from ${GIT_REPO_URL}")
                endif()

                # Handle SSH vs HTTPS URLs differently
                string(FIND "${GIT_REPO_URL}" "git@" SSH_FOUND_INDEX)
                if(SSH_FOUND_INDEX GREATER -1)
                    # SSH format
                    set(NVPRO_GIT_URL ${GIT_BASE_URL}:nvpro-samples/nvpro_core2.git)
                else()
                    # HTTPS format
                    set(NVPRO_GIT_URL ${GIT_BASE_URL}/nvpro-samples/nvpro_core2.git)
                endif()

                # GitHub uses 'master' instead of 'main'
                if("${NVPRO_GIT_TAG}" STREQUAL "main")
                    set(NVPRO_GIT_TAG master)
                endif()

                message(STATUS "Using GitHub nvpro_core2 repository")
            else()
                # Internal repository - reconstruct URL preserving the protocol
                string(REGEX MATCH "^[^/]+//[^/]+/" GIT_BASE_URL "${GIT_REPO_URL}")
                if(NOT GIT_BASE_URL)
                    message(FATAL_ERROR "Failed to extract base URL from ${GIT_REPO_URL}")
                endif()

                set(NVPRO_GIT_URL ${GIT_BASE_URL}devtechproviz/nvpro-samples/nvpro_core2.git)
                message(STATUS "Using internal nvpro_core2 repository")
            endif()
        endif()

        if(NOT NVPRO_GIT_URL)
            message(FATAL_ERROR "Failed to construct git URL for nvpro_core2")
        endif()

        message(STATUS "Will clone from: ${NVPRO_GIT_URL} (branch/tag: ${NVPRO_GIT_TAG})")

         # let's clone the commit we need, depth to 1 so that we do not download the full history
        execute_process( 
            COMMAND git clone --depth 1 --branch ${NVPRO_GIT_TAG} ${NVPRO_GIT_URL} ${CMAKE_BINARY_DIR}/_deps/nvpro_core2
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR} 
        ) 

        # Try to find local installation first
        find_path(NvproCore2_ROOT
            NAMES nvpro_core2/cmake/Setup.cmake
            PATHS
            ${CMAKE_BINARY_DIR}/_deps
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/..
            ${CMAKE_SOURCE_DIR}/../..
        )

        if(NvproCore2_ROOT)
            set(NvproCore2_FOUND TRUE)
        endif()
    endif()
endif()

if(_Print_info)
    message(STATUS "Found nvpro_core2 at: ${NvproCore2_ROOT}")
endif()

if(NvproCore2_FOUND)
    set(NvproCore2_FOUND TRUE)   

    # Include the setup file which will add all the necessary libraries
    # and create the actual targets (nvpro2::nvvk etc)
    include(${NvproCore2_ROOT}/nvpro_core2/cmake/Setup.cmake)
endif()
