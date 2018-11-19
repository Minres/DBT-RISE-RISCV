cmake_minimum_required(VERSION 3.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/sc-components/cmake)

set(ENABLE_SCV TRUE CACHE BOOL "Enable use of SCV")

include(GitFunctions)
get_branch_from_git()
# if we are not on master or develop set the submodules to develop
IF(NOT ${GIT_BRANCH} MATCHES "master") 
	IF(NOT ${GIT_BRANCH} MATCHES "develop") 
		message(STATUS "main branch is '${GIT_BRANCH}', setting submodules to 'develop'")
		set(GIT_BRANCH develop)
	endif()
endif()

### set the directory names of the submodules
set(GIT_SUBMODULES elfio libGIS sc-components dbt-core)
set(GIT_SUBMODULE_DIR_sc-components .)
set(GIT_SUBMODULE_DIR_dbt-core .)
### set each submodules's commit or tag that is to be checked out
### (leave empty if you want master)
#set(GIT_SUBMODULE_VERSION_sc-comp 3af6b9836589b082c19d9131c5d0b7afa8ddd7cd)
set(GIT_SUBMODULE_BRANCH_sc-components ${GIT_BRANCH})
set(GIT_SUBMODULE_BRANCH_dbt-core ${GIT_BRANCH})

include(GNUInstallDirs)
include(Submodules)
include(Conan)

#enable_testing() 

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
	
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
if("${CMAKE_BUILD_TYPE}" STREQUAL "") 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
elseif(NOT(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Werror")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()

setup_conan()

# This line finds the boost lib and headers. 
set(Boost_NO_BOOST_CMAKE ON) #  Don't do a find_package in config mode before searching for a regular boost install.
find_package(Boost COMPONENTS program_options system thread filesystem REQUIRED)

if(DEFINED ENV{LLVM_HOME})
	find_path (LLVM_DIR LLVM-Config.cmake $ENV{LLVM_HOME}/lib/cmake/llvm)
endif(DEFINED ENV{LLVM_HOME})
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
llvm_map_components_to_libnames(llvm_libs support core mcjit x86codegen x86asmparser)

find_package(Threads)
find_package(Tcmalloc)
find_package(ZLIB)
find_package(SystemC)
if(SystemC_FOUND)
        message(STATUS "SystemC headers at ${SystemC_INCLUDE_DIRS}")
        message(STATUS "SystemC library at ${SystemC_LIBRARY_DIRS}")
        if(SCV_FOUND)
            message(STATUS "SCV headers at ${SCV_INCLUDE_DIRS}")
            message(STATUS "SCV library at ${SCV_LIBRARY_DIRS}")
        endif(SCV_FOUND)
        if(CCI_FOUND)
            message(STATUS "CCI headers at ${CCI_INCLUDE_DIRS}")
            message(STATUS "CCI library at ${CCI_LIBRARY_DIRS}")
        endif()
endif(SystemC_FOUND)

set(PROJECT_3PARTY_DIRS external)
include(clang-format)

set(ENABLE_CLANG_TIDY OFF CACHE BOOL "Add clang-tidy automatically to builds")
if (ENABLE_CLANG_TIDY)
    find_program (CLANG_TIDY_EXE NAMES "clang-tidy" PATHS /usr/local/opt/llvm/bin )
    if (CLANG_TIDY_EXE)
        message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
        set(CLANG_TIDY_CHECKS "-*,modernize-*")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=${CLANG_TIDY_CHECKS};-header-filter='${CMAKE_SOURCE_DIR}/*';-fix"
            CACHE STRING "" FORCE)
    else()
        message(AUTHOR_WARNING "clang-tidy not found!")
        set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE) # delete it
    endif()
endif()
  
# Set the version number of your project here (format is MAJOR.MINOR.PATCHLEVEL - e.g. 1.0.0)
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

add_subdirectory(external)
add_subdirectory(dbt-core)
add_subdirectory(sc-components)
add_subdirectory(softfloat)
GET_DIRECTORY_PROPERTY(SOFTFLOAT_INCLUDE_DIRS DIRECTORY softfloat DEFINITION SOFTFLOAT_INCLUDE_DIRS)
add_subdirectory(riscv)
add_subdirectory(platform)

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
