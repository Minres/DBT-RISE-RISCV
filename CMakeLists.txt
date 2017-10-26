cmake_minimum_required(VERSION 3.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/sc-components/cmake)

include(cmake/GitFunctions.cmake)
get_branch_from_git()

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
	
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Werror")
    set(CMAKE_CXX_FLAG_RELEASE "-O3 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_C_FLAG_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAG_DEBUG "-Og -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_C_FLAG_DEBUG "-Og")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()

setup_conan()

find_package(Threads)
find_package(Tcmalloc)

set(PROJECT_3PARTY_DIRS external sr_report sr_signal)
include(clang-format)

add_subdirectory(external)
add_subdirectory(dbt-core)
add_subdirectory(sc-components)
add_subdirectory(riscv)
add_subdirectory(riscv.sc)

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")