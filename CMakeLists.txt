cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

### set the directory names of the submodules
set(GIT_SUBMODULES elfio libGIS sc-components)
set(GIT_SUBMODULE_DIR_sc-components .)
set(GIT_SUBMODULE_DIR_dbt-core .)
### set each submodules's commit or tag that is to be checked out
### (leave empty if you want master)
#set(GIT_SUBMODULE_VERSION_sc-comp 3af6b9836589b082c19d9131c5d0b7afa8ddd7cd)
set(GIT_SUBMODULE_BRANCH_sc-components "develop")

include(GNUInstallDirs)
include(cmake/Submodules.cmake)

#enable_testing() 

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -Werror")
    set(CMAKE_CXX_FLAG_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_C_FLAG_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAG_DEBUG "-Og")
    set(CMAKE_C_FLAG_DEBUG "-Og")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()

FIND_PACKAGE(Threads)

add_subdirectory(external)
add_subdirectory(dbt-core)
add_subdirectory(sc-components)
add_subdirectory(riscv)
add_subdirectory(riscv.sc)
