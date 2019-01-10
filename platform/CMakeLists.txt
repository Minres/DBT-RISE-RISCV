cmake_minimum_required(VERSION 3.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake) # main (top) cmake dir
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake) # project specific cmake dir

# CMake useful variables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib") 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Set the name of your project here
project("platform")

include(Common)
# check that we have averything we need
if(!SystemC_FOUND)
    message( FATAL_ERROR "SystemC library not found." )
endif()

if(!CCI_FOUND)
    message( FATAL_ERROR "SystemC CCI library not found." )
endif()

# This sets the include directory for the reference project. This is the -I flag in gcc.

add_dependent_subproject(dbt-core)
add_dependent_subproject(sc-components)
add_dependent_subproject(riscv)

include_directories(
    ${PROJECT_SOURCE_DIR}/../external/elfio
    ${PROJECT_SOURCE_DIR}/../external/libGIS
    ${Boost_INCLUDE_DIRS}
)

# Mac needed variables (adapt for your needs - http://www.cmake.org/Wiki/CMake_RPATH_handling#Mac_OS_X_and_the_RPATH)
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

## the following setting needs to be consistent with the library
#add_definitions(-DSC_DEFAULT_WRITER_POLICY=SC_MANY_WRITERS)

add_subdirectory(src)

#
# SYSTEM PACKAGING (RPM, TGZ, ...)
# _____________________________________________________________________________

#include(CPackConfig)

#
# CMAKE PACKAGING (for other CMake projects to use this one easily)
# _____________________________________________________________________________

#include(PackageConfigurator)