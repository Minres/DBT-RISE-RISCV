cmake_minimum_required(VERSION 3.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake) # main (top) cmake dir
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake) # project specific cmake dir

# CMake useful variables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib") 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_subdirectory(softfloat)

# Set the name of your project here
project("riscv")

include(Common)

conan_basic_setup()

# This sets the include directory for the reference project. This is the -I flag in gcc.
include_directories(
    ${PROJECT_SOURCE_DIR}/incl
	${SOFTFLOAT_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)
add_dependent_subproject(dbt-core)
add_dependent_subproject(sc-components)
include_directories(
    ${PROJECT_SOURCE_DIR}/incl
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

add_subdirectory(src)

#
# SYSTEM PACKAGING (RPM, TGZ, ...)
# _____________________________________________________________________________

#include(CPackConfig)

#
# CMAKE PACKAGING (for other CMake projects to use this one easily)
# _____________________________________________________________________________

#include(PackageConfigurator)