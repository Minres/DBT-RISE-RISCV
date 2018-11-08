cmake_minimum_required(VERSION 3.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake) # main (top) cmake dir

# CMake useful variables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib") 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Set the name of your project here
project("riscv.sc")

include(Common)

# check that we have averything we need
if(!SystemC_FOUND)
    message( FATAL_ERROR "SystemC library not found." )
endif()

if(!CCI_FOUND)
    message( FATAL_ERROR "SystemC CCI library not found." )
endif()

add_definitions(-DWITH_SYSTEMC) 
include_directories(${SystemC_INCLUDE_DIRS})
link_directories(${SystemC_LIBRARY_DIRS})

include_directories(${CCI_INCLUDE_DIRS})
link_directories(${CCI_LIBRARY_DIRS})

if(SCV_FOUND)   
    add_definitions(-DWITH_SCV)
    include_directories(${SCV_INCLUDE_DIRS})
    link_directories(${SCV_LIBRARY_DIRS})
endif()

# This sets the include directory for the reference project. This is the -I flag in gcc.
include_directories(
    ${PROJECT_SOURCE_DIR}/incl
    ${LLVM_INCLUDE_DIRS}
)

add_dependent_subproject(dbt-core)
add_dependent_subproject(sc-components)
add_dependent_subproject(riscv)

include_directories(
    ${PROJECT_SOURCE_DIR}/incl
    ${PROJECT_SOURCE_DIR}/../riscv/incl
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

# library files
FILE(GLOB RiscVSCHeaders *.h */*.h)

set(LIB_HEADERS ${RiscVSCHeaders} )
set(LIB_SOURCES src/core_complex.cpp )

# Define two variables in order not to repeat ourselves.
set(LIBRARY_NAME riscv.sc)

# Define the library
add_library(${LIBRARY_NAME} ${LIB_SOURCES})

set_target_properties(${LIBRARY_NAME} PROPERTIES
  VERSION ${VERSION}  # ${VERSION} was defined in the main CMakeLists.
  FRAMEWORK FALSE
  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
)

#
# SYSTEM PACKAGING (RPM, TGZ, ...)
# _____________________________________________________________________________

#include(CPackConfig)

#
# CMAKE PACKAGING (for other CMake projects to use this one easily)
# _____________________________________________________________________________

#include(PackageConfigurator)