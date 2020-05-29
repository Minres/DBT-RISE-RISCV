cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake) # main (top) cmake dir
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake) # project specific cmake dir

# CMake useful variables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib") 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Set the name of your project here
project("riscv")

include(Common)

conan_basic_setup()

find_package(Boost COMPONENTS program_options system thread filesystem REQUIRED)

# This sets the include directory for the reference project. This is the -I flag in gcc.
include_directories(
    ${PROJECT_SOURCE_DIR}/incl
	${SOFTFLOAT_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)
add_dependent_subproject(dbt-core)
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

add_subdirectory(softfloat)

# library files
FILE(GLOB RiscVSCHeaders ${CMAKE_CURRENT_SOURCE_DIR}/incl/sysc/*.h ${CMAKE_CURRENT_SOURCE_DIR}/incl/sysc/*/*.h)
set(LIB_HEADERS ${RiscVSCHeaders} )
set(LIB_SOURCES 
	#src/iss/rv32gc.cpp
	src/iss/rv32imac.cpp
	#src/iss/rv64i.cpp
	#src/iss/rv64gc.cpp
	src/iss/mnrv32.cpp
	src/vm/llvm/fp_functions.cpp
	src/vm/llvm/vm_mnrv32.cpp
	#src/vm/llvm/vm_rv32gc.cpp
	#src/vm/llvm/vm_rv32imac.cpp
	#src/vm/llvm/vm_rv64i.cpp
	#src/vm/llvm/vm_rv64gc.cpp
	src/vm/tcc/vm_mnrv32.cpp
	src/vm/interp/vm_mnrv32.cpp
    src/plugin/instruction_count.cpp
    src/plugin/cycle_estimate.cpp)

# Define two variables in order not to repeat ourselves.
set(LIBRARY_NAME riscv)

# Define the library
add_library(${LIBRARY_NAME} ${LIB_SOURCES})
SET(${LIBRARY_NAME} -Wl,-whole-archive -l${LIBRARY_NAME} -Wl,-no-whole-archive)
target_link_libraries(${LIBRARY_NAME} softfloat dbt-core scc-util)
set_target_properties(${LIBRARY_NAME} PROPERTIES
  VERSION ${VERSION}  # ${VERSION} was defined in the main CMakeLists.
  FRAMEWORK FALSE
  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
)

if(SystemC_FOUND)
	set(SC_LIBRARY_NAME riscv_sc)
	add_library(${SC_LIBRARY_NAME} src/sysc/core_complex.cpp)
	add_definitions(-DWITH_SYSTEMC) 
	include_directories(${SystemC_INCLUDE_DIRS})
	
	include_directories(${CCI_INCLUDE_DIRS})
	
	if(SCV_FOUND)   
	    add_definitions(-DWITH_SCV)
	    include_directories(${SCV_INCLUDE_DIRS})
	endif()
	target_link_libraries(${SC_LIBRARY_NAME} ${LIBRARY_NAME})
	target_link_libraries(${SC_LIBRARY_NAME} dbt-core)
	target_link_libraries(${SC_LIBRARY_NAME} softfloat)
	target_link_libraries(${SC_LIBRARY_NAME} scc)
	target_link_libraries(${SC_LIBRARY_NAME} external)
	target_link_libraries(${SC_LIBRARY_NAME} ${llvm_libs})
	target_link_libraries(${SC_LIBRARY_NAME} ${Boost_LIBRARIES} )
	set_target_properties(${SC_LIBRARY_NAME} PROPERTIES
	  VERSION ${VERSION}  # ${VERSION} was defined in the main CMakeLists.
	  FRAMEWORK FALSE
	  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
	)
endif()

project("riscv-sim")

# This is a make target, so you can do a "make riscv-sc"
set(APPLICATION_NAME riscv-sim)

add_executable(${APPLICATION_NAME} src/main.cpp)

# Links the target exe against the libraries
target_link_libraries(${APPLICATION_NAME} ${LIBRARY_NAME})
target_link_libraries(${APPLICATION_NAME} jsoncpp)
target_link_libraries(${APPLICATION_NAME} dbt-core)
target_link_libraries(${APPLICATION_NAME} external)
target_link_libraries(${APPLICATION_NAME} ${llvm_libs})
target_link_libraries(${APPLICATION_NAME} ${Boost_LIBRARIES} )
if (Tcmalloc_FOUND)
    target_link_libraries(${APPLICATION_NAME} ${Tcmalloc_LIBRARIES})
endif(Tcmalloc_FOUND)

# Says how and where to install software
# Targets:
#   * <prefix>/lib/<libraries>
#   * header location after install: <prefix>/include/<project>/*.h
#   * headers can be included by C++ code `#<project>/Bar.hpp>`
install(TARGETS ${LIBRARY_NAME} ${APPLICATION_NAME}
  EXPORT ${PROJECT_NAME}Targets            # for downstream dependencies
  ARCHIVE DESTINATION lib COMPONENT libs   # static lib
  RUNTIME DESTINATION bin COMPONENT libs   # binaries
  LIBRARY DESTINATION lib COMPONENT libs   # shared lib
  FRAMEWORK DESTINATION bin COMPONENT libs # for mac
  PUBLIC_HEADER DESTINATION incl/${PROJECT_NAME} COMPONENT devel   # headers for mac (note the different component -> different package)
  INCLUDES DESTINATION incl             # headers
)



#
# SYSTEM PACKAGING (RPM, TGZ, ...)
# _____________________________________________________________________________

#include(CPackConfig)

#
# CMAKE PACKAGING (for other CMake projects to use this one easily)
# _____________________________________________________________________________

#include(PackageConfigurator)