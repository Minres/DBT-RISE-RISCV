cmake_minimum_required(VERSION 3.12)

project("riscv" VERSION 1.0.0)

include(GNUInstallDirs)

conan_basic_setup()

find_package(Boost COMPONENTS program_options system thread filesystem REQUIRED)
if(WITH_LLVM)
	if(DEFINED ENV{LLVM_HOME})
		find_path (LLVM_DIR LLVM-Config.cmake $ENV{LLVM_HOME}/lib/cmake/llvm)
	endif(DEFINED ENV{LLVM_HOME})
	find_package(LLVM REQUIRED CONFIG)
	message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
	message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
	llvm_map_components_to_libnames(llvm_libs support core mcjit x86codegen x86asmparser)
endif()

#Mac needed variables (adapt for your needs - http://www.cmake.org/Wiki/CMake_RPATH_handling#Mac_OS_X_and_the_RPATH)
#set(CMAKE_MACOSX_RPATH ON)
#set(CMAKE_SKIP_BUILD_RPATH FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(softfloat)

# library files
FILE(GLOB RiscVSCHeaders ${CMAKE_CURRENT_SOURCE_DIR}/incl/sysc/*.h ${CMAKE_CURRENT_SOURCE_DIR}/incl/sysc/*/*.h)
set(LIB_HEADERS ${RiscVSCHeaders} )
set(LIB_SOURCES 
	src/iss/rv32gc.cpp
	src/iss/rv32imac.cpp
	src/iss/rv64i.cpp
	src/iss/rv64gc.cpp
	src/iss/mnrv32.cpp
	src/vm/fp_functions.cpp
	src/vm/tcc/vm_mnrv32.cpp
	src/vm/tcc/vm_rv32gc.cpp
	src/vm/tcc/vm_rv32imac.cpp
	src/vm/tcc/vm_rv64i.cpp
	src/vm/tcc/vm_rv64gc.cpp
	src/vm/interp/vm_mnrv32.cpp
	src/vm/interp/vm_rv32gc.cpp
	src/vm/interp/vm_rv32imac.cpp
	src/vm/interp/vm_rv64i.cpp
	src/vm/interp/vm_rv64gc.cpp
    src/plugin/instruction_count.cpp
    src/plugin/cycle_estimate.cpp
)
if(WITH_LLVM)
set(LIB_SOURCES ${LIB_SOURCES}
	src/vm/llvm/fp_impl.cpp
	src/vm/llvm/vm_mnrv32.cpp
	src/vm/llvm/vm_rv32gc.cpp
	src/vm/llvm/vm_rv32imac.cpp
	src/vm/llvm/vm_rv64i.cpp
	src/vm/llvm/vm_rv64gc.cpp
)
endif()

# Define the library
add_library(riscv SHARED ${LIB_SOURCES})
target_compile_options(riscv PRIVATE -Wno-shift-count-overflow)
target_include_directories(riscv PUBLIC incl ../external/elfio)
target_link_libraries(riscv PUBLIC softfloat scc-util jsoncpp)
target_link_libraries(riscv PUBLIC -Wl,--whole-archive dbt-core -Wl,--no-whole-archive)
set_target_properties(riscv PROPERTIES
  VERSION ${PROJECT_VERSION}
  FRAMEWORK FALSE
  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
)

if(SystemC_FOUND)
	add_library(riscv_sc src/sysc/core_complex.cpp)
	target_compile_definitions(riscv_sc PUBLIC WITH_SYSTEMC) 
	target_include_directories(riscv_sc PUBLIC ../incl ${SystemC_INCLUDE_DIRS} ${CCI_INCLUDE_DIRS})
	
	if(SCV_FOUND)   
	    target_compile_definitions(riscv_sc PUBLIC WITH_SCV)
	    target_include_directories(riscv_sc PUBLIC ${SCV_INCLUDE_DIRS})
	endif()
	target_link_libraries(riscv_sc PUBLIC riscv scc )
	if(WITH_LLVM)
		target_link_libraries(riscv_sc PUBLIC ${llvm_libs})
	endif()
	target_link_libraries(riscv_sc PUBLIC ${Boost_LIBRARIES} )
	set_target_properties(riscv_sc PROPERTIES
	  VERSION ${PROJECT_VERSION}
	  FRAMEWORK FALSE
	  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
	)
endif()

project("riscv-sim")
add_executable(riscv-sim src/main.cpp)
# This sets the include directory for the reference project. This is the -I flag in gcc.
target_include_directories(riscv-sim PRIVATE ../external/libGIS)
if(WITH_LLVM)
	target_compile_definitions(riscv-sim PRIVATE WITH_LLVM)
	target_link_libraries(riscv-sim PUBLIC ${llvm_libs})
endif()
# Links the target exe against the libraries
target_link_libraries(riscv-sim riscv)
target_link_libraries(riscv-sim jsoncpp)
target_link_libraries(riscv-sim external)
target_link_libraries(riscv-sim ${Boost_LIBRARIES} )
if (Tcmalloc_FOUND)
    target_link_libraries(riscv-sim ${Tcmalloc_LIBRARIES})
endif(Tcmalloc_FOUND)

install(TARGETS riscv riscv-sim
  EXPORT ${PROJECT_NAME}Targets            # for downstream dependencies
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libs   # static lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT libs   # binaries
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libs   # shared lib
  FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libs # for mac
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} COMPONENT devel   # headers for mac (note the different component -> different package)
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}             # headers
)



#
# SYSTEM PACKAGING (RPM, TGZ, ...)
# _____________________________________________________________________________

#include(CPackConfig)

#
# CMAKE PACKAGING (for other CMake projects to use this one easily)
# _____________________________________________________________________________

#include(PackageConfigurator)