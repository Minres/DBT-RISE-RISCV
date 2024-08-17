cmake_minimum_required(VERSION 3.18)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
#
###############################################################################
project(dbt-rise-riscv VERSION 2.0.0 LANGUAGES C CXX)

option(UPDATE_EXTERNAL_PROJECT "Whether to pull changes in external projects" ON)

include(GNUInstallDirs)
include(flink)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
if(NOT TARGET dbt-rise-core)
	FetchContent_Declare(
	    dbt_rise_core_git
	    GIT_REPOSITORY  "https://github.com/Minres/DBT-RISE-Core.git"
	    #GIT_TAG         "origin/develop"
	    #GIT_SHALLOW    OFF
	    GIT_TAG         45a9641
	    UPDATE_DISCONNECTED NOT ${UPDATE_EXTERNAL_PROJECT} # When enabled, this option causes the update step to be skipped.
	)
	FetchContent_GetProperties(dbt_rise_core_git)
	if(NOT dbt_rise_core_git_POPULATED)
	  FetchContent_Populate(dbt_rise_core_git)
	endif()
	add_subdirectory(${dbt_rise_core_git_SOURCE_DIR} ${dbt_rise_core_git_BINARY_DIR})
endif()
FetchContent_GetProperties(scc_git)
list(APPEND CMAKE_MODULE_PATH ${scc_git_SOURCE_DIR}/cmake)

find_package(elfio QUIET)
find_package(jsoncpp)
find_package(yaml-cpp)
find_package(Boost COMPONENTS coroutine REQUIRED)

set(CLANG_FORMAT_EXCLUDE_PATTERNS /build/ /.direnv/ softfloat/)
find_package(ClangFormat)

add_subdirectory(softfloat)

set(LIB_SOURCES
    src/iss/plugin/instruction_count.cpp
	src/iss/arch/rv32i.cpp
    src/iss/arch/rv32imc.cpp
    src/iss/arch/rv32imac.cpp
    src/iss/arch/rv32gc.cpp
	src/iss/arch/rv64i.cpp
    src/iss/arch/rv64gc.cpp
    src/vm/interp/vm_rv32i.cpp
    src/vm/interp/vm_rv32imac.cpp
    src/vm/interp/vm_rv32gc.cpp
    src/vm/interp/vm_rv64i.cpp
	src/vm/interp/vm_rv64gc.cpp
    src/vm/fp_functions.cpp
    src/iss/debugger/csr_names.cpp
    src/iss/semihosting/semihosting.cpp
    src/iss/plugin/cycle_estimate.cpp
    src/iss/plugin/instruction_count.cpp
)

if(WITH_LLVM)
    list(APPEND LIB_SOURCES
        src/vm/llvm/vm_rv32imc.cpp
        src/vm/llvm/fp_impl.cpp
    )
endif()

if(WITH_ASMJIT)
    list(APPEND LIB_SOURCES
	src/vm/asmjit/vm_rv32i.cpp
	src/vm/asmjit/vm_rv64i.cpp
    )
endif()

# Define the library
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-shift-count-overflow)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /wd4293)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC src)

target_force_link_libraries(${PROJECT_NAME} PRIVATE dbt-rise-core)

# only re-export the include paths
get_target_property(DBT_CORE_INCL dbt-rise-core INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${PROJECT_NAME} INTERFACE ${DBT_CORE_INCL})
get_target_property(DBT_CORE_DEFS dbt-rise-core INTERFACE_COMPILE_DEFINITIONS)

if(NOT(DBT_CORE_DEFS STREQUAL DBT_CORE_DEFS-NOTFOUND))
    target_compile_definitions(${PROJECT_NAME} INTERFACE ${DBT_CORE_DEFS})
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC elfio::elfio softfloat scc-util  yaml-cpp::yaml-cpp Boost::coroutine)

if(WITH_LLVM)
    find_package(LLVM)
    target_compile_definitions(${PROJECT_NAME} PUBLIC ${LLVM_DEFINITIONS})
    target_include_directories(${PROJECT_NAME} PUBLIC ${LLVM_INCLUDE_DIRS})

    if(BUILD_SHARED_LIBS)
        target_link_libraries(${PROJECT_NAME} PUBLIC ${LLVM_LIBRARIES})
    endif()
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    FRAMEWORK FALSE
)
install(TARGETS ${PROJECT_NAME} COMPONENT ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets # for downstream dependencies
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # static lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # binaries
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # shared lib
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} # for mac
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} # headers for mac (note the different component -> different package)
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # headers
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/incl/iss COMPONENT ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # target directory
    FILES_MATCHING # install only matched files
    PATTERN "*.h" # select header files
)

file(GLOB GEN_YAML_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/contrib/instr/*.yaml)
install(FILES ${GEN_YAML_SOURCES} DESTINATION share/riscv-sim)

###############################################################################
#
###############################################################################
set(CMAKE_INSTALL_RPATH $ORIGIN/../${CMAKE_INSTALL_LIBDIR})
project(riscv-sim)
find_package(Boost COMPONENTS program_options thread REQUIRED)
find_package(fmt)

add_executable(${PROJECT_NAME} src/main.cpp)

foreach(F IN LISTS RISCV_SOURCES)
    if(${F} MATCHES ".*/arch/([^/]*)\.cpp")
        string(REGEX REPLACE ".*/([^/]*)\.cpp" "\\1" CORE_NAME_LC ${F})
        string(TOUPPER ${CORE_NAME_LC} CORE_NAME)
        target_compile_definitions(${PROJECT_NAME} PRIVATE CORE_${CORE_NAME})
    endif()
endforeach()

target_link_libraries(${PROJECT_NAME} PUBLIC dbt-rise-riscv yaml-cpp::yaml-cpp Boost::program_options)
if(TARGET fmt::fmt-header-only)
	target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt-header-only)
else()
	target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)
endif()
target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})

install(TARGETS riscv-sim
    EXPORT ${PROJECT_NAME}Targets # for downstream dependencies
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # static lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # binaries
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # shared lib
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} # for mac
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} # headers for mac (note the different component -> different package)
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # headers
)

if(BUILD_TESTING)
    # ... CMake code to create tests ...
    add_test(NAME riscv-sim-interp
        COMMAND riscv-sim -f ${CMAKE_BINARY_DIR}/../../Firmwares/hello-world/hello --backend interp)

    if(WITH_TCC)
        add_test(NAME riscv-sim-tcc
            COMMAND riscv-sim -f ${CMAKE_BINARY_DIR}/../../Firmwares/hello-world/hello --backend tcc)
    endif()

    if(WITH_LLVM)
        add_test(NAME riscv-sim-llvm
            COMMAND riscv-sim -f ${CMAKE_BINARY_DIR}/../../Firmwares/hello-world/hello --backend llvm)
    endif()

    if(WITH_ASMJIT)
        add_test(NAME riscv-sim-asmjit
            COMMAND riscv-sim -f ${CMAKE_BINARY_DIR}/../../Firmwares/hello-world/hello --backend asmjit)
    endif()
endif()

###############################################################################
#
###############################################################################
if(TARGET scc-sysc)
    project(dbt-rise-riscv_sc VERSION 1.0.0)
    set(LIB_SOURCES
        src/sysc/core_complex.cpp
        src/sysc/register_cores.cpp
    )
    FILE(GLOB GEN_SC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src-gen/sysc/register_*.cpp)
    list(APPEND LIB_SOURCES ${GEN_SC_SOURCES})
    add_library(${PROJECT_NAME} ${LIB_SOURCES})
    target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_SYSTEMC)
    target_link_libraries(${PROJECT_NAME} PUBLIC dbt-rise-riscv scc-sysc)

    # if(WITH_LLVM)
    # target_link_libraries(${PROJECT_NAME} PUBLIC ${llvm_libs})
    # endif()
    set(LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/sysc/core_complex.h)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        FRAMEWORK FALSE
        PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
    )
    install(TARGETS ${PROJECT_NAME} COMPONENT ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets # for downstream dependencies
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # static lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # binaries
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # shared lib
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} # for mac
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sysc # headers for mac (note the different component -> different package)
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # headers
    )
endif()
