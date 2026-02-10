cmake_minimum_required(VERSION 3.24)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
#
###############################################################################
project(dbt-rise-riscv VERSION 2.1.0 LANGUAGES C CXX)

option(UPDATE_EXTERNAL_PROJECT "Whether to pull changes in external projects" ON)

include(GNUInstallDirs)
include(flink)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
if(NOT TARGET dbt-rise-core)
    FetchContent_Declare(
        dbt_rise_core_git
        GIT_REPOSITORY "https://github.com/Minres/DBT-RISE-Core.git"
        GIT_TAG 8f181e19e14e3fe5cfba340b47795c7b48b7844a
        GIT_SHALLOW OFF
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

if(CMAKE_PROJECT_NAME STREQUAL "dbt-rise-riscv")
    set(CLANG_FORMAT_EXCLUDE_PATTERNS /build/ /.direnv/ softfloat/)
    find_package(ClangFormat)
else()
    set(CLANG_FORMAT_EXCLUDE_PATTERNS ${CLANG_FORMAT_EXCLUDE_PATTERNS} /build/ /.direnv/ softfloat/ PARENT_SCOPE)
endif()

add_subdirectory(softvector)
set(LIB_SOURCES
    src/iss/plugin/instruction_count.cpp
    src/iss/arch/rv32i.cpp
    src/iss/arch/rv32imac.cpp
    src/iss/arch/rv32gc.cpp
    src/iss/arch/rv32gcv.cpp
    src/iss/arch/rv64i.cpp
    src/iss/arch/rv64gc.cpp
    src/iss/arch/rv64gcv.cpp
    src/iss/arch/tgc5c.cpp
    src/iss/mem/memory_if.cpp
    src/vm/interp/vm_tgc5c.cpp
    src/vm/interp/vm_rv32i.cpp
    src/vm/interp/vm_rv32imac.cpp
    src/vm/interp/vm_rv32gc.cpp
    src/vm/interp/vm_rv32gcv.cpp
    src/vm/interp/vm_rv64i.cpp
    src/vm/interp/vm_rv64gc.cpp
    src/vm/interp/vm_rv64gcv.cpp
    src/iss/debugger/csr_names.cpp
    src/iss/semihosting/semihosting.cpp
)
if(TARGET yaml-cpp::yaml-cpp)
    list(APPEND LIB_SOURCES
        src/iss/plugin/cycle_estimate.cpp
        src/iss/plugin/instruction_count.cpp
    )
endif()

if(WITH_LLVM)
    list(APPEND LIB_SOURCES
        src/vm/llvm/vm_tgc5c.cpp
        src/vm/llvm/vm_rv32i.cpp
        src/vm/llvm/vm_rv32imac.cpp
        src/vm/llvm/vm_rv32gc.cpp
        src/vm/llvm/vm_rv64i.cpp
        src/vm/llvm/vm_rv64gc.cpp
        src/vm/llvm/fp_impl.cpp
    )
endif()

if(WITH_ASMJIT)
    list(APPEND LIB_SOURCES
        src/vm/asmjit/vm_tgc5c.cpp
        src/vm/asmjit/vm_rv32i.cpp
        src/vm/asmjit/vm_rv32imac.cpp
        src/vm/asmjit/vm_rv32gc.cpp
        src/vm/asmjit/vm_rv32gcv.cpp
        src/vm/asmjit/vm_rv64i.cpp
        src/vm/asmjit/vm_rv64gc.cpp
        src/vm/asmjit/vm_rv64gcv.cpp
    )
endif()

if(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../dbt-rise-custom")
    # custom core files
    FILE(GLOB GEN_ISS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../dbt-rise-custom/src/iss/arch/*.cpp)
    FILE(GLOB GEN_VM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../dbt-rise-custom/src/vm/interp/vm_*.cpp)
    FILE(GLOB GEN_YAML_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../dbt-rise-custom/contrib/instr/*.yaml)
    list(APPEND LIB_SOURCES ${GEN_ISS_SOURCES} ${GEN_VM_SOURCES})

    foreach(FILEPATH ${GEN_ISS_SOURCES})
        get_filename_component(CORE ${FILEPATH} NAME_WE)
        string(TOUPPER ${CORE} CORE)
        list(APPEND LIB_DEFINES CORE_${CORE})
    endforeach()

    #message(STATUS "Core defines are ${LIB_DEFINES}")

    if(WITH_LLVM)
        FILE(GLOB LLVM_GEN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../dbt-rise-custom/src/vm/llvm/vm_*.cpp)
        list(APPEND LIB_SOURCES ${LLVM_GEN_SOURCES})
    endif()

    if(WITH_ASMJIT)
        FILE(GLOB TCC_GEN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../dbt-rise-custom/src/vm/asmjit/vm_*.cpp)
        list(APPEND LIB_SOURCES ${TCC_GEN_SOURCES})
    endif()
endif()
# Define the library
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-shift-count-overflow)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /wd4293)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC src)
if(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../dbt-rise-custom")
    target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/../dbt-rise-custom/src)
endif()

target_force_link_libraries(${PROJECT_NAME} PRIVATE dbt-rise-core)

# only re-export the include paths
get_target_property(DBT_CORE_INCL dbt-rise-core INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${PROJECT_NAME} INTERFACE ${DBT_CORE_INCL})
get_target_property(DBT_CORE_DEFS dbt-rise-core INTERFACE_COMPILE_DEFINITIONS)

if(NOT (DBT_CORE_DEFS STREQUAL DBT_CORE_DEFS-NOTFOUND))
    target_compile_definitions(${PROJECT_NAME} INTERFACE ${DBT_CORE_DEFS})
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC elfio::elfio softfloat softvector scc-util Boost::coroutine)

if(TARGET yaml-cpp::yaml-cpp)
    target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_PLUGINS)
    target_link_libraries(${PROJECT_NAME} PUBLIC yaml-cpp::yaml-cpp)
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

if(Tcmalloc_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${Tcmalloc_LIBRARIES})
endif(Tcmalloc_FOUND)

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
    set(USE_SC_SIGNAL4IRQ OFF CACHE BOOL "Enable the use of ssc_signals for interrupt delivery")

    set(LIB_SOURCES
        src/sysc/core_complex.cpp
        src/sysc/register_cores.cpp
        src/sysc/register_rv32gcv.cpp
        src/sysc/register_rv64gcv.cpp
    )

    if(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../dbt-rise-custom")
        FILE(GLOB GEN_SC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../dbt-rise-custom/src/sysc/register_*.cpp)
        list(APPEND LIB_SOURCES ${GEN_SC_SOURCES})
    endif()
    set(LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/sysc/core_complex.h)

    add_library(${PROJECT_NAME}_sig ${LIB_SOURCES})
    target_compile_definitions(${PROJECT_NAME}_sig PUBLIC SC_SIGNAL_IF)
    target_compile_definitions(${PROJECT_NAME}_sig PUBLIC WITH_SYSTEMC SC_SIGNAL_IF)
    target_link_libraries(${PROJECT_NAME}_sig PUBLIC dbt-rise-riscv scc-sysc)

    # if(WITH_LLVM)
    # target_link_libraries(${PROJECT_NAME} PUBLIC ${llvm_libs})
    # endif()
    set_target_properties(${PROJECT_NAME}_sig PROPERTIES
        VERSION ${PROJECT_VERSION}
        FRAMEWORK FALSE
        PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
    )
    install(TARGETS ${PROJECT_NAME}_sig COMPONENT ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets # for downstream dependencies
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # static lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # binaries
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # shared lib
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} # for mac
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sysc # headers for mac (note the different component -> different package)
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # headers
    )

    add_library(${PROJECT_NAME}_tlm ${LIB_SOURCES})
    target_compile_definitions(${PROJECT_NAME}_tlm PUBLIC WITH_SYSTEMC)
    target_link_libraries(${PROJECT_NAME}_tlm PUBLIC dbt-rise-riscv scc-sysc)

    # if(WITH_LLVM)
    # target_link_libraries(${PROJECT_NAME} PUBLIC ${llvm_libs})
    # endif()
    set_target_properties(${PROJECT_NAME}_tlm PROPERTIES
        VERSION ${PROJECT_VERSION}
        FRAMEWORK FALSE
        PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
    )
    install(TARGETS ${PROJECT_NAME}_tlm COMPONENT ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets # for downstream dependencies
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # static lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # binaries
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # shared lib
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} # for mac
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sysc # headers for mac (note the different component -> different package)
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # headers
    )
    if(USE_SC_SIGNAL4IRQ)
        add_library(${PROJECT_NAME} ALIAS ${PROJECT_NAME}_sig)
    else()
        add_library(${PROJECT_NAME} ALIAS ${PROJECT_NAME}_tlm)
    endif()
endif()

project(elfio-test)
find_package(Boost COMPONENTS program_options thread REQUIRED)

add_executable(${PROJECT_NAME} src/elfio.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC elfio::elfio)
