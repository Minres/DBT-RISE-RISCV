# library files
FILE(GLOB RiscVSCHeaders ${PROJECT_SOURCE_DIR}/incl/sysc/*.h ${PROJECT_SOURCE_DIR}/incl/sysc/*/*.h)
set(LIB_HEADERS ${RiscVSCHeaders} )
set(LIB_SOURCES 
	sysc/aon.cpp
	sysc/BLDC.cpp
	sysc/clint.cpp
	sysc/dcmotor.cpp
	sysc/gpio.cpp
	sysc/h_bridge.cpp
	sysc/hifive1.cpp
	sysc/fe310.cpp
	sysc/mcp_adc.cpp
	sysc/plic.cpp
	sysc/prci.cpp
	sysc/pwm.cpp
	sysc/sc_comm_singleton.cpp
	sysc/spi.cpp
	sysc/system.cpp
	sysc/terminal.cpp
	sysc/uart.cpp	
	CLIParser.cpp )

set(APP_SOURCES sc_main.cpp)

# Define two variables in order not to repeat ourselves.
set(LIBRARY_NAME platform)

## the following setting needs to be consistent with the library
#add_definitions(-DSC_DEFAULT_WRITER_POLICY=SC_MANY_WRITERS)

# Define the library
add_library(${LIBRARY_NAME} SHARED ${LIB_SOURCES})

# Links the target exe against the libraries
target_link_libraries(${LIBRARY_NAME} riscv_sc)
target_link_libraries(${LIBRARY_NAME} dbt-core)
target_link_libraries(${LIBRARY_NAME} softfloat)
target_link_libraries(${LIBRARY_NAME} sc-components)
target_link_libraries(${LIBRARY_NAME} ${CONAN_LIBS_SEASOCKS})
target_link_libraries(${LIBRARY_NAME} external)
target_link_libraries(${LIBRARY_NAME} ${llvm_libs})
target_link_libraries(${LIBRARY_NAME} ${Boost_LIBRARIES} )

set_target_properties(${LIBRARY_NAME} PROPERTIES
  VERSION ${VERSION}  # ${VERSION} was defined in the main CMakeLists.
  FRAMEWORK FALSE
  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
)

# This is a make target, so you can do a "make riscv-sc"
set(APPLICATION_NAME riscv.vp)

include_directories(${PROJECT_SOURCE_DIR}/incl)
include_directories(${CONAN_INCLUDE_DIRS_SEASOCKS})
add_definitions(-DWITH_SYSTEMC) # or -DSC_NO_WRITE_CHECK
include_directories(${SystemC_INCLUDE_DIRS})
include_directories(${CCI_INCLUDE_DIRS})
if(SCV_FOUND)
    add_definitions(-DWITH_SCV)
    include_directories(${SCV_INCLUDE_DIRS})
endif()


link_directories(${SystemC_LIBRARY_DIRS})
link_directories(${CCI_LIBRARY_DIRS})
link_directories(${CONAN_LIB_DIRS_SEASOCKS})

add_executable(${APPLICATION_NAME} ${APP_SOURCES})
# include files for this application
target_include_directories(${APPLICATION_NAME} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
# Links the target exe against the libraries
target_link_libraries(${APPLICATION_NAME} ${LIBRARY_NAME})
target_link_libraries(${APPLICATION_NAME} riscv_sc)
target_link_libraries(${APPLICATION_NAME} dbt-core)
target_link_libraries(${APPLICATION_NAME} softfloat)
target_link_libraries(${APPLICATION_NAME} sc-components)
target_link_libraries(${APPLICATION_NAME} ${CONAN_LIBS_SEASOCKS})
target_link_libraries(${APPLICATION_NAME} external)
target_link_libraries(${APPLICATION_NAME} ${llvm_libs})
target_link_libraries(${APPLICATION_NAME} ${CCI_LIBRARIES} )
target_link_libraries(${APPLICATION_NAME} ${SystemC_LIBRARIES} )
if(SCV_FOUND)
    link_directories(${SCV_LIBRARY_DIRS})
    target_link_libraries (${APPLICATION_NAME} ${SCV_LIBRARIES})
endif()
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

