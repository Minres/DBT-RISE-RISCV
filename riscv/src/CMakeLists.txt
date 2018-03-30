# library files
FILE(GLOB RiscVHeaders *.h)
set(LIB_HEADERS ${RiscVHeaders} )
set(LIB_SOURCES
    iss/rv32imac.cpp
    iss/rv64ia.cpp
    internal/vm_rv32imac.cpp
    internal/vm_rv64ia.cpp
    plugin/instruction_count.cpp
    plugin/cycle_estimate.cpp
)

set(APP_HEADERS )

set(APP_SOURCES main.cpp)

# Define two variables in order not to repeat ourselves.
set(LIBRARY_NAME risc-v)

# Define the library
#add_library(${LIBRARY_NAME} SHARED ${LIB_SOURCES})
add_library(${LIBRARY_NAME} ${LIB_SOURCES})
SET(${LIBRARY_NAME} -Wl,-whole-archive -l${LIBRARY_NAME} -Wl,-no-whole-archive)
set_target_properties(${LIBRARY_NAME} PROPERTIES
  VERSION ${VERSION}  # ${VERSION} was defined in the main CMakeLists.
  FRAMEWORK FALSE
  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
)

# This is a make target, so you can do a "make riscv-sc"
set(APPLICATION_NAME riscv)

add_executable(${APPLICATION_NAME} ${APP_SOURCES})

# Links the target exe against the libraries
target_link_libraries(${APPLICATION_NAME} ${LIBRARY_NAME})
target_link_libraries(${APPLICATION_NAME} dbt-core)
target_link_libraries(${APPLICATION_NAME} sc-components)
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

