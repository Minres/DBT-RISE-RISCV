cmake_minimum_required(VERSION 3.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

# Set the name of your project here
project("external")

include(Common)


include_directories( ${PROJECT_SOURCE_DIR}/libGIS )

FILE(GLOB ElfioHeaders elfio *.hpp)
FILE(GLOB GISHeaders libGis *.h)

set(LIB_HEADERS ${ElfioHeaders} ${GISHeaders})
set(LIB_SOURCES
    libGIS/atmel_generic.c
    libGIS/ihex.c
    libGIS/srecord.c
)

# Define two variables in order not to repeat ourselves.
set(LIBRARY_NAME external)

# Define the library
add_library(${LIBRARY_NAME} ${LIB_SOURCES})


