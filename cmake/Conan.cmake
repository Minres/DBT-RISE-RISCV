macro(setup_conan)
  find_program(conan conan)
  if(NOT EXISTS ${conan})
    message(FATAL_ERROR "Conan is required. Please see README.md")
    return()
  endif()

  if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Darwin)
    set(os Macos)
  else()
    set(os ${CMAKE_HOST_SYSTEM_NAME})
  endif()

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(compiler gcc)
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL AppleClang)
    set(compiler apple-clang)
  else()
    message(FATAL_ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  string(SUBSTRING ${CMAKE_CXX_COMPILER_VERSION} 0 3 compiler_version)

  set(conanfile ${CMAKE_SOURCE_DIR}/conanfile.txt)
  set(conanfile_cmake ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)

  if(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
  	execute_process(COMMAND ${conan} install --build=missing
                   -s build_type=Release
                   ${CMAKE_SOURCE_DIR} RESULT_VARIABLE return_code)
  else()
   	execute_process(COMMAND ${conan} install --build=missing
                   -s build_type=${CMAKE_BUILD_TYPE}
                   ${CMAKE_SOURCE_DIR} RESULT_VARIABLE return_code)
  endif()
  if(NOT ${return_code} EQUAL 0)
    message(FATAL_ERROR "conan install command failed.")
  endif()

  include(${conanfile_cmake})
  conan_basic_setup(TARGETS)   
endmacro()