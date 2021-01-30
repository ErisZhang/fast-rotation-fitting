cmake_minimum_required(VERSION 3.1)
project(fast_rotation_fitting)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

### libIGL options: choose between header only and compiled static library
option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  add_compile_options(-mavx2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  message(FATAL_ERROR "Intel not supported")
elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
  add_compile_options(/arch:AVX2)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_definitions(${CMAKE_CXX_FLAGS} "-mfpmath=sse")
add_definitions(${CMAKE_CXX_FLAGS} "-march=native")
add_definitions(${CMAKE_CXX_FLAGS} "-g")
add_definitions(${CMAKE_CXX_FLAGS} "-Wall")
add_definitions(${CMAKE_CXX_FLAGS} "-w") # suppress compiler warnings


include_directories(src)
include_directories(eigen)
include_directories(src/Singular_Value_Decomposition)

# Add your project files
file(GLOB SRCFILES src/*.h src/*.cpp)

add_executable(benchmarking benchmarking.cpp ${SRCFILES})
target_link_libraries(benchmarking igl::core)

# add_executable(demo demo.cpp ${SRCFILES})
# target_link_libraries(demo igl::core igl::opengl igl::opengl_glfw)

# add_executable(test test.cpp ${SRCFILES})
# target_link_libraries(test igl::core igl::opengl igl::opengl_glfw)

# add_executable(main main.cpp ${SRCFILES})
# target_link_libraries(main igl::core igl::opengl igl::opengl_glfw)
