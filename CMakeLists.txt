cmake_minimum_required(VERSION 3.14)
project(sleef-highway)
enable_testing()

# Just build inline headers for sleef
set(sleef_SOURCE_DIR ${CMAKE_SOURCE_DIR}/sleef)
set(sleef_BINARY_DIR ${CMAKE_BINARY_DIR}/sleef)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs")
set(BUILD_LIBM OFF CACHE BOOL "libsleef will be built.")
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "libsleefgnuabi will be built.")
set(BUILD_TESTS OFF CACHE BOOL "Tests will be built.")
set(ENABLE_LTO ON CACHE BOOL "Enable LTO on GCC or ThinLTO on clang")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs")
set(BUILD_INLINE_HEADERS ON CACHE BOOL "Build header for inlining whole SLEEF functions")
add_subdirectory("sleef")
target_include_directories(sleef INTERFACE "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/sleef/include>")

# Cut out as much of the highway build as possible
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Include contrib/")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Build examples")
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "Install library")
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Enable HWY tests")
add_subdirectory("highway")

add_subdirectory("src/gen-bindings")
add_subdirectory("src/tests")

