cmake_minimum_required(VERSION 3.14)
project(sleef-highway)

# Just build inline headers for sleef
set(sleef_SOURCE_DIR ${CMAKE_SOURCE_DIR}/sleef)
set(sleef_BINARY_DIR ${CMAKE_BINARY_DIR}/sleef)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs")
set(BUILD_LIBM OFF CACHE BOOL "libsleef will be built.")
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "libsleefgnuabi will be built.")
set(BUILD_TESTS OFF CACHE BOOL "Tests will be built.")
set(BUILD_INLINE_HEADERS ON CACHE BOOL "Build header for inlining whole SLEEF functions")
add_subdirectory("sleef")

# Cut out as much of the highway build as possible
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Include contrib/")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Build examples")
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "Install library")
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Enable HWY tests")
add_subdirectory("highway")

# Build just the thread_pool from hwy/congrib
set(HWY_CONTRIB_SOURCES
    highway/hwy/contrib/thread_pool/thread_pool.cc
    highway/hwy/contrib/thread_pool/thread_pool.h
)
add_library(hwy_contrib ${HWY_LIBRARY_TYPE} ${HWY_CONTRIB_SOURCES})
target_link_libraries(hwy_contrib PUBLIC hwy)
target_compile_options(hwy_contrib PRIVATE ${HWY_FLAGS})
set_property(TARGET hwy_contrib PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(hwy_contrib PROPERTIES VERSION ${LIBRARY_VERSION} SOVERSION ${LIBRARY_SOVERSION})
target_include_directories(hwy_contrib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(hwy_contrib PUBLIC cxx_std_11)
set_target_properties(hwy_contrib PROPERTIES
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hwy/hwy.version)
# For GCC __atomic_store_8, see #887
target_link_libraries(hwy_contrib PRIVATE ${ATOMICS_LIBRARIES})

add_subdirectory("src/gen-bindings")
add_subdirectory("src/tests")

