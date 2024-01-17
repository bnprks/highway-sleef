
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

add_executable(all_floats all_floats.cc ${CMAKE_SOURCE_DIR}/src/gen-bindings/PayneHanekReductionTable.cc)
add_dependencies(all_floats hwy_sleef_generated)
set_target_properties(all_floats PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
target_link_libraries(all_floats PUBLIC hwy hwy_contrib profiler sleef)
target_include_directories(all_floats PUBLIC "${CMAKE_BINARY_DIR}")
target_compile_options(all_floats PUBLIC -ffp-contract=off -Wno-ignored-attributes -march=haswell -maes -O2 -g)
