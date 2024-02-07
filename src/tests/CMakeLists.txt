
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library.")
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/e61e332df951b947e858011449a32c3ed5049db7.zip
)
FetchContent_MakeAvailable(benchmark)

cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

list(APPEND TEST_TARGETS measure_ulp identical_to_sleef measure_speed sampling_test mismatch_test)

foreach(TARGET IN LISTS TEST_TARGETS)
  add_executable(${TARGET} ${TARGET}.cc) # ${CMAKE_SOURCE_DIR}/src/gen-bindings/PayneHanekReductionTable.cc)
  add_dependencies(${TARGET} highway_sleef_generated)
  set_target_properties(${TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  target_link_libraries(${TARGET} PUBLIC hwy profiler sleef)
  target_include_directories(${TARGET} PUBLIC "${CMAKE_BINARY_DIR}" "${CMAKE_SOURCE_DIR}")
  target_compile_options(${TARGET} PUBLIC -ffp-contract=off -Wno-ignored-attributes -march=haswell -maes -O2 -g)
endforeach()

target_compile_options(mismatch_test PRIVATE -DHWY_IS_TEST=1)

target_link_libraries(measure_speed PUBLIC benchmark::benchmark)

target_link_libraries(sampling_test PUBLIC hwy_test gtest_main)
target_compile_options(sampling_test PRIVATE -DHWY_IS_TEST=1)

include(GoogleTest)
gtest_discover_tests(sampling_test DISCOVERY_TIMEOUT 60)