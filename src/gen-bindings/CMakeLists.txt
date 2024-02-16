# Setup python3 venv with tree-sitter library
# Taken from https://discourse.cmake.org/t/possible-to-create-a-python-virtual-env-from-cmake-and-then-find-it-with-findpython3/1132
find_package (Python3 COMPONENTS Interpreter)
execute_process (COMMAND "${Python3_EXECUTABLE}" -m venv "${CMAKE_CURRENT_BINARY_DIR}/venv")

## update the environment with VIRTUAL_ENV variable (mimic the activate script)
set (ENV{VIRTUAL_ENV} "${CMAKE_CURRENT_BINARY_DIR}/venv")
## change the context of the search
set (Python3_FIND_VIRTUALENV FIRST)
## unset Python3_EXECUTABLE because it is also an input variable (see documentation, Artifacts Specification section)
unset (Python3_EXECUTABLE)
## Launch a new search
find_package (Python3 COMPONENTS Interpreter)

## install py-treesitter
execute_process(COMMAND "${Python3_EXECUTABLE}" -m pip install tree_sitter tree_sitter_c tree_sitter_cpp)


add_custom_command(OUTPUT "sleef-generated.h"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/translate.py"
        "${sleef_SOURCE_DIR}/src" 
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data"
        sleef-generated.h
    COMMAND cp sleef-generated.h ${CMAKE_SOURCE_DIR}/outputs
    DEPENDS 
        "${CMAKE_CURRENT_SOURCE_DIR}/translate.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/constant_renames.tsv"
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/function_renames.tsv"
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/macro_conditionals.tsv"
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/simd_ops.tsv"
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/types.tsv"
)

add_custom_target(highway_sleef_generated DEPENDS "sleef-generated.h")

add_library(highway_sleef PayneHanekReductionTable.cc)
target_include_directories(highway_sleef INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(highway_sleef INTERFACE hwy)
add_dependencies(highway_sleef highway_sleef_generated)