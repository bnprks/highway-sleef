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
find_package (Python3 COMPONENTS Interpreter Development)

## install py-treesitter
execute_process(COMMAND "${Python3_EXECUTABLE}" -m pip install tree_sitter)


add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tree-sitter-grammars.so"
    COMMAND ${Python3_EXECUTABLE} ARGS ${CMAKE_CURRENT_SOURCE_DIR}/treesitter_utils.py ${CMAKE_SOURCE_DIR}/tree-sitter "tree-sitter-grammars.so"
)

add_custom_target(hwy_sleef_generated
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/translate.py"
        "${sleef_SOURCE_DIR}/src" 
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data"
        "${CMAKE_CURRENT_BINARY_DIR}/tree-sitter-grammars.so" 
        sleef-generated.h
    DEPENDS 
        "${CMAKE_CURRENT_SOURCE_DIR}/translate.py"
        "${CMAKE_CURRENT_BINARY_DIR}/tree-sitter-grammars.so" 
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/function_renames.tsv"
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/simd_ops.tsv"
        "${CMAKE_CURRENT_SOURCE_DIR}/rename_data/types.tsv"
    BYPRODUCTS "sleef-generated.h"
)