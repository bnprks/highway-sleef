# Correctness and speed tests

The correctness tests cover all 32-bit float inputs, providing rock-solid precision guarantees at the cost of extra test
runtime. These tests are also hard-coded to use AVX2 for initial development convenience.

- `identical_to_sleef.cc`: 
    - Tests all functions for bitwise-identical outputs as SLEEF.
    - Allows for mismatches in NaN outputs which seem to be due to some compiler optimizations. 
    - Takes ~6 minutes to run on 4 cores of an i5-1240P
- `measure_ulp.cc`: 
    - Measures maximum ULP for functions using double-precision math to get reference values and perform ULP
 calculations. 
    - Reports whether functions output NaNs when they're supposed to, maximum ULP and worst input/outputs. 
    - Splits out precision results for all floats as well as floats within a specified valid input range. 
    - Takes ~15 minutes to run on 4 cores of an i5-1240P
- `measure_speed.cc`: 
    - Measures execution speed using [Google benchmark](https://github.com/google/benchmark)
    - Takes ~1.5 minutes to run on 1 core of an i5-1240P