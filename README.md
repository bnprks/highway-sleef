# Highway-Sleef

[Sleef](https://sleef.org/) is one of the best vectorized math libraries, created by Professor Naoki Shibata (Nara Institue of Science and Technology). In my opinion, it remains under-utilized due to difficulties in integrating it in downstream projects.

On the other hand, [Highway](https://github.com/google/highway) is a best-in-class portable SIMD library in C++, offering some of the easiest runtime feature detection and dispatching of any SIMD library.

This project aims to expand and improve the math functionality of Highway by performing semi-automated translation of the
excellent math implementations from Sleef.



## Building

Initialize git submodules:
```
git submodule update --init
```

To get LTO for sleef to work, I've configured with cmake as follows:
```
CC=clang CXX=clang++ cmake -B build -S . -G Ninja -DLLVM_AR_COMMAND=llvm-ar-14
```
Replace `llvm-ar-14` with the appropriate version number present on your machine

To run performance or correctness tests:
```
cmake --build build
./build/src/tests/identical_to_sleef
./build/src/tests/measure_ulp
./build/src/tests/measure_speed
```

Example results of these tests are included in `outputs`, along with a copy of the generated result `outputs/sleef-generated.h`

## Results

### Current status:
- All of highway's single-precision functions with 1 argument have translated equivalents from sleef
- All results from the translation are bitwise identical to original SLEEF, with the exception of some NaN outputs (I've noticed some sign-mismatches on NaN's which seem to be dependent on optimization level)
  - This is only with testing on AVX2, so is not guaranteed to hold in other instruction sets
- See [src/gen-bindings/README.md](src/gen-bindings/README.md) for a description of the translation code

### Next steps:
- Implement the rest of the SLEEF operations including double-precision
    - See the [SLEEF documentation](https://sleef.org/purec.xhtml) for a full list of these operations. Highlights include trig functions with arguments automatically multiplied by pi, along with error and gamma function
- Develop testing methodology for functions that cannot be exhaustively tested for all inputs
- Test accuracy and performance on more platforms and Hwy configurations
- Try to merge into upstream hwy library.

### Accuracy
Notes: 

- hwy sometimes produces a NaN when the correct result is not NaN. These inputs have been omitted for comparison. 
- Inputs that cause an Inf when the correct result is not Inf cause Inf to be marked as ULP.
- For many trig functions, SLEEF provides a high and low precision variant
- In a few cases, SLEEF has a more restrictive valid range than hwy, which are marked accordingly


#### Precision in valid range

| Op    | valid range                                                  | Hwy   | Sleef 1 ULP | Sleef 3.5 ULP |
| ----- | ------------------------------------------------------------ | ----- | ----------- | ------------- |
| Exp   | [-FLT_MAX, 104]                                              | 1.03  | 0.988       | N/A           |
| Expm1 | [-FLT_MAX, 104]                                              | 3.55  | 1           | N/A           |
| Log   | (0, FLT_MAX]                                                 | 0.995 | 0.995       | N/A           |
| Log1p | (0, FLT_MAX] for hwy<br />(0, 1e38] for sleef                | 2.47  | 1           | N/A           |
| Log2  | (0, FLT_MAX]                                                 | 1.87  | 0.992       | N/A           |
| Sin   | [-39000, +39000]                                             | 2.47  | 0.904       | 2.47          |
| Cos   | [-39000, +39000]                                             | 2.42  | 0.936       | 2.49          |
| Tan   | [-39000, 39000]                                              | N/A   | 1.4         | 3.46          |
| Sinh  | [-88.722801,88.722801]<br />[-88, 88] for Sleef 3.5 ULP      | 3.4   | 1           | 3.4           |
| Cosh  | [-88, 88]                                                    | N/A   | 0.503       | 1.45          |
| Tanh  | [-FLT_MAX, FLT_MAX]                                          | 2.86  | 1           | 2.86          |
| Asin  | [-1, 1]                                                      | 2.47  | 0.981       | 2.43          |
| Acos  | [-1, 1]                                                      | 1.28  | 0.694       | 1.28          |
| Atan  | [-FLT_MAX, FLT_MAX]                                          | 2.65  | 0.938       | 2.64          |
| Asinh | [-FLT_MAX, FLT_MAX] for hwy<br />[sqrt(-FLT_MAX), sqrt(FLT_MAX)] for sleef | 3.27  | 1           | N/A           |
| Acosh | [1, FLT_MAX] for hwy<br />[1, sqrt(FLT_MAX)] for sleef       | 3.43  | 0.512       | N/A           |
| Atanh | [-1, 1]                                                      | 3.22  | 1           | N/A           |



#### Precision (all floats)

| Op    | Hwy   | Sleef 1 ULP | Sleef 3.5 ULP |
| ----- | ----- | ----------- | ------------- |
| Exp   | 1.03  | 0.988       | N/A           |
| Expm1 | 3.55  | 1           | N/A           |
| Log   | 0.995 | 0.995       | N/A           |
| Log1p | 2.47  | Inf         | N/A           |
| Log2  | 1.87  | 0.992       | N/A           |
| Sin   | Inf   | 1.09        | 2.47          |
| Cos   | Inf   | 1.07        | 2.49          |
| Tan   | N/A   | 1.4         | 3.46          |
| Sinh  | 3.4   | Inf         | Inf           |
| Cosh  | N/A   | Inf         | Inf           |
| Tanh  | 2.86  | 1           | 2.86          |
| Asin  | 2.47  | 0.981       | 2.43          |
| Acos  | 1.28  | 0.694       | 1.28          |
| Atan  | 2.65  | 0.938       | 2.64          |
| Asinh | 3.27  | Inf         | N/A           |
| Acosh | 3.43  | Inf         | N/A           |
| Atanh | 3.22  | 1           | N/A           |



### Speed

Notes:

- Sleef runs more slowly on large inputs methods for sin, cos, and tan. The 1 ULP has a single fallback method, while 3.5 ULP has two fallback methods. The benchmarks for these functions show both a narrow range (-125, 125) and a wide range [-39000, 39000] in the timing table
- The translated version of Sleef sometimes runs more slowly than the original Sleef. One cause I have found for this is that `NearestInt` in hwy performs additional checks sleef can omit without any loss in mathematical precision.
- All results were measured using AVX2 on a Framework laptop with i5-1240P processor. The benchmark task was to repeatedly apply the function on 4,096 random floats with memory clobbering between iterations to avoid optimizing out the code

Units are ns/float, then time relative to best.

| Op     | Range tested                    | Hwy           | Translated 1 ULP | Sleef 1 ULP     | Translated 3.5 ULP | Sleef 3.5 ULP |
| ------ | ------------------------------- | ------------- | ---------------- | --------------- | ------------------ | ------------- |
| Exp    | [-FLT_MAX, 104]                 | 0.5 ns (127%) | 0.4 ns (118%)    | 0.4 ns (100%)   | N/A                | N/A           |
| Expm1  | [-FLT_MAX, 104]                 | 0.6 ns (100%) | 2.1 ns (375%)    | 1.8 ns (322%)   | N/A                | N/A           |
| Log    | (0, FLT_MAX]                    | 0.4 ns (100%) | 1.2 ns (279%)    | 1.2 ns (264%)   | N/A                | N/A           |
| Log1p  | [0, 1e38]                       | 0.4 ns (100%) | 1.0 ns (240%)    | 1.0 ns (223%)   | N/A                | N/A           |
| Log2   | (0, FLT_MAX]                    | 0.4 ns (100%) | 1.2 ns (286%)    | 1.2 ns (280%)   | N/A                | N/A           |
| Sin    | (-125, 125)                     | 0.5 ns (118%) | 0.9 ns (239%)    | 0.8 ns (214%)   | 0.4 ns (100%)      | 0.4 ns (103%) |
| Sin    | [-39000, 39000]                 | 0.3 ns (100%) | 3.7 ns (1080%)   | 3.5 ns (1018%)  | 0.6 ns (161%)      | 0.7 ns (205%) |
| Cos    | (-125, 125)                     | 0.5 ns (121%) | 1.1 ns (266%)    | 0.9 ns (222%)   | 0.4 ns (100%)      | 0.4 ns (102%) |
| Cos v2 | [-39000, 39000]                 | 0.3 ns (100%) | 4.2 ns (1228%)   | 4.6 ns (1349%)  | 0.6 ns (191%)      | 0.6 ns (181%) |
| Tan    | (-125, 125)                     | N/A           | 4.2 ns (619%)    | 4.5 ns (669%)   | 0.7 ns (100%)      | 0.7 ns (107%) |
| Tan v2 | [-39000, 39000]                 | N/A           | 1.6 ns (239%)    | 1.6 ns (234%)   | 0.7 ns (100%)      | 0.8 ns (118%) |
| Sinh   | [-88, 88]                       | 0.8 ns (100%) | 8.0 ns (1056%)   | 8.1 ns (1061%)  | 0.9 ns (119%)      | 0.8 ns (106%) |
| Cosh   | [-88, 88]                       | N/A           | 8.7 ns (387%)    | 7.4 ns (332%)   | 2.2 ns (100%)      | 3.2 ns (141%) |
| Tanh   | [-FLT_MAX, FLT_MAX]             | 0.9 ns (124%) | 2.6 ns (362%)    | 2.0 ns (281%)   | 0.9 ns (119%)      | 0.7 ns (100%) |
| Asin   | [-1, 1]                         | 0.4 ns (106%) | 1.1 ns (309%)    | 1.1 ns (309%)   | 0.4 ns (100%)      | 0.4 ns (102%) |
| Acos   | [-1, 1]                         | 0.5 ns (100%) | 1.5 ns (282%)    | 1.4 ns (268%)   | 0.6 ns (113%)      | 0.6 ns (118%) |
| Atan   | [-FLT_MAX, FLT_MAX]             | 5.6 ns (100%) | 33.2 ns (594%)   | 36.1 ns (646%)  | 6.3 ns (114%)      | 6.0 ns (107%) |
| Asinh  | [-sqrt(FLT_MAX), sqrt(FLT_MAX)] | 1.4 ns (100%) | 44.7 ns (3287%)  | 41.9 ns (3076%) | N/A                | N/A           |
| Acosh  | [1, sqrt(FLT_MAX)]              | 1.2 ns (100%) | 3.1 ns (258%)    | 3.2 ns (261%)   | N/A                | N/A           |
| Atanh  | (-1, 1)                         | 0.8 ns (100%) | 2.1 ns (269%)    | 2.3 ns (289%)   | N/A                | N/A           |

