// Copyright 2020 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <benchmark/benchmark.h>
#include <stdio.h>

#include <algorithm>
#include <cfloat>  // FLT_MAX
#include <cmath>   // std::abs
#include <limits>
#include <memory>
#include <random>

#include "define_dispatchers.h"

// clang-format off
//#undef HWY_TARGET_INCLUDE
//#define HWY_TARGET_INCLUDE "sleef_test.cc"
//#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/tests/test_util.h"
// clang-format on

#include <chrono>
void spin_wait_ms(int ms, int& counter) {
  auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start <
         std::chrono::milliseconds(ms)) {
    counter++;
  }
}

static void fill_float(float* x, size_t n, float min, float max) {
  std::mt19937 gen(125124);  // Standard mersenne_twister_engine with seed
  std::uniform_real_distribution<float> d(min, max);
  for (size_t i = 0; i < n; i++) {
    x[i] = d(gen);
  }
}

static void fill_double(double* x, size_t n, double min, double max) {
  std::mt19937 gen(125124);  // Standard mersenne_twister_engine with seed
  std::uniform_real_distribution<double> d(min, max);
  for (size_t i = 0; i < n; i++) {
    x[i] = d(gen);
  }
}

template <class... Args>
void BM_op_float(benchmark::State& state, Args&&... args) {
  auto args_tuple = std::make_tuple(std::move(args)...);
  auto func = std::get<0>(args_tuple);
  size_t N = std::get<1>(args_tuple);
  float min = std::get<2>(args_tuple);
  float max = std::get<3>(args_tuple);

  auto in = hwy::AllocateAligned<float>(N);
  auto out = hwy::AllocateAligned<float>(N);

  fill_float(in.get(), N, min, max);
  benchmark::ClobberMemory();
  for (auto _ : state) {
    func(in.get(), N, out.get());
    benchmark::ClobberMemory();
  }
}

template <class... Args>
void BM_op_double(benchmark::State& state, Args&&... args) {
  auto args_tuple = std::make_tuple(std::move(args)...);
  auto func = std::get<0>(args_tuple);
  size_t N = std::get<1>(args_tuple);
  double min = std::get<2>(args_tuple);
  double max = std::get<3>(args_tuple);

  auto in = hwy::AllocateAligned<double>(N);
  auto out = hwy::AllocateAligned<double>(N);

  fill_double(in.get(), N, min, max);
  benchmark::ClobberMemory();
  for (auto _ : state) {
    func(in.get(), N, out.get());
    benchmark::ClobberMemory();
  }
}

#if HWY_ONCE

#define BENCHMARK_ALL(OP, TYPE, range_start, range_end)                      \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Hwy, hwy::OP##Hwy<TYPE>, 1 << 12,      \
                    range_start, range_end);                                 \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Translated, hwy::OP##Translated<TYPE>, \
                    1 << 12, range_start, range_end);                        \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Sleef, hwy::OP##Sleef<TYPE>, 1 << 12,  \
                    range_start, range_end);                                 \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Std, OP##Std<TYPE>, 1 << 12,           \
                    range_start, range_end);

#define BENCHMARK_SLEEF_ONLY(OP, TYPE, range_start, range_end)               \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Translated, hwy::OP##Translated<TYPE>, \
                    1 << 12, range_start, range_end);                        \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Sleef, hwy::OP##Sleef<TYPE>, 1 << 12,  \
                    range_start, range_end);

#define BENCHMARK_ALL_NO_HWY(OP, TYPE, range_start, range_end)               \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Translated, hwy::OP##Translated<TYPE>, \
                    1 << 12, range_start, range_end);                        \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Sleef, hwy::OP##Sleef<TYPE>, 1 << 12,  \
                    range_start, range_end);                                 \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Std, OP##Std<TYPE>, 1 << 12,           \
                    range_start, range_end);

#define BENCHMARK_ALL_FAST(OP, TYPE, range_start, range_end)                 \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Hwy, hwy::OP##Hwy<TYPE>, 1 << 12,      \
                    range_start, range_end);                                 \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Translated, hwy::OP##Translated<TYPE>, \
                    1 << 12, range_start, range_end);                        \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Sleef, hwy::OP##Sleef<TYPE>, 1 << 12,  \
                    range_start, range_end);                                 \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##FastTranslated,                        \
                    hwy::OP##FastTranslated<TYPE>, 1 << 12, range_start,     \
                    range_end);                                              \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##FastSleef, hwy::OP##FastSleef<TYPE>,   \
                    1 << 12, range_start, range_end);                        \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Std, OP##Std<TYPE>, 1 << 12,           \
                    range_start, range_end);

#define BENCHMARK_ALL_FAST_NO_HWY(OP, TYPE, range_start, range_end)          \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Translated, hwy::OP##Translated<TYPE>, \
                    1 << 12, range_start, range_end);                        \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Sleef, hwy::OP##Sleef<TYPE>, 1 << 12,  \
                    range_start, range_end);                                 \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##FastTranslated,                        \
                    hwy::OP##FastTranslated<TYPE>, 1 << 12, range_start,     \
                    range_end);                                              \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##FastSleef, hwy::OP##FastSleef<TYPE>,   \
                    1 << 12, range_start, range_end);                        \
  BENCHMARK_CAPTURE(BM_op_##TYPE, OP##Std, OP##Std<TYPE>, 1 << 12,           \
                    range_start, range_end);

BENCHMARK_ALL(Exp, float, -10, 10);
BENCHMARK_ALL(Exp, double, -10, 10);
BENCHMARK_ALL_NO_HWY(Exp2, float, -10, 10);
BENCHMARK_ALL_NO_HWY(Exp2, double, -10, 10);
BENCHMARK_ALL_NO_HWY(Exp10, float, -10, 10);
BENCHMARK_ALL_NO_HWY(Exp10, double, -10, 10);
BENCHMARK_ALL(Expm1, float, -10, 10);
BENCHMARK_ALL(Expm1, double, -10, 10);

BENCHMARK_ALL_FAST(Log, float, 0, 10);
BENCHMARK_ALL_FAST(Log, double, 0, 10);
BENCHMARK_ALL(Log2, float, 0, 10);
BENCHMARK_ALL(Log2, double, 0, 10);
BENCHMARK_ALL_NO_HWY(Log10, float, 0, 10);
BENCHMARK_ALL_NO_HWY(Log10, double, 0, 10);
BENCHMARK_ALL(Log1p, float, 0, 10);
BENCHMARK_ALL(Log1p, double, 0, 10);
// Skip pow
BENCHMARK_ALL_FAST_NO_HWY(Sqrt, float, 0, 10);
BENCHMARK_ALL_FAST_NO_HWY(Sqrt, double, 0, 10);
BENCHMARK_ALL_FAST_NO_HWY(Cbrt, float, -10, 10);
BENCHMARK_ALL_FAST_NO_HWY(Cbrt, double, -10, 10);
// Skip Hypot

// Several tests for each trig function since Sleef uses some
// cutoffs to run faster if all inputs are in a smaller range
BENCHMARK_ALL_FAST(Sin, float, -125, 125);
BENCHMARK_ALL_FAST(Sin, float, -39000, 39000);
BENCHMARK_ALL_FAST(Sin, float, -1e6, 1e6);
BENCHMARK_ALL_FAST(Cos, float, -125, 125);
BENCHMARK_ALL_FAST(Cos, float, -39000, 39000);
BENCHMARK_ALL_FAST(Cos, float, -1e6, 1e6);
BENCHMARK_ALL_FAST_NO_HWY(Tan, float, -50, 50);
BENCHMARK_ALL_FAST_NO_HWY(Tan, float, -125, 125);
BENCHMARK_ALL_FAST_NO_HWY(Tan, float, -39000, 39000);
BENCHMARK_ALL_FAST_NO_HWY(Tan, float, -1e6, 1e6);

BENCHMARK_ALL_FAST(Sin, double, -15, 15);
BENCHMARK_ALL_FAST(Sin, double, -1e6, 1e6);
BENCHMARK_ALL_FAST(Sin, double, -1e17, 1e17);
BENCHMARK_ALL_FAST(Cos, double, -15, 15);
BENCHMARK_ALL_FAST(Cos, double, -1e6, 1e6);
BENCHMARK_ALL_FAST(Cos, double, -1e17, 1e17);
BENCHMARK_ALL_FAST_NO_HWY(Tan, double, -15, 15);
BENCHMARK_ALL_FAST_NO_HWY(Tan, double, -1e6, 1e6);
BENCHMARK_ALL_FAST_NO_HWY(Tan, double, -1e13, 1e13);
BENCHMARK_ALL_FAST_NO_HWY(Tan, double, -1e17, 1e17);

BENCHMARK_ALL_FAST(Asin, float, -1, 1);
BENCHMARK_ALL_FAST(Asin, double, -1, 1);
BENCHMARK_ALL_FAST(Acos, float, -1, 1);
BENCHMARK_ALL_FAST(Acos, double, -1, 1);
BENCHMARK_ALL_FAST(Atan, float, -FLT_MAX, FLT_MAX);
BENCHMARK_ALL_FAST(Atan, double, -DBL_MAX, DBL_MAX);
// Skip Atan2
// Skip SinCos and SinCosPi

BENCHMARK_SLEEF_ONLY(SinPi, float, -10, 10);
BENCHMARK_SLEEF_ONLY(SinPi, double, -10, 10);
BENCHMARK_SLEEF_ONLY(CosPi, float, -10, 10);
BENCHMARK_SLEEF_ONLY(CosPi, double, -10, 10);

BENCHMARK_ALL_FAST(Sinh, float, -10, 10);
BENCHMARK_ALL_FAST(Sinh, double, -10, 10);
BENCHMARK_ALL_FAST_NO_HWY(Cosh, float, -10, 10);
BENCHMARK_ALL_FAST_NO_HWY(Cosh, double, -10, 10);
BENCHMARK_ALL_FAST(Tanh, float, -10, 10);
BENCHMARK_ALL_FAST(Tanh, double, -10, 10);

BENCHMARK_ALL(Asinh, float, -10, 10);
BENCHMARK_ALL(Asinh, double, -10, 10);
BENCHMARK_ALL(Acosh, float, 1, 10);
BENCHMARK_ALL(Acosh, double, -10, 10);
BENCHMARK_ALL(Atanh, float, -1, 1);
BENCHMARK_ALL(Atanh, double, -1, 1);

BENCHMARK_ALL_NO_HWY(Erf, float, -2.5, 2.5);
BENCHMARK_ALL_NO_HWY(Erf, float, -10, 10);
BENCHMARK_ALL_NO_HWY(Erf, double, -2.5, 2.5);
BENCHMARK_ALL_NO_HWY(Erf, double, -10, 10);
BENCHMARK_ALL_NO_HWY(Erfc, float, -10, 10);
BENCHMARK_ALL_NO_HWY(Erfc, double, -10, 10);

// GammaQF / QD have 0.5 cutoffs for an if/else

BENCHMARK_ALL_NO_HWY(Gamma, float, 0, 10);
BENCHMARK_ALL_NO_HWY(Gamma, float, 0, 0.5);
BENCHMARK_ALL_NO_HWY(Gamma, float, -10, 0);
BENCHMARK_ALL_NO_HWY(Gamma, double, 0, 10);
BENCHMARK_ALL_NO_HWY(Gamma, double, 0, 0.5);
BENCHMARK_ALL_NO_HWY(Gamma, double, -10, 0);

BENCHMARK_ALL_NO_HWY(LogGamma, float, 0, 10);
BENCHMARK_ALL_NO_HWY(LogGamma, float, 0, 0.5);
BENCHMARK_ALL_NO_HWY(LogGamma, float, -10, 0);
BENCHMARK_ALL_NO_HWY(LogGamma, double, 0, 10);
BENCHMARK_ALL_NO_HWY(LogGamma, double, 0, 0.5);
BENCHMARK_ALL_NO_HWY(LogGamma, double, -10, 0);


BENCHMARK_MAIN();

#endif
